"""Background memory/skill review — fork the agent to evaluate the turn.

After every turn, ``AIAgent.run_conversation`` may call
:func:`spawn_background_review` to fire off a daemon thread that replays
the conversation snapshot in a forked :class:`AIAgent` and asks itself
"should any skill/memory be saved or updated?".  Writes go straight to
the memory + skill stores.  Main conversation and prompt cache are never
touched.

The fork inherits the parent's live runtime (provider, model, base_url,
credentials, cached system prompt) so it hits the same prefix cache and
uses the same auth.  It runs with a tool whitelist limited to memory and
skill management tools; everything else is denied at runtime.

See the ``hermes-agent-dev`` skill (``references/self-improvement-loop.md``)
for invariants and PR review criteria.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auxiliary-model routing + context-budget helpers (cost reduction).
#
# The review fork historically ran on the parent's main model so it could
# reuse the parent's warmed system-prompt prefix cache (PR #17276). That is
# the cheapest path WHEN the user is on a cheap main model, but on an
# expensive reasoning model (gpt-5.5, opus) every review pays full price for
# the whole replayed transcript across up to 16 iterations.
#
# These helpers let a user route the review fork to a cheap aux model via
# ``auxiliary.background_review.{provider,model}`` (same plumbing as the
# curator and every other aux task). When routed OFF the parent model the
# prefix-cache share is dropped (different cache key) and the replayed
# context is trimmed to a digest, because the fork no longer benefits from
# matching the parent's cached prefix and we want its input bounded.
# ---------------------------------------------------------------------------


def _resolve_review_runtime(agent: Any) -> Dict[str, Any]:
    """Resolve provider/model/credentials for the background-review fork.

    Returns a dict with keys: provider, model, api_key, base_url, api_mode,
    routed (bool — True when routed to a model OTHER than the parent's main
    model, which means the prefix-cache share must be dropped).

    Precedence:
      1. ``auxiliary.background_review.{provider,model}`` when provider is set
         non-"auto" AND model is non-empty → route to that aux model.
      2. Otherwise → inherit the parent's live runtime (historical behaviour),
         preserving the codex_app_server → codex_responses downgrade so the
         agent-loop tools dispatch.
    """
    # Parent live runtime — the historical default, and the fallback.
    parent_runtime = agent._current_main_runtime()
    parent_api_mode = parent_runtime.get("api_mode") or None
    if parent_api_mode == "codex_app_server":
        parent_api_mode = "codex_responses"
    parent = {
        "provider": agent.provider,
        "model": agent.model,
        "api_key": parent_runtime.get("api_key") or None,
        "base_url": parent_runtime.get("base_url") or None,
        "api_mode": parent_api_mode,
        "routed": False,
        "extra_body": None,
    }

    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return parent

    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task = aux.get("background_review", {}) if isinstance(aux.get("background_review"), dict) else {}
    task_provider = (str(task.get("provider", "")).strip() or None)
    task_model = (str(task.get("model", "")).strip() or None)
    task_base_url = (str(task.get("base_url", "")).strip() or None)
    task_api_key = (str(task.get("api_key", "")).strip() or None)
    extra_body = task.get("extra_body") if isinstance(task.get("extra_body"), dict) else None

    # Only route when a concrete provider+model is configured. "auto" or an
    # empty model means "use the main chat model" — keep the cache-share path.
    if not (task_provider and task_provider != "auto" and task_model):
        return parent

    # If the configured aux model is byte-identical to the parent model on the
    # same provider, there is no benefit to dropping the cache-share — treat it
    # as the parent path.
    same_as_parent = (
        task_provider == (agent.provider or "")
        and task_model == (agent.model or "")
    )
    if same_as_parent:
        return parent

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        rp = resolve_runtime_provider(
            requested=task_provider,
            target_model=task_model,
            explicit_api_key=task_api_key,
            explicit_base_url=task_base_url,
        )
        return {
            "provider": rp.get("provider") or task_provider,
            "model": task_model,
            "api_key": rp.get("api_key"),
            "base_url": rp.get("base_url"),
            "api_mode": rp.get("api_mode"),
            "routed": True,
            "extra_body": extra_body,
        }
    except Exception as e:
        logger.debug(
            "Background-review aux routing failed (%s); using main model", e,
            exc_info=True,
        )
        return parent


def _read_review_budget() -> Dict[str, Any]:
    """Read snapshot context-budget knobs from auxiliary.background_review."""
    out = {"max_context_tokens": 48000, "digest_tail_messages": 24, "prepass": False}
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
        task = aux.get("background_review", {}) if isinstance(aux.get("background_review"), dict) else {}
        if isinstance(task.get("max_context_tokens"), int):
            out["max_context_tokens"] = max(0, task["max_context_tokens"])
        if isinstance(task.get("digest_tail_messages"), int):
            out["digest_tail_messages"] = max(1, task["digest_tail_messages"])
        if isinstance(task.get("prepass"), bool):
            out["prepass"] = task["prepass"]
    except Exception:
        pass
    return out


def _rough_token_count(messages: List[Dict]) -> int:
    """Cheap char/2.75 token estimate over a message list (no tokenizer dep)."""
    total = 0
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    t = block.get("text")
                    if isinstance(t, str):
                        total += len(t)
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    total += len(args)
    return int(total / 2.75)


def _msg_text(m: Dict) -> str:
    """Extract plain text from a message's content (str or block list)."""
    c = m.get("content")
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        return " ".join(
            b.get("text", "") for b in c if isinstance(b, dict)
        ).strip()
    return ""


# Lexical cues for turns that tend to carry a durable learning signal. Used to
# FOREGROUND signal-bearing turns in the digest rather than truncating them into
# noise. This is a salience heuristic, not a classifier — the review model still
# makes the final save decision; we just make sure the candidate turns survive
# the digest verbatim instead of being crushed to 250 chars alongside filler.
_USER_SIGNAL_CUES = (
    # corrections / frustration / style + workflow preferences
    "stop ", "don't ", "do not ", "never ", "always ", "i hate", "too verbose",
    "too long", "just give me", "why are you", "you keep", "you always",
    "from now on", "remember", "prefer", "instead of", "not like", "no —",
    "no -", "that's wrong", "thats wrong", "incorrect", "i told you",
    "i've told you", "ive told you", "make that", "going forward",
    # durable persona / environment facts
    "for context", "my setup", "i run", "i use", "i work", "my stack",
    "assume that", "tailor", "i'm on", "im on", "we use", "our ",
)
_ASSISTANT_SIGNAL_CUES = (
    "the fix", "fixed it", "root cause", "the trick", "the issue was",
    "turned out", "the reusable", "recipe", "the approach", "workaround",
    "so the fix", "that fixed", "the cause", "the bug was",
)


def _is_signal_turn(m: Dict) -> bool:
    role = m.get("role")
    text = _msg_text(m).lower()
    if not text:
        return False
    if role == "user":
        return any(cue in text for cue in _USER_SIGNAL_CUES)
    if role == "assistant":
        return any(cue in text for cue in _ASSISTANT_SIGNAL_CUES)
    return False


def _digest_old_messages(old: List[Dict]) -> str:
    """Summarise dropped older turns as a compact, SIGNAL-AWARE text digest.

    A naive digest that uniformly truncates every old turn to ~250 chars buries
    a one-line correction ("stop being verbose") under hundreds of filler lines —
    the review model's attention is then diluted across the haystack exactly as
    it is on the full transcript, so the digest saves cost but cannot *improve*
    capture.

    This version splits the work:

      1. FOREGROUND — turns whose text matches a learning-signal cue
         (corrections, preferences, "remember", durable persona/env facts,
         fix/technique statements) are surfaced near-verbatim under an explicit
         "POTENTIAL LEARNING SIGNALS" header, so the review model sees the
         needle pulled out of the haystack.
      2. ARC — every old turn still gets a short who-said-what / which-tools
         line so the conversation's shape is preserved for context.

    Deterministic, no LLM call. The review model still makes the final
    save/no-op decision; we only raise the salience of the candidate turns.
    """
    signals: List[str] = []
    arc: List[str] = []
    for m in old or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = _msg_text(m).replace("\n", " ")
        if role == "user":
            if text:
                arc.append(f"USER: {text[:300]}")
                if _is_signal_turn(m):
                    signals.append(f"USER said: \"{text[:500]}\"")
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                names = [
                    (tc.get("function") or {}).get("name", "?")
                    for tc in tcs if isinstance(tc, dict)
                ]
                arc.append(f"ASSISTANT[tools: {', '.join(names)}]")
            if text:
                arc.append(f"ASSISTANT: {text[:200]}")
                if _is_signal_turn(m):
                    signals.append(f"ASSISTANT noted: \"{text[:400]}\"")

    parts = [
        "[Earlier conversation digest — older turns summarised to bound review "
        "cost. The recent turns follow VERBATIM after this digest.]"
    ]
    if signals:
        parts.append(
            "\nPOTENTIAL LEARNING SIGNALS from earlier turns (surfaced so they "
            "aren't lost in summarisation — evaluate each for a memory/skill "
            "save just as you would a verbatim turn):\n"
            + "\n".join(f"  • {s}" for s in signals)
        )
    parts.append("\nConversation arc (compressed):\n" + "\n".join(arc))
    return "\n".join(parts)


def _extract_signal_brief(
    old: List[Dict],
    agent_runtime: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Use a cheap auxiliary model to extract + rank durable learning signals.

    The expensive review fork's failure mode on a long noisy session is
    attention dilution: a one-line correction or persona fact is buried under
    hundreds of unrelated turns. This pre-pass spends a small, cheap call on the
    ``background_review`` auxiliary task to do the haystack search — read the
    dropped older turns and return a SHORT ranked list of candidate learnings
    (durable user facts/preferences, corrections, reusable techniques),
    explicitly flagging anything the user asked to remember and anything later
    retracted. The brief is prepended to the digest so the review fork judges a
    clean, complete candidate set.

    Returns the brief text, or None on any failure / empty result (caller falls
    back to the lexical digest). Never raises.
    """
    if not old:
        return None
    transcript = _digest_old_messages(old)  # compact arc + lexical signals
    try:
        from agent.auxiliary_client import call_llm
        prompt = (
            "You are pre-screening an earlier slice of a conversation for a "
            "background self-improvement review. List ONLY durable, reusable "
            "learnings worth saving to long-term memory or a skill:\n"
            "  • durable user facts / environment / identity / standing "
            "constraints (esp. anything the user asked you to remember, assume, "
            "or tailor to);\n"
            "  • explicit corrections of style, workflow, or approach;\n"
            "  • non-trivial techniques / fixes / debugging paths.\n\n"
            "Rank most-important first. For each, one line: TYPE | the learning, "
            "verbatim-ish. EXCLUDE one-off task chatter, transient errors that "
            "resolved, and anything the user RETRACTED (note retractions "
            "explicitly as 'RETRACTED: ...'). If nothing qualifies, reply "
            "exactly 'NONE'.\n\nConversation slice:\n" + transcript[:24000]
        )
        resp = call_llm(
            task="background_review",
            main_runtime=agent_runtime,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=600,
            timeout=60,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or text.strip().upper() == "NONE":
            return None
        return (
            "PRE-SCREENED LEARNING CANDIDATES (a fast model surfaced these from "
            "the earlier turns so they aren't lost — verify each against the "
            "conversation and save the ones that genuinely warrant a memory or "
            "skill; ignore any that don't):\n" + text
        )
    except Exception as e:
        logger.debug("background-review pre-pass failed: %s", e, exc_info=True)
        return None


def _build_review_history(
    messages_snapshot: List[Dict],
    max_context_tokens: int,
    digest_tail_messages: int,
    use_prepass: bool = False,
    agent_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Return the (possibly trimmed) conversation_history for the review fork.

    When the snapshot is within budget (or the guard is disabled), the full
    snapshot is returned unchanged. Otherwise the recent ``digest_tail_messages``
    are kept verbatim and the older prefix is collapsed into a single synthetic
    user-role digest message, preserving role alternation at the seam.

    When ``use_prepass`` is set, a cheap auxiliary model first scans the dropped
    older turns and emits a ranked brief of candidate learnings, which is
    prepended to the digest. This does the haystack search on a cheap model so
    the (expensive) review fork judges a clean, complete signal set rather than
    re-scanning noise — the lever for *improving* capture, not just preserving
    it. Falls back silently to the lexical digest if the pre-pass yields nothing.

    The second return value is metadata for logging/metrics.
    """
    full_tokens = _rough_token_count(messages_snapshot)
    meta = {
        "digested": False,
        "prepass_used": False,
        "full_tokens": full_tokens,
        "replayed_tokens": full_tokens,
        "full_messages": len(messages_snapshot or []),
        "replayed_messages": len(messages_snapshot or []),
    }
    if max_context_tokens <= 0 or full_tokens <= max_context_tokens:
        return list(messages_snapshot or []), meta

    msgs = list(messages_snapshot or [])
    if len(msgs) <= digest_tail_messages:
        return msgs, meta

    tail = msgs[-digest_tail_messages:]
    # Don't open the verbatim tail on a `tool` message — that would orphan it
    # from its assistant(tool_calls) parent and break alternation. Walk back to
    # the nearest assistant or user boundary.
    while tail and isinstance(tail[0], dict) and tail[0].get("role") == "tool":
        digest_tail_messages += 1
        if len(msgs) <= digest_tail_messages:
            return msgs, meta
        tail = msgs[-digest_tail_messages:]

    old = msgs[:-len(tail)]
    brief = _extract_signal_brief(old, agent_runtime=agent_runtime) if use_prepass else None
    digest_body = _digest_old_messages(old)
    if brief:
        digest_body = brief + "\n\n" + digest_body
        meta["prepass_used"] = True
    digest_msg = {"role": "user", "content": digest_body}
    history = [digest_msg] + tail
    meta["digested"] = True
    meta["replayed_tokens"] = _rough_token_count(history)
    meta["replayed_messages"] = len(history)
    return history, meta


def _read_skill_cadence() -> Dict[str, Any]:
    """Read skill-review cadence knobs from the skills config section."""
    out = {
        "skip_tool_free_turns": True,
        "adaptive_backoff": True,
        "adaptive_backoff_after": 3,
    }
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        sk = cfg.get("skills", {}) if isinstance(cfg.get("skills"), dict) else {}
        if isinstance(sk.get("skip_tool_free_turns"), bool):
            out["skip_tool_free_turns"] = sk["skip_tool_free_turns"]
        if isinstance(sk.get("adaptive_backoff"), bool):
            out["adaptive_backoff"] = sk["adaptive_backoff"]
        if isinstance(sk.get("adaptive_backoff_after"), int) and sk["adaptive_backoff_after"] > 0:
            out["adaptive_backoff_after"] = sk["adaptive_backoff_after"]
    except Exception:
        pass
    return out


def effective_skill_interval(base_interval: int, noop_streak: int) -> int:
    """Compute the backed-off skill-review interval for a quiet session.

    Once ``noop_streak`` reaches ``adaptive_backoff_after`` consecutive
    "Nothing to save." reviews, the effective interval grows (×2 per
    additional whole multiple of the threshold) up to a 4× cap, so a session
    that keeps producing nothing stops paying for no-op review forks. Any
    review that saves resets the streak (see ``_run_review_in_thread``).
    Returns ``base_interval`` unchanged when backoff is disabled or the
    streak is below the threshold.
    """
    if base_interval <= 0:
        return base_interval
    cad = _read_skill_cadence()
    if not cad["adaptive_backoff"]:
        return base_interval
    after = cad["adaptive_backoff_after"]
    if noop_streak < after:
        return base_interval
    multiplier = min(4, 1 + (noop_streak // after))
    return base_interval * multiplier


# Module-level type import used by the helpers above.


# Review-prompt strings — used by ``spawn_background_review_thread`` to build
# the user-message that the forked review agent receives.  AIAgent exposes
# them as class attributes (``_MEMORY_REVIEW_PROMPT`` etc.) for back-compat;
# the actual text lives here so future edits are one-place.
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. "
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)

_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update the skill library. Be "
    "ACTIVE — most sessions produce at least one skill update, even if "
    "small. A pass that does nothing is a missed learning opportunity, "
    "not a neutral outcome.\n\n"
    "Target shape of the library: CLASS-LEVEL skills, each with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries. This "
    "shapes HOW you update, not WHETHER you update.\n\n"
    "Signals to look for (any one of these warrants action):\n"
    "  • User corrected your style, tone, format, legibility, or "
    "verbosity. Frustration signals like 'stop doing X', 'this is too "
    "verbose', 'don't format like this', 'why are you explaining', "
    "'just give me the answer', 'you always do Y and I hate it', or an "
    "explicit 'remember this' are FIRST-CLASS skill signals, not just "
    "memory signals. Update the relevant skill(s) to embed the "
    "preference so the next session starts already knowing.\n"
    "  • User corrected your workflow, approach, or sequence of steps. "
    "Encode the correction as a pitfall or explicit step in the skill "
    "that governs that class of task.\n"
    "  • Non-trivial technique, fix, workaround, debugging path, or "
    "tool-usage pattern emerged that a future session would benefit "
    "from. Capture it.\n"
    "  • A skill that got loaded or consulted this session turned out "
    "to be wrong, missing a step, or outdated. Patch it NOW.\n\n"
    "Preference order — prefer the earliest action that fits, but do "
    "pick one when a signal above fired:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the "
    "conversation for skills the user loaded via /skill-name or you "
    "read via skill_view. If any of them covers the territory of the "
    "new learning, PATCH that one first. It is the skill that was in "
    "play, so it's the right one to extend.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). "
    "If no loaded skill fits but an existing class-level skill does, "
    "patch it. Add a subsection, a pitfall, or broaden a trigger.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be "
    "packaged with three kinds of support files — use the right "
    "directory per kind:\n"
    "     • `references/<topic>.md` — session-specific detail (error "
    "transcripts, reproduction recipes, provider quirks) AND "
    "condensed knowledge banks: quoted research, API docs, external "
    "authoritative excerpts, or domain notes you found while working "
    "on the problem. Write it concise and for the value of the task, "
    "not as a full mirror of upstream docs.\n"
    "     • `templates/<name>.<ext>` — starter files meant to be "
    "copied and modified (boilerplate configs, scaffolding, a "
    "known-good example the agent can `reproduce with modifications`).\n"
    "     • `scripts/<name>.<ext>` — statically re-runnable actions "
    "the skill can invoke directly (verification scripts, fixture "
    "generators, deterministic probes, anything the agent should run "
    "rather than hand-type each time).\n"
    "     Add support files via skill_manage action=write_file with "
    "file_path starting 'references/', 'templates/', or 'scripts/'. "
    "The umbrella's SKILL.md should gain a one-line pointer to any "
    "new support file so future agents know it exists.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA SKILL when no existing "
    "skill covers the class. The name MUST be at the class level. "
    "The name MUST NOT be a specific PR number, error string, feature "
    "codename, library-alone name, or 'fix-X / debug-Y / audit-Z-today' "
    "session artifact. If the proposed name only makes sense for "
    "today's task, it's wrong — fall back to (1), (2), or (3).\n\n"
    "User-preference embedding (important): when the user expressed a "
    "style/format/workflow preference, the update belongs in the "
    "SKILL.md body, not just in memory. Memory captures 'who the user "
    "is and what the current situation and state of your operations "
    "are'; skills capture 'how to do this class of task for this "
    "user'. When they complain about how you handled a task, the "
    "skill that governs that task needs to carry the lesson.\n\n"
    "If you notice two existing skills that overlap, note it in your "
    "reply — the background curator handles consolidation at scale.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "Pinned skills (marked via 'hermes curator pin') CAN be improved — "
    "pin only blocks deletion/archive/consolidation by the curator, not "
    "content updates. Patch them when a pitfall or missing step turns up, "
    "same as any other agent-created skill.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture (these become persistent self-imposed constraints "
    "that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "'Nothing to save.' is a real option but should NOT be the "
    "default. If the session ran smoothly with no corrections and "
    "produced no new technique, just say 'Nothing to save.' and stop. "
    "Otherwise, act."
)

_COMBINED_REVIEW_PROMPT = (
    "Review the conversation above and update two things:\n\n"
    "**Memory**: who the user is. Did the user reveal persona, "
    "desires, preferences, personal details, or expectations about "
    "how you should behave? Save facts about the user and durable "
    "preferences with the memory tool. Be ACTIVE here too: a user who "
    "states a standing fact or constraint — their environment/stack "
    "('I run a 3-node Proxmox cluster', 'everything must stay "
    "self-hosted/offline'), their identity/role, or an explicit "
    "'keep this in mind / assume this / tailor to this / remember "
    "this' — is handing you a FIRST-CLASS memory save. Save it even "
    "when it appeared early and the rest of the session moved on to "
    "unrelated work; durable context does not expire because the "
    "conversation drifted. Prefer the most durable, decision-shaping "
    "facts over transient trivia (today's weather, a one-off path): "
    "when several candidates compete for limited memory, keep the "
    "standing constraint, drop the ephemeral note.\n\n"
    "**Skills**: how to do this class of task. Be ACTIVE — most "
    "sessions produce at least one skill update. A pass that does "
    "nothing is a missed learning opportunity, not a neutral outcome.\n\n"
    "Target shape of the skill library: CLASS-LEVEL skills with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries.\n\n"
    "Signals that warrant a skill update (any one is enough):\n"
    "  • User corrected your style, tone, format, legibility, "
    "verbosity, or approach. Frustration is a FIRST-CLASS skill "
    "signal, not just a memory signal. 'stop doing X', 'don't format "
    "like this', 'I hate when you Y' — embed the lesson in the skill "
    "that governs that task so the next session starts fixed.\n"
    "  • Non-trivial technique, fix, workaround, or debugging path "
    "emerged.\n"
    "  • A skill that was loaded or consulted turned out wrong, "
    "missing, or outdated — patch it now.\n\n"
    "Preference order for skills — pick the earliest that fits:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Check what skills were "
    "loaded via /skill-name or skill_view in the conversation. If one "
    "of them covers the learning, PATCH it first. It was in play; "
    "it's the right place.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to "
    "find the right one). Patch it.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella via "
    "skill_manage action=write_file. Three kinds: "
    "`references/<topic>.md` for session-specific detail OR condensed "
    "knowledge banks (quoted research, API docs excerpts, domain "
    "notes) written concise and task-focused; `templates/<name>.<ext>` "
    "for starter files meant to be copied and modified; "
    "`scripts/<name>.<ext>` for statically re-runnable actions "
    "(verification, fixture generators, probes). Add a one-line "
    "pointer in SKILL.md so future agents find them.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. "
    "Name at the class level — NOT a PR number, error string, "
    "codename, library-alone name, or 'fix-X / debug-Y' session "
    "artifact. If the name only fits today's task, fall back to (1), "
    "(2), or (3).\n\n"
    "User-preference embedding: when the user complains about how "
    "you handled a task, update the skill that governs that task — "
    "memory alone isn't enough. Memory says 'who the user is and "
    "what the current situation and state of your operations are'; "
    "skills say 'how to do this class of task for this user'. Both "
    "should carry user-preference lessons when relevant.\n\n"
    "If you notice overlapping existing skills, mention it — the "
    "background curator handles consolidation.\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes, e.g. 'hermes-agent').\n"
    "  • Hub-installed skills (installed via 'hermes skills install').\n"
    "Pinned skills (marked via 'hermes curator pin') CAN be improved — "
    "pin only blocks deletion/archive/consolidation by the curator, not "
    "content updates. Patch them when a pitfall or missing step turns up, "
    "same as any other agent-created skill.\n"
    "If the only skills that need updating are protected, say\n"
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture as skills (these become persistent self-imposed "
    "constraints that bite you later when the environment changes):\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, post-migration path mismatches, 'command not found', "
    "unconfigured credentials, uninstalled packages. The user can fix "
    "these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('browser tools do not "
    "work', 'X tool is broken', 'cannot use Y from execute_code'). These "
    "harden into refusals the agent cites against itself for months "
    "after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended. If retrying worked, the lesson is the retry "
    "pattern, not the original failure.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "Act on whichever of the two dimensions has real signal. If "
    "genuinely nothing stands out on either, say 'Nothing to save.' "
    "and stop — but don't reach for that conclusion as a default."
)



def summarize_background_review_actions(
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
    notification_mode: str = "on",
) -> List[str]:
    """Build the human-facing action summary for a background review pass.

    Walks the review agent's session messages and collects successful memory
    and skill-management actions to surface to the user. Tool messages already
    present in ``prior_snapshot`` are skipped so stale inherited results are
    not re-surfaced as fresh background work (issue #14944).

    ``notification_mode`` controls display detail:
    - ``off``: return no actions.
    - ``on``: generic "Memory updated"/tool messages.
    - ``verbose``: include compact content previews from tool-call arguments.
    """
    mode = str(notification_mode or "on").lower()
    if mode == "off":
        return []
    verbose = mode == "verbose"

    existing_tool_call_ids = set()
    existing_tool_contents = set()
    for prior in prior_snapshot or []:
        if not isinstance(prior, dict) or prior.get("role") != "tool":
            continue
        tcid = prior.get("tool_call_id")
        if tcid:
            existing_tool_call_ids.add(tcid)
        else:
            content = prior.get("content")
            if isinstance(content, str):
                existing_tool_contents.add(content)

    # Map review-agent tool results back to the calls that produced them.  The
    # result JSON only says "Entry added"; the call arguments contain action,
    # target, and content previews.  Restricting to notify_tools also prevents
    # helper tools from surfacing as memory work just because they succeeded.
    notify_tools = {"memory", "skill_manage"}
    all_tool_call_ids: set = set()
    call_details: dict = {}
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            fn_name = fn.get("name", "")
            tcid = tc.get("id")
            if tcid:
                all_tool_call_ids.add(tcid)
            if fn_name not in notify_tools:
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            if tcid:
                call_details[tcid] = {
                    "tool": fn_name,
                    "action": args.get("action", "?"),
                    "target": args.get("target", "memory"),
                    "content": args.get("content", ""),
                    "old_text": args.get("old_text", ""),
                    "operations": args.get("operations") or [],
                    "name": args.get("name", ""),
                    "old_string": args.get("old_string", ""),
                    "new_string": args.get("new_string", ""),
                }

    actions: List[str] = []
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid in existing_tool_call_ids:
            continue
        if not tcid:
            content_str = msg.get("content")
            if isinstance(content_str, str) and content_str in existing_tool_contents:
                continue
        if tcid and all_tool_call_ids and tcid not in call_details:
            continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("success"):
            continue
        message = data.get("message", "")
        detail = call_details.get(tcid, {})
        target = data.get("target", "") or detail.get("target", "")
        is_skill = detail.get("tool") == "skill_manage"

        message_lower = message.lower()
        if not verbose:
            if "created" in message_lower:
                actions.append(message)
                continue
            if "updated" in message_lower:
                actions.append(message)
                continue
            if is_skill and "patched" in message_lower:
                actions.append(message)
                continue

        if is_skill:
            label = "Skill"
        elif target:
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
        else:
            continue

        if verbose:
            action = detail.get("action", "")
            content = detail.get("content", "")
            old_text = detail.get("old_text", "")
            skill_name = detail.get("name", "")
            operations = detail.get("operations") or []
            max_preview = 120
            if is_skill:
                change = data.get("_change", {})
                old_string = change.get("old", "") or detail.get("old_string", "")
                new_string = change.get("new", "") or detail.get("new_string", "")
                description = change.get("description", "")
                if action == "patch" and (old_string or new_string):
                    old_preview = old_string[:80].replace("\n", " ") + (
                        "…" if len(old_string) > 80 else ""
                    )
                    new_preview = new_string[:80].replace("\n", " ") + (
                        "…" if len(new_string) > 80 else ""
                    )
                    actions.append(
                        f"📝 Skill '{skill_name}' patched: "
                        f"\"{old_preview}\" → \"{new_preview}\""
                    )
                elif action == "create" and description:
                    actions.append(f"📝 Skill '{skill_name}' created: {description}")
                elif action == "edit" and description:
                    actions.append(f"📝 Skill '{skill_name}' rewritten: {description}")
                else:
                    actions.append(f"📝 {message}" if message else f"Skill {action}")
            elif operations:
                for op in operations:
                    op = op or {}
                    op_act = op.get("action", "")
                    op_content = (op.get("content") or "")
                    op_old = (op.get("old_text") or "")
                    if op_act == "add" and op_content:
                        preview = op_content[:max_preview] + ("…" if len(op_content) > max_preview else "")
                        actions.append(f"{label} ➕ {preview}")
                    elif op_act == "replace" and op_content:
                        preview = op_content[:max_preview] + ("…" if len(op_content) > max_preview else "")
                        actions.append(f"{label} ✏️ {preview}")
                    elif op_act == "remove" and op_old:
                        preview = op_old[:60] + ("…" if len(op_old) > 60 else "")
                        actions.append(f"{label} ➖ {preview}")
            elif action == "add" and content:
                preview = content[:max_preview] + ("…" if len(content) > max_preview else "")
                actions.append(f"{label} ➕ {preview}")
            elif action == "replace" and content:
                preview = content[:max_preview] + ("…" if len(content) > max_preview else "")
                actions.append(f"{label} ✏️ {preview}")
            elif action == "remove" and old_text:
                preview = old_text[:60] + ("…" if len(old_text) > 60 else "")
                actions.append(f"{label} ➖ {preview}")
            else:
                actions.append(f"{label} updated")
        elif (
            "added" in message_lower
            or "replaced" in message_lower
            or "removed" in message_lower
            or "applied" in message_lower
            or (target and "add" in message.lower())
            or "Entry added" in message
        ):
            actions.append(f"{label} updated")
    return actions


def build_memory_write_metadata(
    agent: Any,
    *,
    write_origin: Optional[str] = None,
    execution_context: Optional[str] = None,
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build provenance metadata for external memory-provider mirrors."""
    metadata: Dict[str, Any] = {
        "write_origin": write_origin or getattr(agent, "_memory_write_origin", "assistant_tool"),
        "execution_context": (
            execution_context
            or getattr(agent, "_memory_write_context", "foreground")
        ),
        "session_id": agent.session_id or "",
        "parent_session_id": agent._parent_session_id or "",
        "platform": agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
        "tool_name": "memory",
    }
    if task_id:
        metadata["task_id"] = task_id
    if tool_call_id:
        metadata["tool_call_id"] = tool_call_id
    return {k: v for k, v in metadata.items() if v not in {None, ""}}


def _run_review_in_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    prompt: str,
) -> None:
    """Worker function executed in the background-review daemon thread.

    Spawns a forked ``AIAgent`` inheriting the parent's runtime, runs the
    review prompt, and surfaces a compact action summary back to the user
    via ``agent._safe_print`` and ``agent.background_review_callback``.
    """
    # Local import to avoid a hard circular dep at module load.
    from run_agent import AIAgent
    from tools.terminal_tool import set_approval_callback as _set_approval_callback

    # Install a non-interactive approval callback on this worker
    # thread so any dangerous-command guard the review agent trips
    # resolves to "deny" instead of falling back to input() -- which
    # deadlocks against the parent's prompt_toolkit TUI (#15216).
    # Same pattern as _subagent_auto_deny in tools/delegate_tool.py.
    def _bg_review_auto_deny(command, description, **kwargs):
        logger.warning(
            "Background review auto-denied dangerous command: %s (%s)",
            command, description,
        )
        return "deny"
    try:
        _set_approval_callback(_bg_review_auto_deny)
    except Exception:
        pass

    review_agent = None
    review_messages: List[Dict] = []
    try:
        with open(os.devnull, "w", encoding="utf-8") as _devnull, \
             contextlib.redirect_stdout(_devnull), \
             contextlib.redirect_stderr(_devnull):
            # Inherit the parent agent's live runtime (provider, model,
            # base_url, api_key, api_mode) so the fork uses the exact
            # same credentials the main turn is using.  Without this,
            # AIAgent.__init__ re-runs auto-resolution from env vars,
            # which fails for OAuth-only providers, session-scoped
            # creds, or credential-pool setups where the resolver can't
            # reconstruct auth from scratch -- producing the spurious
            # "No LLM provider configured" warning at end of turn.
            #
            # _resolve_review_runtime() returns the parent's live runtime by
            # default (routed=False), or — when the user configured
            # auxiliary.background_review.{provider,model} to a cheap aux model
            # — that aux model's runtime (routed=True). The codex_app_server →
            # codex_responses downgrade (so agent-loop tools dispatch) is
            # applied inside the resolver for the parent path.
            _rt = _resolve_review_runtime(agent)
            _routed = bool(_rt.get("routed"))
            # skip_memory=True keeps the review fork from
            # touching external memory plugins (honcho, mem0,
            # supermemory, etc.).  Without it, the fork's
            # __init__ rebuilds its own _memory_manager from
            # config, scoped to the parent's session_id, and
            # run_conversation() then leaks the harness prompt
            # into the user's real memory namespace via three
            # ingestion sites: on_turn_start (cadence + turn
            # message), prefetch_all (recall query), and
            # sync_all (harness prompt + review output recorded
            # as a (user, assistant) turn pair).  Built-in
            # MEMORY.md / USER.md state is re-bound from the
            # parent below so memory(action="add") writes from
            # the review still land on disk; the review just
            # has zero side effects on external providers.
            # Match parent's toolset config so ``tools[]`` is byte-identical
            # in the request body — Anthropic's cache key includes it.
            # (The runtime whitelist below still restricts dispatch.)
            review_agent = AIAgent(
                model=_rt.get("model") or agent.model,
                max_iterations=16,
                quiet_mode=True,
                platform=agent.platform,
                provider=_rt.get("provider") or agent.provider,
                api_mode=_rt.get("api_mode"),
                base_url=_rt.get("base_url") or None,
                api_key=_rt.get("api_key") or None,
                credential_pool=getattr(agent, "_credential_pool", None),
                parent_session_id=agent.session_id,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                skip_memory=True,
            )
            review_agent._memory_write_origin = "background_review"
            review_agent._memory_write_context = "background_review"
            # The review fork pins the parent's cached system prompt and keeps
            # ``tools[]`` byte-identical to the parent so its outbound request
            # hits the same provider cache prefix (see the toolset-parity note
            # above). The between-turns MCP refresh in build_turn_context would
            # add late-connecting MCP tools to this fork and break that parity,
            # so opt the review fork out of it.
            review_agent._skip_mcp_refresh = True
            review_agent._memory_store = agent._memory_store
            review_agent._memory_enabled = agent._memory_enabled
            review_agent._user_profile_enabled = agent._user_profile_enabled
            review_agent._memory_nudge_interval = 0
            review_agent._skill_nudge_interval = 0
            # Suppress all status/warning emits from the fork so the
            # user only sees the final successful-action summary.
            # Without this, mid-review "Iteration budget exhausted",
            # rate-limit retries, compression warnings, and other
            # lifecycle messages bubble up through _emit_status ->
            # _vprint and leak past the stdout redirect (they go via
            # _print_fn/status_callback, which bypass sys.stdout).
            review_agent.suppress_status_output = True
            # Inherit the parent's cached system prompt verbatim so
            # the review fork's outbound HTTP request hits the same
            # Anthropic/OpenRouter prefix cache the parent warmed.
            # Without this, the fork rebuilds the system prompt from
            # scratch (fresh _hermes_now() timestamp, fresh
            # session_id, narrower toolset → different skills_prompt)
            # and the byte-exact prefix-cache key misses. See
            # issue #25322 and PR #17276 for the full analysis +
            # measured impact (~26% end-to-end cost reduction on
            # Sonnet 4.5).
            #
            # ONLY share the parent prefix when the fork runs on the parent's
            # model (the default "auto" path). When routed to a different aux
            # model via auxiliary.background_review.{provider,model}, the
            # parent's cached prompt is for the WRONG model/provider — the
            # cache key would miss regardless, and worse, the system prompt
            # carries model-specific guidance (tool enforcement, vendor
            # self-identification) that doesn't belong to the aux model. Let
            # the routed fork build its own correct system prompt.
            if not _routed:
                review_agent._cached_system_prompt = agent._cached_system_prompt
                # Defensive: pin session_start + session_id to the
                # parent's so any code path that re-renders parts of
                # the system prompt (compression, plugin hooks) still
                # produces byte-identical output. The cached-prompt
                # assignment above already short-circuits the normal
                # rebuild path, but these pins guarantee parity even
                # if a future code path bypasses the cache.
                review_agent.session_start = agent.session_start
            review_agent.session_id = agent.session_id
            # Never let the review fork compress. It shares the parent's
            # session_id, so if it won a compression race it would rotate the
            # parent into a NEW child that the gateway never adopts (the fork
            # is single-lifecycle and dies right after this run_conversation).
            # The foreground turn would then start from the stale parent and
            # compress it again, leaving the same parent with two sibling
            # children (issue #38727). Review also needs full context to
            # produce a good memory/skill summary — compressing would strip
            # detail. Both compression triggers in conversation_loop.py gate on
            # agent.compression_enabled, so this short-circuits both paths.
            review_agent.compression_enabled = False

            from model_tools import get_tool_definitions
            from hermes_cli.plugins import (
                set_thread_tool_whitelist,
                clear_thread_tool_whitelist,
            )

            review_whitelist = {
                t["function"]["name"]
                for t in get_tool_definitions(
                    enabled_toolsets=["memory", "skills"],
                    quiet_mode=True,
                )
            }
            set_thread_tool_whitelist(
                review_whitelist,
                deny_msg_fmt=(
                    "Background review denied non-whitelisted tool: "
                    "{tool_name}. Only memory/skill tools are allowed."
                ),
            )
            try:
                # Build the conversation history to replay into the fork.
                # - Routed to an aux model: the prefix-cache share is already
                #   gone, so trimming to a digest is pure cost win.
                # - Not routed (main-model path): keep the full snapshot so the
                #   message-level prefix cache still hits — UNLESS the snapshot
                #   blows past the hard token ceiling, in which case we digest
                #   anyway to bound the pathological large-session case (the
                #   ~296K-token incident this PR addresses). 0 disables.
                _budget = _read_review_budget()
                _max_ctx = _budget["max_context_tokens"]
                _full_tokens = _rough_token_count(messages_snapshot)
                _engage_digest = _routed or (
                    _max_ctx > 0 and _full_tokens > _max_ctx
                )
                if _engage_digest:
                    # Effective ceiling: the configured cap, or a sane default
                    # when routed with the guard disabled (routing means the
                    # cache-share is gone, so we always bound the replay).
                    _eff_ctx = _max_ctx if _max_ctx > 0 else 48000
                    _review_history, _digest_meta = _build_review_history(
                        messages_snapshot, _eff_ctx, _budget["digest_tail_messages"],
                        use_prepass=_budget["prepass"],
                        agent_runtime=agent._current_main_runtime(),
                    )
                    logger.info(
                        "bg-review context: routed=%s digested=%s prepass=%s "
                        "full=%d→replayed=%d tokens (%d→%d msgs)",
                        _routed, _digest_meta["digested"], _digest_meta.get("prepass_used"),
                        _digest_meta["full_tokens"], _digest_meta["replayed_tokens"],
                        _digest_meta["full_messages"], _digest_meta["replayed_messages"],
                    )
                else:
                    _review_history = messages_snapshot
                review_agent.run_conversation(
                    user_message=(
                        prompt
                        + "\n\nYou can only call memory and skill "
                        "management tools. Other tools will be denied "
                        "at runtime — do not attempt them."
                    ),
                    conversation_history=_review_history,
                )
            finally:
                clear_thread_tool_whitelist()

            # Snapshot review actions before teardown. close() is allowed to
            # clean per-session state, but the user-visible self-improvement
            # summary still needs the completed review agent's tool results.
            review_messages = list(getattr(review_agent, "_session_messages", []))

            # Tear down memory providers while stdout is still
            # redirected so background thread teardown (Honcho flush,
            # Hindsight sync, etc.) stays silent.  The finally block
            # below is a safety net for the exception path.
            try:
                review_agent.shutdown_memory_provider()
            except Exception:
                pass
            try:
                review_agent.close()
            except Exception:
                pass
            review_agent = None

        # Scan the review agent's messages for successful tool actions
        # and surface a compact summary to the user. Tool messages
        # already present in messages_snapshot must be skipped, since
        # the review agent inherits that history and would otherwise
        # re-surface stale "created"/"updated" messages from the prior
        # conversation as if they just happened (issue #14944).
        actions = summarize_background_review_actions(
            review_messages,
            messages_snapshot,
            notification_mode=getattr(agent, "memory_notifications", "on"),
        )

        # Adaptive cadence (idea ④): track the consecutive "Nothing to save."
        # streak on the parent so the trigger logic can back off the review
        # interval on quiet sessions. A review that DID save resets it.
        try:
            if actions:
                agent._review_noop_streak = 0
            else:
                agent._review_noop_streak = (
                    int(getattr(agent, "_review_noop_streak", 0)) + 1
                )
        except Exception:
            pass

        if actions:
            summary = " · ".join(dict.fromkeys(actions))
            agent._safe_print(
                f"  💾 Self-improvement review: {summary}"
            )
            _bg_cb = agent.background_review_callback
            if _bg_cb:
                try:
                    _bg_cb(
                        f"💾 Self-improvement review: {summary}"
                    )
                except Exception:
                    pass

    except Exception as e:
        logger.warning("Background memory/skill review failed: %s", e)
        agent._emit_auxiliary_failure("background review", e)
    finally:
        # Safety-net cleanup for the exception path.  Normal
        # completion already shut down inside redirect_stdout above.
        # Re-open devnull here so any teardown output (Honcho flush,
        # Hindsight sync, background thread joins) stays silent even
        # on the exception path where redirect_stdout already exited.
        if review_agent is not None:
            try:
                with open(os.devnull, "w", encoding="utf-8") as _fn, \
                     contextlib.redirect_stdout(_fn), \
                     contextlib.redirect_stderr(_fn):
                    try:
                        review_agent.shutdown_memory_provider()
                    except Exception:
                        pass
                    try:
                        review_agent.close()
                    except Exception:
                        pass
            except Exception:
                pass
        # Clear the approval callback on this bg-review thread so a
        # recycled thread-id doesn't inherit a stale reference.
        try:
            _set_approval_callback(None)
        except Exception:
            pass


def spawn_background_review_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    review_memory: bool = False,
    review_skills: bool = False,
):
    """Build the review thread target and prompt for a background review.

    Returns a ``(target, prompt)`` tuple.  The caller (``AIAgent._spawn_background_review``)
    owns the actual ``threading.Thread`` construction so test-level patches
    of ``run_agent.threading.Thread`` keep working.
    """
    # Pick the right prompt based on which triggers fired.  Allow per-agent
    # override (the prompts moved to module-level constants but old code paths
    # that set agent._MEMORY_REVIEW_PROMPT etc. directly keep working).
    if review_memory and review_skills:
        prompt = getattr(agent, "_COMBINED_REVIEW_PROMPT", _COMBINED_REVIEW_PROMPT)
    elif review_memory:
        prompt = getattr(agent, "_MEMORY_REVIEW_PROMPT", _MEMORY_REVIEW_PROMPT)
    else:
        prompt = getattr(agent, "_SKILL_REVIEW_PROMPT", _SKILL_REVIEW_PROMPT)

    def _target() -> None:
        _run_review_in_thread(agent, messages_snapshot, prompt)

    return _target, prompt


__all__ = [
    "_MEMORY_REVIEW_PROMPT",
    "_SKILL_REVIEW_PROMPT",
    "_COMBINED_REVIEW_PROMPT",
    "spawn_background_review_thread",
    "summarize_background_review_actions",
    "build_memory_write_metadata",
]
