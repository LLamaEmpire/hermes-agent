# HRM-T0a Step 11 — Host enablement prep (orchestrator profile)

**Status:** preparation only. No live changes applied. No real secrets in this
document. Live enablement requires explicit owner go-ahead and a Critic pass on
this artifact.

This packet captures the exact host state observed at preparation time, the
recommended enablement posture, and the diff/template set the operator will
apply when step 11 actually runs. It does NOT modify config, systemd, or
secrets.

---

## 1. Read-only discovery evidence

All commands below were executed read-only inside the worktree at
`/home/devos/.hermes/hermes-agent/.worktrees/hrm-t0a`.

### 1.1 Active profile and gateway service

```
$ systemctl --user list-units --type=service --all | grep -iE 'hermes|gateway'
  hermes-dashboard-proxy.service        loaded active   running oauth2-proxy — Google OAuth gate for Hermes Dashboard
  hermes-dashboard.service              loaded active   running Hermes Agent Dashboard - Web Mission Control (localhost only)
  hermes-gateway-orchestrator.service   loaded active   running Hermes Agent Gateway - Messaging Platform Integration
  hermes-gateway.service                loaded inactive dead    Hermes Agent Gateway - Messaging Platform Integration
```

The live gateway is the user-scope `hermes-gateway-orchestrator.service`. The
system-scope `hermes-gateway.service` is inactive.

```
$ systemctl --user show hermes-gateway-orchestrator.service \
    -p FragmentPath -p DropInPaths -p EnvironmentFiles -p Environment
Environment=PATH=…  VIRTUAL_ENV=/home/devos/.hermes/hermes-agent/venv \
            HERMES_HOME=/home/devos/.hermes/profiles/orchestrator
FragmentPath=/home/devos/.config/systemd/user/hermes-gateway-orchestrator.service
DropInPaths=
```

Key facts:
- Profile: `orchestrator`.
- `HERMES_HOME=/home/devos/.hermes/profiles/orchestrator`.
- `ExecStart` uses `--profile orchestrator gateway run`.
- No `EnvironmentFile=` is declared and no drop-ins exist. Adding an
  `EnvironmentFile=` requires a new drop-in (template in §5).

### 1.2 Port 8642 listener state

```
$ ss -ltn | grep -E ':8642|:864'   # → no output
```

Nothing is bound to 8642 on any interface. Default-deny is in effect: the API
server platform is not declared in the profile config, so the adapter is never
constructed. This is the expected pre-step-11 state.

### 1.3 Network interfaces and Tailscale evidence

```
$ ip -o -4 addr show
1: lo            inet 127.0.0.1/8
2: eth0          inet 167.233.91.186/32  (public)
3: tailscale0    inet 100.98.1.82/32
4: br-0631…      inet 172.18.0.1/16     (docker bridge)
5: docker0       inet 172.17.0.1/16     (docker bridge)
```

Tailscale presence is independently corroborated by existing listeners already
bound to the tailnet address `100.98.1.82` (dashboard-proxy on 443, syncthing
on 8384, dashboard on 8443, an IPv6 tailnet listener on
`fd7a:115c:a1e0::9a32:153`).

`tailscale ip -4` / `tailscale status` were not run because the shell sandbox
prompts on the binary; the interface evidence above is sufficient.

**Host has a public IPv4 (`167.233.91.186`).** A plaintext public bind on 8642
must be refused; only loopback or Tailscale-only is acceptable.

### 1.4 Live config.yaml state

```
$ head -... /home/devos/.hermes/profiles/orchestrator/config.yaml
```

- No `platforms:` block exists at all.
- No `gateway.topic_default_app_id` (or `gateway.topic: {default_app_id: …}`).
- No `gateway.api_server.*` overrides (concurrency cap defaults to 10).

So step 11 introduces these blocks; it does not edit existing ones.

### 1.5 Posture-validator surfaces (already merged)

- `gateway.platforms.api_server.validate_api_server_posture` — checked at
  `connect()` and at config-load (`gateway/config.py:1270`).
- Step-10 default-deny: a missing `platforms.api_server` block, or
  `enabled: false`, never constructs the adapter.
- Strict-bool guard: quoted YAML scalars (`tls: "true"`,
  `tailscale_only: "yes"`) are now rejected (commit `5371fcb169`). Templates
  below use real YAML booleans.

---

## 2. Recommended enablement posture

**Tailscale-only.** Bind the API server to the verified tailnet address
`100.98.1.82`. Bearer auth (`API_SERVER_KEY`) remains the trust boundary, with
`extra.tailscale_only: true` lifting the network-bind transport-posture gate.

Reason:
- Host has a public IP. Loopback-only would deny tailnet access from the
  owner's phone/laptop, which is the whole point of bringing this surface up
  on the orchestrator gateway. A plaintext public bind is structurally
  refused by `validate_api_server_posture`.
- Tailscale-bound `100.98.1.82` is reachable only over the tailnet — the
  public internet cannot route to it. This matches the dashboard's existing
  Tailscale-fronted posture on this host.
- `tailscale_only: true` is an operator attestation that the bind host is on
  a tailnet interface. The interface evidence in §1.3 backs that attestation.

Fallback (only if the owner wants to defer tailnet exposure): set
`host: 127.0.0.1` and drop `tailscale_only`. Loopback is treated as
non-network-accessible by the validator, so no TLS / Tailscale attestation
is required, but only this box itself can reach the surface (e.g. local CLI
clients, an SSH-tunneled phone session).

---

## 3. `config.yaml` block template

Append to `/home/devos/.hermes/profiles/orchestrator/config.yaml`. Do not
edit any existing keys; this is purely additive.

```yaml
# HRM-T0a step 11 — API server platform (Tailscale-only).
# API_SERVER_KEY is loaded from the systemd EnvironmentFile, NOT this file.
platforms:
  api_server:
    enabled: true
    extra:
      host: 100.98.1.82          # verified tailscale0 IPv4 (see §1.3)
      port: 8642
      tailscale_only: true       # real YAML boolean — quoted strings rejected
      tls: false                 # TLS not terminated upstream on 8642
      # key:  intentionally omitted — supplied via API_SERVER_KEY env

# HRM-T0a step 11 — Active-topic routing parameters.
# default_app_id below is a DECISION POINT (see §9.D1). Replace
# <DEV-OS-APP-ID> with the canonical value once verified.
gateway:
  topic:
    pointer_mode_enabled: true
    slash_ux_enabled: true
    default_app_id: "<DEV-OS-APP-ID>"
```

Notes:
- The validator rejects a non-loopback bind without `tls: true` OR
  `tailscale_only: true`. `tls: false` is included for clarity, but is the
  default and may be omitted.
- `enabled: true` on its own is not enough: without `default_app_id`, the
  step-8 fail-closed path leaves topic routing inert (the active-topic
  pre-pass short-circuits when `default_app_id` is `None`). That is safe but
  it would defeat the point of step 11 — verify the app_id first.
- Loopback fallback diff (if §9.D2 is decided that way): set `host:
  127.0.0.1` and remove `tailscale_only: true`.

---

## 4. `EnvironmentFile` template

Path: `/home/devos/.hermes/profiles/orchestrator/api_server.env`

```ini
# /home/devos/.hermes/profiles/orchestrator/api_server.env
# Owner-only (chmod 600). Loaded by the systemd drop-in in §5.
# Generate with:  openssl rand -hex 32
API_SERVER_KEY=<generate with openssl rand -hex 32>
```

Apply with:

```bash
umask 077
cat > /home/devos/.hermes/profiles/orchestrator/api_server.env <<'EOF'
API_SERVER_KEY=<paste 64-hex output of: openssl rand -hex 32>
EOF
chmod 600 /home/devos/.hermes/profiles/orchestrator/api_server.env
```

Rationale for an EnvironmentFile over config.yaml:
- `config.yaml` is a profile artifact often diffed, backed up, and shipped
  between hosts. Bearer secrets must not ride along with it.
- The API server reads `API_SERVER_KEY` from the process environment when no
  `extra.key` is set in config (`gateway/platforms/api_server.py:882`), so
  the env-only path is the cleanest split: posture in config, secret in env.
- A systemd `EnvironmentFile=` makes the secret reload on `systemctl --user
  daemon-reload` + restart without leaking into `ps`, journal, or the unit
  file itself.

---

## 5. systemd drop-in template

Path: `/home/devos/.config/systemd/user/hermes-gateway-orchestrator.service.d/api-server.conf`

```ini
# Drop-in to load the API server bearer secret from a 0600 env file.
# Created for HRM-T0a step 11. Does not modify the base unit.
[Service]
EnvironmentFile=/home/devos/.hermes/profiles/orchestrator/api_server.env
```

Why a drop-in (not a unit edit):
- The base unit `~/.config/systemd/user/hermes-gateway-orchestrator.service`
  currently has no `EnvironmentFile=` and no `DropInPaths` (§1.1). Editing
  the base unit conflates step 11 with pre-existing unit content; a drop-in
  keeps the change self-contained and trivially reversible (`rm` the
  `.service.d` dir, `daemon-reload`).
- `Environment=API_SERVER_KEY=…` inline in the unit would put the secret in
  the unit file itself (world-readable in user-scope), which is exactly
  what the EnvironmentFile pattern exists to avoid.

If the operator later prefers consolidating, a future change can fold this
into the base unit without touching config.yaml or api_server.env.

---

## 6. Backup procedure (run before any live edit)

```bash
TS="$(date -u +%Y%m%dT%H%M%SZ)"
PROFILE=/home/devos/.hermes/profiles/orchestrator
UNITDIR=/home/devos/.config/systemd/user
BACKUP=/home/devos/.hermes/backups/hrm-t0a-step11-${TS}
mkdir -p "${BACKUP}"
cp -a "${PROFILE}/config.yaml" "${BACKUP}/config.yaml.before"
cp -a "${UNITDIR}/hermes-gateway-orchestrator.service" \
      "${BACKUP}/hermes-gateway-orchestrator.service.before"
# Record git head of the worktree that produced the change
( cd /home/devos/.hermes/hermes-agent/.worktrees/hrm-t0a \
  && git rev-parse HEAD ) > "${BACKUP}/worktree-head.txt"
ls -la "${BACKUP}"
```

The backup directory `~/.hermes/backups/hrm-t0a-step11-<UTC>` is the single
rollback source of truth.

---

## 7. Smoke commands (post-enablement)

Run these in this exact order after the live change. They cover both the
disabled→enabled transition and the
authenticated/unauthenticated/off-tailnet matrix.

### 7.1 Before enabling (sanity — must all match §1)

```bash
ss -ltn | grep ':8642' || echo 'no listener (expected pre-enable)'
systemctl --user is-active hermes-gateway-orchestrator.service
```

### 7.2 After enabling (Tailscale-only posture)

On the box itself:

```bash
# Loopback path is NOT bound when tailscale_only host=100.98.1.82.
# This should refuse — there is no listener on 127.0.0.1:8642.
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8642/health || true
#   expected: connect refused / 000

# Tailnet path, unauthenticated — should be 200 for /health (public) and
# 401 for any authed endpoint.
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://100.98.1.82:8642/health
#   expected: 200

curl -sS -o /dev/null -w '%{http_code}\n' \
  http://100.98.1.82:8642/v1/models
#   expected: 401

# Tailnet path, authenticated.
SOURCE_ENV=/home/devos/.hermes/profiles/orchestrator/api_server.env
KEY="$(grep '^API_SERVER_KEY=' "${SOURCE_ENV}" | cut -d= -f2-)"
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer ${KEY}" \
  http://100.98.1.82:8642/v1/models
#   expected: 200
```

From another tailnet device (phone/laptop):

```bash
# Authenticated, with a strong key (>=16 chars enforced by the validator).
curl -sS http://100.98.1.82:8642/v1/capabilities \
  -H "Authorization: Bearer <paste-key>" | head
#   expected: JSON capabilities envelope
```

Off-tailnet (public internet) verification: from any device NOT on the
tailnet, attempt `curl http://167.233.91.186:8642/health` and
`http://100.98.1.82:8642/health`. Both must time out / connection-refuse.
The public IP must never expose 8642.

### 7.3 Topic-routing smoke

Once the surface is up:

```bash
# /topic list on a chat already permitted by Telegram allow-list should now
# return an empty pointer set (the slash UX is reachable) rather than the
# legacy thread-derived behaviour. Verify by sending `/topic list` in the
# Telegram chat and observing the new UX.
```

A 500 on `/topic` after enablement is the canonical "default_app_id is
wrong / missing" signal (step 8 fail-closed). Roll back to the previous
`config.yaml` (§8) before debugging.

### 7.4 Disabled-mode sanity (regression guard)

To prove default-deny still holds, temporarily set `enabled: false` in
`platforms.api_server`, reload, and re-run §7.2 — port 8642 must vanish
from `ss -ltn`. Restore to `enabled: true` afterwards. This step is
optional; skip it if the change is being smoke-tested under a tight window.

---

## 8. Rollback procedure

The change is three artifacts: a config.yaml diff, a new env file, and a
new systemd drop-in. Each is independently reversible.

```bash
TS_DIR=<paste backup dir from §6>
PROFILE=/home/devos/.hermes/profiles/orchestrator
UNITDIR=/home/devos/.config/systemd/user

# 1. Restore config.yaml.
cp -a "${TS_DIR}/config.yaml.before" "${PROFILE}/config.yaml"

# 2. Remove the systemd drop-in.
rm -rf "${UNITDIR}/hermes-gateway-orchestrator.service.d"

# 3. Remove the env file (secret).
shred -u "${PROFILE}/api_server.env" 2>/dev/null \
  || rm -f "${PROFILE}/api_server.env"

# 4. Reload systemd + restart gateway.
systemctl --user daemon-reload
systemctl --user restart hermes-gateway-orchestrator.service

# 5. Confirm rollback.
ss -ltn | grep ':8642' || echo 'rolled back — no listener'
systemctl --user is-active hermes-gateway-orchestrator.service
```

The worktree's git history (this commit + prior step-0..step-10 commits)
remains untouched by rollback — code is unaffected, only configuration
reverts.

---

## 9. Decision points still open

These must be resolved by Sebastian before the live change.

### D1 — Canonical dev-os `topic_default_app_id`

Owner instruction: "Use canonical dev-os topic_default_app_id after
verifying from registry/config; if not verifiable, stop with decision point
rather than guessing."

Verification attempted, but the worktree's read sandbox is scoped to
`/home/devos/.hermes/hermes-agent/.worktrees/hrm-t0a` and cannot read the
`development-os` repo (`/home/devos/workspace/development-os/`). The
canonical app_id therefore could not be confirmed from this seat.

Observed in this worktree:
- `gateway/active_topic.py` is explicit that the dev-os boundary owns the
  registry; v1 takes `app_id` as an injected input.
- All `tests/gateway/test_topic_*.py` fixtures use the placeholder
  `app_id="hermes-agent"`. This is convenient, but it is a TEST fixture,
  not a registry attestation.

Required: confirm the value from the dev-os topic registry / config
before substituting `<DEV-OS-APP-ID>` in §3. Do not guess.

### D2 — Tailscale-only vs loopback-only

Recommendation: Tailscale-only on `100.98.1.82` (§2).
Counter-position: loopback-only (`127.0.0.1`) is the most conservative
posture and matches `hermes-dashboard.service`'s "localhost only" stance.
The recommendation tilts to Tailscale-only because the orchestrator
gateway is the principal-facing surface and tailnet reach is the practical
value of step 11; the dashboard's loopback stance is enforced separately
by the `hermes-dashboard-proxy` oauth2-proxy on tailnet 443.

### D3 — `max_concurrent_runs` cap

The API server defaults to 10 in-flight runs (`api_server.py:940`). The
orchestrator profile may want a tighter cap given its delegation fan-out
(`max_concurrent_children: 3`). Optional knob, not blocking:

```yaml
gateway:
  api_server:
    max_concurrent_runs: 3   # match delegation concurrency
```

### D4 — CORS

No browser frontend is in play for the orchestrator surface today.
`cors_origins` deliberately omitted. Revisit only if a browser client (e.g.
Open WebUI) is later attached.

---

## 10. What this packet does NOT cover

- Pushing the worktree, merging step-11 to `main`, or restarting the
  gateway. All of those wait on owner go-ahead.
- Dashboard or webhook surface changes. Step 11 is API-server only.
- Tailscale ACL audit (separately the operator's responsibility on the
  tailnet side).
