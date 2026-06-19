/**
 * ssh-connection.cjs
 *
 * Pure, electron-free OpenSSH ControlMaster connection manager for Desktop SSH
 * remote mode. Uses the system `ssh` client (not a JS SSH library) so it
 * inherits ~/.ssh/config, the agent, jump hosts (ProxyJump), and hardware keys
 * for free — the same rationale as tools/environments/ssh.py.
 *
 * Kept standalone (no `require('electron')`) so it can be unit-tested with
 * `node --test` — same pattern as connection-config.cjs / dashboard-token.cjs.
 * main.cjs requires this and wires it into the electron-coupled lifecycle.
 *
 * Conventions mirrored from tools/environments/ssh.py:
 *   - ControlMaster=auto + ControlPersist so one TCP/auth handshake is reused
 *     across exec/forward operations.
 *   - Hashed control-socket filename under a short tmpdir to stay under the
 *     104-byte sun_path limit macOS enforces on Unix domain sockets
 *     (ssh.py:53-67 rationale applies verbatim).
 *   - BatchMode=yes for every programmatic invocation — a spawned ssh must
 *     never hang on an interactive prompt (passphrase / 2FA). If auth needs
 *     interactivity we fail fast and tell the user to load the key into their
 *     agent.
 *
 * Host-key policy: StrictHostKeyChecking=accept-new (trust-on-first-use, log
 * the fingerprint), never `no`. A host-key *change* fails closed with the
 * verbatim OpenSSH error surfaced to the UI.
 *
 * Every operation is raced against a hard timeout. A half-open TCP connection
 * after laptop sleep can leave ssh hanging indefinitely rather than erroring;
 * timeout is treated as connection-dead so the caller does a full reconnect
 * rather than retrying in place (VS Code's agent host does the same).
 */

const { spawn } = require('node:child_process')
const crypto = require('node:crypto')
const net = require('node:net')
const os = require('node:os')
const path = require('node:path')
const fs = require('node:fs')

const DEFAULT_CONNECT_TIMEOUT_MS = 15_000
const DEFAULT_EXEC_TIMEOUT_MS = 20_000
const DEFAULT_FORWARD_TIMEOUT_MS = 15_000
const CONTROL_PERSIST_SECONDS = 300

// ---------------------------------------------------------------------------
// Token / secret redaction
// ---------------------------------------------------------------------------

// Every lifecycle log line in SSH mode passes through this before it reaches
// rememberLog/desktop.log. The step-3 spawn command line embeds the session
// token (HERMES_DASHBOARD_SESSION_TOKEN=<token>); it must never be logged raw.
// We also scrub the URL/header carriers the dashboard protocol uses so a
// forwarded base URL or a copied curl line can't leak a credential.
//
// Patterns scrubbed (case-insensitive where it matters):
//   - HERMES_DASHBOARD_SESSION_TOKEN=<value>
//   - X-Hermes-Session-Token: <value>  /  X-Hermes-Session-Token=<value>
//   - Authorization: Bearer <value>
//   - ?token=<value> / &token=<value>     (the WS auth param)
//   - ?ticket=<value> / &ticket=<value>   (the OAuth ws-ticket param)
const _REDACTIONS = [
  [/(HERMES_DASHBOARD_SESSION_TOKEN=)(\S+)/g, '$1<redacted>'],
  [/(X-Hermes-Session-Token["']?\s*[:=]\s*["']?)([^\s"'&]+)/gi, '$1<redacted>'],
  [/(Authorization["']?\s*:\s*Bearer\s+)(\S+)/gi, '$1<redacted>'],
  [/([?&](?:token|ticket)=)([^\s&"']+)/gi, '$1<redacted>']
]

function redactSecrets(text) {
  let out = String(text == null ? '' : text)
  for (const [re, repl] of _REDACTIONS) {
    out = out.replace(re, repl)
  }
  return out
}

// ---------------------------------------------------------------------------
// Control-socket path
// ---------------------------------------------------------------------------

// Hash user@host:port to a short, stable, filesystem-safe socket id. Stable
// across reconnects so ControlMaster reuse works; short so the full path stays
// well under sun_path's 104-byte limit even under macOS's deeply nested
// $TMPDIR (/var/folders/xx/yy/T/...). Mirrors ssh.py:53-67.
function controlSocketPath(user, host, port, baseDir) {
  const dir = baseDir || path.join(os.tmpdir(), 'hermes-desktop-ssh')
  const id = crypto.createHash('sha256').update(`${user}@${host}:${port}`).digest('hex').slice(0, 16)
  return path.join(dir, `${id}.sock`)
}

// ---------------------------------------------------------------------------
// Command construction (pure — the unit tests exercise these directly)
// ---------------------------------------------------------------------------

function baseSshOptions(controlPath, connectTimeoutMs) {
  const connectSecs = Math.max(1, Math.round((connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS) / 1000))
  return [
    '-o', `ControlPath=${controlPath}`,
    '-o', 'ControlMaster=auto',
    '-o', `ControlPersist=${CONTROL_PERSIST_SECONDS}`,
    '-o', 'BatchMode=yes',
    '-o', 'StrictHostKeyChecking=accept-new',
    '-o', `ConnectTimeout=${connectSecs}`
  ]
}

// Per-host args shared by exec, the master open, and forward control commands:
// non-default port and explicit identity file.
function hostArgs({ port, keyPath }) {
  const args = []
  if (port && Number(port) !== 22) {
    args.push('-p', String(port))
  }
  if (keyPath) {
    args.push('-i', keyPath)
  }
  return args
}

function target(user, host) {
  return user ? `${user}@${host}` : host
}

// `ssh <opts> <host> <remoteCommand>` — one-shot over the control connection.
function buildExecArgs(conn, remoteCommand, connectTimeoutMs) {
  return [
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    target(conn.user, conn.host),
    remoteCommand
  ]
}

// `ssh -O <op> <opts> <host>` — control-command against the running master
// (check / forward / cancel / exit). -O commands don't take a remote command.
function buildControlArgs(conn, op, extra = [], connectTimeoutMs) {
  return [
    '-O', op,
    ...extra,
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    target(conn.user, conn.host)
  ]
}

// Open the master explicitly: `-M -N -f` puts ssh into the background once the
// master is up, so the spawn resolves when the connection is established (or
// fails fast under BatchMode if auth is non-interactive-only).
function buildMasterArgs(conn, connectTimeoutMs) {
  return [
    '-M', '-N', '-f',
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    target(conn.user, conn.host)
  ]
}

// Interactive `ssh -tt` for the INTERIM remote terminal (component 5, SSH mode
// only). Reuses the existing ControlMaster socket so NO new auth handshake
// happens — the master is already open, so this attaches instantly and never
// prompts (BatchMode stays safe here for that reason). `-tt` forces a PTY even
// though our stdio is a node-pty, so the remote sees a real terminal.
//
// When a remoteCwd is given we cd into it (best-effort) then exec the user's
// login shell so the prompt/rc files load; an unreadable cwd falls back to
// $HOME rather than failing the session.
//
// NOTE (tracked): this is the interim path until the dashboard /api/terminal
// WebSocket lands (specs/desktop-remote-terminal.md). Once that ships, the
// terminal rides the tunnel like every other socket and cwd-follows-session
// behavior becomes uniform; delete this path then.
function buildInteractiveSshArgs(conn, remoteCwd, connectTimeoutMs) {
  const args = [
    '-tt',
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    target(conn.user, conn.host)
  ]
  const cwd = String(remoteCwd || '').trim()
  if (cwd) {
    // cd then exec a login shell; quote the path; tolerate a missing dir.
    const q = `'${cwd.replace(/'/g, `'\\''`)}'`
    args.push(`cd ${q} 2>/dev/null; exec "$SHELL" -l`)
  } else {
    args.push('exec "$SHELL" -l')
  }
  return args
}

// Local forward spec for `-O forward -L <local>:<remoteHost>:<remotePort>`.
// Bind the local end to 127.0.0.1 ONLY — never 0.0.0.0 — so the tunnel does
// not re-expose the remote dashboard to the client's LAN.
function forwardSpec(localPort, remotePort, remoteHost = '127.0.0.1') {
  return `127.0.0.1:${localPort}:${remoteHost}:${remotePort}`
}

// ---------------------------------------------------------------------------
// Error classification — distinct, actionable messages for the UI
// ---------------------------------------------------------------------------

const SSH_ERROR = {
  UNREACHABLE: 'unreachable',
  AUTH_FAILED: 'auth-failed',
  HOST_KEY_CHANGED: 'host-key-changed',
  TIMEOUT: 'timeout',
  UNKNOWN: 'unknown'
}

// Map raw ssh stderr to a stable error kind. Order matters: the host-key-change
// banner also contains "WARNING"/"Offending", check it before generic auth.
function classifySshError(stderr) {
  const text = String(stderr || '')
  if (/REMOTE HOST IDENTIFICATION HAS CHANGED|Host key verification failed|Offending (?:key|ECDSA|RSA|ED25519)/i.test(text)) {
    return SSH_ERROR.HOST_KEY_CHANGED
  }
  if (/Permission denied|Too many authentication failures|no matching host key|publickey|password|keyboard-interactive/i.test(text)) {
    return SSH_ERROR.AUTH_FAILED
  }
  if (/Could not resolve hostname|Connection refused|Connection timed out|No route to host|Network is unreachable|Operation timed out|port \d+: Connection/i.test(text)) {
    return SSH_ERROR.UNREACHABLE
  }
  return SSH_ERROR.UNKNOWN
}

function sshErrorMessage(kind, conn, stderr) {
  const host = target(conn.user, conn.host)
  switch (kind) {
    case SSH_ERROR.HOST_KEY_CHANGED:
      return (
        `The host key for ${host} has CHANGED since you last connected. ` +
        `This could be a man-in-the-middle attack, or the server was reinstalled. ` +
        `SSH refused to connect. Verify the change is expected, then remove the old key ` +
        `with \`ssh-keygen -R ${conn.host}\` and reconnect.\n\n${String(stderr || '').trim()}`
      )
    case SSH_ERROR.AUTH_FAILED:
      return (
        `SSH authentication to ${host} failed. Desktop runs ssh non-interactively ` +
        `(BatchMode), so a key requiring a passphrase or 2FA must be loaded into your ` +
        `ssh-agent first (e.g. \`ssh-add ~/.ssh/id_ed25519\`), or set an IdentityFile in ` +
        `~/.ssh/config. Original error: ${String(stderr || '').trim()}`
      )
    case SSH_ERROR.UNREACHABLE:
      return `Could not reach ${host} over SSH. Check the host, port, and your network. Original error: ${String(stderr || '').trim()}`
    case SSH_ERROR.TIMEOUT:
      return `SSH operation to ${host} timed out. The connection may be half-open (e.g. after sleep); reconnecting.`
    default:
      return `SSH error connecting to ${host}: ${String(stderr || '').trim() || 'unknown failure'}`
  }
}

// ---------------------------------------------------------------------------
// Spawn helper — runs an ssh invocation, races it against a hard timeout
// ---------------------------------------------------------------------------

// Resolves { code, stdout, stderr }. On timeout the child is SIGKILLed and the
// promise rejects with err.kind = TIMEOUT. `spawnFn` is injectable for tests.
function runSsh(args, { timeoutMs, spawnFn = spawn, stdin = 'ignore' } = {}) {
  return new Promise((resolve, reject) => {
    let child
    try {
      child = spawnFn('ssh', args, { stdio: [stdin === 'ignore' ? 'ignore' : 'pipe', 'pipe', 'pipe'] })
    } catch (error) {
      reject(error)
      return
    }

    let stdout = ''
    let stderr = ''
    let settled = false

    const timer = setTimeout(() => {
      if (settled) return
      settled = true
      try {
        child.kill('SIGKILL')
      } catch {
        // already gone
      }
      const err = new Error(`ssh timed out after ${timeoutMs}ms`)
      err.kind = SSH_ERROR.TIMEOUT
      reject(err)
    }, timeoutMs)

    child.stdout?.on('data', d => {
      stdout += d.toString()
    })
    child.stderr?.on('data', d => {
      stderr += d.toString()
    })
    child.on('error', error => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      reject(error)
    })
    child.on('close', code => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve({ code, stdout, stderr })
    })
  })
}

// ---------------------------------------------------------------------------
// SshConnection — the public manager
// ---------------------------------------------------------------------------

class SshConnection {
  /**
   * @param {{host:string, user?:string, port?:number, keyPath?:string}} cfg
   * @param {{ spawnFn?, rememberLog?, controlDir?, connectTimeoutMs?, execTimeoutMs?, forwardTimeoutMs? }} [opts]
   */
  constructor(cfg, opts = {}) {
    if (!cfg || !cfg.host) {
      throw new Error('SshConnection requires a host.')
    }
    this.host = cfg.host
    this.user = cfg.user || ''
    this.port = cfg.port ? Number(cfg.port) : 22
    this.keyPath = cfg.keyPath || ''
    this.controlPath = controlSocketPath(this.user, this.host, this.port, opts.controlDir)

    this._spawnFn = opts.spawnFn || spawn
    this._log = typeof opts.rememberLog === 'function' ? opts.rememberLog : () => {}
    this._connectTimeoutMs = opts.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS
    this._execTimeoutMs = opts.execTimeoutMs ?? DEFAULT_EXEC_TIMEOUT_MS
    this._forwardTimeoutMs = opts.forwardTimeoutMs ?? DEFAULT_FORWARD_TIMEOUT_MS
    this._opened = false
  }

  // Lifecycle logging — ALWAYS through redaction.
  _logLine(msg) {
    this._log(redactSecrets(`[ssh] ${msg}`))
  }

  // Throw a classified, UI-ready error from an ssh result/exception.
  _fail(stderrOrErr, fallbackKind = SSH_ERROR.UNKNOWN) {
    if (stderrOrErr && stderrOrErr.kind === SSH_ERROR.TIMEOUT) {
      const err = new Error(sshErrorMessage(SSH_ERROR.TIMEOUT, this))
      err.kind = SSH_ERROR.TIMEOUT
      return err
    }
    const stderr = typeof stderrOrErr === 'string' ? stderrOrErr : stderrOrErr?.message || ''
    const kind = stderr ? classifySshError(stderr) : fallbackKind
    const err = new Error(sshErrorMessage(kind, this, stderr))
    err.kind = kind
    return err
  }

  // Open the persistent ControlMaster. Idempotent: if a master socket is
  // already alive (`-O check` succeeds), this is a no-op.
  async open() {
    if (await this.isAlive()) {
      this._opened = true
      return
    }
    // Ensure the control-socket directory exists — OpenSSH will not create
    // intermediate dirs for ControlPath, so a fresh box (no prior hermes-ssh
    // socket dir under $TMPDIR) would otherwise fail before the first connect.
    // 0o700: the socket grants command execution on the master; keep it private.
    try {
      fs.mkdirSync(path.dirname(this.controlPath), { recursive: true, mode: 0o700 })
    } catch {
      // best effort — a pre-existing dir or a races-with-another-conn mkdir is fine
    }
    const args = buildMasterArgs(this, this._connectTimeoutMs)
    this._logLine(`opening control master to ${target(this.user, this.host)}:${this.port}`)
    let result
    try {
      result = await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn })
    } catch (error) {
      throw this._fail(error, SSH_ERROR.UNREACHABLE)
    }
    if (result.code !== 0) {
      throw this._fail(result.stderr, SSH_ERROR.UNREACHABLE)
    }
    this._opened = true
    this._logLine('control master established')
  }

  // `-O check` against the master socket. True iff the master is alive.
  async isAlive() {
    const args = buildControlArgs(this, 'check', [], this._connectTimeoutMs)
    try {
      const result = await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn })
      return result.code === 0
    } catch {
      return false
    }
  }

  // One-shot remote command over the control connection. Resolves the trimmed
  // stdout; rejects with a classified error on non-zero exit or timeout.
  async exec(remoteCommand, { timeoutMs } = {}) {
    const args = buildExecArgs(this, remoteCommand, this._connectTimeoutMs)
    let result
    try {
      result = await runSsh(args, { timeoutMs: timeoutMs ?? this._execTimeoutMs, spawnFn: this._spawnFn })
    } catch (error) {
      throw this._fail(error)
    }
    if (result.code !== 0) {
      throw this._fail(result.stderr)
    }
    return result.stdout
  }

  // Establish a local→remote forward against the running master.
  // 127.0.0.1:<localPort> → <remoteHost>:<remotePort>.
  async forward(localPort, remotePort, remoteHost = '127.0.0.1') {
    const spec = forwardSpec(localPort, remotePort, remoteHost)
    const args = buildControlArgs(this, 'forward', ['-L', spec], this._connectTimeoutMs)
    this._logLine(`forwarding 127.0.0.1:${localPort} -> ${remoteHost}:${remotePort}`)
    let result
    try {
      result = await runSsh(args, { timeoutMs: this._forwardTimeoutMs, spawnFn: this._spawnFn })
    } catch (error) {
      throw this._fail(error)
    }
    if (result.code !== 0) {
      throw this._fail(result.stderr)
    }
  }

  // Cancel a previously-established forward. Best-effort: a failure here is
  // logged but not thrown (the master close tears everything down anyway).
  async cancelForward(localPort, remotePort, remoteHost = '127.0.0.1') {
    const spec = forwardSpec(localPort, remotePort, remoteHost)
    const args = buildControlArgs(this, 'cancel', ['-L', spec], this._connectTimeoutMs)
    try {
      await runSsh(args, { timeoutMs: this._forwardTimeoutMs, spawnFn: this._spawnFn })
      this._logLine(`cancelled forward 127.0.0.1:${localPort}`)
    } catch (error) {
      this._logLine(`cancelForward failed (ignored): ${error.message}`)
    }
  }

  // Tear down the master. Best-effort; never throws.
  async close() {
    if (!this._opened) return
    const args = buildControlArgs(this, 'exit', [], this._connectTimeoutMs)
    try {
      await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn })
      this._logLine('control master closed')
    } catch (error) {
      this._logLine(`close failed (ignored): ${error.message}`)
    } finally {
      this._opened = false
    }
  }
}

// ---------------------------------------------------------------------------
// Free local port — for the tunnel's local end. Bind 127.0.0.1:0, read the
// kernel-assigned port, release. There is a benign TOCTOU window between
// release and the forward grabbing it; the forward failing is caught upstream
// and retried with a fresh port.
// ---------------------------------------------------------------------------

function pickLocalPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address()
      server.close(() => resolve(port))
    })
  })
}

module.exports = {
  CONTROL_PERSIST_SECONDS,
  DEFAULT_CONNECT_TIMEOUT_MS,
  DEFAULT_EXEC_TIMEOUT_MS,
  DEFAULT_FORWARD_TIMEOUT_MS,
  SSH_ERROR,
  SshConnection,
  baseSshOptions,
  buildControlArgs,
  buildExecArgs,
  buildInteractiveSshArgs,
  buildMasterArgs,
  classifySshError,
  controlSocketPath,
  forwardSpec,
  hostArgs,
  pickLocalPort,
  redactSecrets,
  runSsh,
  sshErrorMessage,
  target
}
