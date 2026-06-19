/**
 * Tests for electron/ssh-connection.cjs.
 *
 * Run with: node --test electron/ssh-connection.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Pure, electron-free: command construction, secret redaction, error
 * classification, and the SshConnection lifecycle are exercised with an
 * injected fake `spawn` so no real ssh process is started.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const {
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
  redactSecrets,
  sshErrorMessage,
  target
} = require('./ssh-connection.cjs')

// --- secret redaction -------------------------------------------------------

test('redactSecrets scrubs the spawn-time session token env var', () => {
  const line = 'setsid env HERMES_DASHBOARD_SESSION_TOKEN=abc123deadbeef HERMES_DESKTOP=1 hermes dashboard'
  const out = redactSecrets(line)
  assert.ok(!out.includes('abc123deadbeef'))
  assert.match(out, /HERMES_DASHBOARD_SESSION_TOKEN=<redacted>/)
  // non-secret env vars are preserved
  assert.match(out, /HERMES_DESKTOP=1/)
})

test('redactSecrets scrubs ?token= and ?ticket= URL params', () => {
  assert.match(redactSecrets('ws://127.0.0.1:5000/api/ws?token=supersecret'), /\?token=<redacted>/)
  assert.match(redactSecrets('ws://127.0.0.1:5000/api/ws?ticket=onetimeticket'), /\?ticket=<redacted>/)
  assert.match(redactSecrets('GET /x?a=1&token=zzz HTTP'), /&token=<redacted>/)
  assert.ok(!redactSecrets('?token=supersecret').includes('supersecret'))
})

test('redactSecrets scrubs Authorization and X-Hermes-Session-Token headers', () => {
  assert.match(redactSecrets('Authorization: Bearer tok_9999'), /Authorization: Bearer <redacted>/)
  assert.ok(!redactSecrets('Authorization: Bearer tok_9999').includes('tok_9999'))
  assert.match(redactSecrets('X-Hermes-Session-Token: hdr_888'), /X-Hermes-Session-Token: ?<redacted>/)
  assert.ok(!redactSecrets('X-Hermes-Session-Token: hdr_888').includes('hdr_888'))
})

test('redactSecrets handles null/undefined and non-secret text untouched', () => {
  assert.equal(redactSecrets(null), '')
  assert.equal(redactSecrets(undefined), '')
  assert.equal(redactSecrets('uname -s -m'), 'uname -s -m')
})

// --- control-socket path ----------------------------------------------------

test('controlSocketPath is stable, short, and host-distinct', () => {
  const a = controlSocketPath('me', 'box1', 22, '/tmp/d')
  const a2 = controlSocketPath('me', 'box1', 22, '/tmp/d')
  const b = controlSocketPath('me', 'box2', 22, '/tmp/d')
  assert.equal(a, a2, 'same triple → same socket (ControlMaster reuse)')
  assert.notEqual(a, b, 'different host → different socket')
  // 16 hex chars + .sock keeps the basename short for sun_path 104-byte limit
  assert.match(a, /\/[0-9a-f]{16}\.sock$/)
})

// --- command construction ---------------------------------------------------

test('baseSshOptions carries the house ControlMaster/BatchMode/accept-new policy', () => {
  const opts = baseSshOptions('/tmp/x.sock', 15000)
  const joined = opts.join(' ')
  assert.match(joined, /ControlPath=\/tmp\/x\.sock/)
  assert.match(joined, /ControlMaster=auto/)
  assert.match(joined, /ControlPersist=\d+/)
  assert.match(joined, /BatchMode=yes/)
  assert.match(joined, /StrictHostKeyChecking=accept-new/)
  assert.match(joined, /ConnectTimeout=15/)
  assert.ok(!joined.includes('StrictHostKeyChecking=no'), 'never disables host-key checking')
})

test('hostArgs adds -p only for non-default port and -i only with a key', () => {
  assert.deepEqual(hostArgs({ port: 22 }), [])
  assert.deepEqual(hostArgs({ port: 2222 }), ['-p', '2222'])
  assert.deepEqual(hostArgs({ port: 22, keyPath: '/k' }), ['-i', '/k'])
  assert.deepEqual(hostArgs({ port: 2200, keyPath: '/k' }), ['-p', '2200', '-i', '/k'])
})

test('target builds user@host or bare host', () => {
  assert.equal(target('me', 'box'), 'me@box')
  assert.equal(target('', 'box'), 'box')
})

test('buildExecArgs ends with host then the remote command', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildExecArgs(conn, 'command -v hermes', 15000)
  assert.equal(args[args.length - 1], 'command -v hermes')
  assert.equal(args[args.length - 2], 'me@box')
  assert.ok(args.includes('BatchMode=yes'))
})

test('buildControlArgs places -O <op> first and never appends a remote command', () => {
  const conn = { user: 'me', host: 'box', port: 2222, keyPath: '/k', controlPath: '/tmp/x.sock' }
  const args = buildControlArgs(conn, 'forward', ['-L', forwardSpec(5000, 6000)], 15000)
  assert.equal(args[0], '-O')
  assert.equal(args[1], 'forward')
  assert.ok(args.includes('-L'))
  assert.ok(args.includes('127.0.0.1:5000:127.0.0.1:6000'))
  assert.equal(args[args.length - 1], 'me@box')
})

test('buildMasterArgs requests a backgrounded master (-M -N -f)', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildMasterArgs(conn, 15000)
  assert.ok(args.includes('-M'))
  assert.ok(args.includes('-N'))
  assert.ok(args.includes('-f'))
})

test('forwardSpec binds the local end to 127.0.0.1 only', () => {
  assert.equal(forwardSpec(5000, 6000), '127.0.0.1:5000:127.0.0.1:6000')
  assert.ok(forwardSpec(5000, 6000).startsWith('127.0.0.1:'))
  assert.ok(!forwardSpec(5000, 6000).startsWith('0.0.0.0'))
})

test('buildInteractiveSshArgs requests a PTY, reuses the control master, execs a login shell', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildInteractiveSshArgs(conn, '', 15000)
  assert.equal(args[0], '-tt', 'forces a PTY so the remote sees a real terminal')
  assert.ok(args.join(' ').includes('ControlPath=/tmp/x.sock'), 'reuses the existing master (no new auth)')
  assert.equal(args[args.length - 2], 'me@box')
  assert.equal(args[args.length - 1], 'exec "$SHELL" -l')
})

test('buildInteractiveSshArgs cds into the remote cwd (best-effort) before the shell', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildInteractiveSshArgs(conn, '/home/me/project', 15000)
  const remoteCmd = args[args.length - 1]
  assert.match(remoteCmd, /^cd '\/home\/me\/project' 2>\/dev\/null; exec "\$SHELL" -l$/)
})

test('buildInteractiveSshArgs single-quotes a cwd with quotes safely', () => {
  const conn = { user: 'me', host: 'box', port: 22, keyPath: '', controlPath: '/tmp/x.sock' }
  const args = buildInteractiveSshArgs(conn, "/tmp/a'b", 15000)
  // the embedded quote must be escaped, not break out of the quoting
  assert.ok(args[args.length - 1].startsWith("cd '/tmp/a'"))
  assert.ok(args[args.length - 1].includes('exec "$SHELL" -l'))
})

// --- error classification ---------------------------------------------------

test('classifySshError detects a changed host key (fail-closed)', () => {
  assert.equal(
    classifySshError('@@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@@'),
    SSH_ERROR.HOST_KEY_CHANGED
  )
  assert.equal(classifySshError('Host key verification failed.'), SSH_ERROR.HOST_KEY_CHANGED)
  assert.equal(classifySshError('Offending ECDSA key in /home/u/.ssh/known_hosts:5'), SSH_ERROR.HOST_KEY_CHANGED)
})

test('classifySshError detects auth failure', () => {
  assert.equal(classifySshError('Permission denied (publickey).'), SSH_ERROR.AUTH_FAILED)
  assert.equal(classifySshError('Too many authentication failures'), SSH_ERROR.AUTH_FAILED)
})

test('classifySshError detects unreachable', () => {
  assert.equal(classifySshError('ssh: Could not resolve hostname nope'), SSH_ERROR.UNREACHABLE)
  assert.equal(classifySshError('connect to host x port 22: Connection refused'), SSH_ERROR.UNREACHABLE)
})

test('sshErrorMessage gives actionable guidance for auth and host-key-change', () => {
  const conn = { user: 'me', host: 'box', port: 22 }
  assert.match(sshErrorMessage(SSH_ERROR.AUTH_FAILED, conn, 'Permission denied'), /ssh-agent|ssh-add|IdentityFile/)
  assert.match(sshErrorMessage(SSH_ERROR.HOST_KEY_CHANGED, conn, 'CHANGED'), /ssh-keygen -R box/)
})

// --- SshConnection lifecycle with injected fake spawn -----------------------

// A fake child process that emits a scripted result on next tick.
function fakeChild({ code = 0, stdout = '', stderr = '', errorEvent = null, hang = false } = {}) {
  const child = new EventEmitter()
  child.stdout = new EventEmitter()
  child.stderr = new EventEmitter()
  child.kill = () => {
    child._killed = true
  }
  if (hang) {
    return child // never emits close → drives the timeout path
  }
  process.nextTick(() => {
    if (errorEvent) {
      child.emit('error', errorEvent)
      return
    }
    if (stdout) child.stdout.emit('data', Buffer.from(stdout))
    if (stderr) child.stderr.emit('data', Buffer.from(stderr))
    child.emit('close', code)
  })
  return child
}

// Build a spawnFn that returns scripted children per ssh invocation, recording
// the args it was called with.
function scriptedSpawn(scripts) {
  const calls = []
  let i = 0
  const fn = (_cmd, args) => {
    calls.push(args)
    const script = typeof scripts === 'function' ? scripts(args, i) : scripts[Math.min(i, scripts.length - 1)]
    i += 1
    return fakeChild(script || {})
  }
  fn.calls = calls
  return fn
}

test('open() establishes the master when not already alive', async () => {
  // `-O check` fails first (not alive) → master opens (code 0). Track which
  // ssh ops ran rather than re-probing with the same always-failing check.
  const ops = []
  const spawnFn = scriptedSpawn(args => {
    ops.push(args.includes('check') ? 'check' : args.includes('-M') ? 'master' : 'other')
    if (args.includes('check')) return { code: 255, stderr: 'no control path' }
    return { code: 0 }
  })
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await conn.open()
  assert.deepEqual(ops, ['check', 'master'], 'probes liveness first, then opens the master')
})

test('open() is a no-op when the master is already alive', async () => {
  const ops = []
  const spawnFn = scriptedSpawn(args => {
    ops.push(args.includes('check') ? 'check' : 'master')
    return { code: 0 } // check succeeds → already alive
  })
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await conn.open()
  assert.deepEqual(ops, ['check'], 'alive master → no second spawn to open it')
})

test('open() creates the control-socket directory if it does not exist', async () => {
  const dir = path.join(os.tmpdir(), `hermes-ssh-test-${process.pid}-${Date.now()}`)
  assert.ok(!fs.existsSync(dir), 'precondition: control dir absent')
  const spawnFn = scriptedSpawn(args => (args.includes('check') ? { code: 255 } : { code: 0 }))
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: dir })
  try {
    await conn.open()
    assert.ok(fs.existsSync(dir), 'open() created the control-socket directory before spawning ssh')
  } finally {
    try {
      fs.rmSync(dir, { recursive: true, force: true })
    } catch {
      /* ignore */
    }
  }
})

test('open() surfaces a classified auth error', async () => {
  const spawnFn = scriptedSpawn(args => {
    if (args.includes('check')) return { code: 255 }
    return { code: 255, stderr: 'Permission denied (publickey).' }
  })
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await assert.rejects(() => conn.open(), err => {
    assert.equal(err.kind, SSH_ERROR.AUTH_FAILED)
    assert.match(err.message, /ssh-agent|ssh-add/)
    return true
  })
})

test('exec() returns stdout on success and rejects (classified) on failure', async () => {
  const okSpawn = scriptedSpawn([{ code: 0, stdout: 'Linux\n' }])
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn: okSpawn, controlDir: '/tmp/d' })
  assert.equal((await conn.exec('uname -s')).trim(), 'Linux')

  const failSpawn = scriptedSpawn([{ code: 1, stderr: 'ssh: Could not resolve hostname box' }])
  const conn2 = new SshConnection({ host: 'box', user: 'me' }, { spawnFn: failSpawn, controlDir: '/tmp/d' })
  await assert.rejects(() => conn2.exec('uname -s'), err => {
    assert.equal(err.kind, SSH_ERROR.UNREACHABLE)
    return true
  })
})

test('exec() treats a hung ssh as a timeout (half-open connection)', async () => {
  const spawnFn = scriptedSpawn([{ hang: true }])
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await assert.rejects(() => conn.exec('uname -s', { timeoutMs: 30 }), err => {
    assert.equal(err.kind, SSH_ERROR.TIMEOUT)
    return true
  })
})

test('forward() issues -O forward with a loopback-bound -L spec', async () => {
  const spawnFn = scriptedSpawn([{ code: 0 }])
  const conn = new SshConnection({ host: 'box', user: 'me' }, { spawnFn, controlDir: '/tmp/d' })
  await conn.forward(5000, 6000)
  const args = spawnFn.calls[0]
  assert.equal(args[0], '-O')
  assert.equal(args[1], 'forward')
  assert.ok(args.includes('127.0.0.1:5000:127.0.0.1:6000'))
})

test('lifecycle logging passes through redaction', async () => {
  const logs = []
  const spawnFn = scriptedSpawn(args => (args.includes('check') ? { code: 255 } : { code: 0 }))
  const conn = new SshConnection(
    { host: 'box', user: 'me' },
    { spawnFn, controlDir: '/tmp/d', rememberLog: l => logs.push(l) }
  )
  await conn.open()
  // none of the emitted log lines may carry a raw token-shaped secret
  for (const line of logs) {
    assert.ok(!/token=[^<]/.test(line))
  }
  assert.ok(logs.some(l => l.includes('[ssh]')))
})
