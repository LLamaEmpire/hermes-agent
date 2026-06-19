/**
 * ssh-config.cjs
 *
 * Pure, electron-free helpers for reading the user's OpenSSH client config:
 *   - parseSshConfigHosts(text): extract concrete `Host` aliases for the
 *     settings UI's host suggestions, filtering wildcard/negated patterns.
 *   - collectSshConfigHosts(rootPath, deps): read ~/.ssh/config and follow
 *     `Include` directives (read-only — we NEVER write that file).
 *   - parseSshGOutput(text): parse `ssh -G <host>` key/value output into the
 *     resolved hostname/user/port/identityfile for display + normalization.
 *
 * Kept standalone (no `require('electron')`) so it can be unit-tested with
 * `node --test`. main.cjs requires this and wires the fs + `ssh -G` exec in.
 */

const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

// Pull concrete host aliases out of an ssh_config body. A `Host` line can list
// several patterns; we keep only literal aliases (no `*`, `?`, or `!` negation)
// since those are the ones a user can actually connect to by name.
function parseSshConfigHosts(text) {
  const hosts = []
  const seen = new Set()
  for (const rawLine of String(text || '').split('\n')) {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) continue
    const m = /^host\s+(.+)$/i.exec(line)
    if (!m) continue
    for (const pattern of m[1].split(/\s+/)) {
      if (!pattern || pattern.includes('*') || pattern.includes('?') || pattern.startsWith('!')) {
        continue
      }
      if (!seen.has(pattern)) {
        seen.add(pattern)
        hosts.push(pattern)
      }
    }
  }
  return hosts
}

// Extract `Include` paths from an ssh_config body (relative paths resolve under
// ~/.ssh). Globs are expanded by the caller's fs deps when supported; here we
// just return the raw tokens for the collector to resolve.
function parseSshConfigIncludes(text) {
  const includes = []
  for (const rawLine of String(text || '').split('\n')) {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) continue
    const m = /^include\s+(.+)$/i.exec(line)
    if (!m) continue
    for (const token of m[1].split(/\s+/)) {
      if (token) includes.push(token)
    }
  }
  return includes
}

// Read ~/.ssh/config and any files it Includes, returning a de-duplicated list
// of concrete host aliases. Read-only; bounded include depth to avoid cycles.
// `deps` injects { readFile, homeDir, globSync } for tests.
function collectSshConfigHosts(rootPath, deps = {}) {
  const readFile =
    deps.readFile ||
    (p => {
      try {
        return fs.readFileSync(p, 'utf8')
      } catch {
        return null
      }
    })
  const homeDir = deps.homeDir || os.homedir()
  const root = rootPath || path.join(homeDir, '.ssh', 'config')
  const sshDir = path.join(homeDir, '.ssh')

  const out = []
  const seen = new Set()
  const visited = new Set()

  const resolveIncludePath = token => {
    if (token.startsWith('~/')) return path.join(homeDir, token.slice(2))
    if (path.isAbsolute(token)) return token
    return path.join(sshDir, token)
  }

  const walk = (filePath, depth) => {
    if (depth > 8 || visited.has(filePath)) return
    visited.add(filePath)
    const text = readFile(filePath)
    if (text == null) return
    for (const host of parseSshConfigHosts(text)) {
      if (!seen.has(host)) {
        seen.add(host)
        out.push(host)
      }
    }
    for (const token of parseSshConfigIncludes(text)) {
      const target = resolveIncludePath(token)
      // Optional glob expansion (token may contain * — e.g. config.d/*).
      const expanded = deps.globSync ? deps.globSync(target) : [target]
      for (const p of expanded) {
        walk(p, depth + 1)
      }
    }
  }

  walk(root, 0)
  return out
}

// Parse `ssh -G <host>` output. Keys are lowercased by ssh; we surface the ones
// the settings UI cares about. Returns { hostname, user, port, identityFile }.
function parseSshGOutput(text) {
  const out = { hostname: null, user: null, port: null, identityFile: null }
  for (const rawLine of String(text || '').split('\n')) {
    const line = rawLine.trim()
    if (!line) continue
    const sp = line.indexOf(' ')
    if (sp === -1) continue
    const key = line.slice(0, sp).toLowerCase()
    const value = line.slice(sp + 1).trim()
    if (key === 'hostname' && !out.hostname) out.hostname = value
    else if (key === 'user' && !out.user) out.user = value
    else if (key === 'port' && !out.port) out.port = Number.parseInt(value, 10) || null
    else if (key === 'identityfile' && !out.identityFile) out.identityFile = value
  }
  return out
}

module.exports = {
  collectSshConfigHosts,
  parseSshConfigHosts,
  parseSshConfigIncludes,
  parseSshGOutput
}
