/**
 * The proxy every update transport in the Desktop must honor, as one answer.
 *
 * The Desktop's update traffic has three transports (the channel resolver's
 * metadata fetches, electron-updater's native session, and the env inherited
 * by the Python source-check / hand-off children). A GUI-launched app never
 * inherits the shell's HTTP(S)_PROXY env (#60049), so on proxied networks
 * every one of them dials direct and the "Check for Updates…" button dies
 * with a network error while `hermes update` in a proxied shell works.
 *
 * The explicit opt-in is config.yaml's `updates.proxy` — the same key the
 * CLI updater reads — with the ambient proxy env still honored when the app
 * was launched from a proxied shell. Auto-probing localhost listeners or the
 * OS system proxy is deliberately NOT done: routing update traffic through
 * whatever happens to listen locally is a product decision, not a default.
 */
const PROXY_ENV_KEYS = ['https_proxy', 'http_proxy', 'HTTPS_PROXY', 'HTTP_PROXY'] as const
const ALL_PROXY_ENV_KEYS = ['all_proxy', 'ALL_PROXY'] as const
const PROXY_SCHEMES = new Set(['http', 'https', 'socks5', 'socks5h'])

/** The stripped proxy URL when *value* is a usable `scheme://host` URL, else null. */
export function validProxyUrl(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }

  const candidate = value.trim()

  if (!candidate) {
    return null
  }

  let parsed: URL

  try {
    parsed = new URL(candidate)
  } catch {
    return null
  }

  return PROXY_SCHEMES.has(parsed.protocol.replace(':', '')) && parsed.hostname ? candidate : null
}

/** The validated `updates.proxy` value from config.yaml, else null (never throws). */
export function proxyFromConfig(config: unknown): string | null {
  const updates =
    config && typeof config === 'object' && 'updates' in config
      ? (config as Record<string, unknown>).updates
      : null

  if (!updates || typeof updates !== 'object') {
    return null
  }

  return validProxyUrl((updates as Record<string, unknown>).proxy)
}

/** Config first, then ambient env (a proxied shell launch still wins over nothing). */
export function resolveProxyFromSources(env: NodeJS.ProcessEnv, config: unknown): string | null {
  const configured = proxyFromConfig(config)

  if (configured) {
    return configured
  }

  for (const key of [...PROXY_ENV_KEYS, ...ALL_PROXY_ENV_KEYS]) {
    const value = env[key]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return null
}

export interface ProxyResolution {
  proxy: string | null
  /** Env with the proxy exported for children that only read env (git, Python urllib). */
  env: NodeJS.ProcessEnv
}

/**
 * The single proxy answer for update traffic: `updates.proxy` from config,
 * else the ambient env, else null. The returned `env` overlays the proxy onto
 * the given base (ambient wins, config only fills unset keys) so hand-off and
 * source-check children inherit it exactly like `hermes update` in a proxied
 * shell would.
 */
export function resolveUpdateProxy(
  env: NodeJS.ProcessEnv,
  config: unknown,
  base: NodeJS.ProcessEnv = env
): ProxyResolution {
  const proxy = resolveProxyFromSources(env, config)

  if (!proxy) {
    return { proxy: null, env: base }
  }

  const overlay: NodeJS.ProcessEnv = { ...base }

  for (const key of PROXY_ENV_KEYS) {
    if (!overlay[key]) {
      overlay[key] = proxy
    }
  }

  return { proxy, env: overlay }
}

/** `session.setProxy` rules string for a resolved proxy URL, or null. */
export function proxyRulesFor(proxy: string): string | null {
  let parsed: URL

  try {
    parsed = new URL(proxy)
  } catch {
    return null
  }

  const scheme = parsed.protocol.replace(':', '')
  const mapped = scheme === 'socks5' || scheme === 'socks5h' ? `socks5://${parsed.host}` : `${scheme}://${parsed.host}`

  return `http=${mapped};https=${mapped}`
}

/**
 * Read config.yaml for `updates.proxy` without the full config loader: the
 * updater must work before any backend exists, on a machine whose config may
 * not parse for unrelated reasons. One targeted regex over the `updates:` block
 * — the same document `updates.desktop_feed_base_url` is already read from by
 * updater/feed-config.ts's stricter YAML parse; here a read failure simply
 * means "no configured proxy", and ambient env still applies.
 */
export async function readConfiguredUpdateProxy(
  configPath: string,
  readFile: (path: string) => Promise<string> = async path => (await import('node:fs/promises')).readFile(path, 'utf8')
): Promise<string | null> {
  try {
    const text = await readFile(configPath)
    const lines = text.split('\n')
    let insideUpdates = false
    let candidate: string | null = null

    for (const line of lines) {
      if (/^updates:\s*(?:#.*)?$/.test(line)) {
        insideUpdates = true
        continue
      }

      if (insideUpdates) {
        if (/^\S/.test(line)) {
          break
        }

        const inline = /^[\s]+proxy:\s*["']?([^"'\n#]+?)["']?\s*(?:#.*)?$/.exec(line)

        if (inline) {
          candidate = inline[1]
          break
        }
      }
    }

    return candidate === null ? null : validProxyUrl(candidate)
  } catch {
    return null
  }
}

/** A fetch bound to the given proxy via undici's ProxyAgent dispatcher (#60049). */
export function proxyFetchFactory(): (proxy: string) => typeof fetch {
  const agentCache = new Map<string, typeof fetch>()

  return (proxy: string): typeof fetch => {
    const cached = agentCache.get(proxy)

    if (cached) {
      return cached
    }

    const proxied: typeof fetch = (input, init) => {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { ProxyAgent, fetch: undiciFetch } = require('undici') as typeof import('undici')
      return undiciFetch(input as any, { ...(init as any), dispatcher: new ProxyAgent(proxy) }) as any
    }

    agentCache.set(proxy, proxied)

    return proxied
  }
}
