/**
 * Conversation recording middleware for Hono.
 *
 * Intercepts POST requests to API endpoints, captures the request body
 * and response body (including streaming), then emits a [CONV] line to
 * stdout for the Electron host to capture.
 *
 * Only active when COPILOT_PROXY_CONVERSATION_LOG=1 environment variable is set.
 * Does not modify any request or response — purely observational.
 */

import type { Context, Next } from 'hono'
import { randomUUID } from 'node:crypto'

// ─── Session tracking ──────────────────────────────────────────────

const SESSION_TIMEOUT_MS = 15 * 60 * 1000
const sessions = new Map<string, { id: string, lastSeen: number }>()

function getSessionId(clientType: string, model: string): string {
  const key = `${clientType}::${model}`
  const now = Date.now()
  const existing = sessions.get(key)
  if (existing && (now - existing.lastSeen) < SESSION_TIMEOUT_MS) {
    existing.lastSeen = now
    return existing.id
  }
  const id = `sess_${randomUUID().slice(0, 8)}`
  sessions.set(key, { id, lastSeen: now })
  return id
}

// ─── Client detection ──────────────────────────────────────────────

function detectClientType(path: string, headers: Headers): string {
  if (path.includes('/messages')) {
    const beta = headers.get('anthropic-beta') || ''
    if (/claude-code/.test(beta)) return 'claude-code'
    return 'anthropic'
  }
  if (path.includes('/responses')) return 'openai-responses'

  const ua = (headers.get('user-agent') || '').toLowerCase()
  if (ua.includes('cursor')) return 'cursor'
  if (ua.includes('continue')) return 'continue'
  if (ua.includes('cline')) return 'cline'
  return 'openai'
}

// ─── Emit helper ───────────────────────────────────────────────────

function emit(data: Record<string, unknown>): void {
  process.stdout.write(`[CONV]${JSON.stringify(data)}\n`)
}

// ─── Extract readable content from various response formats ────────

function extractResponseContent(body: any, _path: string): { content?: string, toolCalls?: unknown[], usage?: unknown } {
  if (!body) return {}

  // OpenAI chat completions format
  if (body.choices?.[0]?.message) {
    const msg = body.choices[0].message
    return {
      content: msg.content || undefined,
      toolCalls: msg.tool_calls?.length ? msg.tool_calls : undefined,
      usage: body.usage,
    }
  }

  // Anthropic messages format
  if (body.content && Array.isArray(body.content)) {
    return {
      content: body.content.filter((b: any) => b.type === 'text').map((b: any) => b.text).join(''),
      toolCalls: body.content.filter((b: any) => b.type === 'tool_use'),
      usage: body.usage,
    }
  }

  // Responses API format
  if (body.output) {
    const texts = Array.isArray(body.output)
      ? body.output.filter((o: any) => o.type === 'message').flatMap((o: any) =>
        (o.content || []).filter((c: any) => c.type === 'output_text').map((c: any) => c.text),
      )
      : [String(body.output)]
    return {
      content: texts.join('') || undefined,
      usage: body.usage,
    }
  }

  return {}
}

// ─── SSE stream collector ──────────────────────────────────────────

function collectSSEStream(
  stream: ReadableStream<Uint8Array>,
  _path: string,
): { passthrough: ReadableStream<Uint8Array>, collected: Promise<{ content: string, toolCalls?: unknown[], usage?: unknown }> } {
  const [branch1, branch2] = stream.tee()
  const decoder = new TextDecoder()

  const collected = (async () => {
    const parts: string[] = []
    const toolCalls: unknown[] = []
    let usage: unknown
    const reader = branch2.getReader()

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = decoder.decode(value, { stream: true })
        // Parse SSE lines
        for (const line of text.split('\n')) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') continue
          try {
            const parsed = JSON.parse(data)

            // OpenAI chat completion chunks
            const delta = parsed.choices?.[0]?.delta
            if (delta?.content) parts.push(delta.content)
            if (delta?.tool_calls) toolCalls.push(...delta.tool_calls)
            if (parsed.usage) usage = parsed.usage

            // Anthropic content_block_delta
            if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
              parts.push(parsed.delta.text)
            }
            if (parsed.type === 'message_delta' && parsed.usage) {
              usage = parsed.usage
            }

            // Responses API events
            if (parsed.type === 'response.output_text.delta' && parsed.delta) {
              parts.push(parsed.delta)
            }
          } catch { /* skip unparseable lines */ }
        }
      }
    } catch { /* stream read error */ } finally {
      reader.releaseLock()
    }

    return {
      content: parts.join(''),
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      usage,
    }
  })()

  return { passthrough: branch1, collected }
}

// ─── Middleware ─────────────────────────────────────────────────────

const TARGET_PATHS = ['/messages', '/completions', '/responses']

export async function conversationMiddleware(c: Context, next: Next) {
  // Skip if not enabled
  if (process.env.COPILOT_PROXY_CONVERSATION_LOG !== '1') {
    return next()
  }

  // Only intercept POST requests to API endpoints
  if (c.req.method !== 'POST') return next()
  const path = c.req.path
  if (!TARGET_PATHS.some(p => path.includes(p))) return next()

  const startTime = Date.now()

  // Clone request to read body without consuming it
  let requestBody: any = null
  try {
    requestBody = await c.req.json()
    // Hono caches parsed JSON, so the handler will still get it
  } catch { /* ignore */ }

  // Run the actual handler
  await next()

  // After handler completes, capture response
  const res = c.res
  if (!res) return

  const clientType = detectClientType(path, c.req.raw.headers)
  const model = requestBody?.model || 'unknown'
  const isStream = (res.headers.get('content-type') || '').includes('text/event-stream')

  const baseEntry = {
    id: randomUUID(),
    timestamp: new Date().toISOString(),
    sessionId: getSessionId(clientType, model),
    endpoint: path,
    client: {
      type: clientType,
      anthropicBeta: c.req.header('anthropic-beta'),
    },
    model,
    stream: isStream,
    request: {
      messages: requestBody?.messages,
      system: requestBody?.system,
      input: requestBody?.input,
    },
  }

  if (isStream && res.body) {
    // Streaming: tee the response stream
    const { passthrough, collected } = collectSSEStream(res.body, path)

    // Replace response body with the passthrough branch (client gets original data)
    c.res = new Response(passthrough, {
      status: res.status,
      headers: res.headers,
    })

    // Collect in background, emit when done
    collected.then((responseData) => {
      emit({
        ...baseEntry,
        response: responseData,
        durationMs: Date.now() - startTime,
      })
    }).catch(() => { /* ignore collection errors */ })
  } else {
    // Non-streaming: clone and read response body
    try {
      const cloned = res.clone()
      const body = await cloned.json()
      const responseData = extractResponseContent(body, path)
      emit({
        ...baseEntry,
        response: responseData,
        durationMs: Date.now() - startTime,
      })
    } catch { /* ignore */ }
  }
}
