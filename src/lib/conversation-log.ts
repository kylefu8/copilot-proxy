/**
 * Conversation logging utility.
 *
 * Emits structured conversation records to stdout with a [CONV] prefix
 * so the Electron host can capture and store them separately from regular logs.
 *
 * Only emits when COPILOT_PROXY_CONVERSATION_LOG=1 environment variable is set.
 */

import { randomUUID } from 'node:crypto'
import { isClaudeCodeRequest } from '~/routes/messages/count-tokens-handler'

export interface ConversationEntry {
  id: string
  timestamp: string
  endpoint: string
  client: {
    type: string // 'claude-code' | 'anthropic' | 'openai' | 'openai-responses'
    userAgent?: string
    anthropicBeta?: string
  }
  model: string
  stream: boolean
  request: {
    messages?: unknown[]
    system?: unknown
    input?: unknown
  }
  response: {
    content?: string
    toolCalls?: unknown[]
    usage?: unknown
  }
  durationMs: number
}

// Session tracking: grouped by (clientType + model), timeout 15 min
const sessions = new Map<string, { id: string, lastSeen: number }>()
const SESSION_TIMEOUT_MS = 15 * 60 * 1000

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

export function isConversationLogEnabled(): boolean {
  return process.env.COPILOT_PROXY_CONVERSATION_LOG === '1'
}

export function detectClientType(
  endpoint: string,
  anthropicBeta?: string,
  userAgent?: string,
): string {
  if (endpoint.includes('/messages')) {
    if (isClaudeCodeRequest(anthropicBeta)) return 'claude-code'
    return 'anthropic'
  }
  if (endpoint.includes('/responses')) return 'openai-responses'

  // OpenAI chat-completions — try to distinguish by User-Agent
  if (userAgent) {
    const ua = userAgent.toLowerCase()
    if (ua.includes('cursor')) return 'cursor'
    if (ua.includes('continue')) return 'continue'
    if (ua.includes('cline')) return 'cline'
  }
  return 'openai'
}

export function emitConversation(entry: ConversationEntry & { sessionId?: string }): void {
  if (!isConversationLogEnabled()) return

  const sessionId = entry.sessionId ?? getSessionId(entry.client.type, entry.model)
  const record = { ...entry, sessionId }

  // Write as a single JSON line with [CONV] prefix for Electron to intercept
  const line = `[CONV]${JSON.stringify(record)}`
  process.stdout.write(`${line}\n`)
}

/**
 * Helper to collect streaming text chunks into a complete response string.
 */
export function createStreamCollector() {
  const parts: string[] = []
  let toolCalls: unknown[] = []
  let usage: unknown = undefined

  return {
    addText(text: string) {
      parts.push(text)
    },
    addToolCall(tc: unknown) {
      toolCalls.push(tc)
    },
    setUsage(u: unknown) {
      usage = u
    },
    getResult() {
      return {
        content: parts.join(''),
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage,
      }
    },
  }
}
