// Helper to call the server
async function postWithRetry(path: string, body: any, retries = 0, backoffMs = 300) {
  let attempt = 0;
  while (attempt <= retries) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 300000); // 5 min timeout for long turns
      const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        console.warn(`${path} non-OK:`, res.status, text);
        throw new Error(`status ${res.status}`);
      }
      return await res.json();
    } catch (err) {
      if (attempt === retries) {
        console.warn(`${path} failed after retries:`, err);
        return null;
      }
      await new Promise((r) => setTimeout(r, backoffMs * Math.pow(2, attempt)));
      attempt++;
    }
  }
  return null;
}

export async function callChatAPI(message: string, conversationId: string) {
  return postWithRetry("/api/chat", { conversation_id: conversationId, message });
}

export async function callHumanReplyAPI(message: string, conversationId: string) {
  return postWithRetry("/api/human_reply", { conversation_id: conversationId, message });
}

export async function callHumanBackAPI(conversationId: string) {
  return postWithRetry("/api/human_back", { conversation_id: conversationId });
}

export type StreamEvent = {
  event: string;
  data: any;
};

export async function startChatStream(
  message: string,
  conversationId: string,
  onEvent: (e: StreamEvent) => void
): Promise<void> {
  const res = await fetch("/api/chat/stream?mode=runner", {
    // Use runner mode by default to preserve agent workflow and handoffs
    // Switch to direct by appending ?mode=direct if needed
    // e.g., fetch("/api/chat/stream?mode=direct", ...)
    cache: "no-store",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, message }),
    // Append mode=runner to ensure handovers/tools via Runner path
    // Note: Next.js route reads this query param
  });
  if (!res.ok || !res.body) {
    throw new Error(`stream status ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    // SSE frames are separated by blank line
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const lines = frame.split("\n");
      let event = "message";
      let dataStr = "";
      for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }
      if (dataStr) {
        try {
          const data = JSON.parse(dataStr);
          onEvent({ event, data });
        } catch {
          // ignore parse errors
        }
      }
    }
  }
}
