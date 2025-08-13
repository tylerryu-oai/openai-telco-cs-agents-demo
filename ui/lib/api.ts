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
