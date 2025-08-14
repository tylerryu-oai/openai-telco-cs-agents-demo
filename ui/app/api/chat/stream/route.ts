export const runtime = 'nodejs';

export async function POST(req: Request) {
  const url = new URL(req.url);
  const mode = url.searchParams.get('mode') || 'direct';
  const target = mode === 'runner' ? 'http://127.0.0.1:8000/chat_stream' : 'http://127.0.0.1:8000/chat_stream_direct';
  const r = await fetch(target, {
    method: 'POST',
    body: await req.arrayBuffer(),
    headers: { 'Content-Type': 'application/json', 'Connection': 'keep-alive' },
  });
  // Proxy the streaming body through unchanged
  const headers = new Headers(r.headers);
  headers.set('Content-Type', 'text/event-stream');
  headers.set('Cache-Control', 'no-cache, no-transform');
  headers.set('Connection', 'keep-alive');
  headers.set('X-Accel-Buffering', 'no');
  return new Response(r.body, { status: r.status, headers });
}


