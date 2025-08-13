export const runtime = 'nodejs';

export async function POST(req: Request) {
  const body = await req.arrayBuffer();
  const r = await fetch('http://127.0.0.1:8000/chat', {
    method: 'POST',
    body,
    headers: { 'Content-Type': 'application/json', 'Connection': 'close' },
  });
  const buf = await r.arrayBuffer();
  return new Response(buf, { status: r.status, headers: { 'Content-Type': 'application/json' } });
}


