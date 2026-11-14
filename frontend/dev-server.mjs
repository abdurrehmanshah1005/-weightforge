import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('./public/', import.meta.url));
const backend = new URL(process.env.WEIGHTFORGE_BACKEND_URL || 'http://127.0.0.1:8001');
http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname.startsWith('/api/')) {
    const upstream = http.request({ hostname: backend.hostname, port: backend.port,
      path: url.pathname.slice(4) + url.search, method: req.method,
      headers: { ...req.headers, host: backend.host } }, reply => {
      res.writeHead(reply.statusCode, reply.headers); reply.pipe(res);
    });
    upstream.on('error', () => { if (!res.headersSent) res.writeHead(503, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ detail: 'Backend unavailable. Start the API on port 8001.' })); });
    req.pipe(upstream); req.on('aborted', () => upstream.destroy()); return;
  }
  let relative;
  try { relative = decodeURIComponent(url.pathname).replace(/^\/static\//, '/'); }
  catch { res.writeHead(400); res.end(); return; }
  let full = path.resolve(root, '.' + relative);
  if (full !== path.resolve(root) && !full.startsWith(root)) { res.writeHead(403); res.end(); return; }
  if (relative === '/') full = path.join(root, 'index.html');
  else if (!path.extname(full)) full += '.html';
  if (!fs.existsSync(full) || !fs.statSync(full).isFile()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': ({ '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript' })[path.extname(full)] || 'application/octet-stream' });
  fs.createReadStream(full).pipe(res);
}).listen(8000, '127.0.0.1', () => console.log('WeightForge frontend: http://127.0.0.1:8000'));
