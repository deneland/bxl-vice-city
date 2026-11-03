import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';

const METRICS_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../data/metrics');

/** Dev-only sink for viewer telemetry: POST /__metrics -> data/metrics/sessions.jsonl */
function metricsSink(): Plugin {
  return {
    name: 'metrics-sink',
    apply: 'serve',
    configureServer(server) {
      mkdirSync(resolve(METRICS_DIR, 'snapshots'), { recursive: true });
      server.middlewares.use('/__metrics', (req, res) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end(); return; }
        const parts: Buffer[] = [];
        req.on('data', (c: Buffer) => parts.push(c));
        req.on('end', () => {
          try {
            const msg = JSON.parse(Buffer.concat(parts).toString('utf8'));
            msg.received = new Date().toISOString();
            if (msg.type === 'snapshot' && typeof msg.image === 'string') {
              const png = Buffer.from(msg.image.split(',')[1] ?? '', 'base64');
              const file = resolve(METRICS_DIR, 'snapshots', `${msg.id}.png`);
              writeFileSync(file, png);
              msg.image = file;
            }
            appendFileSync(resolve(METRICS_DIR, 'sessions.jsonl'), JSON.stringify(msg) + '\n');
            res.statusCode = 204;
          } catch (e) {
            res.statusCode = 400;
          }
          res.end();
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [metricsSink()],
  server: { host: '127.0.0.1', port: 5173 },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
