/**
 * Robot learning loop storage.
 *   PUT  /blobs/<key>                 store an episode file in R2
 *   GET  /blobs/<key>                 fetch it back (videos, keyframes, parquet)
 *   POST /runs/<run_id>/iterations    index one loop iteration in D1
 *   GET  /runs                        every run with its first/latest held-out success
 *   GET  /runs/<run_id>               all iterations of a run
 *   GET  /                            small dashboard
 */
export interface Env {
  RUNS: R2Bucket;
  DB: D1Database;
  API_TOKEN?: string;
}

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data, null, 2), { status, headers: { "content-type": "application/json" } });

function authorized(req: Request, env: Env): boolean {
  if (!env.API_TOKEN) return true; // local dev
  return req.headers.get("authorization") === `Bearer ${env.API_TOKEN}`;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const path = decodeURIComponent(url.pathname);
    const write = req.method === "PUT" || req.method === "POST";
    if (write && !authorized(req, env)) return json({ error: "unauthorized" }, 401);

    if (path.startsWith("/blobs/")) {
      const key = path.slice("/blobs/".length);
      if (!key || key.includes("..")) return json({ error: "bad key" }, 400);
      if (req.method === "PUT") {
        const obj = await env.RUNS.put(key, req.body, {
          httpMetadata: { contentType: req.headers.get("content-type") ?? "application/octet-stream" },
        });
        return json({ key, size: obj?.size ?? 0 });
      }
      if (req.method === "GET") {
        const obj = await env.RUNS.get(key);
        if (!obj) return json({ error: "not found" }, 404);
        const headers = new Headers();
        obj.writeHttpMetadata(headers);
        return new Response(obj.body, { headers });
      }
    }

    const iterMatch = path.match(/^\/runs\/([\w.-]+)\/iterations$/);
    if (iterMatch && req.method === "POST") {
      const r = (await req.json()) as Record<string, any>;
      await env.DB.prepare(
        `INSERT OR REPLACE INTO iterations
         (run_id, iteration, version, brain, heldout_success, candidate_success, train_success, accepted,
          changes, failure_counts, diagnosis, params, seconds)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)`,
      )
        .bind(
          iterMatch[1], r.iteration, r.version, r.brain ?? null, r.heldout_success, r.candidate_success ?? null,
          r.train_success ?? null, r.accepted ? 1 : 0, JSON.stringify(r.changes ?? {}),
          JSON.stringify(r.failure_counts ?? {}), r.diagnosis ?? null, JSON.stringify(r.params ?? {}), r.seconds ?? null,
        )
        .run();
      return json({ ok: true });
    }

    if (path === "/runs" && req.method === "GET") {
      const { results } = await env.DB.prepare(
        `SELECT run_id, brain, COUNT(*) AS iterations, SUM(accepted) AS accepted,
                MIN(train_success) AS worst_train, MAX(heldout_success) AS best_heldout, MAX(created_at) AS updated
         FROM iterations GROUP BY run_id ORDER BY updated DESC`,
      ).all();
      return json(results);
    }

    const runMatch = path.match(/^\/runs\/([\w.-]+)$/);
    if (runMatch && req.method === "GET") {
      const { results } = await env.DB.prepare(`SELECT * FROM iterations WHERE run_id = ?1 ORDER BY iteration`)
        .bind(runMatch[1])
        .all();
      return json(results.map((r: any) => ({
        ...r, changes: JSON.parse(r.changes), failure_counts: JSON.parse(r.failure_counts), params: JSON.parse(r.params),
      })));
    }

    if (path === "/" && req.method === "GET") return new Response(DASHBOARD, { headers: { "content-type": "text/html" } });
    return json({ error: "not found" }, 404);
  },
};

const DASHBOARD = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Learning Loop</title>
<style>body{font:14px system-ui;margin:24px;max-width:900px;color:#0b0b0b;background:#fcfcfb}
table{border-collapse:collapse;width:100%}td,th{padding:6px 8px;border-bottom:1px solid #e6e5e0;text-align:left;vertical-align:top}
.no{color:#52514e}code{font-size:12px}</style></head><body>
<h1>Robot learning loop runs</h1><div id="runs">loading…</div><div id="detail"></div>
<script>
const pct = v => v == null ? "–" : Math.round(v * 100) + "%";
async function load() {
  const runs = await (await fetch("/runs")).json();
  document.getElementById("runs").innerHTML = "<table><tr><th>Run</th><th>Brain</th><th>Iterations</th><th>Accepted</th><th>Best held-out</th></tr>" +
    runs.map(r => '<tr><td><a href="#" data-run="' + r.run_id + '">' + r.run_id + "</a></td><td>" + r.brain + "</td><td>" +
      r.iterations + "</td><td>" + r.accepted + "</td><td>" + pct(r.best_heldout) + "</td></tr>").join("") + "</table>";
  document.querySelectorAll("[data-run]").forEach(a => a.onclick = e => { e.preventDefault(); show(a.dataset.run); });
}
async function show(id) {
  const its = await (await fetch("/runs/" + id)).json();
  document.getElementById("detail").innerHTML = "<h2>" + id + "</h2><table><tr><th>#</th><th>Held-out</th><th>Candidate</th><th>Gate</th><th>Changes</th><th>Diagnosis</th></tr>" +
    its.map(i => "<tr><td>" + i.iteration + "</td><td>" + pct(i.heldout_success) + "</td><td>" + pct(i.candidate_success) + "</td><td>" +
      (i.accepted ? "accepted" : '<span class="no">rejected</span>') + "</td><td><code>" + JSON.stringify(i.changes) + "</code></td><td>" +
      (i.diagnosis || "") + "</td></tr>").join("") + '</table><p><img style="max-width:100%" src="/blobs/runs/' + id + '/success_curve.png"></p>';
}
load();
</script></body></html>`;
