# Robot learning loop (simulation)

Run -> Capture -> Find failure -> Fix (Codex brain) -> Run again (regression gate), on a simulated Franka Panda
in MuJoCo. Built as a starter for the Alloy Robotics x OpenAI Physical AI Hackathon (Sydney, 5 Dec 2026).

## Setup
```bash
uv sync
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda trs_so_arm100 unitree_go2
```

## See the robots
```bash
uv run python scripts/show_robots.py              # PNGs in out/robots/
uv run python scripts/show_robots.py --view panda # interactive window (also so_arm100, go2)
```

## Run the loop
```bash
uv run python -m rll.loop --brain heuristic --iters 4   # offline rules
uv run python -m rll.loop --brain codex --iters 4       # Codex reads report + controller + keyframes
uv run python scripts/compare_runs.py <run_a> <run_b>   # out/compare.png
uv run pytest -q
```
Each run writes `out/runs/<run_id>/`: per-iteration failure report, replayed failure episodes
(`steps.parquet`, `video.mp4`, `keyframes.png`), the brain's workspace and proposal, gate result,
accepted `params_vN.json`, `history.json`, `success_curve.png`.

## Pieces
| Loop step | Code |
|---|---|
| Run | `rll/env.py` (seeded domain randomization), `policy/controller.py` + `policy/params.json` |
| Capture | `rll/runner.py` (every control step: poses, contacts, gripper, phase), `rll/replay.py` (bit-exact replay + video) |
| Find failure | `rll/detect.py` (knocked / missed_grasp / slip / misplaced / timeout + step + phase), `rll/analyze.py` (drivers by tercile) |
| Fix | `rll/brains.py`: `CodexBrain` = `codex exec -s read-only --output-schema`, gets history of accepted/rejected tries |
| Run again | `rll/gate.py`: 100 held-out seeds, accept only on gain with <=3% regressions |
| Store | `rll/storage.py` -> Cloudflare Worker in `cloud/` (R2 blobs, D1 iteration index, dashboard at `/`) |

## Cloudflare
```bash
cd cloud && npm i
npx wrangler d1 migrations apply rll --local && npx wrangler dev --local   # local R2 + D1
RLL_WORKER_URL=http://127.0.0.1:8787 uv run python -m rll.loop --iters 2
# deploy: wrangler login; wrangler r2 bucket create robot-learning-runs; wrangler d1 create rll
#   (paste database_id into wrangler.jsonc); wrangler d1 migrations apply rll --remote;
#   wrangler secret put API_TOKEN; wrangler deploy; then set RLL_WORKER_URL + RLL_WORKER_TOKEN
```
