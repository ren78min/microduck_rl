"""Headless evaluation battery for the OneLegHop task.

Runs each policy for exactly ONE episode per env (num_envs episodes, so envs
that fail fast don't contribute more episodes than survivors), under
deploy-like conditions (final-stage DR ranges, pushes on unless --no-pushes),
and prints one comparison table.

Policies (repeat --policy to compare several in one table):
    zero                 hold HOME (action 0) — never lifts a foot
    random               N(0, 1) actions ≈ an untrained policy (init std 1.0)
    path/to/policy.onnx  exported policy (normalizer baked — the deploy path)
    path/to/model_N.pt   local checkpoint, exported to ONNX first

Examples:
    uv run scripts/eval_one_leg_hop.py --policy zero --policy random \\
        --policy logs/rsl_rl/one_leg_hop_right/<run>/model_4999.pt
    uv run scripts/eval_one_leg_hop.py --policy policy.onnx --no-pushes --json out.json

Metrics (per episode, aggregated over episodes):
    success     episode reached the 10 s time-out with no failure
    failure     which termination fired (fell_over / illegal_contact / nan_state)
    survival    episode duration (s)
    speed       net displacement along the INITIAL heading / duration (m/s) —
                ground truth "how fast did it go", independent of the reward
    path speed  ∫ heading-frame forward velocity dt / duration (what the
                reward integrates; > speed when the robot curves)
    lateral     |sideways displacement| w.r.t. the initial heading (m)
    yaw drift   |heading change| start → end (deg)
    hops/s      support-foot touch-downs after a ≥ 30 ms flight with the free
                foot up as well (flights before the support foot's first
                touch-down are the spawn drop, not hops — ignored)
    flight      mean flight duration (ms), and % of post-grace time airborne
    free up     % of post-grace time the free foot is off the ground
    peak |a_z|  largest trunk vertical acceleration after the first 0.3 s
                (landing harshness — gearbox load proxy on the real robot)
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks import microduck_one_leg_hop_env_cfg as hop

TASK_ID = "Mjlab-OneLegHop-Flat-MicroDuck"
FOOT_INDEX = {"left": 0, "right": 1}

# Final-stage DR (end of the training curricula) — the conditions the policy
# must hold up under, not the easy stage-0 values a play cfg starts at.
FINAL_COM_RANGE = 0.015
FINAL_HEAD_COM_RANGE = 0.01

MIN_FLIGHT_S = hop.HOP_AIR_TIME_MIN
IGNORE_AZ_FIRST_S = 0.3


def build_env(num_envs: int, device: str, hop_leg: str, pushes: bool) -> ManagerBasedRlEnv:
    cfg = hop.make_microduck_one_leg_hop_env_cfg(play=True, hop_leg=hop_leg)
    cfg.scene.num_envs = num_envs
    # Curricula would restart at stage 0 here; pin the DR they control instead.
    cfg.curriculum = {}
    if "randomize_com" in cfg.events:
        cfg.events["randomize_com"].params["ranges"] = (-FINAL_COM_RANGE, FINAL_COM_RANGE)
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["ranges"] = (-FINAL_HEAD_COM_RANGE, FINAL_HEAD_COM_RANGE)
    if not pushes:
        cfg.events.pop("push_robot", None)
    return ManagerBasedRlEnv(cfg=cfg, device=device)


class OnnxPolicy:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        # The export fixes batch = 1 (the runtime's contract) → feed row by row.
        self.batch1 = inp.shape[0] == 1

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.detach().cpu().numpy().astype(np.float32)
        if self.batch1:
            out = np.concatenate([self.sess.run(None, {self.inp: x[i : i + 1]})[0] for i in range(len(x))])
        else:
            out = self.sess.run(None, {self.inp: x})[0]
        return torch.as_tensor(out, device=obs.device)


def load_policy(spec: str, hop_leg: str):
    if spec == "zero":
        return "zero", lambda obs: torch.zeros(obs.shape[0], 14, device=obs.device)
    if spec == "random":
        return "random", lambda obs: torch.randn(obs.shape[0], 14, device=obs.device)
    p = Path(spec)
    if p.suffix == ".onnx":
        return p.name, OnnxPolicy(str(p))
    if p.suffix == ".pt":
        from mjlab_microduck.export import ExportConfig, run_export

        onnx_path = Path(tempfile.mkdtemp()) / f"{p.stem}.onnx"
        task = TASK_ID if hop_leg == hop.HOP_LEG else None
        assert task, "local .pt export only for the configured HOP_LEG; export to ONNX yourself"
        run_export(task, ExportConfig(onnx_file=str(onnx_path), checkpoint_file=str(p)))
        return f"{p.parent.name[-20:]}/{p.name}", OnnxPolicy(str(onnx_path))
    raise ValueError(f"unknown policy spec: {spec}")


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@torch.no_grad()
def evaluate(env: ManagerBasedRlEnv, policy, hop_leg: str, seed: int) -> dict:
    torch.manual_seed(seed)
    N, dt, dev = env.num_envs, env.step_dt, env.device
    robot = env.scene["robot"]
    feet = env.scene.sensors["feet_ground_contact"]
    sup, free = FOOT_INDEX[hop_leg], FOOT_INDEX["left" if hop_leg == "right" else "right"]
    term_names = [n for n in env.termination_manager.active_terms if n != "time_out"]
    max_steps = int(round(env.max_episode_length_s / dt)) + 5

    obs, _ = env.reset()

    def z(): return torch.zeros(N, device=dev)
    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    start_yaw = _yaw(robot.data.root_link_quat_w).clone()
    last_xy, last_yaw = start_xy.clone(), start_yaw.clone()
    steps, post_grace, free_up, airborne_steps = z(), z(), z(), z()
    path, flight, n_hops, flight_total, peak_az = z(), z(), z(), z(), z()
    touched = torch.zeros(N, dtype=torch.bool, device=dev)  # support foot landed once
    prev_vz = robot.data.root_link_lin_vel_w[:, 2].clone()
    done_once = torch.zeros(N, dtype=torch.bool, device=dev)
    outcome = ["" for _ in range(N)]
    rec = {k: torch.full((N,), float("nan"), device=dev) for k in (
        "survival", "speed", "path_speed", "lateral", "yaw_drift", "hops_per_s",
        "flight_ms", "airborne_pct", "free_up_pct", "peak_az")}

    for _ in range(max_steps):
        if done_once.all():
            break
        obs, _, terminated, truncated, _ = env.step(policy(obs["actor"]))
        dones = (terminated | truncated).bool()
        live = ~dones  # state of done envs is already the NEXT episode's

        # ── per-step accumulation on envs still inside their episode ──────────
        steps += live.float()
        t = steps * dt
        vx, _ = m._heading_frame_lin_vel(robot)
        path += vx * dt * live
        found = feet.data.found
        sup_down, free_down = found[:, sup] > 0, found[:, free] > 0
        g = live & (t > hop.ONE_LEG_GRACE_S)
        post_grace += g.float()
        free_up += (g & ~free_down).float()
        touched |= live & sup_down
        in_flight = live & touched & ~sup_down & ~free_down
        flight = torch.where(in_flight, flight + dt, flight)
        landed = live & sup_down & (flight >= MIN_FLIGHT_S)
        n_hops += landed.float()
        flight_total += torch.where(landed, flight, torch.zeros_like(flight))
        airborne_steps += (g & in_flight).float()
        flight = torch.where(sup_down | free_down, torch.zeros_like(flight), flight)
        vz = robot.data.root_link_lin_vel_w[:, 2]
        az = ((vz - prev_vz) / dt).abs()
        peak_az = torch.where(live & (t > IGNORE_AZ_FIRST_S), torch.maximum(peak_az, az), peak_az)
        prev_vz = vz.clone()
        last_xy = torch.where(live.unsqueeze(-1), robot.data.root_link_pos_w[:, :2], last_xy)
        last_yaw = torch.where(live, _yaw(robot.data.root_link_quat_w), last_yaw)

        # ── finalize each env's FIRST episode ─────────────────────────────────
        fin = dones & ~done_once
        if fin.any():
            dur = (steps * dt).clamp(min=dt)
            d = last_xy - start_xy
            hx, hy = torch.cos(start_yaw), torch.sin(start_yaw)
            fwd, lat = d[:, 0] * hx + d[:, 1] * hy, -d[:, 0] * hy + d[:, 1] * hx
            yaw_d = torch.rad2deg(torch.atan2(torch.sin(last_yaw - start_yaw), torch.cos(last_yaw - start_yaw)).abs())
            pg = post_grace.clamp(min=1)
            vals = {
                "survival": dur, "speed": fwd / dur, "path_speed": path / dur,
                "lateral": lat.abs(), "yaw_drift": yaw_d, "hops_per_s": n_hops / dur,
                "flight_ms": torch.where(n_hops > 0, 1000 * flight_total / n_hops.clamp(min=1), torch.full_like(dur, float("nan"))),
                "airborne_pct": 100 * airborne_steps / pg, "free_up_pct": 100 * free_up / pg,
                "peak_az": peak_az,
            }
            for k, v in vals.items():
                rec[k] = torch.where(fin, v, rec[k])
            tout = env.termination_manager.time_outs
            for i in fin.nonzero().flatten().tolist():
                if tout[i] and not terminated[i]:
                    outcome[i] = "success"
                else:
                    hit = [n for n in term_names if env.termination_manager.get_term(n)[i]]
                    outcome[i] = "+".join(hit) or "terminated"
            done_once |= fin

        # reset accumulators of every env that just reset (bookkeeping only
        # matters for first episodes, but keep it uniform)
        if dones.any():
            for buf in (steps, post_grace, free_up, airborne_steps, path, flight, n_hops, flight_total, peak_az):
                buf[dones] = 0.0
            start_xy[dones] = robot.data.root_link_pos_w[dones, :2]
            start_yaw[dones] = _yaw(robot.data.root_link_quat_w[dones])
            last_xy[dones], last_yaw[dones] = start_xy[dones], start_yaw[dones]
            prev_vz[dones] = robot.data.root_link_lin_vel_w[dones, 2]
            touched[dones] = False

    assert done_once.all(), "some envs never finished an episode"
    return {k: v.cpu().numpy() for k, v in rec.items()} | {"outcome": outcome}


def summarize(name: str, r: dict) -> dict:
    out = np.array(r["outcome"])
    ok = out == "success"
    n = len(out)

    def pct(a, q): return float(np.nanpercentile(a, q)) if np.isfinite(a).any() else float("nan")
    def mean(a): return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

    fails = {k: int(c) for k, c in zip(*np.unique(out[~ok], return_counts=True))} if (~ok).any() else {}
    s = {
        "policy": name, "episodes": n,
        "success_pct": 100 * ok.mean(),
        "failures": fails,
        "survival_s_p50": pct(r["survival"], 50),
        "speed_mean": mean(r["speed"]),
        "speed_p10": pct(r["speed"], 10), "speed_p50": pct(r["speed"], 50), "speed_p90": pct(r["speed"], 90),
        "speed_success_mean": mean(r["speed"][ok]) if ok.any() else float("nan"),
        "path_speed_mean": mean(r["path_speed"]),
        "lateral_m_p50": pct(r["lateral"], 50),
        "yaw_drift_deg_p50": pct(r["yaw_drift"], 50),
        "hops_per_s": mean(r["hops_per_s"]),
        "flight_ms": mean(r["flight_ms"]),
        "airborne_pct": mean(r["airborne_pct"]),
        "free_up_pct": mean(r["free_up_pct"]),
        "peak_az_p50": pct(r["peak_az"], 50), "peak_az_p95": pct(r["peak_az"], 95),
    }
    return s


ROWS = [
    ("episodes", "episodes", "{:.0f}"),
    ("success_pct", "success (10 s, no failure) %", "{:.1f}"),
    ("failures", "failures", None),
    ("survival_s_p50", "survival p50 (s)", "{:.2f}"),
    ("speed_success_mean", "SPEED, successful eps (m/s)", "{:+.3f}"),
    ("speed_mean", "speed all eps, incl. falls (m/s)", "{:+.3f}"),
    ("speed_p10", "  p10", "{:+.3f}"),
    ("speed_p50", "  p50", "{:+.3f}"),
    ("speed_p90", "  p90", "{:+.3f}"),
    ("path_speed_mean", "path speed mean (m/s)", "{:+.3f}"),
    ("lateral_m_p50", "lateral drift p50 (m)", "{:.3f}"),
    ("yaw_drift_deg_p50", "yaw drift p50 (deg)", "{:.1f}"),
    ("hops_per_s", "hops / s", "{:.2f}"),
    ("flight_ms", "flight per hop (ms)", "{:.0f}"),
    ("airborne_pct", "airborne after grace %", "{:.1f}"),
    ("free_up_pct", "free foot up after grace %", "{:.1f}"),
    ("peak_az_p50", "peak |a_z| p50 (m/s²)", "{:.0f}"),
    ("peak_az_p95", "peak |a_z| p95 (m/s²)", "{:.0f}"),
]


def print_table(summaries: list[dict]) -> None:
    w0 = max(len(r[1]) for r in ROWS)
    cols = [s["policy"] for s in summaries]
    w = max(14, *(len(c) for c in cols))
    print("\n" + "metric".ljust(w0) + " | " + " | ".join(c.rjust(w) for c in cols))
    print("-" * (w0 + (w + 3) * len(cols) + 1))
    for key, label, fmt in ROWS:
        cells = []
        for s in summaries:
            v = s[key]
            if fmt is None:
                cells.append(", ".join(f"{k} {c}" for k, c in v.items()) or "-")
            else:
                cells.append("n/a" if isinstance(v, float) and math.isnan(v) else fmt.format(v))
        if fmt is None and any(len(c) > w for c in cells):
            print(label.ljust(w0) + " |")
            for col, c in zip(cols, cells):
                print(f"  {col}: {c}")
        else:
            print(label.ljust(w0) + " | " + " | ".join(c.rjust(w) for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--num-envs", type=int, default=512, help="= episodes per policy")
    ap.add_argument("--hop-leg", default=hop.HOP_LEG, choices=("right", "left"))
    ap.add_argument("--no-pushes", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    policies = [load_policy(p, args.hop_leg) for p in args.policy]
    env = build_env(args.num_envs, args.device, args.hop_leg, pushes=not args.no_pushes)
    summaries = []
    for name, pol in policies:
        print(f"[eval] {name}: {args.num_envs} episodes, hop leg {args.hop_leg}, "
              f"pushes {'off' if args.no_pushes else 'on'}")
        summaries.append(summarize(name, evaluate(env, pol, args.hop_leg, args.seed)))
    print_table(summaries)
    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
