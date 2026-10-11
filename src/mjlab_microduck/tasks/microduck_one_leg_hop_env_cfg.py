"""Microduck OneLegHop — hop forward on ONE leg (HOP_LEG flag) as fast as
possible without falling, on flat ground.

Perpetual policy: the robot starts STANDING on both feet (HOME + the velocity
recipe's reset noise), shifts onto the hopping leg, tucks the free leg, and
hops forward along its own heading. Train the right-leg and left-leg policies
as separate runs (flip HOP_LEG; wandb experiment name follows it).

Built on make_microduck_velocity_env_cfg, so DR / obs noise / IMU+encoder DR /
delays / NaN guards / the 61D actor obs layout are the proven walking recipe.
Only the robot model, sensors, terminations, rewards and curricula change.

Key design decisions:
  - ROBOT = groundcontact model (trunk, head, both hips/shanks collide with
    the floor). The walk model collides on the feet only: a hopper that drags
    its tucked knee or face-plants would pass through the floor there.
  - "Do not fail" is a HARD gate, not a nudge: the episode terminates when
    anything but the support foot touches the floor (trunk/head/hips/shanks:
    always; free foot: after ONE_LEG_GRACE_S of lift-off grace), or tilt
    > 70°. A one-shot failure penalty makes the price explicit on top of the
    lost future reward.
  - Speed reward is LINEAR in heading-frame forward speed (not a saturating
    tracking Gaussian): "maximum velocity" needs gradient at high speed. It
    integrates to distance hopped, so it can't be farmed by oscillating, and
    it is paid only while the free foot is up.
  - Support-foot air time (bounded window) rewards a real flight phase so the
    policy hops rather than stick-slip shuffles on one foot.
  - No velocity command: the twist slot is zero-padded (tiny ranges, keeps
    its input neurons alive per AGENTS.md); head/body slots likewise. The
    yaw-rate tracking term (twist ang ≈ 0) keeps the hop straight.
  - Regularizers are LOW at start (hopping is violent and must be discovered
    first); action-rate smoothing and the |a_z| landing-impact tax ramp in
    after discovery. Pushes ramp in once hopping exists.

Symmetry must stay OFF (inherently one-legged).

RISK (unverified, 2026-10-06): whether 14 XL330s can lift this 0.74 kg robot
into a flight phase off ONE leg was not established in sim before training
(open-loop crouch/extend probes toppled — inconclusive). Watch
Episode_Reward/hop_air_time and Metrics in the first ~500 iters: if it stays
≈ 0 while hop_forward rises, the policy has found a one-foot shuffle instead.
"""

import math
from copy import deepcopy

# ── Hopping leg: "right" or "left" ────────────────────────────────────────────
HOP_LEG = "right"
assert HOP_LEG in ("right", "left")

NUM_STEPS_PER_ENV = 24

EPISODE_LENGTH_S = 10.0

# Lift-off grace: spawned on both feet, the free foot may touch the floor for
# this long before touching becomes a failure.
ONE_LEG_GRACE_S = 1.0

# Sanity clamp on the linear speed reward — far above anything reachable
# (walking tops out ~0.4 m/s). Not a target.
HOP_MAX_SPEED = 2.0

# Support-foot flight window (s) that counts as a hop.
HOP_AIR_TIME_MIN = 0.03
HOP_AIR_TIME_MAX = 0.25

# Pushes (ramped in by curriculum once hopping exists; final < walking's ±0.3
# because one-leg support has a far smaller support polygon).
PUSH_FINAL_RANGE = (-0.2, 0.2)

# Bodies whose floor contact is a failure in the groundcontact model (every
# colliding body except the two feet; same names in the backlash twin).
NON_FOOT_COLLISION_BODIES = ("trunk_base", "hip_l", "hip_l_2", "leg", "leg_2", "jaw_soft")

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import rewards as envs_rewards
from mjlab.managers import (
    CurriculumTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# feet_ground_contact column order (velocity recipe): LEFT = 0, RIGHT = 1.
_FOOT_INDEX = {"left": 0, "right": 1}

# Velocity-recipe rewards that are walking-specific (twist-gated gait shaping,
# leg pose pulled to HOME — which would forbid tucking the free leg — head
# droop EMA, and angular momentum, a motion-blocker for a hop).
_DROPPED_REWARDS = (
    "track_linear_velocity",
    "pose",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "angular_momentum",
    "head_pose_bias",
)
_DROPPED_CURRICULA = ("standing_envs", "head_pose_bias_weight", "head_pose_range")


def make_microduck_one_leg_hop_env_cfg(
    play: bool = False,
    hop_leg: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck OneLegHop env cfg (flat ground only).

    ``hop_leg`` overrides the module-level HOP_LEG flag (used by tests).
    """
    hop_leg = hop_leg or HOP_LEG
    assert hop_leg in ("right", "left")
    free_leg = "left" if hop_leg == "right" else "right"
    support_idx = _FOOT_INDEX[hop_leg]
    free_idx = _FOOT_INDEX[free_leg]

    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Robot + sensors ───────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=tuple(rf"^{b}$" for b in NON_FOOT_COLLISION_BODIES),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (body_ground_cfg,)
    feet = "feet_ground_contact"

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=microduck_mdp.one_leg_illegal_contact,
        params={
            "body_sensor_name": body_ground_cfg.name,
            "feet_sensor_name": feet,
            "free_foot_index": free_idx,
            "grace_s": ONE_LEG_GRACE_S,
        },
    )
    nan_params = cfg.terminations["nan_state"].params
    nan_params["sensor_names"] = tuple(nan_params["sensor_names"]) + (body_ground_cfg.name,)

    # ── Commands: no velocity command — twist zero-padded ─────────────────────
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x = (-0.01, 0.01)
    tw.ranges.lin_vel_y = (-0.01, 0.01)
    tw.ranges.ang_vel_z = (-0.01, 0.01)
    tw.rel_turn_in_place_envs = 0.0
    # Same sampling, but no viewer joystick (its slider floor is 0.1 → the
    # ±0.01 ranges above crashed `uv run play`).
    cfg.commands["twist"] = microduck_mdp.ZeroPaddedVelocityCommandCfg(**vars(tw))
    # head_pose keeps the velocity recipe's step-0 (small) ranges: the head is
    # a useful counterweight, but no posture commands are part of this task.

    # ── Rewards ───────────────────────────────────────────────────────────────
    for name in _DROPPED_REWARDS:
        cfg.rewards.pop(name, None)

    # Primary objective: distance hopped (linear speed), free foot up.
    cfg.rewards["hop_forward"] = RewardTermCfg(
        func=microduck_mdp.hop_forward_velocity,
        weight=8.0,
        params={"sensor_name": feet, "free_foot_index": free_idx, "max_speed": HOP_MAX_SPEED},
    )
    # Real flight phase on the support foot.
    cfg.rewards["hop_air_time"] = RewardTermCfg(
        func=microduck_mdp.support_foot_air_time,
        weight=1.0,
        params={
            "sensor_name": feet,
            "support_foot_index": support_idx,
            "free_foot_index": free_idx,
            "threshold_min": HOP_AIR_TIME_MIN,
            "threshold_max": HOP_AIR_TIME_MAX,
        },
    )
    # Survival in single support / flight.
    cfg.rewards["one_leg_alive"] = RewardTermCfg(
        func=microduck_mdp.free_foot_up_alive,
        weight=0.5,
        params={"sensor_name": feet, "free_foot_index": free_idx},
    )
    # Failure price (fall, illegal contact, NaN) on top of lost future reward.
    cfg.rewards["termination"] = RewardTermCfg(func=envs_rewards.is_terminated, weight=-10.0)
    # Free foot on the floor during the grace — lift off early.
    cfg.rewards["free_foot_contact"] = RewardTermCfg(
        func=microduck_mdp.free_foot_contact_cost,
        weight=-1.0,
        params={"sensor_name": feet, "free_foot_index": free_idx},
    )
    # Keep the hop straight: yaw rate → twist ang (≈ 0), no sideways drift.
    cfg.rewards["track_angular_velocity"].weight = 1.0
    cfg.rewards["hop_lateral_velocity"] = RewardTermCfg(
        func=microduck_mdp.hop_lateral_velocity_l2,
        weight=-1.0,
    )
    # Upright: looser than walking's 2.0 / std²=0.05 — a hop pitches.
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.1)
    # Head: free counterweight, gently held near HOME.
    cfg.rewards["head_pose_tracking"].weight = 0.5
    # Motion-blocker kept LOW (dynamic task).
    cfg.rewards["body_ang_vel"].weight = -0.01
    # Landing impact |a_z| (self-negating → POSITIVE weight). 0 during
    # discovery, ramped by the curriculum below.
    cfg.rewards["landing_impact"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.0,
    )
    # action_rate_l2 / dof_pos_limits / self_collisions / body_pose_tracking
    # (weight 0) carry over from the velocity recipe; action_rate is re-ramped.

    # ── Critic: grace-elapsed clock (actor layout is frozen at 61D) ───────────
    cfg.observations["critic"].terms["grace_clock"] = ObservationTermCfg(
        func=microduck_mdp.episode_time_frac,
        params={"horizon_s": ONE_LEG_GRACE_S},
    )

    # ── Pushes ────────────────────────────────────────────────────────────────
    if "push_robot" in cfg.events:
        push = cfg.events["push_robot"]
        if play:
            push.interval_range_s = (3.0, 6.0)
            push.params["velocity_range"] = {"x": PUSH_FINAL_RANGE, "y": PUSH_FINAL_RANGE}
        else:
            push.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
            cfg.curriculum["push_range"] = CurriculumTermCfg(
                func=microduck_mdp.push_curriculum,
                params={
                    "event_name": "push_robot",
                    "push_stages": [
                        {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                        {"step": 750 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
                        {"step": 1500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                        {"step": 2250 * NUM_STEPS_PER_ENV, "velocity_range": {"x": PUSH_FINAL_RANGE, "y": PUSH_FINAL_RANGE}},
                    ],
                },
            )

    # ── Curricula ─────────────────────────────────────────────────────────────
    for name in _DROPPED_CURRICULA:
        cfg.curriculum.pop(name, None)
    # Smoothness AFTER skill discovery (reward mass here ≈ 5/step vs walking's
    # ~11, so the final -0.3 is about walking's -1.0 in relative terms).
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.01},
        {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.05},
        {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.1},
        {"step": 1750 * NUM_STEPS_PER_ENV, "weight": -0.2},
        {"step": 2250 * NUM_STEPS_PER_ENV, "weight": -0.3},
    ]
    cfg.rewards["action_rate_l2"].weight = -0.01
    # Landing-impact tax: a 50 m/s² landing costs 0.1 → 0.25/step at the end.
    cfg.curriculum["landing_impact_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "landing_impact",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 0.002},
                {"step": 2000 * NUM_STEPS_PER_ENV, "weight": 0.005},
            ],
        },
    )

    return cfg


MicroduckOneLegHopRlCfg = deepcopy(MicroduckRlCfg)
MicroduckOneLegHopRlCfg.algorithm.symmetry_cfg = None
MicroduckOneLegHopRlCfg.experiment_name = f"one_leg_hop_{HOP_LEG}"
MicroduckOneLegHopRlCfg.run_name = f"one_leg_hop_{HOP_LEG}"
MicroduckOneLegHopRlCfg.max_iterations = 5_000
