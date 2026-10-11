"""OneLegHop cfg invariants + mdp function regressions (CPU only)."""

from types import SimpleNamespace

import mujoco
import pytest
import torch

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_GROUNDCONTACT_BACKLASH_XML,
    MICRODUCK_GROUNDCONTACT_XML,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_one_leg_hop_env_cfg as hop
from mjlab_microduck.tasks.microduck_one_leg_hop_env_cfg import (
    MicroduckOneLegHopRlCfg,
    make_microduck_one_leg_hop_env_cfg,
)


def _colliding_bodies(xml) -> set[str]:
    m = mujoco.MjModel.from_xml_path(str(xml))
    return {
        m.body(m.geom_bodyid[g]).name
        for g in range(m.ngeom)
        if (m.geom_contype[g] & 1) or (m.geom_conaffinity[g] & 1)
    }


@pytest.mark.parametrize("xml", [MICRODUCK_GROUNDCONTACT_XML, MICRODUCK_GROUNDCONTACT_BACKLASH_XML])
def test_failure_sensor_covers_every_non_foot_colliding_body(xml):
    # If a model revision adds/renames a colliding body, a floor contact on it
    # would silently stop counting as a failure.
    feet = {"ankle_left", "ankle_right"}
    assert _colliding_bodies(xml) - feet == set(hop.NON_FOOT_COLLISION_BODIES)


def test_uses_groundcontact_model():
    cfg = make_microduck_one_leg_hop_env_cfg()
    assert cfg.scene.entities["robot"] is MICRODUCK_STANDUP_ROBOT_CFG
    assert cfg.scene.terrain.terrain_type == "plane"


@pytest.mark.parametrize("leg,support,free", [("right", 1, 0), ("left", 0, 1)])
def test_foot_indices_follow_hop_leg(leg, support, free):
    cfg = make_microduck_one_leg_hop_env_cfg(hop_leg=leg)
    # feet_ground_contact column order is LEFT, RIGHT
    feet = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    assert feet.primary.pattern.startswith("^(left_foot_collision|right_foot_collision)")
    r = cfg.rewards
    assert r["hop_forward"].params["free_foot_index"] == free
    assert r["hop_air_time"].params["support_foot_index"] == support
    assert r["hop_air_time"].params["free_foot_index"] == free
    assert r["free_foot_contact"].params["free_foot_index"] == free
    assert cfg.terminations["illegal_contact"].params["free_foot_index"] == free


def test_reward_signs():
    r = make_microduck_one_leg_hop_env_cfg().rewards
    for name in ("hop_forward", "hop_air_time", "one_leg_alive", "upright", "track_angular_velocity"):
        assert r[name].weight > 0, name
    # mjlab-style costs (>= 0) → negative weight
    for name in ("termination", "free_foot_contact", "hop_lateral_velocity", "action_rate_l2", "body_ang_vel"):
        assert r[name].weight < 0, name
    # self-negating |a_z| → weight >= 0 (0 until the curriculum ramps it)
    assert r["landing_impact"].weight >= 0
    # the primary objective dominates the positive stack
    assert r["hop_forward"].weight * 0.3 > r["one_leg_alive"].weight + r["upright"].weight


def test_walking_terms_removed():
    cfg = make_microduck_one_leg_hop_env_cfg()
    for name in hop._DROPPED_REWARDS:
        assert name not in cfg.rewards, name
    for name in hop._DROPPED_CURRICULA:
        assert name not in cfg.curriculum, name


def test_command_slots_zero_padded_but_alive():
    cfg = make_microduck_one_leg_hop_env_cfg()
    tw = cfg.commands["twist"]
    for lo, hi in (tw.ranges.lin_vel_x, tw.ranges.lin_vel_y, tw.ranges.ang_vel_z):
        assert lo < 0 < hi and hi <= 0.01
    # 61D actor layout: command slots kept in order
    terms = list(cfg.observations["actor"].terms)
    assert terms[-3:] == ["command", "head_command", "body_command"]
    # grace clock is critic-only
    assert "grace_clock" in cfg.observations["critic"].terms
    assert "grace_clock" not in cfg.observations["actor"].terms


def test_curricula_and_terminations_wired():
    cfg = make_microduck_one_leg_hop_env_cfg()
    assert "illegal_contact" in cfg.terminations
    assert "fell_over" in cfg.terminations
    assert "body_ground_contact" in cfg.terminations["nan_state"].params["sensor_names"]
    assert cfg.events["expand_bam_friction_fields"].mode == "startup"
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["weight"] == cfg.rewards["action_rate_l2"].weight  # starts ~0
    assert cfg.curriculum["push_range"].params["push_stages"][0]["velocity_range"]["x"] == (0.0, 0.0)
    assert "push_range" not in make_microduck_one_leg_hop_env_cfg(play=True).curriculum


def test_rl_cfg():
    assert MicroduckOneLegHopRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckOneLegHopRlCfg.experiment_name.startswith("one_leg_hop_")
    assert MicroduckOneLegHopRlCfg.actor.obs_normalization


# ── mdp functions on a fake env ───────────────────────────────────────────────

def _fake_env(found_feet, found_body, t_steps, vel_w, quat_w):
    n = len(t_steps)
    sensors = {
        "feet": SimpleNamespace(data=SimpleNamespace(
            found=torch.tensor(found_feet, dtype=torch.float32),
            current_air_time=torch.zeros(n, 2),
        )),
        "body": SimpleNamespace(data=SimpleNamespace(found=torch.tensor(found_body, dtype=torch.float32))),
    }
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_w=torch.tensor(vel_w, dtype=torch.float32),
        root_link_quat_w=torch.tensor(quat_w, dtype=torch.float32),
    ))
    scene = type("S", (), {"sensors": sensors, "__getitem__": lambda self, k: robot})()
    return SimpleNamespace(
        scene=scene,
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor(t_steps),
        termination_manager=SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)),
    )


_ID = [1.0, 0.0, 0.0, 0.0]
_YAW90 = [0.7071068, 0.0, 0.0, 0.7071068]


def test_illegal_contact_grace_and_body_hits():
    # env0: free(left) foot down inside grace → ok; env1: same after grace → fail;
    # env2: trunk touches inside grace → fail; env3: only support foot down → ok.
    env = _fake_env(
        found_feet=[[1, 1], [1, 1], [0, 1], [0, 1]],
        found_body=[[0] * 6, [0] * 6, [1, 0, 0, 0, 0, 0], [0] * 6],
        t_steps=[10, 60, 10, 500],
        vel_w=[[0, 0, 0]] * 4,
        quat_w=[_ID] * 4,
    )
    out = microduck_mdp.one_leg_illegal_contact(env, "body", "feet", free_foot_index=0, grace_s=1.0)
    assert out.tolist() == [False, True, True, False]


def test_forward_velocity_is_heading_frame_linear_and_gated():
    # env0: facing +x moving +x 0.5, left foot up → 0.5
    # env1: facing +y (yaw 90°) moving +y 0.5 → 0.5 (heading frame, not world x)
    # env2: same as env0 but left foot down → 0
    # env3: moving backward → negative (linear, not clamped at 0)
    env = _fake_env(
        found_feet=[[0, 1], [0, 1], [1, 1], [0, 0]],
        found_body=[[0] * 6] * 4,
        t_steps=[100] * 4,
        vel_w=[[0.5, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [-0.3, 0, 0]],
        quat_w=[_ID, _YAW90, _ID, _ID],
    )
    v = microduck_mdp.hop_forward_velocity(env, "feet", free_foot_index=0, max_speed=2.0)
    assert torch.allclose(v, torch.tensor([0.5, 0.5, 0.0, -0.3]), atol=1e-5)
    lat = microduck_mdp.hop_lateral_velocity_l2(env)
    assert torch.allclose(lat, torch.zeros(4), atol=1e-6)


def test_support_air_time_needs_both_feet_up_in_window():
    env = _fake_env(
        found_feet=[[0, 0], [0, 0], [1, 0], [0, 0]],
        found_body=[[0] * 6] * 4,
        t_steps=[100] * 4,
        vel_w=[[0, 0, 0]] * 4,
        quat_w=[_ID] * 4,
    )
    # right (support, col 1) air time: in window, too long, in window but left down, too short
    env.scene.sensors["feet"].data.current_air_time = torch.tensor(
        [[0.0, 0.1], [0.0, 0.5], [0.0, 0.1], [0.0, 0.01]]
    )
    r = microduck_mdp.support_foot_air_time(env, "feet", 1, 0, 0.03, 0.25)
    assert r.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_twist_slot_skips_viewer_joystick():
    # Regression: viser's joystick "Max" slider has a floor of 0.1, so the
    # ±0.01 zero-padded twist ranges crashed `uv run play` at GUI creation.
    for play in (False, True):
        tw = make_microduck_one_leg_hop_env_cfg(play=play).commands["twist"]
        assert isinstance(tw, microduck_mdp.ZeroPaddedVelocityCommandCfg)
        assert tw.ranges.lin_vel_x == (-0.01, 0.01)
        assert tw.rel_turn_in_place_envs == 0.0
    assert microduck_mdp.ZeroPaddedVelocityCommand.create_gui(None) is None
