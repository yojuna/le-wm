#!/usr/bin/env python3
"""CPU tests for Phase B dump/probe/drift helpers and --horizon plumbing."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase_b import (  # noqa: E402
    ACTION_DIM,
    CEM_HORIZON,
    HISTORY,
    PUSHT_CONTACT_RADIUS,
    REACHER_FACTORS,
    action_convention_for_collector,
    collector_uses_set_state,
    contact_events,
    dump_default_out_dir,
    effective_rank,
    factor_names_for_env,
    imagine_closed_loop,
    imagine_path,
    load_dump,
    pack_action_token,
    pad_action,
    remaining_pose_error,
    resolve_dump_collector,
    save_dump,
    shuffle_future_actions,
    summarize_drift,
    validate_oracle_actor,
)
from eval_live import EnvSpec, _apply_horizon  # noqa: E402


def test_pad_action():
    a = np.array([1.0, 2.0], dtype=np.float32)
    p = pad_action(a, action_dim=10)
    assert p.shape == (10,)
    assert p[0] == 1.0 and p[1] == 2.0 and p[9] == 0.0


def test_shuffle_keeps_history():
    rng = np.random.default_rng(0)
    acts = np.arange(20, dtype=np.float32).reshape(10, 2)
    sh = shuffle_future_actions(acts, history=3, rng=rng)
    assert np.allclose(sh[:3], acts[:3])
    assert sh.shape == acts.shape
    # tail permutation of original tail (same multiset)
    assert sorted(sh[3:].reshape(-1).tolist()) == sorted(acts[3:].reshape(-1).tolist())


def test_summarize_drift_excludes_history():
    err = np.zeros(8, dtype=np.float64)
    err[HISTORY:] = np.arange(1, 8 - HISTORY + 1, dtype=np.float64)
    s = summarize_drift(err, history=HISTORY)
    assert s["mean_all_frames"] < s["mean_predicted_only"]
    cem_idx = HISTORY + CEM_HORIZON - 1
    assert s["at_h5_index"] == float(err[cem_idx])


def test_remaining_pose_zero_at_end():
    states = np.zeros((5, 7), dtype=np.float64)
    states[:, 0] = np.linspace(0, 10, 5)
    rem = remaining_pose_error("pusht", states)
    assert rem[-1] == 0.0
    assert rem[0] > rem[-2]


def test_linear_probe_recovers_linear_factor():
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    z = rng.normal(size=(400, 32))
    w = rng.normal(size=32)
    y = z @ w
    tr, va = z[:300], z[300:]
    ytr, yva = y[:300], y[300:]
    model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
    model.fit(tr, ytr)
    r2 = r2_score(yva, model.predict(va))
    assert r2 > 0.95, r2


def test_dump_roundtrip(tmp_path: Path | None = None):
    out = Path("/tmp/phase_b_dump_test") if tmp_path is None else tmp_path
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dump.npz"
    z = np.random.randn(4, 6, 8).astype(np.float32)
    save_dump(
        path,
        {
            "z": z,
            "remaining_k": np.arange(4 * 6).reshape(4, 6).astype(np.float32),
            "meta": {"env": "pusht", "n_segments": 4},
        },
    )
    loaded = load_dump(path)
    assert loaded["z"].shape == (4, 6, 8)
    assert loaded["meta"]["env"] == "pusht"


def test_apply_horizon():
    spec = EnvSpec(
        env_name="swm/PushT-v1",
        hf_repo="x",
        ckpt_dir="hf_pusht",
        horizon=5,
        receding_horizon=5,
    )

    @dataclass
    class A:
        horizon: int = 0
        receding_horizon: int = 0

    assert _apply_horizon(spec, A()).horizon == 5
    s2 = _apply_horizon(spec, A(horizon=2))
    assert s2.horizon == 2 and s2.receding_horizon == 2
    s3 = _apply_horizon(spec, A(horizon=8, receding_horizon=3))
    assert s3.horizon == 8 and s3.receding_horizon == 3


def test_episode_len_pixels_fallback():
    from eval_logging.pairs import EpisodeTraj

    ep = EpisodeTraj(seed=0, pixels=[np.zeros((4, 4, 3))] * 3)
    assert len(ep) == 3


def test_pack_token_is_tile_not_zero_pad():
    a = np.array([1.0, -0.5], dtype=np.float32)
    tok = pack_action_token(a)
    assert tok.shape == (ACTION_DIM,)
    assert np.allclose(tok[:2], a)
    # tile: [1, -0.5, 1, -0.5, ...] not zero-pad
    assert np.allclose(tok, np.tile(a, 5))
    padded = pad_action(a)
    assert not np.allclose(tok, padded)
    assert np.allclose(padded[2:], 0.0)


def test_action_mode_diverse_not_kinematic():
    assert resolve_dump_collector("pusht", action_mode="diverse") == "random"
    assert resolve_dump_collector("pusht") == "kinematic"
    assert resolve_dump_collector("reacher") == "random"
    assert resolve_dump_collector("pusht", collector="kinematic") == "kinematic"
    out = dump_default_out_dir(Path("/x"), "pusht", "random", 0)
    assert "phase_b_dump_diverse" in str(out)
    kin = dump_default_out_dir(Path("/x"), "pusht", "kinematic", 1)
    assert "phase_b_dump_diverse" not in str(kin)


def test_shuffle_permutes_rows_not_values_across_columns():
    rng = np.random.default_rng(1)
    acts = np.arange(20, dtype=np.float32).reshape(10, 2)
    sh = shuffle_future_actions(acts, history=3, rng=rng)
    assert np.allclose(sh[:3], acts[:3])
    orig = [tuple(r.tolist()) for r in acts[3:]]
    got = [tuple(r.tolist()) for r in sh[3:]]
    assert sorted(orig) == sorted(got)
    assert got != orig  # seed 1 should actually shuffle


def test_random_collector_is_physics_not_set_state():
    from eval_logging.pairs import _make_collection_policy

    assert collector_uses_set_state("kinematic")
    assert not collector_uses_set_state("random")
    assert action_convention_for_collector("random") == "from_state_step"
    assert action_convention_for_collector("kinematic") == "into_state_fd"
    pol = _make_collection_policy(
        object(), env_name="swm/PushT-v1", seed=0, collector="random"
    )
    assert pol is None


def test_reacher_factor_names():
    names = factor_names_for_env("reacher", 8)
    assert names == REACHER_FACTORS
    assert "factor_2" not in names
    assert names[2] == "qvel_0"
    assert names[6] == "target_x"


def test_effective_rank_identity_vs_rank1():
    rng = np.random.default_rng(0)
    full = rng.normal(size=(2000, 16))
    r_full = effective_rank(full)
    assert r_full > 10, r_full
    t = rng.normal(size=(2000, 1))
    rank1 = t * np.linspace(0.1, 2.0, 16)
    r1 = effective_rank(rank1)
    assert r1 < 2.5, r1


def test_oracle_actor_rejects_kinematic():
    try:
        validate_oracle_actor("kinematic")
        raise AssertionError("kinematic oracle must be rejected")
    except ValueError as exc:
        assert "kinematic" in str(exc).lower()
    assert validate_oracle_actor("oracle_replay") == "oracle_replay"
    assert validate_oracle_actor("cem_l2") == "cem"
    assert validate_oracle_actor("goal_push") == "goal_push"
    assert validate_oracle_actor("weak") == "weak"


def test_oracle_window_band_and_action_len():
    from eval_logging.oracle_bank import (
        OracleReplayPolicy,
        load_oracle_bank,
        save_oracle_bank,
        window_oracle_pairs,
    )
    from eval_logging.pairs import EpisodeTraj, TrajectoryBank
    import tempfile
    ep = EpisodeTraj(seed=0)
    # 30 states: start at origin, slide block_x so t=0→25 pose ≈ 22.5
    for t in range(30):
        x = 0.9 * t  # pose at t=25 is 22.5 ∈ (20, 25]
        state = np.asarray([0.0, 0.0, x, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ep.state.append(state)
        ep.proprio.append(state[:2])
        ep.pixels.append(np.full((4, 4, 3), t, dtype=np.uint8))
        ep.action.append(np.asarray([0.1 * t, -0.2], dtype=np.float32))
    bank = TrajectoryBank(episodes=[ep], env_name="swm/PushT-v1", collector="weak")
    pairs = window_oracle_pairs(bank, window=25, stride=25, num_eval=None)
    assert len(pairs) == 1, len(pairs)
    p = pairs[0]
    assert p.oracle_actions is not None and p.oracle_actions.shape == (25, 2)
    assert 20.0 <= p.pos_progress <= 25.0
    assert p.path_pixels is not None and p.path_pixels.shape[0] == 26

    # pose 10 is below the short_horizon band
    ep2 = EpisodeTraj(seed=1)
    for t in range(30):
        x = 0.3 * t  # t=25 → 7.5, out of band
        state = np.asarray([0.0, 0.0, x, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ep2.state.append(state)
        ep2.proprio.append(state[:2])
        ep2.pixels.append(np.zeros((4, 4, 3), dtype=np.uint8))
        ep2.action.append(np.zeros(2, dtype=np.float32))
    bank2 = TrajectoryBank(episodes=[ep2], env_name="swm/PushT-v1", collector="weak")
    assert window_oracle_pairs(bank2, window=25, stride=25) == []

    pol = OracleReplayPolicy([p.oracle_actions])
    pol.begin_pair(0)
    a0 = pol.get_action(None)
    assert a0.shape == (1, 2)
    np.testing.assert_allclose(a0[0], p.oracle_actions[0])

    tmp = Path(tempfile.mkdtemp(prefix="lewm_oracle_bank_"))
    save_oracle_bank(tmp, pairs, extra={"oracle_source": "weak"})
    loaded, meta = load_oracle_bank(tmp)
    assert len(loaded) == 1
    assert meta["action_pack"] == "tile_block"
    np.testing.assert_allclose(loaded[0].oracle_actions, p.oracle_actions)


class _DummyLiveP:
    """Next z = last z + 0.1 * sum(last action token)."""

    def action_encoder(self, act):
        return act.float()

    def predict(self, emb, act_emb):
        delta = act_emb[:, -1, :].sum(dim=-1, keepdim=True)
        return emb[:, -1:, :] + 0.1 * delta.unsqueeze(-1)


class _DummyIdentityP:
    def action_encoder(self, act):
        return act.float()

    def predict(self, emb, act_emb):
        return emb[:, -1:, :]


def test_imagine_path_length_and_history():
    model = _DummyIdentityP()
    L, D = 8, 4
    z = torch.randn(L, D)
    acts = np.random.randn(L, 2).astype(np.float32)
    out = imagine_path(model, z, acts, device=torch.device("cpu"), history=HISTORY)
    assert out.shape == (L, D)
    assert torch.allclose(out[:HISTORY], z[:HISTORY])


def test_imagine_closed_loop_large_m_matches_open_loop():
    model = _DummyLiveP()
    L, D = 12, 4
    z = torch.randn(L, D)
    acts = np.random.randn(L, 2).astype(np.float32)
    ol = imagine_path(model, z, acts, device=torch.device("cpu"))
    cl = imagine_closed_loop(model, z, acts, m=99, device=torch.device("cpu"))
    assert torch.allclose(ol, cl)
    cl1 = imagine_closed_loop(model, z, acts, m=1, device=torch.device("cpu"))
    assert cl1.shape == (L, D)
    assert torch.allclose(cl1[:HISTORY], z[:HISTORY])


def test_contact_events_pusht():
    st = np.zeros((4, 7), dtype=np.float64)
    st[:, :2] = 100.0
    st[:, 2:4] = 100.0
    ev = contact_events(st, env="pusht")
    assert ev["contact"].all()
    st2 = np.zeros((3, 7), dtype=np.float64)
    st2[:, :2] = 256.0
    st2[:, 2:4] = 256.0
    st2[0, 2:4] = 5.0
    ev2 = contact_events(st2, env="pusht")
    assert ev2["wall"][0]
    assert not ev2["contact"][0]
    empty = contact_events(st, env="reacher")
    assert not empty["any"].any()
    assert PUSHT_CONTACT_RADIUS == 45.0


def test_dummy_p_shuffle_increases_drift():
    rng = np.random.default_rng(0)
    model = _DummyLiveP()
    L, D = 12, 4
    acts = rng.normal(size=(L, 2)).astype(np.float32)
    # build a consistent z_true by rolling DummyLiveP with true actions
    z0 = torch.zeros(L, D)
    z0[0] = torch.ones(D)
    # fill teacher-forced history as a ramp
    for t in range(1, HISTORY):
        z0[t] = z0[t - 1] + 0.01
    z_hat_true = imagine_path(model, z0, acts, device=torch.device("cpu"))
    sh = shuffle_future_actions(acts, HISTORY, np.random.default_rng(2))
    z_hat_sh = imagine_path(model, z0, sh, device=torch.device("cpu"))
    err_true = torch.linalg.vector_norm(z_hat_true[HISTORY:] - z_hat_true[HISTORY:], dim=-1)
    # compare shuffled imagination to true-action imagination
    gap = float(
        torch.linalg.vector_norm(z_hat_sh[HISTORY:] - z_hat_true[HISTORY:], dim=-1).mean()
    )
    assert gap > 0.05, gap
    assert float(err_true.mean()) == 0.0


if __name__ == "__main__":
    test_pad_action()
    test_pack_token_is_tile_not_zero_pad()
    test_action_mode_diverse_not_kinematic()
    test_shuffle_keeps_history()
    test_shuffle_permutes_rows_not_values_across_columns()
    test_summarize_drift_excludes_history()
    test_remaining_pose_zero_at_end()
    test_linear_probe_recovers_linear_factor()
    test_dump_roundtrip()
    test_apply_horizon()
    test_episode_len_pixels_fallback()
    test_random_collector_is_physics_not_set_state()
    test_reacher_factor_names()
    test_effective_rank_identity_vs_rank1()
    test_oracle_actor_rejects_kinematic()
    test_oracle_window_band_and_action_len()
    test_imagine_path_length_and_history()
    test_imagine_closed_loop_large_m_matches_open_loop()
    test_contact_events_pusht()
    test_dummy_p_shuffle_increases_drift()
    print("all phase_b tests passed")
