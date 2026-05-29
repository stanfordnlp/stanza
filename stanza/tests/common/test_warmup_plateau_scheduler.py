"""
Unit tests for WarmupThenPlateauScheduler.

Run with:
    pytest test_warmup_plateau_scheduler.py -v

All phase durations (freeze, warmup, patience, cooldown) are in training
*steps*.  step(num_steps) advances the internal counter by num_steps; the
current phase is determined by _total_steps relative to num_freeze_steps
and num_freeze_steps + num_warmup_steps.

Initial state
-------------
After construction, _total_steps=0, and the initial LR is applied
immediately (freeze → 0, no-freeze no-warmup → base_lr, warmup start → 0
if num_freeze_steps=0 and num_warmup_steps>0, since scale=0/M=0).

Warmup ramp
-----------
scale = _warmup_steps_elapsed / num_warmup_steps, clamped to [0, 1].
The scale starts at 0 (at the first warmup step) and reaches 1.0 exactly
when _warmup_steps_elapsed == num_warmup_steps.
"""

import warnings
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD

from stanza.models.common.warmup_plateau_scheduler import WarmupThenPlateauScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_LR = 1e-3
EVAL_STEPS = 100  # default steps per eval cycle used throughout tests


def make_scheduler(freeze, warmup, base_lr=BASE_LR, n_groups=1, **kwargs):
    """Return (optimizer, scheduler)."""
    if n_groups == 1:
        model = nn.Linear(2, 2)
        opt = AdamW(model.parameters(), lr=base_lr)
    else:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        params = list(model.parameters())
        mid = len(params) // 2
        lrs = [base_lr * (i + 1) for i in range(n_groups)]
        opt = SGD(
            [{"params": params[:mid], "lr": lrs[0]},
             {"params": params[mid:], "lr": lrs[1]}],
        )
    sched = WarmupThenPlateauScheduler(opt, freeze, warmup, **kwargs)
    return opt, sched


def lr(sched_or_opt):
    """First param-group LR."""
    obj = sched_or_opt.optimizer if hasattr(sched_or_opt, "optimizer") else sched_or_opt
    return obj.param_groups[0]["lr"]


def approx(x):
    return pytest.approx(x, rel=1e-6)


# ---------------------------------------------------------------------------
# 1. Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_factor_gte_1_raises(self):
        with pytest.raises(ValueError, match="factor"):
            make_scheduler(0, 0, factor=1.0)

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            make_scheduler(0, 0, mode="banana")

    def test_bad_threshold_mode_raises(self):
        with pytest.raises(ValueError, match="threshold_mode"):
            make_scheduler(0, 0, threshold_mode="banana")

    def test_min_lr_wrong_length_raises(self):
        with pytest.raises(ValueError, match="min_lr length"):
            make_scheduler(0, 0, n_groups=2, min_lr=[0.0])

    def test_negative_freeze_raises(self):
        with pytest.raises(ValueError, match="num_freeze_steps"):
            make_scheduler(-1, 0)

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="num_warmup_steps"):
            make_scheduler(0, -1)

    def test_negative_patience_raises(self):
        with pytest.raises(ValueError, match="patience"):
            make_scheduler(0, 0, patience=-1)

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown"):
            make_scheduler(0, 0, cooldown=-1)

    def test_base_lrs_captured(self):
        opt, sched = make_scheduler(0, 0, patience=None)
        assert sched._base_lrs == [BASE_LR]

    def test_plateau_start_step_property(self):
        _, sched = make_scheduler(300, 400)
        assert sched._plateau_start_step == 700

    def test_warmup_start_step_property(self):
        _, sched = make_scheduler(300, 400)
        assert sched._warmup_start_step == 300


# ---------------------------------------------------------------------------
# 2. Initial LR applied at construction
# ---------------------------------------------------------------------------

class TestInitialLR:
    def test_freeze_warmup_initial_lr_is_zero(self):
        """When freeze > 0, LR should be 0 immediately after construction."""
        opt, _ = make_scheduler(200, 0, patience=None)
        assert lr(opt) == 0.0

    def test_no_freeze_no_warmup_initial_lr_is_base(self):
        """freeze=0, warmup=0: LR should be base_lr from the start."""
        opt, _ = make_scheduler(0, 0, patience=None)
        assert lr(opt) == approx(BASE_LR)

    def test_no_freeze_with_warmup_initial_lr_is_zero(self):
        """freeze=0, warmup>0: scale=0/M=0 at construction."""
        opt, _ = make_scheduler(0, 400, patience=None)
        assert lr(opt) == 0.0


# ---------------------------------------------------------------------------
# 3. Freeze phase
# ---------------------------------------------------------------------------

class TestFreezePhase:
    def test_lr_stays_zero_throughout_freeze(self):
        opt, sched = make_scheduler(300, 0, mode="max", patience=9999)
        sched.step(100)
        assert lr(opt) == 0.0
        sched.step(100)
        assert lr(opt) == 0.0

    def test_phase_label_during_freeze(self):
        _, sched = make_scheduler(300, 0, patience=None)
        assert sched._current_phase() == "freeze"
        sched.step(299)
        assert sched._current_phase() == "freeze"

    def test_freeze_exits_at_correct_step(self):
        opt, sched = make_scheduler(300, 0, patience=None)
        sched.step(300)  # total=300, exactly at warmup_start → no longer freeze
        assert sched._current_phase() != "freeze"

    def test_freeze_multiple_param_groups(self):
        opt, sched = make_scheduler(200, 0, n_groups=2, patience=None)
        sched.step(100)
        for g in opt.param_groups:
            assert g["lr"] == 0.0


# ---------------------------------------------------------------------------
# 4. Warmup phase
# ---------------------------------------------------------------------------

class TestWarmupPhase:
    def test_warmup_ramp_is_linear(self):
        """scale = warmup_steps_elapsed / num_warmup_steps."""
        freeze, warmup = 0, 400
        opt, sched = make_scheduler(freeze, warmup, patience=None)
        # After construction: scale=0, lr=0
        assert lr(opt) == 0.0

        sched.step(100)  # elapsed=100, scale=0.25
        assert lr(opt) == approx(BASE_LR * 0.25)

        sched.step(100)  # elapsed=200, scale=0.5
        assert lr(opt) == approx(BASE_LR * 0.5)

        sched.step(100)  # elapsed=300, scale=0.75
        assert lr(opt) == approx(BASE_LR * 0.75)

        sched.step(100)  # elapsed=400, scale=1.0
        assert lr(opt) == approx(BASE_LR * 1.0)

    def test_warmup_reaches_base_lr_at_boundary(self):
        opt, sched = make_scheduler(0, 500, patience=None)
        sched.step(500)
        assert lr(opt) == approx(BASE_LR)

    def test_warmup_after_freeze(self):
        freeze, warmup = 200, 400
        opt, sched = make_scheduler(freeze, warmup, patience=None)
        sched.step(200)  # exits freeze, enters warmup at elapsed=0
        assert lr(opt) == 0.0  # scale=0/400=0

        sched.step(200)  # warmup elapsed=200, scale=0.5
        assert lr(opt) == approx(BASE_LR * 0.5)

        sched.step(200)  # warmup elapsed=400, scale=1.0
        assert lr(opt) == approx(BASE_LR * 1.0)

    def test_phase_label_during_warmup(self):
        _, sched = make_scheduler(100, 200, patience=None)
        sched.step(150)  # total=150, in warmup (100-300)
        assert sched._current_phase() == "warmup"

    def test_warmup_ramps_each_param_group(self):
        freeze, warmup = 0, 400
        opt, sched = make_scheduler(freeze, warmup, n_groups=2, patience=None)
        base_lrs = [BASE_LR * (i + 1) for i in range(2)]
        sched.step(200)  # scale=0.5
        for g, blr in zip(opt.param_groups, base_lrs):
            assert g["lr"] == approx(blr * 0.5)


# ---------------------------------------------------------------------------
# 5. Phase boundary spanning
# ---------------------------------------------------------------------------

class TestBoundarySpanning:
    def test_single_step_spans_freeze_into_warmup(self):
        """A step() call that crosses the freeze→warmup boundary should credit
        only the warmup portion toward warmup_steps_elapsed."""
        freeze, warmup = 200, 400
        opt, sched = make_scheduler(freeze, warmup, patience=None)
        # 300 steps: 200 in freeze, 100 in warmup → scale = 100/400 = 0.25
        sched.step(300)
        assert sched._current_phase() == "warmup"
        assert sched._warmup_steps_elapsed == 100
        assert lr(opt) == approx(BASE_LR * 0.25)

    def test_single_step_spans_warmup_into_plateau(self):
        """A step() call that crosses the warmup→plateau boundary should put
        us in plateau with the LR at base_lr (multiplier=1.0)."""
        opt, sched = make_scheduler(0, 200, patience=None)
        sched.step(300)  # 200 warmup + 100 plateau
        assert sched._current_phase() == "plateau"
        assert lr(opt) == approx(BASE_LR)

    def test_single_step_spans_all_three_phases(self):
        """One giant step() that covers freeze + warmup + plateau."""
        opt, sched = make_scheduler(100, 200, patience=None)
        sched.step(500)  # 100 freeze + 200 warmup + 200 plateau
        assert sched._current_phase() == "plateau"
        assert lr(opt) == approx(BASE_LR)

    def test_boundary_step_with_metrics(self):
        """When a step spans warmup→plateau, metrics should be accepted."""
        opt, sched = make_scheduler(0, 200, mode="max", patience=9999, factor=0.5)
        sched.step(300, metrics=0.8)  # crosses into plateau; should not raise
        assert sched._plateau_phase_started


# ---------------------------------------------------------------------------
# 6. Plateau entry
# ---------------------------------------------------------------------------

class TestPlateauEntry:
    def test_lr_at_base_on_plateau_entry(self):
        opt, sched = make_scheduler(100, 100, mode="max", patience=9999)
        sched.step(200, metrics=0.8)
        assert lr(opt) == approx(BASE_LR)

    def test_no_freeze_no_warmup_starts_at_base_lr(self):
        opt, sched = make_scheduler(0, 0, patience=None)
        assert lr(opt) == approx(BASE_LR)

    def test_metrics_always_optional(self):
        """metrics=None must never raise, regardless of phase."""
        _, sched = make_scheduler(100, 100, patience=9999)
        sched.step(100)           # freeze, no score
        sched.step(100)           # warmup, no score
        sched.step(100)           # plateau, no score — must not raise

    def test_none_metrics_in_plateau_skips_tracking(self):
        """metrics=None in plateau advances the clock but does not update
        _steps_without_improvement or _best."""
        _, sched = make_scheduler(0, 0, mode="max", patience=9999)
        sched.step(EVAL_STEPS, metrics=0.8)
        bad_before = sched._steps_without_improvement
        sched.step(EVAL_STEPS, metrics=None)
        assert sched._steps_without_improvement == bad_before
        assert sched._best == pytest.approx(0.8)

    def test_uniform_call_convention(self):
        """Caller can always pass (num_steps, dev_score) without inspecting phase."""
        opt, sched = make_scheduler(200, 200, mode="max", patience=5000, factor=0.5)
        for _ in range(40):
            sched.step(EVAL_STEPS, metrics=0.8)
        assert sched._current_phase() == "plateau"
        assert lr(opt) == approx(BASE_LR)

    def test_num_steps_must_be_positive(self):
        _, sched = make_scheduler(0, 0, patience=None)
        with pytest.raises(ValueError, match="num_steps"):
            sched.step(0)

    def test_plateau_phase_started_flag(self):
        _, sched = make_scheduler(100, 100, mode="max", patience=9999)
        assert not sched._plateau_phase_started
        sched.step(100)   # freeze done, entering warmup
        assert not sched._plateau_phase_started
        sched.step(100, metrics=0.8)  # warmup done, first plateau step
        assert sched._plateau_phase_started

    def test_best_resets_on_plateau_entry(self):
        _, sched = make_scheduler(100, 0, mode="max", patience=9999)
        sched.step(100, metrics=0.5)
        assert sched._best == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 7. Plateau decay (step-based)
# ---------------------------------------------------------------------------

class TestPlateauDecay:
    """
    patience=1000 steps, EVAL_STEPS=100.
    Decay fires when steps_without_improvement > 1000.
    That means 10 bad evals = exactly 1000 steps (not triggered),
    11 bad evals = 1100 steps > 1000 (triggered on the 11th).
    """

    PATIENCE = 1000

    def _sched(self, **kwargs):
        return make_scheduler(0, 0, mode="max", patience=self.PATIENCE,
                              factor=0.5, **kwargs)

    def test_no_decay_while_improving(self):
        opt, sched = self._sched()
        for i in range(20):
            sched.step(EVAL_STEPS, metrics=0.8 + i * 0.01)
        assert lr(opt) == approx(BASE_LR)

    def test_decay_fires_when_patience_exceeded(self):
        opt, sched = self._sched()
        sched.step(EVAL_STEPS, metrics=0.9)       # sets best
        for _ in range(10):
            sched.step(EVAL_STEPS, metrics=0.9)   # 1000 bad steps, not > 1000
        assert lr(opt) == approx(BASE_LR)
        sched.step(EVAL_STEPS, metrics=0.9)       # 1100 bad steps → decay
        assert lr(opt) == approx(BASE_LR * 0.5)

    def test_decay_not_triggered_at_exactly_patience(self):
        opt, sched = self._sched()
        sched.step(EVAL_STEPS, metrics=0.9)       # best
        for _ in range(10):
            sched.step(EVAL_STEPS, metrics=0.9)   # exactly 1000 bad steps
        assert lr(opt) == approx(BASE_LR)

    def test_large_eval_stride_decays_sooner(self):
        """600-step evals accumulate bad steps faster."""
        opt, sched = self._sched()
        sched.step(100, metrics=0.9)   # best
        sched.step(600, metrics=0.9)   # 600 bad steps
        assert lr(opt) == approx(BASE_LR)
        sched.step(600, metrics=0.9)   # 1200 bad steps → decay
        assert lr(opt) == approx(BASE_LR * 0.5)

    def test_eval_cadence_invariance(self):
        """
        Identical training, different eval cadences → decay at the same
        cumulative bad-step count.

        Scenario A: 12 evals × 100 steps (one sets best, 11 are bad = 1100 steps).
        Scenario B: 3 evals × 400 steps (one sets best, then 400+800=1200 bad steps,
                    decay on eval 3 at 1200 steps, which is > 1000).

        Both scenarios should produce a decayed LR; the first decay in each
        occurs at the earliest eval where cumulative bad steps exceed patience.
        """
        def run(eval_stride, n_evals):
            opt, sched = self._sched()
            sched.step(eval_stride, metrics=0.9)  # sets best
            for _ in range(n_evals - 1):
                sched.step(eval_stride, metrics=0.9)
            return lr(opt)

        lr_a = run(100, 12)   # 11 × 100 = 1100 bad steps
        lr_b = run(400, 4)    # 3 × 400 = 1200 bad steps

        assert lr_a == approx(BASE_LR * 0.5)
        assert lr_b == approx(BASE_LR * 0.5)

    def test_decay_factor_applied_correctly(self):
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000, factor=0.3)
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)
        assert lr(opt) == approx(BASE_LR * 0.3)

    def test_multiple_decays_compound(self):
        opt, sched = self._sched()
        sched.step(EVAL_STEPS, metrics=0.9)       # best
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)   # first decay
        assert lr(opt) == approx(BASE_LR * 0.5)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.5)   # second decay
        assert lr(opt) == approx(BASE_LR * 0.25)

    def test_improvement_resets_step_counter(self):
        opt, sched = self._sched()
        sched.step(EVAL_STEPS, metrics=0.8)   # best
        for _ in range(5):
            sched.step(EVAL_STEPS, metrics=0.8)   # 500 bad steps
        sched.step(EVAL_STEPS, metrics=0.9)   # improvement → reset
        for _ in range(10):
            sched.step(EVAL_STEPS, metrics=0.9)   # 1000 bad steps (not > 1000)
        assert lr(opt) == approx(BASE_LR)

    def test_mode_min_decays_on_stagnant(self):
        opt, sched = make_scheduler(0, 0, mode="min", patience=1000, factor=0.5)
        sched.step(EVAL_STEPS, metrics=0.5)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.5)
        assert lr(opt) == approx(BASE_LR * 0.5)

    def test_mode_min_no_decay_while_improving(self):
        opt, sched = make_scheduler(0, 0, mode="min", patience=1000, factor=0.5)
        for i in range(20):
            sched.step(EVAL_STEPS, metrics=0.9 - i * 0.01)
        assert lr(opt) == approx(BASE_LR)


# ---------------------------------------------------------------------------
# 8. Threshold & threshold_mode
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_rel_threshold_blocks_marginal_improvement(self):
        # threshold=0.1 rel: must beat best * 1.1; 1.05 < 1.1 → bad
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                    factor=0.5, threshold=0.1)
        sched.step(EVAL_STEPS, metrics=1.0)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=1.05)
        assert lr(opt) == approx(BASE_LR * 0.5)

    def test_rel_threshold_accepts_sufficient_improvement(self):
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                    factor=0.5, threshold=0.1)
        score = 1.0
        for _ in range(20):
            sched.step(EVAL_STEPS, metrics=score)
            score *= 1.15  # each score beats best * 1.1
        assert lr(opt) == approx(BASE_LR)

    def test_abs_threshold_blocks_marginal_improvement(self):
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                    factor=0.5, threshold=0.1, threshold_mode="abs")
        sched.step(EVAL_STEPS, metrics=1.0)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=1.05)  # 1.05 < 1.0 + 0.1 → bad
        assert lr(opt) == approx(BASE_LR * 0.5)


# ---------------------------------------------------------------------------
# 9. Cooldown (step-based)
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_cooldown_delays_next_decay(self):
        """
        patience=1000, cooldown=500, factor=0.5.

        step 0  : 100 steps, score=0.9 → best, bad=0
        step 1  : 600 steps, score=0.9 → bad=600
        step 2  : 600 steps, score=0.9 → bad=1200 > 1000 → DECAY, cd=500, bad=0
        step 3  : 300 steps, score=0.5 → in cooldown, cd=max(0,500-300)=200
        step 4  : 300 steps, score=0.5 → cd=max(0,200-300)=0, cd ends → bad=0
        step 5  : 600 steps, score=0.5 → not in cd, bad=600
        step 6  : 600 steps, score=0.5 → bad=1200 > 1000 → SECOND DECAY
        """
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                    factor=0.5, cooldown=500)
        sched.step(100,  metrics=0.9)
        sched.step(600,  metrics=0.9)
        sched.step(600,  metrics=0.9)   # first decay
        assert lr(opt) == approx(BASE_LR * 0.5)
        sched.step(300,  metrics=0.5)   # cd=200
        assert lr(opt) == approx(BASE_LR * 0.5)
        sched.step(300,  metrics=0.5)   # cd→0, bad reset
        assert lr(opt) == approx(BASE_LR * 0.5)
        sched.step(600,  metrics=0.5)   # bad=600
        assert lr(opt) == approx(BASE_LR * 0.5)
        sched.step(600,  metrics=0.5)   # bad=1200 → second decay
        assert lr(opt) == approx(BASE_LR * 0.25)

    def test_cooldown_zero_means_no_delay(self):
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                    factor=0.5, cooldown=0)
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)
        assert lr(opt) == approx(BASE_LR * 0.5)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.5)
        assert lr(opt) == approx(BASE_LR * 0.25)


# ---------------------------------------------------------------------------
# 10. min_lr floor
# ---------------------------------------------------------------------------

class TestMinLR:
    def test_scalar_min_lr_respected(self):
        min_lr = BASE_LR * 0.4
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                    factor=0.1, min_lr=min_lr)
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)
        assert lr(opt) >= min_lr - 1e-10

    def test_per_group_min_lr(self):
        min_lrs = [BASE_LR * 0.5, BASE_LR * 0.3]
        opt, sched = make_scheduler(0, 0, n_groups=2, mode="max", patience=100,
                                    factor=0.01, min_lr=min_lrs)
        for _ in range(20):
            sched.step(EVAL_STEPS, metrics=0.0)
        for g, ml in zip(opt.param_groups, min_lrs):
            assert g["lr"] >= ml - 1e-10

    def test_eps_prevents_tiny_decay(self):
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                    factor=0.5, min_lr=BASE_LR)
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)
        assert lr(opt) == approx(BASE_LR)


# ---------------------------------------------------------------------------
# 11. Multiple param groups
# ---------------------------------------------------------------------------

class TestMultipleParamGroups:
    def test_each_group_decays_from_own_base_lr(self):
        opt, sched = make_scheduler(0, 0, n_groups=2, mode="max",
                                    patience=1000, factor=0.5)
        base_lrs = [BASE_LR * (i + 1) for i in range(2)]
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)
        for g, blr in zip(opt.param_groups, base_lrs):
            assert g["lr"] == approx(blr * 0.5)

    def test_warmup_ramps_each_group_proportionally(self):
        opt, sched = make_scheduler(0, 400, n_groups=2, patience=None)
        base_lrs = [BASE_LR * (i + 1) for i in range(2)]
        sched.step(200)   # scale = 200/400 = 0.5
        for g, blr in zip(opt.param_groups, base_lrs):
            assert g["lr"] == approx(blr * 0.5)


# ---------------------------------------------------------------------------
# 12. State dict / checkpointing
# ---------------------------------------------------------------------------

class TestStateDict:
    def _partial_run(self, n_evals, eval_steps=EVAL_STEPS):
        opt, sched = make_scheduler(200, 200, mode="max", patience=1000, factor=0.5)
        # plateau_start=400; first 4 evals of 100 steps are non-plateau
        for i in range(n_evals):
            total_before = sched._total_steps
            will_plateau = (total_before + eval_steps) >= sched._plateau_start_step
            sched.step(eval_steps, metrics=0.8 if will_plateau else None)
        return opt, sched

    def test_roundtrip_preserves_total_steps(self):
        _, sched = self._partial_run(6)
        sd = sched.state_dict()
        _, sched2 = make_scheduler(200, 200, mode="max", patience=1000, factor=0.5)
        sched2.load_state_dict(sd)
        assert sched2._total_steps == sched._total_steps

    def test_roundtrip_preserves_best(self):
        _, sched = self._partial_run(6)
        sd = sched.state_dict()
        _, sched2 = make_scheduler(200, 200, mode="max", patience=1000, factor=0.5)
        sched2.load_state_dict(sd)
        assert sched2._best == sched._best

    def test_roundtrip_preserves_steps_without_improvement(self):
        _, sched = self._partial_run(8)
        sd = sched.state_dict()
        _, sched2 = make_scheduler(200, 200, mode="max", patience=1000, factor=0.5)
        sched2.load_state_dict(sd)
        assert sched2._steps_without_improvement == sched._steps_without_improvement

    def test_roundtrip_preserves_plateau_multipliers(self):
        _, sched = self._partial_run(16)  # enough to trigger decay
        sd = sched.state_dict()
        _, sched2 = make_scheduler(200, 200, mode="max", patience=1000, factor=0.5)
        sched2.load_state_dict(sd)
        assert sched2._plateau_multipliers == sched._plateau_multipliers

    def test_roundtrip_restores_lr_to_optimizer(self):
        opt, sched = self._partial_run(16)
        sd = sched.state_dict()
        opt2, sched2 = make_scheduler(200, 200, mode="max", patience=1000, factor=0.5)
        sched2.load_state_dict(sd)
        assert lr(opt2) == approx(lr(opt))

    def test_resume_continues_decay_at_same_step(self):
        """After a checkpoint, decay fires at the same cumulative step count."""
        opt, sched = make_scheduler(0, 0, mode="max", patience=1000, factor=0.5)
        sched.step(EVAL_STEPS, metrics=0.9)   # best
        for _ in range(9):
            sched.step(EVAL_STEPS, metrics=0.9)  # 900 bad steps
        assert sched._steps_without_improvement == 900
        sd = sched.state_dict()

        opt2, sched2 = make_scheduler(0, 0, mode="max", patience=1000, factor=0.5)
        sched2.load_state_dict(sd)
        sched2.step(200, metrics=0.9)   # 900+200=1100 > 1000 → decay
        assert lr(opt2) == approx(BASE_LR * 0.5)

    def test_state_dict_contains_all_keys(self):
        _, sched = make_scheduler(100, 100, mode="max", patience=1000)
        sd = sched.state_dict()
        for key in ("_total_steps", "_warmup_steps_elapsed", "_best",
                    "_steps_without_improvement", "_cooldown_steps_remaining",
                    "_plateau_multipliers", "_plateau_phase_started"):
            assert key in sd


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_freeze_zero_warmup_immediate_plateau(self):
        _, sched = make_scheduler(0, 0, patience=9999)
        assert sched._current_phase() == "plateau"
        assert lr(sched.optimizer) == approx(BASE_LR)

    def test_patience_zero_any_bad_step_decays(self):
        """patience=0: any nonzero bad steps → steps_without_improvement > 0 → decay."""
        opt, sched = make_scheduler(0, 0, mode="max", patience=0, factor=0.5)
        sched.step(EVAL_STEPS, metrics=0.9)   # best, bad=0 (not > 0)
        assert lr(opt) == approx(BASE_LR)
        sched.step(EVAL_STEPS, metrics=0.8)   # bad=100 > 0 → decay
        assert lr(opt) == approx(BASE_LR * 0.5)

    def test_warmup_scale_clamped_at_1(self):
        """Overshoot: if more steps are taken than num_warmup_steps, scale stays at 1."""
        opt, sched = make_scheduler(0, 200, patience=None)
        sched.step(500)   # 300 steps past warmup end
        assert sched._warmup_steps_elapsed == 200
        assert lr(opt) == approx(BASE_LR)

    def test_very_large_single_step(self):
        """A single step() covering all phases should land correctly in plateau."""
        opt, sched = make_scheduler(100, 200, patience=None)
        sched.step(10_000)
        assert sched._current_phase() == "plateau"
        assert lr(opt) == approx(BASE_LR)

    def test_plateau_multipliers_not_used_in_warmup(self):
        """_plateau_multipliers should not affect the LR during warmup."""
        opt, sched = make_scheduler(0, 400, patience=None)
        # Manually corrupt multipliers to ensure they don't bleed in
        sched._plateau_multipliers = [0.1]
        sched.step(200)   # scale = 200/400 = 0.5 (ignores multiplier)
        assert lr(opt) == approx(BASE_LR * 0.5)


# ---------------------------------------------------------------------------
# 14. No-decay mode (patience=None)
# ---------------------------------------------------------------------------

class TestNoDecayMode:
    def test_factor_gte_1_allowed(self):
        make_scheduler(0, 0, patience=None, factor=1.0)
        make_scheduler(0, 0, patience=None, factor=99.0)

    def test_lr_constant_after_warmup(self):
        opt, sched = make_scheduler(0, 0, patience=None)
        for _ in range(20):
            sched.step(EVAL_STEPS)
        assert lr(opt) == approx(BASE_LR)

    def test_metrics_not_required(self):
        _, sched = make_scheduler(0, 0, patience=None)
        for _ in range(5):
            sched.step(EVAL_STEPS)  # no metrics; must not raise

    def test_metrics_silently_ignored(self):
        opt, sched = make_scheduler(0, 0, patience=None)
        for _ in range(5):
            sched.step(EVAL_STEPS, metrics=0.0)
        assert lr(opt) == approx(BASE_LR)

    def test_freeze_and_warmup_unaffected(self):
        opt, sched = make_scheduler(200, 400, patience=None)
        sched.step(200)   # through freeze
        assert lr(opt) == 0.0
        sched.step(200)   # halfway through warmup
        assert lr(opt) == approx(BASE_LR * 0.5)
        sched.step(200)   # end of warmup
        assert lr(opt) == approx(BASE_LR)

    def test_state_dict_roundtrip(self):
        opt, sched = make_scheduler(100, 100, patience=None)
        for _ in range(5):
            sched.step(EVAL_STEPS)
        sd = sched.state_dict()
        opt2, sched2 = make_scheduler(100, 100, patience=None)
        sched2.load_state_dict(sd)
        for _ in range(5):
            sched2.step(EVAL_STEPS, metrics=0.0)
        assert lr(opt2) == approx(BASE_LR)


# ---------------------------------------------------------------------------
# 15. status() method
# ---------------------------------------------------------------------------

class TestStatus:
    def test_phase_freeze(self):
        _, sched = make_scheduler(200, 200, patience=None)
        s = sched.status()
        assert s.phase == "freeze"

    def test_phase_warmup(self):
        _, sched = make_scheduler(100, 200, patience=None)
        sched.step(100)
        s = sched.status()
        assert s.phase == "warmup"

    def test_phase_plateau(self):
        _, sched = make_scheduler(0, 0, patience=None)
        s = sched.status()
        assert s.phase == "plateau"

    def test_total_steps(self):
        _, sched = make_scheduler(0, 0, patience=None)
        sched.step(123)
        assert sched.status().total_steps == 123

    def test_lrs_matches_optimizer(self):
        opt, sched = make_scheduler(0, 0, patience=None)
        s = sched.status()
        assert s.lrs == [g["lr"] for g in opt.param_groups]

    def test_warmup_progress_none_outside_warmup(self):
        _, sched = make_scheduler(100, 200, patience=None)
        assert sched.status().warmup_progress is None   # freeze
        sched.step(300)
        assert sched.status().warmup_progress is None   # plateau

    def test_warmup_progress_during_warmup(self):
        _, sched = make_scheduler(0, 400, patience=None)
        sched.step(100)
        s = sched.status()
        assert s.phase == "warmup"
        assert s.warmup_progress == pytest.approx(0.25)

    def test_warmup_progress_at_completion(self):
        _, sched = make_scheduler(0, 400, patience=None)
        sched.step(400)
        # Now in plateau; warmup_progress should be None
        assert sched.status().warmup_progress is None

    def test_best_metric_none_before_plateau(self):
        _, sched = make_scheduler(100, 100, patience=9999)
        sched.step(100)   # still in warmup
        assert sched.status().best_metric is None

    def test_best_metric_none_patience_none(self):
        _, sched = make_scheduler(0, 0, patience=None)
        sched.step(EVAL_STEPS)
        assert sched.status().best_metric is None

    def test_best_metric_after_first_score(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=9999)
        sched.step(EVAL_STEPS, metrics=0.75)
        assert sched.status().best_metric == pytest.approx(0.75)

    def test_best_metric_updates_on_improvement(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=9999)
        sched.step(EVAL_STEPS, metrics=0.75)
        sched.step(EVAL_STEPS, metrics=0.90)
        assert sched.status().best_metric == pytest.approx(0.90)

    def test_steps_without_improvement_none_outside_plateau(self):
        _, sched = make_scheduler(100, 0, patience=9999)
        assert sched.status().steps_without_improvement is None  # freeze

    def test_steps_without_improvement_accumulates(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=9999)
        sched.step(EVAL_STEPS, metrics=0.8)   # sets best, bad=0
        sched.step(EVAL_STEPS, metrics=0.8)   # bad += 100
        sched.step(EVAL_STEPS, metrics=0.8)   # bad += 100
        assert sched.status().steps_without_improvement == 200

    def test_patience_remaining_counts_down(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=1000)
        sched.step(EVAL_STEPS, metrics=0.8)   # best, bad=0
        sched.step(EVAL_STEPS, metrics=0.8)   # bad=100
        sched.step(EVAL_STEPS, metrics=0.8)   # bad=200
        assert sched.status().patience_remaining == 800

    def test_patience_remaining_none_outside_plateau(self):
        _, sched = make_scheduler(100, 0, patience=1000)
        assert sched.status().patience_remaining is None

    def test_patience_remaining_none_in_cooldown(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                  factor=0.5, cooldown=500)
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)   # triggers decay → cooldown
        s = sched.status()
        assert s.in_cooldown
        assert s.patience_remaining is None

    def test_in_cooldown_false_by_default(self):
        _, sched = make_scheduler(0, 0, patience=9999)
        sched.step(EVAL_STEPS, metrics=0.8)
        assert not sched.status().in_cooldown

    def test_in_cooldown_true_after_decay(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                  factor=0.5, cooldown=500)
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)
        assert sched.status().in_cooldown

    def test_cooldown_steps_remaining_none_when_not_in_cooldown(self):
        _, sched = make_scheduler(0, 0, patience=9999)
        sched.step(EVAL_STEPS, metrics=0.8)
        assert sched.status().cooldown_steps_remaining is None

    def test_cooldown_steps_remaining_counts_down(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                  factor=0.5, cooldown=500)
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)   # decay → cd=500
        assert sched.status().cooldown_steps_remaining == 500
        sched.step(200, metrics=0.5)   # drains 200 steps of cooldown
        assert sched.status().cooldown_steps_remaining == 300

    def test_str_contains_phase(self):
        _, sched = make_scheduler(0, 0, patience=None)
        assert "plateau" in str(sched.status())

    def test_str_contains_lr(self):
        _, sched = make_scheduler(0, 0, patience=None)
        assert "lr=" in str(sched.status())

    def test_str_contains_patience_remaining_in_plateau(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=1000)
        sched.step(EVAL_STEPS, metrics=0.8)
        assert "patience_remaining" in str(sched.status())

    def test_str_contains_cooldown_when_active(self):
        _, sched = make_scheduler(0, 0, mode="max", patience=1000,
                                  factor=0.5, cooldown=500)
        sched.step(EVAL_STEPS, metrics=0.9)
        for _ in range(11):
            sched.step(EVAL_STEPS, metrics=0.9)
        assert "cooldown_remaining" in str(sched.status())


# ---------------------------------------------------------------------------
# 16. reset_optimizer_on_unfreeze
# ---------------------------------------------------------------------------

class TestResetOptimizerOnUnfreeze:
    def test_state_cleared_at_unfreeze(self):
        """Optimizer state for all parameters should be empty after the first
        step() that exits the freeze phase."""
        opt, sched = make_scheduler(200, 200, patience=None,
                                    reset_optimizer_on_unfreeze=True)
        # Seed the optimizer state by doing a real forward/backward pass
        # (or just poke it directly, which is simpler in a unit test).
        model_params = list(opt.param_groups[0]["params"])
        for p in model_params:
            opt.state[p]["dummy_accum"] = torch.zeros_like(p)

        assert any(opt.state for _ in [None])  # state is non-empty before unfreeze

        sched.step(200)  # total=200, exits freeze → reset fires
        for p in model_params:
            assert p not in opt.state, "optimizer state should be cleared at unfreeze"

    def test_state_cleared_exactly_once(self):
        """The reset must not fire a second time on subsequent steps."""
        opt, sched = make_scheduler(200, 200, patience=None,
                                    reset_optimizer_on_unfreeze=True)
        sched.step(200)  # exits freeze, reset fires, _optimizer_reset_done=True

        # Seed some state after the reset
        for p in opt.param_groups[0]["params"]:
            opt.state[p]["post_reset"] = torch.zeros_like(p)

        sched.step(100)  # still in warmup, reset must NOT fire again
        for p in opt.param_groups[0]["params"]:
            assert "post_reset" in opt.state[p], (
                "optimizer state should not be cleared on subsequent steps"
            )

    def test_reset_fires_on_boundary_spanning_step(self):
        """A single step() that covers both freeze and warmup should still reset."""
        opt, sched = make_scheduler(200, 200, patience=None,
                                    reset_optimizer_on_unfreeze=True)
        for p in opt.param_groups[0]["params"]:
            opt.state[p]["accum"] = torch.zeros_like(p)

        sched.step(350)  # spans freeze→warmup boundary in one call
        for p in opt.param_groups[0]["params"]:
            assert p not in opt.state

    def test_reset_fires_when_spanning_all_phases(self):
        """A step() that spans freeze→warmup→plateau should still reset."""
        opt, sched = make_scheduler(100, 100, patience=None,
                                    reset_optimizer_on_unfreeze=True)
        for p in opt.param_groups[0]["params"]:
            opt.state[p]["accum"] = torch.zeros_like(p)

        sched.step(1000)  # covers all three phases
        for p in opt.param_groups[0]["params"]:
            assert p not in opt.state

    def test_no_reset_when_flag_false(self):
        """With reset_optimizer_on_unfreeze=False (default), state is untouched."""
        opt, sched = make_scheduler(200, 200, patience=None,
                                    reset_optimizer_on_unfreeze=False)
        for p in opt.param_groups[0]["params"]:
            opt.state[p]["accum"] = torch.zeros_like(p)

        sched.step(200)
        for p in opt.param_groups[0]["params"]:
            assert "accum" in opt.state[p], "state should be preserved when flag is False"

    def test_no_reset_when_no_freeze(self):
        """With num_freeze_steps=0 there is no freeze→warmup transition,
        so the reset should never fire even if the flag is True."""
        opt, sched = make_scheduler(0, 200, patience=None,
                                    reset_optimizer_on_unfreeze=True)
        for p in opt.param_groups[0]["params"]:
            opt.state[p]["accum"] = torch.zeros_like(p)

        sched.step(100)  # in warmup from the start, no freeze boundary
        for p in opt.param_groups[0]["params"]:
            assert "accum" in opt.state[p]

    def test_optimizer_reset_done_flag_set(self):
        _, sched = make_scheduler(200, 0, patience=None,
                                  reset_optimizer_on_unfreeze=True)
        assert not sched._optimizer_reset_done
        sched.step(200)
        assert sched._optimizer_reset_done

    def test_optimizer_reset_done_not_set_when_still_frozen(self):
        _, sched = make_scheduler(200, 0, patience=None,
                                  reset_optimizer_on_unfreeze=True)
        sched.step(100)   # still in freeze
        assert not sched._optimizer_reset_done

    def test_reset_works_with_multiple_param_groups(self):
        opt, sched = make_scheduler(200, 200, n_groups=2, patience=None,
                                    reset_optimizer_on_unfreeze=True)
        for group in opt.param_groups:
            for p in group["params"]:
                opt.state[p]["accum"] = torch.zeros_like(p)

        sched.step(200)
        for group in opt.param_groups:
            for p in group["params"]:
                assert p not in opt.state

    def test_state_dict_preserves_reset_done_flag(self):
        """After a checkpoint round-trip, _optimizer_reset_done is restored
        so the reset does not fire again on resumption."""
        opt, sched = make_scheduler(200, 200, patience=None,
                                    reset_optimizer_on_unfreeze=True)
        sched.step(200)   # fires reset, sets _optimizer_reset_done=True
        assert sched._optimizer_reset_done

        sd = sched.state_dict()
        opt2, sched2 = make_scheduler(200, 200, patience=None,
                                      reset_optimizer_on_unfreeze=True)
        sched2.load_state_dict(sd)
        assert sched2._optimizer_reset_done

        # Seed state after reload and confirm next step doesn't clear it
        for p in opt2.param_groups[0]["params"]:
            opt2.state[p]["post_reload"] = torch.zeros_like(p)
        sched2.step(100)
        for p in opt2.param_groups[0]["params"]:
            assert "post_reload" in opt2.state[p]

    def test_state_dict_contains_reset_done_key(self):
        _, sched = make_scheduler(100, 100, patience=None,
                                  reset_optimizer_on_unfreeze=True)
        assert "_optimizer_reset_done" in sched.state_dict()
