"""
WarmupThenPlateauScheduler
==========================
A single LR scheduler that chains three phases, all measured in training
*steps* for a consistent time axis regardless of evaluation frequency:

  1. Freeze   – first num_freeze_steps steps at LR = 0
  2. Warmup   – next num_warmup_steps steps linearly ramping 0 → base_lr
  3. Plateau  – ReduceLROnPlateau-style decay thereafter

Pass 0 for num_freeze_steps or num_warmup_steps to skip those phases.

Each call to step(num_steps) advances the internal step counter by num_steps
(the number of training steps taken since the last evaluation).  In the
plateau phase, step() also requires a metrics value; it is not required
during freeze or warmup.

Pass patience=None to disable plateau decay entirely; the scheduler then
acts as a pure freeze-then-warmup-then-constant schedule, and metrics is
never required in step().

Pass reset_optimizer_on_unfreeze=True to clear the optimizer's accumulated
state (momentum buffers, second-moment estimates, etc.) at the moment the
freeze phase ends.  This prevents stale accumulator history built up during
freeze from miscalibrating adaptive optimizers (Adadelta, Adam, RMSprop)
when the parameters first start being updated.

Because every duration is expressed in steps, the LR schedule is completely
independent of how frequently you choose to evaluate.
"""

from dataclasses import dataclass

from torch.optim import Optimizer


@dataclass
class SchedulerStatus:
    """Snapshot of WarmupThenPlateauScheduler state, returned by status()."""
    phase: str
    total_steps: int
    lrs: list
    warmup_progress: float | None
    best_metric: float | None
    steps_without_improvement: int | None
    patience_remaining: int | None
    in_cooldown: bool
    cooldown_steps_remaining: int | None

    def __str__(self) -> str:
        parts = [f"phase={self.phase}", f"total_steps={self.total_steps}"]
        lr_str = ", ".join(f"{lr:.2e}" for lr in self.lrs)
        parts.append(f"lr=[{lr_str}]")
        if self.warmup_progress is not None:
            parts.append(f"warmup={self.warmup_progress:.1%}")
        if self.best_metric is not None:
            parts.append(f"best={self.best_metric:.4g}")
        if self.steps_without_improvement is not None:
            parts.append(f"bad_steps={self.steps_without_improvement}")
        if self.patience_remaining is not None:
            parts.append(f"patience_remaining={self.patience_remaining}")
        if self.in_cooldown:
            parts.append(f"cooldown_remaining={self.cooldown_steps_remaining}")
        return "SchedulerStatus(" + ", ".join(parts) + ")"


class WarmupThenPlateauScheduler:
    """
    Parameters
    ----------
    optimizer : Optimizer
        The optimizer whose LR groups are managed.  The per-group
        ``lr`` values at construction time are treated as the *maximum*
        (post-warmup) LR for each group.
    num_freeze_steps : int
        Number of training steps to hold LR at 0 before warmup begins.
        Pass 0 to skip.
    num_warmup_steps : int
        Number of training steps to linearly ramp from 0 → base_lr.
        Pass 0 to skip (LR jumps straight to base_lr after freeze, or
        from the very start if num_freeze_steps is also 0).
    mode : str
        ``'min'`` or ``'max'`` – whether a lower or higher metric is
        better (same semantics as ReduceLROnPlateau).
    factor : float
        Factor by which the LR is multiplied on each plateau decay
        (default 0.1).  Must be < 1.  Ignored when patience is None.
    patience : int or None
        Number of *training steps* with no improvement to tolerate
        before decaying (default 10_000).  Pass ``None`` to disable
        decay entirely; the LR stays constant after warmup and metrics
        is not required in step().
    cooldown : int
        Number of *training steps* to wait after a decay before
        resuming patience counting (default 0).
    threshold : float
        Minimum change to count as an improvement (default 1e-4).
    threshold_mode : str
        ``'rel'`` (improvement relative to best) or ``'abs'``
        (absolute improvement).  Same as ReduceLROnPlateau.
    min_lr : float or list[float]
        Lower bound(s) on the LR (default 0).
    eps : float
        Minimum decay magnitude; smaller decays are ignored
        (default 1e-8).
    verbose : bool
        Print a message when the LR is decayed (default False).
    reset_optimizer_on_unfreeze : bool
        If True, the optimizer's accumulated state is cleared for all
        parameter groups at the moment the freeze phase ends (i.e. on
        the first step() call that moves total_steps past
        num_freeze_steps).  This is recommended when using adaptive
        optimizers (Adadelta, Adam, AdamW, RMSprop) with a non-zero
        freeze period, to prevent accumulator history built up during
        freeze from producing a miscalibrated effective LR at the start
        of warmup.  Has no effect when num_freeze_steps=0 (default False).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_freeze_steps: int,
        num_warmup_steps: int,
        # --- plateau args ---
        mode: str = "min",
        factor: float = 0.1,
        patience: int | None = 10_000,
        cooldown: int = 0,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        min_lr: float | list[float] = 0.0,
        eps: float = 1e-8,
        verbose: bool = False,
        reset_optimizer_on_unfreeze: bool = False,
    ):
        if patience is not None and factor >= 1.0:
            raise ValueError(f"factor must be < 1.0, got {factor}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(
                f"threshold_mode must be 'rel' or 'abs', got {threshold_mode!r}"
            )
        if num_freeze_steps < 0:
            raise ValueError(f"num_freeze_steps must be >= 0, got {num_freeze_steps}")
        if num_warmup_steps < 0:
            raise ValueError(f"num_warmup_steps must be >= 0, got {num_warmup_steps}")
        if patience is not None and patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {cooldown}")

        self.optimizer = optimizer
        self.num_freeze_steps = num_freeze_steps
        self.num_warmup_steps = num_warmup_steps
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.eps = eps
        self.verbose = verbose
        self.reset_optimizer_on_unfreeze = reset_optimizer_on_unfreeze

        # Capture base LRs from the optimizer before we touch anything
        self._base_lrs: list[float] = [g["lr"] for g in optimizer.param_groups]

        # Normalise min_lr to a per-group list
        num_groups = len(optimizer.param_groups)
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != num_groups:
                raise ValueError(
                    f"min_lr length ({len(min_lr)}) must match "
                    f"number of param groups ({num_groups})"
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * num_groups

        # Step-based phase tracking
        self._total_steps: int = 0
        self._warmup_steps_elapsed: int = 0
        # Tracks whether the optimizer reset at unfreeze has been done,
        # so it fires exactly once even if a step() call spans the boundary.
        self._optimizer_reset_done: bool = False

        # Plateau counters
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._steps_without_improvement: int = 0
        self._cooldown_steps_remaining: int = 0
        self._plateau_multipliers: list[float] = [1.0] * num_groups
        self._plateau_phase_started: bool = False

        # Apply the initial LR immediately
        self._apply_lr()

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @property
    def _warmup_start_step(self) -> int:
        return self.num_freeze_steps

    @property
    def _plateau_start_step(self) -> int:
        return self.num_freeze_steps + self.num_warmup_steps

    def _current_phase(self) -> str:
        if self._total_steps < self._warmup_start_step:
            return "freeze"
        if self._total_steps < self._plateau_start_step:
            return "warmup"
        return "plateau"

    # ------------------------------------------------------------------
    # LR computation and application
    # ------------------------------------------------------------------

    def _current_lrs(self) -> list[float]:
        phase = self._current_phase()

        if phase == "freeze":
            return [0.0] * len(self._base_lrs)

        if phase == "warmup":
            if self.num_warmup_steps == 0:
                scale = 1.0
            else:
                scale = min(self._warmup_steps_elapsed / self.num_warmup_steps, 1.0)
            return [
                max(blr * scale, mlr)
                for blr, mlr in zip(self._base_lrs, self.min_lrs)
            ]

        # plateau: base_lr scaled by accumulated decay multipliers
        return [
            max(blr * mult, mlr)
            for blr, mult, mlr in zip(
                self._base_lrs, self._plateau_multipliers, self.min_lrs
            )
        ]

    def _apply_lr(self):
        for group, lr in zip(self.optimizer.param_groups, self._current_lrs()):
            group["lr"] = lr

    # ------------------------------------------------------------------
    # Public step interface
    # ------------------------------------------------------------------

    def step(self, num_steps: int, metrics: float | None = None):
        """
        Advance the scheduler by num_steps training steps.

        Can be called the same way at every evaluation regardless of the
        current phase — just always pass num_steps and metrics and let the
        scheduler decide what to do with them.

        Parameters
        ----------
        num_steps : int
            Number of training steps taken since the last call to step().
            Must be positive.
        metrics : float, optional
            Validation metric for this evaluation.  Always safe to pass
            regardless of the current phase; silently ignored during freeze,
            warmup, and when patience=None.  If None during the plateau
            phase, the step advances the internal clock but does not update
            improvement tracking (useful for mid-epoch calls without a score).
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be a positive integer, got {num_steps}")

        prev_phase = self._current_phase()
        self._total_steps += num_steps

        # Reset optimizer state on the first step that exits the freeze phase.
        if (self.reset_optimizer_on_unfreeze
                and not self._optimizer_reset_done
                and prev_phase == "freeze"
                and self._current_phase() != "freeze"):
            self._reset_optimizer_state()
            self._optimizer_reset_done = True

        # Accumulate steps that fell inside the warmup window.
        # A single step() call may span a phase boundary, so we only count
        # the portion that actually landed in warmup.
        if self.num_warmup_steps > 0:
            warmup_end = self._plateau_start_step
            warmup_start = self._warmup_start_step
            prev_total = self._total_steps - num_steps
            steps_in_warmup = (
                min(self._total_steps, warmup_end)
                - max(prev_total, warmup_start)
            )
            if steps_in_warmup > 0:
                self._warmup_steps_elapsed = min(
                    self._warmup_steps_elapsed + steps_in_warmup,
                    self.num_warmup_steps,
                )

        # Non-plateau phases: just update the LR and return.
        if self._current_phase() != "plateau":
            self._apply_lr()
            return

        # ---- plateau phase ----

        if self.patience is None:
            self._apply_lr()
            return

        if not self._plateau_phase_started:
            self._best = float("inf") if self.mode == "min" else float("-inf")
            self._steps_without_improvement = 0
            self._cooldown_steps_remaining = 0
            self._plateau_phase_started = True

        # Steps that fell inside the plateau window for this call.
        steps_in_plateau = self._total_steps - max(
            self._total_steps - num_steps, self._plateau_start_step
        )

        if self._in_cooldown():
            self._cooldown_steps_remaining = max(
                0, self._cooldown_steps_remaining - steps_in_plateau
            )
            if not self._in_cooldown():
                self._steps_without_improvement = 0
        elif metrics is not None:
            # Only update improvement tracking when a real score is provided.
            if self._is_better(metrics, self._best):
                self._best = metrics
                self._steps_without_improvement = 0
            else:
                self._steps_without_improvement += steps_in_plateau

            if self._steps_without_improvement > self.patience:
                self._decay_lrs()
                self._cooldown_steps_remaining = self.cooldown
                self._steps_without_improvement = 0

        self._apply_lr()

    # ------------------------------------------------------------------
    # Optimizer reset
    # ------------------------------------------------------------------

    def _reset_optimizer_state(self):
        """Clear per-parameter optimizer state for all param groups.

        PyTorch optimizers store their running statistics (momentum buffers,
        squared-gradient accumulators, etc.) in optimizer.state, keyed by
        parameter tensor.  Clearing these entries causes the optimizer to
        reinitialise them from scratch on the next step(), as if the
        parameters had never been seen before.
        """
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.optimizer.state.pop(p, None)

    # ------------------------------------------------------------------
    # Plateau internals
    # ------------------------------------------------------------------

    def _in_cooldown(self) -> bool:
        return self._cooldown_steps_remaining > 0

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            threshold_val = (best * (1.0 - self.threshold) if self.threshold_mode == "rel"
                             else best - self.threshold)
            return current < threshold_val
        else:
            threshold_val = (best * (1.0 + self.threshold) if self.threshold_mode == "rel"
                             else best + self.threshold)
            return current > threshold_val

    def _decay_lrs(self):
        for i, (group, min_lr) in enumerate(
            zip(self.optimizer.param_groups, self.min_lrs)
        ):
            old_lr = group["lr"]
            new_lr = max(old_lr * self.factor, min_lr)
            if old_lr - new_lr > self.eps:
                group["lr"] = new_lr
                self._plateau_multipliers[i] *= self.factor
                if self.verbose:
                    print(
                        f"  WarmupThenPlateauScheduler: group {i} LR "
                        f"{old_lr:.4e} → {new_lr:.4e}"
                    )

    # ------------------------------------------------------------------
    # Status (for logging)
    # ------------------------------------------------------------------

    def status(self) -> "SchedulerStatus":
        """
        Return a snapshot of the scheduler's current state, suitable for
        logging.  All values reflect the state *after* the most recent
        step() call.

        Returns
        -------
        SchedulerStatus
            A dataclass with the following fields:

            phase : str
                Current phase: ``'freeze'``, ``'warmup'``, or ``'plateau'``.
            total_steps : int
                Total training steps elapsed so far.
            lrs : list[float]
                Current learning rate for each param group.
            warmup_progress : float or None
                Fraction of warmup completed (0.0-1.0).  None outside warmup.
            best_metric : float or None
                Best metric seen so far in the plateau phase.  None if the
                plateau phase has not started yet or patience is None.
            steps_without_improvement : int or None
                Training steps accumulated without a new best metric.  None
                outside the plateau phase or when patience is None.
            patience_remaining : int or None
                Steps remaining before the next LR decay
                (patience - steps_without_improvement).  None outside
                the plateau phase, when in cooldown, or when patience is None.
            in_cooldown : bool
                True if the scheduler is in the post-decay cooldown period.
            cooldown_steps_remaining : int or None
                Steps left in the current cooldown period.  None when not in
                cooldown or when patience is None.
        """
        phase = self._current_phase()
        lrs = [g["lr"] for g in self.optimizer.param_groups]

        warmup_progress = None
        if phase == "warmup" and self.num_warmup_steps > 0:
            warmup_progress = self._warmup_steps_elapsed / self.num_warmup_steps

        in_plateau = phase == "plateau" and self.patience is not None

        best_metric = (
            self._best if in_plateau and self._plateau_phase_started else None
        )
        steps_without_improvement = (
            self._steps_without_improvement if in_plateau else None
        )
        in_cooldown = in_plateau and self._in_cooldown()
        cooldown_steps_remaining = (
            self._cooldown_steps_remaining if in_cooldown else None
        )
        patience_remaining = (
            self.patience - self._steps_without_improvement
            if in_plateau and not in_cooldown
            else None
        )

        return SchedulerStatus(
            phase=phase,
            total_steps=self._total_steps,
            lrs=lrs,
            warmup_progress=warmup_progress,
            best_metric=best_metric,
            steps_without_improvement=steps_without_improvement,
            patience_remaining=patience_remaining,
            in_cooldown=in_cooldown,
            cooldown_steps_remaining=cooldown_steps_remaining,
        )

    # ------------------------------------------------------------------
    # State dict (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "num_freeze_steps": self.num_freeze_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "cooldown": self.cooldown,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "min_lrs": self.min_lrs,
            "eps": self.eps,
            "_total_steps": self._total_steps,
            "_warmup_steps_elapsed": self._warmup_steps_elapsed,
            "_best": self._best,
            "_steps_without_improvement": self._steps_without_improvement,
            "_cooldown_steps_remaining": self._cooldown_steps_remaining,
            "_plateau_multipliers": self._plateau_multipliers,
            "_plateau_phase_started": self._plateau_phase_started,
            "_optimizer_reset_done": self._optimizer_reset_done,
        }

    def load_state_dict(self, state_dict: dict):
        state_dict = dict(state_dict)  # don't mutate the caller's dict
        self._total_steps = state_dict["_total_steps"]
        self._warmup_steps_elapsed = state_dict["_warmup_steps_elapsed"]
        self._best = state_dict["_best"]
        self._steps_without_improvement = state_dict["_steps_without_improvement"]
        self._cooldown_steps_remaining = state_dict["_cooldown_steps_remaining"]
        self._plateau_multipliers = state_dict["_plateau_multipliers"]
        self._plateau_phase_started = state_dict["_plateau_phase_started"]
        self._optimizer_reset_done = state_dict["_optimizer_reset_done"]
        self._apply_lr()
