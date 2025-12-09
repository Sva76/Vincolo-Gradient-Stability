"""
controller.py
==============

Core implementation of the Vincolo Gradient Stability Controller.

Vincolo is a lightweight wrapper around a standard training loop that:

- Tracks a smoothed stability signal S(t) derived from the loss trajectory
- Computes a "shock" level as |S(t) - S(t-1)|
- Detects instability events when shock exceeds a threshold
- Temporarily damps the learning rate during recovery
- Exposes a simple `.update(loss)` API returning a stats dict

This module is agnostic to:
- model architecture
- optimizer implementation
- training backend (PyTorch, Tinker, etc.)

You only need to:
1) instantiate `StabilityController`
2) call `update(loss_value)` at each step
3) use `stats["eta_t"]` as your stabilized learning rate
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class ControllerState:
    """Internal state snapshot of the stability controller."""
    step: int = 0
    S_t: float = 0.0              # current stability signal
    S_prev: float = 0.0           # previous stability signal
    in_recovery: bool = False     # whether we are in recovery mode
    recovery_steps_left: int = 0  # remaining steps in recovery
    last_loss: Optional[float] = None


class StabilityController:
    """
    Vincolo Gradient Stability Controller.

    Parameters
    ----------
    eta_base : float
        Base learning rate used in stable regime.
    shock_threshold : float
        Threshold on |S(t) - S(t-1)| that triggers recovery mode.
    window : int
        Smoothing window length used for the stability signal S(t).
        Higher values -> slower, smoother response.
    recovery_factor : float
        Multiplicative factor applied to eta_base during recovery.
        Example: eta_t = eta_base * recovery_factor.
    max_recovery_steps : int
        Maximum number of steps to stay in recovery after a shock.
    critical_shock : Optional[float]
        Optional higher threshold above which an event is marked as "critical".
        If None, no critical flag is set.
    """

    def __init__(
        self,
        eta_base: float = 1e-3,
        shock_threshold: float = 1.5,
        window: int = 50,
        recovery_factor: float = 0.1,
        max_recovery_steps: int = 3,
        critical_shock: Optional[float] = None,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")

        self.eta_base = float(eta_base)
        self.shock_threshold = float(shock_threshold)
        self.window = int(window)
        self.recovery_factor = float(recovery_factor)
        self.max_recovery_steps = int(max_recovery_steps)
        self.critical_shock = float(critical_shock) if critical_shock is not None else None

        # Smoothing coefficient for S(t) (EMA style)
        # alpha ~ 2 / (window + 1) is a common choice
        self.alpha = 2.0 / (self.window + 1.0)

        self.state = ControllerState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the internal state of the controller."""
        self.state = ControllerState()

    def update(self, loss_value: float) -> Dict[str, Any]:
        """
        Update stability statistics with a new loss value.

        This should be called once per training step.

        Parameters
        ----------
        loss_value : float
            The scalar training loss at current step.

        Returns
        -------
        stats : Dict[str, Any]
            Dictionary containing:
            - "step": current step index (int)
            - "loss": current loss (float)
            - "S_t": stability signal S(t) (float)
            - "shock": |S(t) - S(t-1)| (float)
            - "eta_t": stabilized learning rate (float)
            - "regime": "NORMAL" or "RECOVERY" (str)
            - "in_recovery": bool
            - "critical": bool (if critical_shock is set)
        """
        st = self.state
        st.step += 1

        # --- 1) Update stability signal S(t) via EMA of |Δloss| ---
        if st.last_loss is None:
            # first step: no delta yet
            delta = 0.0
            S_new = 0.0
        else:
            delta = abs(float(loss_value) - float(st.last_loss))
            # EMA update: S_t = (1 - alpha) * S_prev + alpha * delta
            S_new = (1.0 - self.alpha) * st.S_t + self.alpha * delta

        st.S_prev = st.S_t
        st.S_t = S_new
        st.last_loss = float(loss_value)

        # --- 2) Compute shock = |S(t) - S(t-1)| ---
        shock = abs(st.S_t - st.S_prev)

        # --- 3) Decide regime (NORMAL vs RECOVERY) ---
        regime = "NORMAL"

        # Case A: already in recovery
        if st.in_recovery:
            regime = "RECOVERY"
            st.recovery_steps_left -= 1
            if st.recovery_steps_left <= 0:
                # Exit recovery
                st.in_recovery = False
                st.recovery_steps_left = 0
                regime = "NORMAL"

        # Case B: not in recovery -> check if we must enter
        if not st.in_recovery and shock >= self.shock_threshold:
            st.in_recovery = True
            st.recovery_steps_left = self.max_recovery_steps
            regime = "RECOVERY"

        # --- 4) Compute stabilized learning rate eta_t ---
        if st.in_recovery:
            eta_t = self.eta_base * self.recovery_factor
        else:
            eta_t = self.eta_base

        # --- 5) Critical flag (optional) ---
        critical = False
        if self.critical_shock is not None and shock >= self.critical_shock:
            critical = True

        # --- 6) Prepare stats dictionary for logging / external use ---
        stats: Dict[str, Any] = {
            "step": st.step,
            "loss": float(loss_value),
            "S_t": float(st.S_t),
            "shock": float(shock),
            "eta_t": float(eta_t),
            "regime": regime,
            "in_recovery": bool(st.in_recovery),
            "critical": critical,
        }

        return stats

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Return a shallow copy of the internal state as a dict.
        Useful for debugging or logging.
        """
        return asdict(self.state)

    def __repr__(self) -> str:
        return (
            f"StabilityController(eta_base={self.eta_base}, "
            f"shock_threshold={self.shock_threshold}, "
            f"window={self.window}, "
            f"recovery_factor={self.recovery_factor}, "
            f"max_recovery_steps={self.max_recovery_steps}, "
            f"critical_shock={self.critical_shock})"
        )
