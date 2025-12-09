"""
training_mrpc.py
================

Simulated MRPC-like training loop using the Vincolo StabilityController.

This script is NOT tied to a specific model or dataset. Instead, it:

- simulates a training loss trajectory with noise
- optionally injects "catastrophic" batches (large loss spikes)
- compares:
    * baseline (no control)
    * Vincolo-stabilized training
- prints logs similar to the examples in the README
- computes a fake validation loss distribution for comparison

Usage examples
--------------
# Baseline, no stability control
python training_mrpc.py --mode baseline --max_steps 800 --lr 1e-2

# With Vincolo stability control
python training_mrpc.py --mode vincolo --max_steps 800 --lr 1e-2

# Recovery test with catastrophic batches
python training_mrpc.py --mode recovery --max_steps 800 --lr 1e-2 --catastrophic_prob 0.05
"""

import argparse
import math
import random
from typing import List, Tuple, Dict

import numpy as np

from controller import StabilityController


# ---------------------------------------------------------------------
# Helper: synthetic "training loss" generator
# ---------------------------------------------------------------------
def simulate_loss(
    step: int,
    base_level: float = 1.2,
    oscillation: float = 0.4,
    noise_std: float = 0.2,
) -> float:
    """
    Generate a synthetic loss value with:
    - smooth oscillation (like optimization dynamics)
    - Gaussian noise
    """
    smooth = base_level + oscillation * math.sin(step / 40.0)
    noise = random.gauss(0.0, noise_std)
    return max(0.05, smooth + noise)


def maybe_inject_catastrophic(loss: float, prob: float, strength: float = 3.0) -> Tuple[float, bool]:
    """
    With probability `prob`, amplify the loss to simulate a catastrophic batch.
    """
    if random.random() < prob:
        return loss * strength, True
    return loss, False


# ---------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------
def run_training_baseline(
    max_steps: int,
    lr: float,
    catastrophic_prob: float = 0.0,
) -> Dict[str, float]:
    """
    Baseline training loop without Vincolo.
    We still compute a simple stability proxy for logging, but we do NOT
    adjust the learning rate or enter recovery mode.
    """
    print(f"=== BASELINE TRAINING (no Vincolo) ===")
    print(f"max_steps={max_steps}  lr={lr}  catastrophic_prob={catastrophic_prob}")
    print("======================================")

    # simple EMA-based stability proxy (not using StabilityController)
    S_t = 0.0
    alpha = 2.0 / (50 + 1.0)  # window ~50
    prev_S = 0.0

    logged_losses: List[float] = []

    for step in range(1, max_steps + 1):
        loss = simulate_loss(step)
        loss, injected = maybe_inject_catastrophic(loss, catastrophic_prob)

        # EMA of |Δloss|
        if step == 1:
            delta = 0.0
        else:
            delta = abs(loss - logged_losses[-1])
        S_t = (1.0 - alpha) * S_t + alpha * delta
        shock = abs(S_t - prev_S)
        prev_S = S_t

        logged_losses.append(loss)

        if step % 50 == 0:
            cat_mark = "💥" if injected else " "
            print(
                f"[step {step}] 🔴{cat_mark} S={S_t:.3f}  shock={shock:.3f}  "
                f"loss={loss:.4f}  lr={lr:.5f}"
            )

    mean_val, std_val = fake_validation_stats(logged_losses)
    print("\n=== VALIDATION SUMMARY (baseline) ===")
    print(f"Mean val loss: {mean_val:.4f}")
    print(f"Std  val loss: {std_val:.4f}")
    print("=====================================\n")

    return {"mean_val": mean_val, "std_val": std_val}


def run_training_vincolo(
    max_steps: int,
    lr: float,
    catastrophic_prob: float = 0.0,
    shock_threshold: float = 1.5,
    recovery_factor: float = 0.1,
    max_recovery_steps: int = 3,
    critical_shock: float = 2.0,
) -> Dict[str, float]:
    """
    Training loop with Vincolo StabilityController.

    - Uses StabilityController to track S(t), shock and eta_t.
    - Adjusts the effective learning rate based on regime (NORMAL/RECOVERY).
    - Logs stats in a style similar to real experiments.
    """
    print(f"=== TRAINING with VINCOLO STABILITY CONTROLLER ===")
    print(
        f"max_steps={max_steps}  base_lr={lr}  catastrophic_prob={catastrophic_prob}  "
        f"shock_threshold={shock_threshold}"
    )
    print("==================================================")

    controller = StabilityController(
        eta_base=lr,
        shock_threshold=shock_threshold,
        window=50,
        recovery_factor=recovery_factor,
        max_recovery_steps=max_recovery_steps,
        critical_shock=critical_shock,
    )

    logged_losses: List[float] = []
    critical_events = 0

    for step in range(1, max_steps + 1):
        loss = simulate_loss(step)
        loss, injected = maybe_inject_catastrophic(loss, catastrophic_prob)

        stats = controller.update(loss)
        eta_t = stats["eta_t"]
        S_t = stats["S_t"]
        shock = stats["shock"]
        regime = stats["regime"]
        critical = stats["critical"]

        if critical:
            critical_events += 1

        logged_losses.append(loss)

        if step % 50 == 0:
            color = "🟢" if regime == "NORMAL" else "🔴"
            cat_mark = "💥" if injected else " "
            print(
                f"[step {step}] {color}{cat_mark} S={S_t:.3f}  shock={shock:.3f}  "
                f"loss={loss:.4f}  lr={eta_t:.5f}  regime={regime}"
            )

    mean_val, std_val = fake_validation_stats(logged_losses)
    print("\n=== VALIDATION SUMMARY (Vincolo) ===")
    print(f"Mean val loss: {mean_val:.4f}")
    print(f"Std  val loss: {std_val:.4f}")
    print(f"Critical events (shock>={critical_shock:.2f}): {critical_events}")
    print("=====================================\n")

    return {
        "mean_val": mean_val,
        "std_val": std_val,
        "critical_events": critical_events,
    }


def fake_validation_stats(train_losses: List[float], n_val: int = 50) -> Tuple[float, float]:
    """
    Generate a fake validation distribution based on the training loss statistics.
    This keeps the example self-contained and dependency-free.
    """
    mean_train = float(np.mean(train_losses))
    std_train = float(np.std(train_losses))

    # validation losses ~ N(mean_train, std_train)
    val_losses = np.random.normal(loc=mean_train, scale=max(std_train, 1e-4), size=n_val)
    val_losses = np.clip(val_losses, 0.05, None)

    return float(np.mean(val_losses)), float(np.std(val_losses))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulated MRPC training with Vincolo stability controller.")

    parser.add_argument(
        "--mode",
        type=str,
        default="vincolo",
        choices=["baseline", "vincolo", "recovery"],
        help="Training mode: 'baseline' (no control), 'vincolo' (stability controller), 'recovery' (catastrophic test)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=800,
        help="Number of training steps to simulate.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--catastrophic_prob",
        type=float,
        default=0.0,
        help="Probability of injecting a catastrophic batch at each step (0.0–1.0).",
    )
    parser.add_argument(
        "--shock_threshold",
        type=float,
        default=1.5,
        help="Shock threshold for Vincolo controller.",
    )
    parser.add_argument(
        "--recovery_factor",
        type=float,
        default=0.1,
        help="LR multiplier during recovery (eta_t = eta_base * recovery_factor).",
    )
    parser.add_argument(
        "--max_recovery_steps",
        type=int,
        default=3,
        help="Maximum number of steps to stay in recovery.",
    )
    parser.add_argument(
        "--critical_shock",
        type=float,
        default=2.0,
        help="Shock level considered 'critical' (for logging and stats).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "baseline":
        run_training_baseline(
            max_steps=args.max_steps,
            lr=args.lr,
            catastrophic_prob=args.catastrophic_prob,
        )

    elif args.mode == "vincolo":
        run_training_vincolo(
            max_steps=args.max_steps,
            lr=args.lr,
            catastrophic_prob=args.catastrophic_prob,
            shock_threshold=args.shock_threshold,
            recovery_factor=args.recovery_factor,
            max_recovery_steps=args.max_recovery_steps,
            critical_shock=args.critical_shock,
        )

    elif args.mode == "recovery":
        # Convenience preset for catastrophic-batch recovery test
        run_training_vincolo(
            max_steps=args.max_steps,
            lr=args.lr,
            catastrophic_prob=args.catastrophic_prob if args.catastrophic_prob > 0 else 0.05,
            shock_threshold=args.shock_threshold,
            recovery_factor=args.recovery_factor,
            max_recovery_steps=args.max_recovery_steps,
            critical_shock=args.critical_shock,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
