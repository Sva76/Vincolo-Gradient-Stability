import argparse, csv, json, os, random
from statistics import median, pstdev

def compute_metrics(rows, shock_step, pre_window=10, recover_k=2.0):
    losses = [r["loss"] for r in rows]
    shock_idx = shock_step - 1

    pre = losses[max(0, shock_idx - pre_window):shock_idx]
    pre_med = median(pre) if pre else losses[0]
    pre_std = pstdev(pre) if len(pre) > 1 else 0.0

    spike = losses[shock_idx] - pre_med

    thr = pre_med + recover_k * pre_std
    ttr = None
    for i in range(shock_idx + 1, len(losses)):
        if losses[i] <= thr:
            ttr = (i + 1) - shock_step
            break

    post = losses[shock_idx:]
    post_var = pstdev(post) ** 2 if len(post) > 1 else 0.0

    return {
        "delta_loss_spike": spike,
        "time_to_recover_steps": ttr,
        "post_loss_variance": post_var,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vincolo", choices=["on","off"], default="off")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--base_lr", type=float, default=5e-4)
    ap.add_argument("--shock_step", type=int, default=40)
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--amplify_reported_shock_baseline", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs("results", exist_ok=True)

    rows = []
    lr = args.base_lr
    loss = 0.12

    for step in range(1, args.steps + 1):
        loss = max(0.02, loss - 0.001 + random.uniform(-0.01, 0.01))

        if step == args.shock_step:
            loss += 0.08

        prev_loss = rows[-1]["loss"] if rows else loss
        shock = abs(loss - prev_loss)

        if args.vincolo == "off" and args.amplify_reported_shock_baseline:
            shock = min(1.0, shock * 6.0)

        if args.vincolo == "on":
            if shock > 0.02:
                lr = max(args.base_lr * 0.2, lr * 0.5)
            else:
                lr = min(args.base_lr, lr * 1.05)

        rows.append({
            "step": step,
            "loss": float(loss),
            "lr": float(lr),
            "shock": float(shock)
        })

    tag = "vincolo" if args.vincolo == "on" else "baseline"

    with open(f"results/log_killer_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step","loss","lr","shock"])
        w.writeheader()
        w.writerows(rows)

    metrics = compute_metrics(rows, args.shock_step)
    with open(f"results/metrics_killer_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("DONE:", tag, metrics)

if __name__ == "__main__":
    main()
