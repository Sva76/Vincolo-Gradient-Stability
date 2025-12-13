import csv
import matplotlib.pyplot as plt
import os

def read_csv(path):
    xs, loss = [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["step"]))
            loss.append(float(row["loss"]))
    return xs, loss

def main():
    baseline_csv = "results/log_killer_baseline.csv"
    vincolo_csv = "results/log_killer_vincolo.csv"
    out = "results/loss_vs_step_killer.png"

    xb, lb = read_csv(baseline_csv)
    xv, lv = read_csv(vincolo_csv)

    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(xb, lb, label="baseline")
    plt.plot(xv, lv, label="vincolo")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)

    print("Wrote:", out)

if __name__ == "__main__":
    main()
