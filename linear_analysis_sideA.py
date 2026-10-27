"""
Plot piezo position against groove width from a groove_piezo_*.csv file.

For every groove number in the CSV:
    x = average of width_um over all rows with that groove number
    y = the piezo position of that groove (the same for all its rows)
    horizontal error bar = ± N_SD standard deviations of width_um for that groove

A straight line piezo_pos = gradient * width_um + intercept is fitted through the
groove averages, and the gradient and intercept are printed and shown on the plot.
"""

import csv
import glob
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt 
import numpy as np

from constants import general as gen
from constants import cutter_positions as cutter


class LinearAnalysis:
    def __init__(self, logger, log_folder, csv_path):
        self.logger = logger
        self.log_folder = log_folder
        self.csv_path = csv_path

    def summarise_grooves(self, path):
        """One row per groove number: groove, piezo_pos, n, mean, sd."""
        widths = defaultdict(list)
        positions = defaultdict(list)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                g = int(row["groove"])
                widths[g].append(float(row["width_um"]))
                positions[g].append(float(row["piezo_pos"]))

        rows = []
        for g in sorted(widths):
            if g in gen.EXCLUDE_GROOVES or len(widths[g]) < gen.MIN_N:
                continue
            pos_counts = Counter(positions[g])
            pos = pos_counts.most_common(1)[0][0]
            if len(pos_counts) > 1:
                print(f"WARNING: groove {g} has more than one piezo position {dict(pos_counts)} - using {pos}")
            w = np.asarray(widths[g])
            rows.append(dict(groove=g, piezo_pos=pos, n=len(w), mean=w.mean(),
                            sd=w.std(ddof=1) if len(w) > 1 else 0.0))
        return rows


    def linear_fit(self, rows):
        """Least-squares line piezo_pos = gradient * width + intercept, with standard errors."""
        x = np.array([r["mean"] for r in rows])
        y = np.array([r["piezo_pos"] for r in rows])
        if len(x) < 2:
            raise SystemExit("Need at least 2 grooves for a linear fit.")
        if len(x) >= 3:
            (gradient, intercept), cov = np.polyfit(x, y, 1, cov=True)
            gradient_err, intercept_err = np.sqrt(np.diag(cov))
        else:
            gradient, intercept = np.polyfit(x, y, 1)
            gradient_err = intercept_err = float("nan")
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        return dict(gradient=gradient, intercept=intercept, gradient_err=gradient_err,
                    intercept_err=intercept_err, r2=r2, n=len(x))


    def plot(self, rows, csv_path, fit):
        x = np.array([r["mean"] for r in rows])
        y = np.array([r["piezo_pos"] for r in rows])
        xerr = gen.N_SD * np.array([r["sd"] for r in rows])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(x, y, xerr=xerr, fmt="o", markersize=6, capsize=5, elinewidth=1.5,
                    label=f"average width per groove (±{gen.N_SD} SD)")

        seen = defaultdict(int)                       # stagger labels of grooves at the same position
        for r in rows:
            k = seen[r["piezo_pos"]]
            seen[r["piezo_pos"]] += 1
            ax.annotate(f"g{r['groove']}", (r["mean"], r["piezo_pos"]),
                        textcoords="offset points", xytext=(6, 6 + 11 * k), fontsize=8, color="0.35")

        if fit is not None:
            xs = np.linspace((x - xerr).min(), (x + xerr).max(), 100)
            ax.plot(xs, fit["gradient"] * xs + fit["intercept"], "--", color="0.4", label="linear fit")
            ax.text(0.98, 0.03,
                    f"piezo = gradient × width + intercept\n"
                    f"gradient  = {fit['gradient']:.5f} ± {fit['gradient_err']:.5f} per µm\n"
                    f"intercept = {fit['intercept']:.4f} ± {fit['intercept_err']:.4f}\n"
                    f"R² = {fit['r2']:.4f}   ({fit['n']} grooves)",
                    transform=ax.transAxes, ha="right", va="bottom", multialignment="left",
                    fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))

        ax.set_xlabel("Average groove width (µm)")
        ax.set_ylabel("Piezo position")
        ax.set_title(os.path.basename(csv_path))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()

        csv_path = Path(csv_path)
        png = csv_path.with_name(f"{csv_path.stem}_plot.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"\nSaved plot to {png}")

    def determine_if_headscrew_needs_adjustment(self, fit):
        mmum = fit['gradient']
        y_int = fit['intercept']
        max_um_possible = (cutter.PIEZO_UPPER_LIMIT - y_int)/mmum
        min_um_possible = (cutter.PIEZO_LOWER_LIMIT - y_int)/mmum

        if max_um_possible < cutter.MAX_SAFE_GW:
            #head is too high up needs to be shifted downwards
            delta_um = cutter.MAX_SAFE_GW - max_um_possible
            delta_mm = (delta_um * mmum)

            piezo_pos_at_min_um = (mmum*cutter.PIEZO_LOWER_LIMIT) + y_int
            safe_shifting_distance = piezo_pos_at_min_um - cutter.PIEZO_LOWER_LIMIT

            if safe_shifting_distance < delta_mm:
                #not enough room in piezo range to capture desired range
                #recommend to user to replace the spring
                #null value corresponds to replacing the spring
                safe_offset_required = None
                return [True, safe_offset_required]

            else:
                #safe to shift piezo response in the negative x in mm/um graph aim for the center of the range
                safe_offset_required = delta_mm + (0.5*safe_shifting_distance)
                
                #nagative value corresponds to shifting the head down
                safe_offset_required = safe_offset_required*(-1)
                return [True, safe_offset_required]


        elif min_um_possible > cutter.MAX_SAFE_GW:
            #head is too low down needs to be shifted upwards
            delta_um = max_um_possible - cutter.MAX_SAFE_GW
            delta_mm = (delta_um * mmum)

            piezo_pos_at_max_um = (mmum*cutter.MAX_SAFE_GW) + y_int
            safe_shifting_distance = cutter.PIEZO_UPPER_LIMIT - piezo_pos_at_max_um

            if safe_shifting_distance < delta_mm:
                #not enough space to shift graph without reducing the piezo range at the other end
                #recommend to the user to replace spring
                safe_offset_required = None
                return [True, safe_offset_required]
            else:
                #shift piezo response in mmum in the positive x direction
                safe_offset_required = delta_mm + (0.5*safe_shifting_distance)

                #positive value corresponds to shifting the head up
                return [True, safe_offset_required]


        else:
            #False means no adjustments are required
            return [False, None]


    def run_analysis(self):
        path = self.csv_path
        if path is None:
            files = sorted(glob.glob("groove_piezo_*.csv"), key=os.path.getmtime)
            if not files:
                raise SystemExit("No groove_piezo_*.csv found here - set CSV_FILE at the top of the file.")
            path = files[-1]

        rows = self.summarise_grooves(path)
        if not rows:
            raise SystemExit("No grooves with enough rows to plot.")

        print(f"{'groove':>6} {'piezo':>6} {'n':>4} {'avg um':>8} {'sd um':>6} {f'±{gen.N_SD}sd um':>9}")
        for r in rows:
            print(f"{r['groove']:>6} {r['piezo_pos']:>6.2f} {r['n']:>4} {r['mean']:>8.2f} "
                f"{r['sd']:>6.2f} {gen.N_SD * r['sd']:>9.2f}")

        fit = None
        if gen.FIT:
            fit = self.linear_fit(rows)
            print(f"\nLinear fit over {fit['n']} grooves:  piezo_pos = gradient * mm_um + intercept")
            print(f"  gradient  = {fit['gradient']:.6f} ± {fit['gradient_err']:.6f}  (mm per µm)")
            print(f"  intercept = {fit['intercept']:.4f} ± {fit['intercept_err']:.4f}  (mm at 0 µm)")
            print(f"  R²        = {fit['r2']:.4f}")

        self.plot(rows, path, fit)
        needs_adjustment = self.determine_if_headscrew_needs_adjustment(fit)
        return fit, needs_adjustment