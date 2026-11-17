"""Run the whole possession pipeline (KH/main.txt) and write the solution CSV.

    python main.py

main.txt, step by step, and what runs for each line:

    first run the initial_points script         -> initial_points.py
                                                   (+ control.py: the control function that
                                                    restart and pass_sprint use, computed once)
    then run the restart_points script          -> restart.py
    then run the pass_sprint script             -> pass_sprint.py
    then run the possession script with no
      classifier (if training needed)           -> possession_script.py         -> out/possession_pass1.csv
    then train the classifier (if needed)       -> classifier.py --labels out/possession_pass1.csv
    then run the possession script with the
      classifier                                -> possession_script.py --classifier ...  -> out/final.csv

main.txt originally ended with a second pass_sprint run and a second possession run. They
were removed: the second pass_sprint run found only ~30 new passes (the first run had already
found ~450), and the final submission came out byte-identical with or without them. A pass
between two possessions of the same team already gets that team when the loose frames are
filled. pass_sprint.py still accepts --possession if you want to try it by hand.

Finally the submission file out/final_submission.csv is format-checked (the same checks as the
organisers' validate_submission.py) and copied to solution/<team>.csv.

Each step runs as its own Python process, exactly as if typed by hand, so every step reads
the previous steps' files from cache/ and out/ and nothing is shared in memory.

Options
-------
    --model tree|knn     classifier to use (default tree)
    --no-per-stretch     label missing-ball stretches frame by frame (classifier as written
                         in possession_script.txt) instead of one majority label per stretch
    --team NAME          solution file name (default ASKNYC -> solution/ASKNYC.csv)
    --fresh              delete cache/ and out/ first, so even the tracking is re-parsed
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

KH_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(KH_DIR, "solution")
N_FRAMES = 199319


def run(step, args):
    """Run one pipeline script in its own process and stop everything if it fails."""
    print("\n" + "=" * 92)
    print("[{}] python {} {}".format(step, args[0], " ".join(args[1:])).rstrip())
    print("=" * 92, flush=True)
    t0 = time.time()
    subprocess.run([sys.executable] + args, cwd=KH_DIR, check=True)
    print("({:.1f} s)".format(time.time() - t0), flush=True)


def validate(path):
    """The organisers' format checks (SUBMISSION.md). Raises ValueError on any problem."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if [c.strip().lower() for c in df.columns] != ["frame", "label"]:
        raise ValueError("header must be exactly frame,label")
    if len(df) != N_FRAMES:
        raise ValueError("expected {} rows, found {}".format(N_FRAMES, len(df)))
    if not np.array_equal(df.iloc[:, 0].astype(int).to_numpy(), np.arange(N_FRAMES)):
        raise ValueError("frame column must be 0..{} in order".format(N_FRAMES - 1))
    labels = df.iloc[:, 1].str.strip().str.upper()
    bad = ~labels.isin(["H", "A", "D", "X"])
    if bad.any():
        raise ValueError("{} invalid labels (blank or not H/A/D/X)".format(int(bad.sum())))
    return labels.value_counts().to_dict()


def main():
    ap = argparse.ArgumentParser(description="Run the KH possession pipeline end to end.")
    ap.add_argument("--model", choices=["tree", "knn"], default="tree")
    ap.add_argument("--no-per-stretch", action="store_true")
    ap.add_argument("--team", default="ASKNYC")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.fresh:
        for d in ("cache", "out"):
            shutil.rmtree(os.path.join(KH_DIR, d), ignore_errors=True)

    cls = ["--classifier", args.model] + ([] if args.no_per_stretch else ["--per-stretch"])
    t0 = time.time()

    run("1/7 initial points", ["initial_points.py"])
    run("2/7 control function", ["control.py"])
    run("3/7 restart points", ["restart.py"])
    run("4/7 pass / sprint", ["pass_sprint.py"])
    run("5/7 possession, no classifier", ["possession_script.py", "--name", "possession_pass1"])
    run("6/7 train classifier", ["classifier.py", "--labels", "out/possession_pass1.csv",
                                 "--model", args.model, "--rebuild-features"])
    run("7/7 possession with classifier", ["possession_script.py", "--name", "final"] + cls)

    final = os.path.join(KH_DIR, "out", "final_submission.csv")
    counts = validate(final)
    os.makedirs(SOLUTION_DIR, exist_ok=True)
    target = os.path.join(SOLUTION_DIR, args.team + ".csv")
    shutil.copyfile(final, target)

    print("\n" + "=" * 92)
    print("DONE in {:.0f} s".format(time.time() - t0))
    print("solution: {}   (format OK: {} rows, {})".format(
        os.path.relpath(target, KH_DIR), N_FRAMES,
        ", ".join("{} {}".format(k, counts.get(k, 0)) for k in "HADX")))


if __name__ == "__main__":
    main()
