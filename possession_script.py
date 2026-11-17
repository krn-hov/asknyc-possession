"""Step 4 of the pipeline (see KH/possession_script.txt and KH/main.txt): the possession script.

It walks through the match frame by frame and decides f(t), the possession at video frame t,
from everything the earlier steps produced:

    initial_points.py   frames with a known restarting team
    restart.py          frames where the ball is out of play
    pass_sprint.py      gaps that are a completed pass / sprint / carry by one team
    control.py          is a player controlling the ball, which team, who is nearest
    classifier.py       a guess for frames nothing else can decide (later runs only)

The rules (possession_script.txt), applied in this order for each frame t
--------------------------------------------------------------------------
    start with the first initial point frame (of each half)
    if t is a pass_sprint frame             -> f(t) = its pass_sprint team
    elif t is a restart frame                -> f(t) = 0 (neither team)
    elif the ball is visible and t is not an initial point:
        if f(t-1) is a team:
            if control(t)                    -> f(t) = f(t-1)           (see note 1)
            else                             -> f(t) = 0
        if f(t-1) is 0 (or unassigned, note 3):
            if the closest player is within PROXIMITY_M and control(t)
                                             -> f(t) = team of the closest player (note 2)
            else                             -> f(t) = 0
    elif t is an initial point               -> f(t) = its known team
    elif the ball dropout is shorter than SHORT_BLANK_S -> f(t) = f(t-1)  (note 6)
    elif this is the first pass (no classifier) -> f(t) stays unassigned
    else                                     -> f(t) = classifier(t)

Notes 5 and 6 were added after comparing with the team's earlier analysis/ pipeline: that
pipeline's possessions were far more stable (few under 2 s), and these two of its ideas fix
this script's main weakness, possessions split by a brief touch or a brief tracking dropout.

Interpretation notes (decisions made while implementing; change them here if you disagree)
------------------------------------------------------------------------------------------
1. "if control(t) then f(t) = f(t-1)": if control.py says the ball is controlled by the
   *other* team, f(t) becomes that team. Otherwise a direct tackle (H in control straight
   into A in control, with no uncontrolled frame between) could never change possession.
   CHALLENGE.md defines a change of possession as exactly this: the opponent gaining
   controlled possession.
2. "team of the closest player" is used only when control.py's controlling team *is* the
   closest player's team. If someone else controls the ball, the closest player is not the
   one with it, so f(t) = 0.
3. On the first pass, frames with the ball missing are unassigned. When the ball reappears,
   an unassigned f(t-1) is treated like 0: a team must be seen controlling the ball before
   it is assigned.
4. Frames of a half before its first initial point (the kick-off) are not walked. They are
   0 where restart.py says the ball is dead, and unassigned otherwise.
5. Sticky possession changes (from analysis/'s "sustained nearest team before a flip").
   Possession only passes to the other team once that team has controlled the ball for
   SWITCH_S (0.5 s) in total, all within PENDING_MAX_FRAMES (3 s). A touch, deflection or
   one-frame control blip is shorter, and CHALLENGE.md says none of these changes possession.
   Until then the frame keeps what it would otherwise have had: the old team, or 0. Once
   confirmed, the switch is back-dated to the new team's first frame of control, because a
   possession starts at the controlled gain, not 0.5 s later. The comparison is with the last
   team that had possession, even across a stretch of 0 (most changes go team -> 0 -> other
   team). After a restart there is no last team, so the first control counts immediately.
6. Short ball dropouts (from analysis/'s "short blank -> hold last team"). If the ball is
   missing for less than SHORT_BLANK_S (2 s), f(t) = f(t-1) instead of unassigned or the
   classifier. A two-second dropout is the tracker losing the ball, not a change of
   possession, and a classifier guess there could split the possession in two.

Labels in the raw output (out/<name>.csv)
--------------------------------------------
    H / A   f(t) is a team
    D       f(t) = 0 because the ball is out of play (a restart frame)
    0       f(t) = 0 with the ball in play (nobody controls it)
    X       unassigned (first pass), or outside the graded halves
The classifier trains on this file: it reads H and A (and "0"/"D" with --include-zero).

The submission (out/<name>_submission.csv)
---------------------------------------------
The official format has only H / A / D / X, so 0 and unassigned frames have to be resolved:
  * restart frames                      -> D
  * a run of live-0 / unassigned frames -> the team that had the ball just before it, because
    a loose or contested ball stays with the last team in possession (CHALLENGE.md). If the
    frame just before the run is D (play just restarted) or the half's start, the run takes
    the *next* team instead: whoever controls the ball first after a restart most likely
    took it. If neither side has a team, X.
  * frames outside the graded halves    -> X
With --keep-x, the live-0 / unassigned frames are written as X instead and left to the
scorer (whose spellF1 carries the previous label forward).

Usage (run from the KH folder, after initial_points, control, restart, pass_sprint)
-------------------------------------------------------------------------------------
    # first pass, no classifier                     -> out/possession_pass1.csv
    python possession_script.py
    # with a trained classifier                     -> out/possession.csv
    python possession_script.py --classifier tree
    # same, one majority label per stretch of missing ball (stops H/A flicker)
    python possession_script.py --classifier tree --per-stretch

main.py runs all of this in the order main.txt gives, and checks the final file's format.
"""

import argparse
import os

import numpy as np
import pandas as pd

import control as C
import initial_points as IP
import pass_sprint as PS
import restart as R
import tracking as T

# --------------------------------------------------------------------------------------------
# Parameters and state codes
# --------------------------------------------------------------------------------------------
PROXIMITY_M = C.PZ_RADIUS      # "proximity threshold": the same 1.5 m zone control.py uses
SWITCH_S = 0.5                  # s of control a new team needs before possession switches (note 5)
SWITCH_FRAMES = int(round(SWITCH_S * T.FPS))
PENDING_MAX_FRAMES = int(3.0 * T.FPS)   # the switch's control must come within 3 s (note 5)
SHORT_BLANK_S = 2.0             # s. Shorter ball dropouts keep the last team (note 6)

HOME, AWAY = T.HOME, T.AWAY    # 0, 1
ZERO = 2                        # f(t) = 0, neither team
UNASSIGNED = -1


# --------------------------------------------------------------------------------------------
# The frame-by-frame walk
# --------------------------------------------------------------------------------------------
def classifier_predictions(need, kind, per_stretch=False):
    """Classifier output for the frames in `need` (bool mask), as state codes.

    per_stretch=True: every contiguous stretch of `need` frames gets ONE label, the majority
    of the classifier's per-frame predictions inside it. A frame-by-frame classifier
    flickers between H and A within one stretch of missing ball, and each flicker splits a
    possession in two, which the ranking metric (spellF1) punishes. This option is an
    addition to possession_script.txt.
    """
    import classifier as K
    pred = np.full(len(need), UNASSIGNED, dtype=np.int8)
    frames = np.flatnonzero(need)
    if not len(frames):
        return pred
    lab = K.classify(frames, kind)
    pred[frames] = np.select([lab == "H", lab == "A", lab == "0"], [HOME, AWAY, ZERO], UNASSIGNED)
    if per_stretch:
        m = np.r_[False, need, False].astype(int)
        starts = np.flatnonzero(np.diff(m) == 1)
        ends = np.flatnonzero(np.diff(m) == -1)
        for s, e in zip(starts, ends):
            vals, counts = np.unique(pred[s:e], return_counts=True)
            pred[s:e] = vals[np.argmax(counts)]
    return pred


def short_blanks(ball_visible, half):
    """True on ball-missing frames whose missing stretch lasts less than SHORT_BLANK_S."""
    missing = ~ball_visible & (half > 0)
    m = np.r_[False, missing, False].astype(int)
    starts = np.flatnonzero(np.diff(m) == 1)
    ends = np.flatnonzero(np.diff(m) == -1)
    out = np.zeros(len(missing), dtype=bool)
    for s, e in zip(starts, ends):
        if e - s < SHORT_BLANK_S * T.FPS:
            out[s:e] = True
    return out


def walk(ip_label, restart, ps_label, ctrl, half, classifier_pred=None, short_blank=None):
    """Apply possession_script.txt (+ notes 1, 5, 6) to every frame. Returns the state array f."""
    n = len(ip_label)
    f = np.full(n, UNASSIGNED, dtype=np.int8)
    f[restart] = ZERO                                   # note 4: frames before the walk starts
    if short_blank is None:
        short_blank = np.zeros(n, dtype=bool)

    # plain Python lists: much faster than numpy scalars inside a 160k-step loop
    ip_l, rs_l, ps_l = ip_label.tolist(), restart.tolist(), ps_label.tolist()
    vis_l, ctrl_l, cteam_l = ctrl["ball_visible"].tolist(), ctrl["control"].tolist(), ctrl["team"].tolist()
    nteam_l = ctrl["nearest_team"].tolist()
    ndist_l = np.nan_to_num(ctrl["nearest_dist"], nan=np.inf).tolist()
    cls_l = classifier_pred.tolist() if classifier_pred is not None else None
    sb_l = short_blank.tolist()

    for h, (a, b) in T.GRADED_HALVES.items():
        ips = np.flatnonzero(ip_label[a:b + 1] >= 0)
        start = a + int(ips[0]) if len(ips) else a        # first initial point of the half

        # note 5: sticky possession changes
        last_team = None        # the team that last had possession (None after a stoppage)
        pending = None          # [team, first frame, frames of control seen] while confirming

        def assign(t, team, hold):
            """Give frame t to `team` with control, applying the switch confirmation."""
            nonlocal last_team, pending
            if last_team is None or team == last_team:
                f[t] = team
                last_team, pending = team, None
                return
            if pending is None or pending[0] != team or t - pending[1] > PENDING_MAX_FRAMES:
                pending = [team, t, 0]
            pending[2] += 1
            if pending[2] >= SWITCH_FRAMES:                 # confirmed: back-date the switch
                first = pending[1]
                f[first:t + 1] = np.where(np.asarray(rs_l[first:t + 1]), f[first:t + 1], team)
                last_team, pending = team, None
            else:
                f[t] = hold                                 # not yet: keep what we had

        for t in range(start, b + 1):
            prev = f[t - 1] if t > start else UNASSIGNED
            if ps_l[t] >= 0:                                # pass_sprint frame
                f[t] = ps_l[t]
                last_team, pending = ps_l[t], None
            elif rs_l[t]:                                   # restart frame
                f[t] = ZERO
                last_team, pending = None, None
            elif vis_l[t] and ip_l[t] < 0:                  # ball visible, not an initial point
                if prev in (HOME, AWAY):
                    if ctrl_l[t]:
                        assign(t, cteam_l[t], hold=prev)                  # note 1 + 5
                    else:
                        f[t] = ZERO
                else:                                                 # prev is 0 or unassigned
                    close = ndist_l[t] < PROXIMITY_M
                    if close and ctrl_l[t] and cteam_l[t] == nteam_l[t]:   # note 2
                        assign(t, nteam_l[t], hold=ZERO)                  # note 5
                    else:
                        f[t] = ZERO
            elif ip_l[t] >= 0:                              # initial point
                f[t] = ip_l[t]
                last_team, pending = ip_l[t], None
            elif sb_l[t]:                                   # note 6: short dropout
                f[t] = prev if prev in (HOME, AWAY) else ZERO
            elif cls_l is None:                             # first pass: keep unassigned
                f[t] = UNASSIGNED
            else:                                           # classifier
                f[t] = cls_l[t]
                if cls_l[t] in (HOME, AWAY):
                    last_team, pending = cls_l[t], None
    return f


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------
def raw_labels(f, restart, half):
    """H / A / D / 0 / X per frame (see module doc)."""
    lab = np.full(len(f), "X", dtype=object)
    lab[f == HOME] = "H"
    lab[f == AWAY] = "A"
    lab[(f == ZERO) & restart] = "D"
    lab[(f == ZERO) & ~restart] = "0"
    lab[half == 0] = "X"
    return lab


def submission_labels(raw, half, keep_x=False):
    """Resolve 0 / unassigned frames into the official H / A / D / X (see module doc)."""
    sub = raw.copy()
    unresolved = (raw == "0") | (raw == "X")
    sub[unresolved] = "X"
    if keep_x:
        sub[half == 0] = "X"
        return sub
    for h, (a, b) in T.GRADED_HALVES.items():
        seg = raw[a:b + 1]
        un = (seg == "0") | (seg == "X")
        m = np.r_[False, un, False].astype(int)
        starts = np.flatnonzero(np.diff(m) == 1)
        ends = np.flatnonzero(np.diff(m) == -1) - 1
        for s, e in zip(starts, ends):
            before = seg[s - 1] if s > 0 else "D"
            after = seg[e + 1] if e + 1 < len(seg) else "D"
            if before in ("H", "A"):
                fill = before                       # loose ball stays with the last team
            elif after in ("H", "A"):
                fill = after                        # just after a restart: next team in control
            else:
                fill = "X"
            sub[a + s:a + e + 1] = fill
    sub[half == 0] = "X"
    return sub


def _possessions(labels, half):
    """Number and mean length (s) of possessions = maximal runs of H or of A within a half."""
    lengths = []
    for h, (a, b) in T.GRADED_HALVES.items():
        seg = labels[a:b + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(seg)]
        lengths += [(e - s) / T.FPS for s, e in zip(starts, ends) if seg[s] in ("H", "A")]
    return len(lengths), (np.mean(lengths) if lengths else 0.0)


def main():
    ap = argparse.ArgumentParser(description="Frame-by-frame possession (possession_script.txt).")
    ap.add_argument("--classifier", choices=["none", "tree", "knn"], default="none",
                    help="none = first pass (ball-missing frames stay unassigned)")
    ap.add_argument("--name", help="output base name (default: possession_pass1 without a "
                                   "classifier, possession with one)")
    ap.add_argument("--per-stretch", action="store_true",
                    help="one majority classifier label per stretch of missing ball "
                         "(prevents H/A flicker; an addition to possession_script.txt)")
    ap.add_argument("--keep-x", action="store_true",
                    help="write unresolved live frames as X instead of filling them")
    args = ap.parse_args()
    name = args.name or ("possession_pass1" if args.classifier == "none" else "possession")

    data = T.load_tracking()
    half = data["half"]
    ctrl = C.load_control()
    ip_label = np.load(IP.OUT_NPZ)["label"]
    restart = np.load(R.CACHE_FILE)["restart"]
    ps_label = np.load(PS.CACHE_FILE)["label"]

    short_blank = short_blanks(ctrl["ball_visible"], half)
    cls_pred = None
    if args.classifier != "none":
        need = ((half > 0) & ~ctrl["ball_visible"] & (ip_label < 0) & (ps_label < 0)
                & ~restart & ~short_blank)
        cls_pred = classifier_predictions(need, args.classifier, args.per_stretch)

    f = walk(ip_label, restart, ps_label, ctrl, half, cls_pred, short_blank)
    raw = raw_labels(f, restart, half)
    sub = submission_labels(raw, half, keep_x=args.keep_x)

    frames = np.arange(len(f))
    raw_path = os.path.join(T.OUT_DIR, name + ".csv")
    sub_path = os.path.join(T.OUT_DIR, name + "_submission.csv")
    pd.DataFrame({"frame": frames, "label": raw}).to_csv(raw_path, index=False)
    pd.DataFrame({"frame": frames, "label": sub}).to_csv(sub_path, index=False)

    # ---- summary -------------------------------------------------------------------------
    g = half > 0
    print("classifier: {}".format(args.classifier))
    print("raw labels on graded frames ({}):".format(int(g.sum())))
    for k in ("H", "A", "D", "0", "X"):
        print("  {}  {:>7}  {:5.1f}%".format(k, int((raw[g] == k).sum()), 100.0 * (raw[g] == k).mean()))
    print("submission labels on graded frames:")
    for k in ("H", "A", "D", "X"):
        print("  {}  {:>7}  {:5.1f}%".format(k, int((sub[g] == k).sum()), 100.0 * (sub[g] == k).mean()))
    n_pos, mean_len = _possessions(sub, half)
    print("possessions in the submission: {}   mean {:.1f} s".format(n_pos, mean_len))
    print("\nwrote {}\n      {}".format(T.rel(raw_path), T.rel(sub_path)))


if __name__ == "__main__":
    main()
