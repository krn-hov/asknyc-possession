"""Step 3 of the pipeline (see KH/pass_sprint.txt and KH/main.txt): completed passes and sprints.

Idea (from pass_sprint.txt)
---------------------------
While a team keeps the ball, there are moments when no player is controlling it: the ball is
travelling between two teammates (a completed pass), or a player knocks it ahead and chases it
(a sprint / carry). In both, the ball leaves the proximity zone of the team in possession,
so control.py says "nobody", but possession has not changed. This step finds those gaps and
gives them to the team that had the ball on both sides.

Procedure
---------
1. Possession anchors. On every frame that is not a restart (restart.py), we take the team in
   control from control.py. Initial points (initial_points.py) are anchors too, with their
   known team, so the pass that takes a free kick or throw-in is also found. On the second
   run of this step (main.txt), the anchors instead come from the possession script's output
   CSV (--possession), so gaps between the possessions it found are re-examined.
2. Sandwiched gaps. A gap is a maximal run of frames with no anchor. Only gaps with the
   *same* team on both sides are considered (A ... gap ... A). A gap between different teams
   is a change of possession and belongs to the possession script, not here. A gap that
   contains a restart frame is skipped, because the ball went dead in between.
3. Each gap is then classified with the ball's velocity where it is tracked and the
   distance it travelled:

   short   the gap lasts at most SHORT_GAP_S (0.5 s). The ball barely left the player's
           zone: a heavy touch while dribbling, or a first-time lay-off. Accepted as
           possession.

   pass    a completed pass from one teammate to another:
             * the ball travels at least PASS_MIN_DIST (3 m) from where the gap started
               (the farthest point it reaches, so a one-two that comes back still counts)
             * the ball moves at least PASS_MIN_SPEED (4 m/s, a soft pass). This uses the
               median tracked ball speed inside the gap if the ball is seen on at least
               MIN_VISIBLE frames, otherwise the distance covered divided by the gap length
             * the gap lasts at most MAX_PASS_S (5 s, long enough for a long ball)
             * it is not a sprint (below). Because the distance is measured from the
               passer's feet, a one-two that comes back to the same player is still a pass

   sprint  a player carries the ball at pace and keeps it:
             * the same player has the ball at both ends of the gap: the same track slot,
               or the receiving player stands within SAME_PLAYER_DIST of the carrier
               (track slots sometimes swap identity)
             * that player's mean speed over the gap is at least SPRINT_MIN_SPEED (5 m/s,
               about 18 km/h, running)
             * whenever the ball is tracked inside the gap it is within CARRY_MAX_DIST (5 m)
               of the carrier: knocked ahead, never given away
             * the gap lasts at most MAX_SPRINT_S (4 s)

   carry   (an addition to pass_sprint.txt) the same player keeps the ball, at any speed, and
           whenever it is tracked it is within CLOSE_CARRY_DIST (2.5 m) of them. This is
           a slow dribble, or a player shielding the ball just outside the 1.5 m control
           zone. CHALLENGE.md lists "a player keeps the ball" as a sign of control.

   Rejected gaps stay unassigned for the possession script and classifier. The reasons:
     too_long       longer than any pass or sprint we accept
     opponent_near  an opponent was the nearest player and inside the proximity zone for
                    more than OPP_MAX_FRAMES frames. It may have been an interception or a
                    contest, which is not for this step to decide
     restart_inside the ball went dead inside the gap
     unclear        neither a pass nor a sprint matched

Outputs
-------
cache/pass_sprint.npz   label (n,) int8  -1 = not a pass/sprint frame, 0 = H, 1 = A
                           kind  (n,) int8  0 none, 1 short, 2 pass, 3 sprint, 4 carry
out/pass_sprint.csv     one row per sandwiched gap: its measurements and the decision

Run from the KH folder (after initial_points.py, control.py and restart.py):

    python pass_sprint.py                                     # first run
    python pass_sprint.py --possession out/possession.csv  # later runs

A later run (--possession) adds its finds to the cached labels of the earlier run instead of
replacing them. Run without --possession to start fresh.
"""

import argparse
import csv
import os
from collections import Counter

import numpy as np
import pandas as pd

import control as C
import initial_points as IP
import restart as R
import tracking as T

# --------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------
SHORT_GAP_S = 0.5         # s. Gaps this short are kept with the team (ball barely left the zone).
MAX_PASS_S = 5.0          # s. Longest pass flight + reception we accept.
MAX_SPRINT_S = 4.0        # s. Longest carry gap between controlled touches we accept.
PASS_MIN_DIST = 3.0       # m. The ball must travel at least this far for a pass.
PASS_MIN_SPEED = 4.0      # m/s. Slower than a soft pass is not a pass.
MIN_VISIBLE = 3           # frames. Visible frames needed to trust the measured ball speed.
SPRINT_MIN_SPEED = 5.0    # m/s (~18 km/h). Carrier's mean speed for a sprint / carry.
CARRY_MAX_DIST = 5.0      # m. Ball never farther than this from the carrier during a carry.
SAME_PLAYER_DIST = 1.5    # m. Receiver this close to the carrier = the same person (slot swap).
OPP_MAX_FRAMES = 5        # frames. Opponent-nearest-in-zone frames tolerated in a gap.
CLOSE_CARRY_DIST = 2.5    # m. Ball always this close to the same player = slow carry / shielding.

NONE, SHORT, PASS, SPRINT, CARRY = 0, 1, 2, 3, 4
KIND_NAMES = {SHORT: "short", PASS: "pass", SPRINT: "sprint", CARRY: "carry"}

CACHE_FILE = os.path.join(T.CACHE_DIR, "pass_sprint.npz")
OUT_CSV = os.path.join(T.OUT_DIR, "pass_sprint.csv")


# --------------------------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------------------------
def anchors_from_control(ctrl, restart, ip_label):
    """Team per frame from control.py on non-restart frames, plus the initial points.

    Also returns the slot in control (initial points take the nearest player as the taker).
    """
    team = np.where(restart, -1, ctrl["team"]).astype(np.int8)
    slot = np.where(restart, -1, ctrl["slot"]).astype(np.int16)
    is_ip = ip_label >= 0
    team[is_ip] = ip_label[is_ip]
    slot[is_ip] = np.where(slot[is_ip] >= 0, slot[is_ip], ctrl["nearest_slot"][is_ip])
    return team, slot


def anchors_from_csv(path, ctrl, restart):
    """Team per frame from a possession CSV (frame,label with H / A), for the later runs."""
    df = pd.read_csv(path, dtype={"label": str})
    lab = df["label"].fillna("").str.strip().str.upper().to_numpy()
    team = np.full(T.N_FRAMES, -1, dtype=np.int8)
    team[df["frame"].to_numpy()[lab == "H"]] = T.HOME
    team[df["frame"].to_numpy()[lab == "A"]] = T.AWAY
    team[restart] = -1
    # the slot on each side of a gap: whoever controls, else the nearest player of that team
    slot = np.where(ctrl["team"] == team, ctrl["slot"], -1).astype(np.int16)
    near_ok = (slot < 0) & (ctrl["nearest_team"] == team)
    slot[near_ok] = ctrl["nearest_slot"][near_ok]
    return team, slot


# --------------------------------------------------------------------------------------------
# Gap classification
# --------------------------------------------------------------------------------------------
def _visible_pos(ball, t, step, limit=25):
    """Ball position at t, or the nearest visible frame stepping by `step` (up to `limit`)."""
    for k in range(limit):
        u = t + k * step
        if 0 <= u < len(ball) and not np.isnan(ball[u, 0]):
            return ball[u].astype(float)
    return np.array([np.nan, np.nan])


def classify_gap(g0, g1, team_x, giver, receiver, data, ctrl, restart):
    """Measure one sandwiched gap [g0, g1] and decide short / pass / sprint / reject."""
    ball, players = data["ball"], data["players"]
    dur = (g1 - g0 + 1) / T.FPS
    row = {"start": g0, "end": g1, "duration_s": round(dur, 2), "team": T.TEAM_LABEL[team_x],
           "giver_slot": int(giver), "receiver_slot": int(receiver)}

    if restart[g0:g1 + 1].any():
        return row, NONE, "restart_inside"
    if dur > max(MAX_PASS_S, MAX_SPRINT_S):
        return row, NONE, "too_long"

    other = 1 - team_x
    opp = (ctrl["nearest_team"][g0:g1 + 1] == other) & \
          (np.nan_to_num(ctrl["nearest_dist"][g0:g1 + 1], nan=np.inf) < C.PZ_RADIUS)
    row["opp_frames"] = int(opp.sum())
    if opp.sum() > OPP_MAX_FRAMES:
        return row, NONE, "opponent_near"
    if dur <= SHORT_GAP_S:
        return row, SHORT, "short"

    # --- ball measurements -----------------------------------------------------------------
    start = _visible_pos(ball, g0 - 1, -1)
    end = _visible_pos(ball, g1 + 1, +1)
    inside = ball[g0:g1 + 1].astype(float)
    seen = ~np.isnan(inside[:, 0])
    pts = np.vstack([start[None], inside[seen], end[None]])
    pts = pts[~np.isnan(pts[:, 0])]
    reach = float(np.max(np.linalg.norm(pts - pts[0], axis=1))) if len(pts) > 1 else np.nan
    net = float(np.linalg.norm(end - start)) if not np.isnan(start[0] + end[0]) else np.nan
    if seen.sum() >= MIN_VISIBLE:
        speed_est = float(np.nanmedian(data["ball_speed"][g0:g1 + 1][seen]))
    else:
        path = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) if len(pts) > 1 else np.nan
        speed_est = float(path / (dur + 2.0 / T.FPS)) if not np.isnan(path) else np.nan
    row.update({"ball_visible_frames": int(seen.sum()), "ball_reach_m": _r(reach),
                "ball_net_m": _r(net), "ball_speed_est": _r(speed_est)})

    # --- carrier measurements (the giver's track through the gap) --------------------------
    carrier_speed, carrier_ball_max, same_player = np.nan, np.nan, giver == receiver and giver >= 0
    if giver >= 0:
        cv = np.linalg.norm(data["player_vel"][g0:g1 + 1, giver], axis=1)
        carrier_speed = float(np.nanmean(cv)) if np.any(~np.isnan(cv)) else np.nan
        cp = players[g0:g1 + 1, giver].astype(float)
        dist_cb = np.linalg.norm(inside - cp, axis=1)
        if np.any(~np.isnan(dist_cb)):
            carrier_ball_max = float(np.nanmax(dist_cb))
        if not same_player and receiver >= 0 and g1 + 1 < len(players):
            d_sp = np.linalg.norm(players[g1 + 1, receiver].astype(float)
                                  - players[g1 + 1, giver].astype(float))
            same_player = bool(d_sp <= SAME_PLAYER_DIST)
    row.update({"carrier_speed": _r(carrier_speed), "carrier_ball_max_m": _r(carrier_ball_max),
                "same_player": same_player})

    # --- decisions ---------------------------------------------------------------------------
    # Sprint is tested first: if the same runner has the ball at both ends and it never
    # strayed more than CARRY_MAX_DIST from them, it was a carry, however fast the ball moved.
    # Otherwise a fast ball that went far is a pass. reach is measured from where the ball was
    # when the gap opened (at the passer's feet), so a one-two that returns to the passer
    # still qualifies: same_player is True but the ball went far, so it is not a sprint.
    is_sprint = (dur <= MAX_SPRINT_S and same_player
                 and not np.isnan(carrier_speed) and carrier_speed >= SPRINT_MIN_SPEED
                 and (np.isnan(carrier_ball_max) or carrier_ball_max <= CARRY_MAX_DIST))
    is_pass = (dur <= MAX_PASS_S
               and not np.isnan(reach) and reach >= PASS_MIN_DIST
               and not np.isnan(speed_est) and speed_est >= PASS_MIN_SPEED)
    is_carry = (dur <= MAX_SPRINT_S and same_player
                and not np.isnan(carrier_ball_max) and carrier_ball_max <= CLOSE_CARRY_DIST)
    if is_sprint:
        return row, SPRINT, "sprint"
    if is_pass:
        return row, PASS, "pass"
    if is_carry:
        return row, CARRY, "carry"
    return row, NONE, "unclear"


def _r(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else round(v, 2)


def find_pass_sprint(anchor_team, anchor_slot, data, ctrl, restart):
    """Scan every sandwiched gap. Returns (label array, kind array, CSV rows)."""
    n = len(anchor_team)
    half = data["half"]
    label = np.full(n, -1, dtype=np.int8)
    kind = np.zeros(n, dtype=np.int8)
    rows = []

    # gaps = runs of "no anchor" inside graded halves
    no_anchor = (anchor_team < 0) & (half > 0)
    m = np.r_[False, no_anchor, False].astype(int)
    starts = np.flatnonzero(np.diff(m) == 1)
    ends = np.flatnonzero(np.diff(m) == -1) - 1
    for g0, g1 in zip(starts, ends):
        if g0 == 0 or g1 + 1 >= n:
            continue
        before, after = anchor_team[g0 - 1], anchor_team[g1 + 1]
        if before < 0 or before != after:                 # not sandwiched by the same team
            continue
        if half[g0 - 1] != half[g1 + 1]:
            continue
        row, k, decision = classify_gap(g0, g1, int(before), int(anchor_slot[g0 - 1]),
                                        int(anchor_slot[g1 + 1]), data, ctrl, restart)
        row["decision"] = decision
        rows.append(row)
        if k != NONE:
            label[g0:g1 + 1] = before
            kind[g0:g1 + 1] = k
    return label, kind, rows


def main():
    ap = argparse.ArgumentParser(description="Find completed passes and sprints.")
    ap.add_argument("--possession", help="possession CSV (frame,label) to use as anchors "
                                         "instead of control.py (later runs in main.txt)")
    args = ap.parse_args()

    data = T.load_tracking()
    ctrl = C.load_control()
    restart = np.load(R.CACHE_FILE)["restart"]
    ip_label = np.load(IP.OUT_NPZ)["label"]

    if args.possession:
        team, slot = anchors_from_csv(args.possession, ctrl, restart)
        source = args.possession
    else:
        team, slot = anchors_from_control(ctrl, restart, ip_label)
        source = "control.py + initial points"

    label, kind, rows = find_pass_sprint(team, slot, data, ctrl, restart)
    if args.possession and os.path.exists(CACHE_FILE):
        # Later runs ADD to what earlier runs found: the gaps found against the possession
        # file are new, and the passes/sprints already found stay valid.
        prev = np.load(CACHE_FILE)
        keep = (prev["label"] >= 0) & (label < 0)
        label[keep], kind[keep] = prev["label"][keep], prev["kind"][keep]
    np.savez_compressed(CACHE_FILE, label=label, kind=kind)
    fields = ["start", "end", "duration_s", "team", "giver_slot", "receiver_slot", "opp_frames",
              "ball_visible_frames", "ball_reach_m", "ball_net_m", "ball_speed_est",
              "carrier_speed", "carrier_ball_max_m", "same_player", "decision"]
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary -------------------------------------------------------------------------
    g = data["half"] > 0
    print("anchors from: {}".format(source))
    print("sandwiched gaps: {}".format(len(rows)))
    for decision, k in Counter(r["decision"] for r in rows).most_common():
        frames = sum(r["end"] - r["start"] + 1 for r in rows if r["decision"] == decision)
        print("  {:<15} {:>5} gaps  {:>6} frames".format(decision, k, frames))
    print("frames given to a team (all runs so far): {} ({:.1f}% of graded)   H {}  A {}".format(
        int((label >= 0).sum()), 100.0 * (label >= 0)[g].mean(),
        int((label == T.HOME).sum()), int((label == T.AWAY).sum())))
    print("\nwrote {}\n      {}".format(T.rel(CACHE_FILE), T.rel(OUT_CSV)))


if __name__ == "__main__":
    main()
