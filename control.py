"""The control function (see KH/control.txt): is the ball under a player's control at frame t?

control.txt asks three questions:
  1. Is the ball relatively steady and moving with the player?
  2. Does the ball's velocity drop sharply, suggesting a player has taken control of it?
  3. If more than one team is in the proximity zone, is the closest player successfully
     guarding the ball?

Following the team's decision, this module re-implements the individual-player-possession
algorithm of Vidal-Codina et al. (2022), "Automatic event detection in football using tracking
data", in the form databallpy 0.8.1 uses it (databallpy/features/player_possession.py), with
the same default parameters. The control.txt checks are then layered on top. Nothing is
imported from databallpy. The code works directly on our 46-slot arrays from tracking.py.

Step A: proximity zone (PZ). As in databallpy, for every frame the player nearest the ball is
    a candidate if they are within PZ_RADIUS (1.5 m). The ball missing means no candidate.

Step B: Vidal-Codina gain validation (databallpy defaults). Consecutive frames with the same
    candidate form a *PZ run*. A run counts as a real gain of possession only if, while the
    ball is in that player's zone, the ball changes speed or direction:
      * speed: |ball speed[t] - ball speed[t-1]| > BV_THRESHOLD (5 m/s) for some t in the run
      * angle: the direction the ball entered the zone (p[start] - p[start-1]) and the
        direction it left (p[end+1] - p[end]) differ by more than BA_THRESHOLD (10 degrees)
      * the run lasts at least MIN_FRAMES_PZ frames (0, i.e. any length)
    A ball that rolls straight past a player at constant speed fails both and is not a gain.
    After a valid gain the player keeps possession through later PZ runs of their own. A
    touch by another player that is *not* a valid gain does not take possession away,
    matching CHALLENGE.md ("a simple touch without control", "a deflection").
    Departure from databallpy: an extension across a gap is only allowed if the gap is at
    most MAX_EXTEND_FRAMES (2 s), so possession is never carried through a long stoppage or
    a long tracking dropout. databallpy has no such limit, and also ends possession one frame
    before the last PZ frame; we include that frame.

Step C: the control.txt checks.
    1. moving_with: the ball is in the PZ and its velocity matches the nearest player's
       velocity, |v_ball - v_player| <= REL_SPEED_MAX (3 m/s), as a median over a centred
       STEADY_WINDOW (7 frames, 0.28 s). This is a dribble or a ball at the player's feet.
    2. reception: the ball is in the PZ and, compared with the fastest it moved in the last
       RECEPTION_LOOKBACK frames (0.5 s), its speed has dropped by at least DROP_ABS (4 m/s)
       and to at most DROP_FRAC (50%) of that peak. This is a first touch killing a pass.
       Once a player has received the ball, the rest of that PZ run counts as controlled.
    3. guarded: if a player of the *other* team is also within PZ_RADIUS of the ball
       (contested), the controller must be closer than the nearest opponent by at least
       GUARD_MARGIN (0.5 m), and must have held that margin for the last GUARD_FRAMES frames
       (0.4 s). A player shielding the ball keeps it; a 50-50 does not count as control.

Final rule, for each frame t with the ball visible:
    controller(t) = the Vidal-Codina possessor if there is one, otherwise the PZ candidate if
                    moving_with or reception says they have the ball, otherwise nobody
    control(t)    = controller(t) exists AND guarded(t)
Where the ball is missing, control(t) is False and ball_visible(t) is False. Callers (the
possession script) treat those frames separately, as possession_script.txt specifies.

Outputs
-------
compute_control(data) returns a dict of per-frame arrays. Running this file saves them to
cache/control.npz. The keys:
    control       bool   the answer to control.txt
    team          int8   team in control (0 home, 1 away, -1 nobody)
    slot          int16  track slot in control (0-based, -1 nobody)
    nearest_slot  int16  nearest player to the ball (-1 if ball missing)
    nearest_team  int8
    nearest_dist  float  metres to the nearest player (NaN if ball missing)
    dist_home, dist_away  metres to the nearest home / away player
    in_pz, vc_slot, moving_with, reception, contested, guarded, ball_visible
                         every intermediate step, so each decision can be inspected

Other steps can call control(t) or team_in_control(t) for single frames, or load_control()
for the arrays.

Run from the KH folder:

    python control.py
"""

import os

import numpy as np
import pandas as pd

import tracking as T

# --------------------------------------------------------------------------------------------
# Parameters. The first four are databallpy's defaults for Vidal-Codina et al. (2022).
# --------------------------------------------------------------------------------------------
PZ_RADIUS = 1.5            # m. Possession / proximity zone radius.
BV_THRESHOLD = 5.0         # m/s. Ball speed change that validates a gain.
BA_THRESHOLD = 10.0        # degrees. Ball direction change that validates a gain.
MIN_FRAMES_PZ = 0          # frames. Minimum PZ run length for a gain.

MAX_EXTEND_FRAMES = 50     # frames (2 s). Longest gap a possession is carried across.
REL_SPEED_MAX = 3.0        # m/s. Ball vs player velocity difference for "moving with".
STEADY_WINDOW = 7          # frames. Centred window for the moving-with median.
RECEPTION_LOOKBACK = 12    # frames (~0.5 s). Window to find the pre-reception peak speed.
DROP_ABS = 4.0             # m/s. Minimum speed drop for a reception.
DROP_FRAC = 0.5            # the speed must fall to this share of the peak or below.
GUARD_MARGIN = 0.5         # m. How much closer than the nearest opponent the controller must be.
GUARD_FRAMES = 10          # frames (0.4 s). How long the margin must have held.

CACHE_FILE = os.path.join(T.CACHE_DIR, "control.npz")


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _runs(values, ignore=-1):
    """(start, end_inclusive, value) for each maximal run of equal values, skipping `ignore`."""
    values = np.asarray(values)
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change - 1, len(values) - 1]
    keep = values[starts] != ignore
    return starts[keep], ends[keep], values[starts[keep]]


def _angle_deg(u, v):
    """Smallest angle in degrees between row vectors u and v (NaN if either is missing/zero)."""
    dot = np.sum(u * v, axis=1)
    norm = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _rolling(series, window, how, center=False, min_periods=1):
    """Rolling statistic over a 1-D array, NaN-aware (pandas does the bookkeeping)."""
    roll = pd.Series(series).rolling(window, center=center, min_periods=min_periods)
    return getattr(roll, how)().to_numpy()


# --------------------------------------------------------------------------------------------
# Step A: proximity zone
# --------------------------------------------------------------------------------------------
def proximity(data):
    """Distances from the ball to every player, and the nearest player per frame."""
    ball, players, team = data["ball"], data["players"], data["team"]
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)            # (n, 46), NaN-safe
    filled = np.where(np.isnan(dist), np.inf, dist)
    nearest_slot = np.argmin(filled, axis=1).astype(np.int16)
    nearest_dist = filled[np.arange(len(dist)), nearest_slot]
    none = ~np.isfinite(nearest_dist)                                   # ball or all players missing
    nearest_slot[none] = -1
    nearest_dist[none] = np.nan
    nearest_team = np.where(none, -1, team[np.clip(nearest_slot, 0, None)]).astype(np.int8)
    dist_home = np.min(np.where(team == T.HOME, filled, np.inf), axis=1)
    dist_away = np.min(np.where(team == T.AWAY, filled, np.inf), axis=1)
    dist_home[~np.isfinite(dist_home)] = np.nan
    dist_away[~np.isfinite(dist_away)] = np.nan
    return {"dist": dist, "nearest_slot": nearest_slot, "nearest_dist": nearest_dist,
            "nearest_team": nearest_team, "dist_home": dist_home, "dist_away": dist_away}


# --------------------------------------------------------------------------------------------
# Step B: Vidal-Codina possession, as in databallpy
# --------------------------------------------------------------------------------------------
def vidal_codina_possession(ball, ball_speed, pz_slot):
    """Per-frame possessor slot (-1 = none) from the PZ candidate per frame. See module doc."""
    n = len(pz_slot)
    starts, ends, players = _runs(pz_slot, ignore=-1)
    if len(starts) == 0:
        return np.full(n, -1, dtype=np.int16)

    # speed condition: any |speed[t] - speed[t-1]| > BV_THRESHOLD inside the run
    dspeed = np.abs(np.diff(np.r_[0.0, ball_speed]))
    big = np.nan_to_num(dspeed, nan=0.0) > BV_THRESHOLD
    cum = np.r_[0, np.cumsum(big)]
    speed_ok = (cum[ends + 1] - cum[starts]) > 0

    # angle condition: entering vs leaving direction of the ball
    before = np.clip(starts - 1, 0, n - 1)
    after = np.clip(ends + 1, 0, n - 1)
    incoming = ball[starts] - ball[before]
    outgoing = ball[after] - ball[ends]
    angle_ok = np.nan_to_num(_angle_deg(incoming, outgoing), nan=0.0) > BA_THRESHOLD

    length_ok = (ends - starts + 1) >= MIN_FRAMES_PZ
    valid = length_ok & (speed_ok | angle_ok)

    # ball-loss logic: a possession lasts from a valid gain through the same player's later
    # runs, until another player makes a valid gain (or the gap grows beyond the cap)
    possessor = np.full(n, -1, dtype=np.int16)
    current, seg_start, seg_end = -1, 0, -1
    for s, e, p, ok in zip(starts, ends, players, valid):
        if p == current and s - seg_end - 1 <= MAX_EXTEND_FRAMES:
            seg_end = e                                   # same player again: extend
        elif ok:
            if current >= 0:
                possessor[seg_start:seg_end + 1] = current
            current, seg_start, seg_end = p, s, e         # new valid gain
        elif p == current:                                # same player, but gap too long and
            possessor[seg_start:seg_end + 1] = current    # this run is not itself a gain
            current = -1
        # else: another player's touch without a valid gain changes nothing
    if current >= 0:
        possessor[seg_start:seg_end + 1] = current
    return possessor


# --------------------------------------------------------------------------------------------
# Step C: the control.txt checks
# --------------------------------------------------------------------------------------------
def moving_with_player(data, nearest_slot, in_pz):
    """Check 1: the ball moves with the nearest player (dribble / ball at feet)."""
    n = len(nearest_slot)
    idx = np.clip(nearest_slot, 0, None)
    pv = data["player_vel"][np.arange(n), idx]                        # nearest player's velocity
    rel = np.linalg.norm(data["ball_vel"] - pv, axis=1)
    rel[nearest_slot < 0] = np.nan
    rel_med = _rolling(rel, STEADY_WINDOW, "median", center=True, min_periods=STEADY_WINDOW // 2 + 1)
    return in_pz & (np.nan_to_num(rel_med, nan=np.inf) <= REL_SPEED_MAX)


def reception(ball_speed, in_pz, nearest_slot):
    """Check 2: a sharp speed drop inside the PZ, carried through the rest of that PZ run."""
    peak = _rolling(ball_speed, RECEPTION_LOOKBACK + 1, "max", min_periods=1)
    s = np.nan_to_num(ball_speed, nan=np.inf)
    drop = in_pz & (peak - s >= DROP_ABS) & (s <= DROP_FRAC * peak)
    # sticky within the PZ run of the same player
    sticky = np.zeros_like(drop)
    pz_slot = np.where(in_pz, nearest_slot, -1)
    for st, en, _ in zip(*_runs(pz_slot, ignore=-1)):
        hits = np.flatnonzero(drop[st:en + 1])
        if len(hits):
            sticky[st + hits[0]:en + 1] = True
    return sticky


def guarded(controller_team, controller_dist, dist_home, dist_away):
    """Check 3: if an opponent is also in the PZ, is the controller safely closer?

    The margin must have held over the last GUARD_FRAMES frames. Frames before the current
    controller had the ball count as "not held", so a contested ball is never controlled
    during its first GUARD_FRAMES frames. That is intended: a 50-50 is not control.
    """
    opp_dist = np.where(controller_team == T.HOME, dist_away, dist_home)
    opp_dist = np.nan_to_num(opp_dist, nan=np.inf)
    contested = (controller_team >= 0) & (opp_dist < PZ_RADIUS)
    with np.errstate(invalid="ignore"):        # inf - inf where nobody is near: not ok
        margin_ok = np.nan_to_num(opp_dist - controller_dist, nan=-np.inf) >= GUARD_MARGIN
    # the margin must have held for the last GUARD_FRAMES frames (rolling minimum of a 0/1 flag)
    held = _rolling(margin_ok.astype(float), GUARD_FRAMES, "min", min_periods=1) >= 1.0
    return contested, ~contested | held


# --------------------------------------------------------------------------------------------
# Everything together
# --------------------------------------------------------------------------------------------
def compute_control(data=None):
    """Run steps A-C over every frame. Returns the dict described in the module docstring."""
    data = data or T.load_tracking()
    ball = data["ball"].astype(float)
    speed = data["ball_speed"].astype(float)
    ball_visible = ~np.isnan(ball[:, 0])

    prox = proximity(data)
    nearest_slot, nearest_dist = prox["nearest_slot"], prox["nearest_dist"]
    in_pz = ball_visible & (np.nan_to_num(nearest_dist, nan=np.inf) < PZ_RADIUS)
    pz_slot = np.where(in_pz, nearest_slot, -1).astype(np.int16)

    vc_slot = vidal_codina_possession(ball, speed, pz_slot)
    moving = moving_with_player(data, nearest_slot, in_pz)
    received = reception(speed, in_pz, nearest_slot)

    # controller: Vidal-Codina possessor first, otherwise the PZ candidate with evidence
    slot = np.where(vc_slot >= 0, vc_slot, np.where(in_pz & (moving | received), nearest_slot, -1))
    slot = np.where(ball_visible, slot, -1).astype(np.int16)
    team = np.where(slot >= 0, data["team"][np.clip(slot, 0, None)], -1).astype(np.int8)
    ctrl_dist = prox["dist"][np.arange(len(slot)), np.clip(slot, 0, None)]
    ctrl_dist = np.where(slot >= 0, ctrl_dist, np.nan)

    contested, is_guarded = guarded(team, np.nan_to_num(ctrl_dist, nan=np.inf),
                                    prox["dist_home"], prox["dist_away"])
    control = (slot >= 0) & is_guarded

    return {
        "control": control,
        "team": np.where(control, team, -1).astype(np.int8),
        "slot": np.where(control, slot, -1).astype(np.int16),
        "nearest_slot": nearest_slot, "nearest_team": prox["nearest_team"],
        "nearest_dist": nearest_dist.astype(np.float32),
        "dist_home": prox["dist_home"].astype(np.float32),
        "dist_away": prox["dist_away"].astype(np.float32),
        "in_pz": in_pz, "vc_slot": vc_slot, "moving_with": moving, "reception": received,
        "contested": contested, "guarded": is_guarded, "ball_visible": ball_visible,
    }


_CACHE = None


def load_control(force=False):
    """The control arrays, from cache/control.npz (computed and saved if missing)."""
    global _CACHE
    if _CACHE is None or force:
        if os.path.exists(CACHE_FILE) and not force:
            _CACHE = dict(np.load(CACHE_FILE))
        else:
            _CACHE = compute_control()
            np.savez_compressed(CACHE_FILE, **_CACHE)
    return _CACHE


def control(t):
    """control.txt for one frame: True if a player controls the ball at video frame t."""
    return bool(load_control()["control"][t])


def team_in_control(t):
    """0 (home), 1 (away) or -1 (nobody) at video frame t."""
    return int(load_control()["team"][t])


def main():
    c = load_control(force=True)
    np.savez_compressed(CACHE_FILE, **c)

    half = T.half_of_frame(len(c["control"]))
    g = half > 0
    vis = g & c["ball_visible"]
    pct = lambda m: 100.0 * m.sum() / max(vis.sum(), 1)
    print("graded frames {}, ball visible on {} ({:.1f}%)".format(
        int(g.sum()), int(vis.sum()), 100.0 * vis.sum() / g.sum()))
    print("\nshare of ball-visible graded frames")
    print("  ball in a player's PZ ({} m)      {:5.1f}%".format(PZ_RADIUS, pct(vis & c["in_pz"])))
    print("  Vidal-Codina possession            {:5.1f}%".format(pct(vis & (c["vc_slot"] >= 0))))
    print("  moving with player                 {:5.1f}%".format(pct(vis & c["moving_with"])))
    print("  after a reception                  {:5.1f}%".format(pct(vis & c["reception"])))
    print("  contested (both teams in PZ)       {:5.1f}%".format(pct(vis & c["contested"])))
    print("  CONTROL                            {:5.1f}%   (H {:.1f}%, A {:.1f}%)".format(
        pct(vis & c["control"]), pct(vis & (c["team"] == T.HOME)), pct(vis & (c["team"] == T.AWAY))))
    runs = _runs(np.where(g, c["team"], -1), ignore=-1)
    if len(runs[0]):
        lengths = (runs[1] - runs[0] + 1) / T.FPS
        print("\ncontrolled spells: {}  median {:.2f} s  90th pct {:.2f} s".format(
            len(lengths), np.median(lengths), np.percentile(lengths, 90)))
    print("\nwrote {}".format(T.rel(CACHE_FILE)))


if __name__ == "__main__":
    main()
