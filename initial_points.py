"""Step 1 of the pipeline (see KH/initial_points.txt and KH/main.txt): initial points.

Idea (from initial_points.txt)
------------------------------
Initial points are frames whose possession value we already know from the Smart Tagging. Every
set piece (goal kick, corner kick, free kick, throw-in, kick-off at the start and kick-off after a
goal) restarts play with a known team on the ball. Later steps (the possession script) start from
these frames and carry possession forward.

What the data actually gives us, and why this script is more than a lookup
---------------------------------------------------------------------------
A set-piece tag is a *window*, not an instant: 53 of the 59 set-piece tags are 35 s long and the 6
kick-offs are 5 s long. The window typically starts while the ball is still dead (often with no
ball tracked at all), the restart happens somewhere inside it, and the rest of the window is live
play that may already contain a change of possession. Copying the tag's team over the whole
window would therefore label dead-ball frames as possession and could run over a turnover.

So for each tag this script finds the **restart frame**: the moment the ball is put back in play.
That single frame is the initial point, with the restarting team as its value. The frames between
the start of the tag window and the restart frame are recorded too (`dead_from`), because the ball
is out of play there. The restart step can use them later. They are not initial points.

How the restart frame is found
------------------------------
Kick-offs (both kinds) are always taken from the centre spot. The restart frame is the first
frame in the tag window where the tracked ball is within KICKOFF_RADIUS of the centre spot and
moving at kick speed (see find_kickoff for the fallbacks). Centre-spot anchoring matters because around one kick-off the tracked ball glides in a smooth
line at ~10 m/s for several seconds before the real kick (a tracker prediction, not a real ball),
so a speed-only rule would fire too early there.

All other set pieces use a "dead, then kicked" pattern, with three conditions on frame t:
  1. kicked: the ball is visible, its speed at t is at least KICK_SPEED, and the median speed
     over the next KICK_CONFIRM frames is also at least KICK_SPEED. A single noisy frame
     cannot trigger it.
  2. dead before: over the DEAD_LOOKBACK frames before t, at least DEAD_SHARE of the frames have
     the ball missing or moving slower than CARRY_SPEED. A dead ball is either invisible to the
     tracker, lying still, or being carried in a player's hands (throw-ins), and all three look
     like this. Those frames must lie inside the tag window: frames before the window belong
     to the previous phase of play.
  3. legal place: where the kick starts (the last visible ball position just before t) is where
     that kind of restart has to be taken, with a tolerance PLACE_TOL for tracking error:
       throw-in    near a touchline
       corner      near a corner flag
       goal kick   in or just around a goal area (six-yard box)
       free kick   anywhere
The restart frame is the first frame in the window meeting all three. If none meets all three,
the place condition is dropped ("relaxed_place"), then the dead-before condition too
("first_kick"). If nothing at all is found, the tag gives no initial point ("not_found"). The
CSV records which rule fired for every tag, so weak detections can be reviewed or ignored.

Which team the initial point belongs to
---------------------------------------
* Tags with a Team value (corners, goal kicks, free kicks): that team, "tag". The Team tag on a
  set piece is the team taking it.
* Kick-off after a goal: the team that conceded, i.e. the opposite of the most recent GOALS tag,
  "goal". Each GOALS interval ends one second into its kick-off window.
* Throw-ins and the start-of-half kick-offs have Team = N/A in the tagging, so the taker is
  inferred from tracking: the team of the player nearest the restart's origin, within
  TAKER_RADIUS, "nearest". For a throw-in the ball is often first tracked in flight, so its
  origin is traced back along its velocity to the touchline (restart_origin). If no player is
  close enough the tag gets a restart frame but no team, so it is not an initial point.
For every tag the nearest-player team is also recorded (`nearest_team`), so you can see where
the tag and the tracking disagree.

Outputs
-------
out/initial_points.csv    one row per set-piece tag (all detections and diagnostics)
cache/initial_points.npz  per-frame arrays used by the next steps:
    label     (n,) int8   -1 = not an initial point, 0 = home (H), 1 = away (A)
    dead_from (n,) bool   frames inside a set-piece window before its detected restart
                          (ball dead), whether or not the restarting team is known

Run from the KH folder:

    python initial_points.py

Thresholds are generic football/physics values, chosen from the laws of the game and from how
the ball signal looks inside the tag windows. None are tied to particular frames of this match,
and none were tuned on the public ground truth. That was used only to confirm that a set piece's
Team tag names the team taking it.
"""

import csv
import os

import numpy as np

import tracking as T

# --------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------
KICK_SPEED = 5.0          # m/s. A kicked or thrown ball moves at least this fast.
KICK_CONFIRM = 5          # frames (0.2 s). The kick must persist this long (median speed).
CARRY_SPEED = 3.0         # m/s. Slower than this = still, or carried in hands before a throw.
DEAD_LOOKBACK = 25        # frames (1 s) checked before a candidate kick.
DEAD_SHARE = 0.8          # share of those frames that must look dead.
PLACE_TOL = 3.0           # metres of tracking slack around each legal restart area.
CORNER_RADIUS = 4.0       # metres from a corner flag.
KICKOFF_RADIUS = 4.0      # metres from the centre spot.
TAKER_RADIUS = 3.0        # metres. The nearest player must be this close to count as the taker.
SPOT_PLAYER_RADIUS = 2.0  # metres. A player this close to the centre spot = kicker over the ball.
MAX_BACKPROJECT_S = 1.5   # seconds. Longest flight traced back to find a throw-in's origin.

SET_PIECE_CODE = "SET PIECES"
GOALS_CODE = "GOALS"

OUT_CSV = os.path.join(T.OUT_DIR, "initial_points.csv")
OUT_NPZ = os.path.join(T.CACHE_DIR, "initial_points.npz")


# --------------------------------------------------------------------------------------------
# Restart areas (the place condition)
# --------------------------------------------------------------------------------------------
def in_restart_area(kind, x, y):
    """Is (x, y) in metres a legal place for this kind of restart, allowing PLACE_TOL?

    The geometry is the laws of the game: throw-ins from the touchline, corners from the
    corner arc, goal kicks from inside the goal area (5.5 m deep, 18.32 m wide), kick-offs
    from the centre spot. Free kicks can be anywhere.
    """
    L, W = T.PITCH_LENGTH, T.PITCH_WIDTH
    if np.isnan(x) or np.isnan(y):
        return False
    if kind == "throw-in":
        return y <= PLACE_TOL or y >= W - PLACE_TOL
    if kind == "corner kick":
        corners = ((0, 0), (0, W), (L, 0), (L, W))
        return min(np.hypot(x - cx, y - cy) for cx, cy in corners) <= CORNER_RADIUS
    if kind == "goal kick":
        near_goal_line = x <= 5.5 + PLACE_TOL or x >= L - 5.5 - PLACE_TOL
        return near_goal_line and abs(y - W / 2) <= 18.32 / 2 + PLACE_TOL
    if kind.startswith("kick off"):
        return np.hypot(x - L / 2, y - W / 2) <= KICKOFF_RADIUS
    return True  # free kick, or any type we do not know: no place constraint


# --------------------------------------------------------------------------------------------
# Restart-frame detection
# --------------------------------------------------------------------------------------------
def _last_visible_before(ball, t, max_back=KICK_CONFIRM):
    """Ball position at t-1 (or the nearest earlier visible frame, up to max_back back).

    This approximates where the kick started: at t the ball has already moved up to ~0.7 m.
    It falls back to the position at t itself when the ball only appears at the kick.
    """
    for k in range(1, max_back + 1):
        if t - k >= 0 and not np.isnan(ball[t - k, 0]):
            return ball[t - k]
    return ball[t]


def find_kickoff(ball, speed, players, start, end):
    """Restart frame of a kick-off. Returns (frame or None, method name).

    First choice: the first frame in [start, end) with the ball within KICKOFF_RADIUS of the
    centre spot *and* moving at KICK_SPEED or more ("centre_spot"). Requiring movement means
    a ball tracked lying on the spot before the whistle does not count as the kick. Next, the
    first frame with the ball near the spot at any speed ("centre_spot_still"). If the ball is
    never tracked near the spot, the kick-off is placed at the first frame where a player stands
    within SPOT_PLAYER_RADIUS of the centre spot ("player_at_spot"). At a kick-off only the
    kicking team may be inside the centre circle, so a player on the spot is the kicker
    standing over the ball.
    """
    near = [t for t in range(start, end) if in_restart_area("kick off", ball[t, 0], ball[t, 1])]
    moving = [t for t in near if not np.isnan(speed[t]) and speed[t] >= KICK_SPEED]
    if moving:
        return moving[0], "centre_spot"
    if near:
        return near[0], "centre_spot_still"
    cx, cy = T.PITCH_LENGTH / 2, T.PITCH_WIDTH / 2
    for t in range(start, end):
        d = np.hypot(players[t, :, 0] - cx, players[t, :, 1] - cy)
        if np.any(d <= SPOT_PLAYER_RADIUS):
            return t, "player_at_spot"
    return None, "not_found"


def find_restart(ball, speed, start, end, kind):
    """Restart frame of a non-kick-off set piece. Returns (frame or None, method name).

    Tries the strict rule (kicked + dead before + legal place), then relaxes the place
    condition, then the dead-before condition. See the module docstring.
    """
    n = len(ball)
    visible = ~np.isnan(ball[:, 0])
    # "looks dead": missing, or slower than a carried ball
    deadish = ~visible | (np.nan_to_num(speed, nan=0.0) < CARRY_SPEED)

    kicked, dead_before, placed = [], [], []
    for t in range(start, min(end, n - KICK_CONFIRM)):
        if not visible[t] or speed[t] < KICK_SPEED:
            continue
        ahead = speed[t:t + KICK_CONFIRM]
        ahead = ahead[~np.isnan(ahead)]
        if len(ahead) == 0 or np.median(ahead) < KICK_SPEED:
            continue
        kicked.append(t)
        # The dead stretch must lie inside the tag window: the window opens at the stoppage,
        # so frames before it belong to the previous phase of play and may still be live.
        lo = t - DEAD_LOOKBACK
        is_dead = lo >= start and deadish[lo:t].mean() >= DEAD_SHARE
        x, y = _last_visible_before(ball, t)
        is_placed = in_restart_area(kind, x, y)
        if is_dead:
            dead_before.append(t)
            if is_placed:
                placed.append(t)

    if placed:
        return placed[0], "strict"
    if dead_before:
        return dead_before[0], "relaxed_place"
    if kicked:
        return kicked[0], "first_kick"
    return None, "not_found"


def restart_origin(data, t, kind):
    """Best estimate of where the restart was taken from, for finding the taker.

    Normally the last visible ball position just before t. For throw-ins the ball is often
    first tracked in flight, metres away from the thrower, so if that position is not already
    near a touchline it is projected backwards along the ball's velocity to the touchline it
    came from (at most MAX_BACKPROJECT_S seconds back).
    """
    x, y = _last_visible_before(data["ball"], t + 1)     # +1 so frame t itself comes first
    if kind != "throw-in" or np.isnan(x) or in_restart_area("throw-in", x, y):
        return x, y
    vx, vy = data["ball_vel"][t]
    if np.isnan(vy) or abs(vy) < 0.5:
        return x, y
    line_y = 0.0 if vy > 0 else T.PITCH_WIDTH          # moving away from that touchline
    dt = min((y - line_y) / vy, MAX_BACKPROJECT_S)
    if dt <= 0:
        return x, y
    return x - vx * dt, y - vy * dt


def nearest_player(data, t, kind=""):
    """(slot, team, distance m) of the player nearest the restart origin at frame t."""
    bx, by = restart_origin(data, t, kind)
    if np.isnan(bx):
        return -1, -1, np.nan
    d = np.hypot(data["players"][t, :, 0] - bx, data["players"][t, :, 1] - by)
    if np.all(np.isnan(d)):
        return -1, -1, np.nan
    slot = int(np.nanargmin(d))
    return slot, int(data["team"][slot]), float(d[slot])


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def build_initial_points(data=None, tags=None):
    """Detect every set-piece restart. Returns (rows, label, dead_from). See module doc."""
    data = data or T.load_tracking()
    tags = tags or T.load_smart_tagging()
    n = len(data["ball"])
    ball, speed = data["ball"].astype(float), data["ball_speed"].astype(float)

    goals = sorted((t for t in tags if t["code"] == GOALS_CODE), key=lambda t: t["end_frame"])
    set_pieces = sorted((t for t in tags if t["code"] == SET_PIECE_CODE),
                        key=lambda t: t["start_frame"])

    label = np.full(n, -1, dtype=np.int8)
    dead_from = np.zeros(n, dtype=bool)
    rows = []

    for tag in set_pieces:
        kind = (tag["tags"].get("Type") or "").strip().lower()
        a, b = tag["start_frame"], min(tag["end_frame"], n)

        if kind.startswith("kick off"):
            frame, method = find_kickoff(ball, speed, data["players"], a, b)
        else:
            frame, method = find_restart(ball, speed, a, b, kind)

        row = {
            "tag_id": tag["id"], "type": tag["tags"].get("Type"),
            "tag_team": {T.HOME: "H", T.AWAY: "A"}.get(tag["team"], "N/A"),
            "window_start": a, "window_end": b, "restart_frame": "",
            "label": "", "team_source": "", "method": method,
            "nearest_team": "", "taker_dist_m": "", "half": "",
        }
        if frame is None:
            rows.append(row)
            continue

        slot, near_team, dist = nearest_player(data, frame, kind)
        near_ok = slot >= 0 and dist <= TAKER_RADIUS

        # Decide the restarting team, in order of how much we trust the source.
        team, source = None, ""
        if tag["team"] is not None:
            team, source = tag["team"], "tag"
        elif kind == "kick off (after goal)":
            prior = [g for g in goals if g["end_frame"] <= b and g["team"] is not None]
            if prior:
                team, source = 1 - prior[-1]["team"], "goal"   # the conceding team kicks off
        if team is None and near_ok:
            team, source = near_team, "nearest"

        row.update({
            "restart_frame": frame,
            "nearest_team": {T.HOME: "H", T.AWAY: "A"}.get(near_team, "") if near_ok else "",
            "taker_dist_m": "" if np.isnan(dist) else round(dist, 2),
            "half": int(data["half"][frame]),
        })
        # The ball is dead from the start of the window up to the restart, whether or not
        # we know who took it.
        dead_from[a:frame] = True
        if team is not None:
            row.update({"label": T.TEAM_LABEL[team], "team_source": source})
            label[frame] = team
        rows.append(row)

    return rows, label, dead_from


def main():
    rows, label, dead_from = build_initial_points()

    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT_NPZ, label=label, dead_from=dead_from)

    # ---- summary -------------------------------------------------------------------------
    from collections import Counter
    print("set-piece tags: {}".format(len(rows)))
    print("initial points: {}  (H {}, A {})".format(
        int((label >= 0).sum()), int((label == T.HOME).sum()), int((label == T.AWAY).sum())))
    print("\ndetection method per type")
    by_type = Counter((r["type"], r["method"]) for r in rows)
    for (kind, method), k in sorted(by_type.items()):
        print("  {:<22} {:<14} {:>3}".format(kind, method, k))
    print("\nteam source: {}".format(dict(Counter(r["team_source"] or "none" for r in rows))))
    both = [r for r in rows if r["label"] and r["nearest_team"]]
    agree = sum(r["label"] == r["nearest_team"] for r in both)
    print("label agrees with nearest player at restart: {}/{}".format(agree, len(both)))
    offsets = [(r["restart_frame"] - r["window_start"]) / T.FPS for r in rows if r["label"]]
    if offsets:
        print("restart offset into tag window: median {:.1f} s, range {:.1f}-{:.1f} s".format(
            np.median(offsets), min(offsets), max(offsets)))
    outside = sum(1 for r in rows if r["label"] and r["half"] == 0)
    if outside:
        print("WARNING: {} initial point(s) fall outside a graded half".format(outside))
    print("\nwrote {}\n      {}".format(T.rel(OUT_CSV), T.rel(OUT_NPZ)))


if __name__ == "__main__":
    main()
