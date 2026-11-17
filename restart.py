"""Step 2 of the pipeline: restart points.

Idea
----
"All frames that the ball is out of play and not an initial point are restart points. It could
be because the ball is out of bounds or any referee stoppage, or during halftime, or before
kick-off, or after the final whistle."

The possession script gives every restart frame the value 0 (neither team), and in the final
submission those frames become D (dead ball).

How out-of-play frames are found
--------------------------------
Seven sources, each tagged with a bit in `reason` so every decision can be traced:

  1  OUTSIDE_HALF   Pre-match, halftime and post-match: every frame outside the two graded
                    halves published in DATA.md.
  2  PRE_KICKOFF    Inside a graded half but before that half's kick-off. The kick-off frame is
                    the "Kick Off (Start)" initial point found by initial_points.py; it is not
                    hard-coded. (The 2nd half is graded ~2 minutes before its kick-off.)
  4  TAG_WINDOW     Inside a Smart Tagging set-piece window, before the detected restart
                    (`dead_from` from initial_points.py). The ball is waiting to be put back
                    in play.
  8  BALL_OUT       The tracked ball is more than OUT_MARGIN metres beyond a touchline or goal
                    line for at least OUT_MIN_FRAMES frames. The ball is dead from the first of
                    those frames until play *resumes* (see below).
 16  STOPPAGE       A referee stoppage that left no tag (a foul, an injury, a throw-in the
                    tagging missed). Two rules, either one is enough:
                    a) the players are calm and nobody is playing the ball, for at least
                       STOP_MIN_S seconds. "Nobody playing it" means the ball is missing or
                       slower than a carried ball, and no player controls it (control.py).
                    b) the players are calm for at least LONG_CALM_S seconds, whatever the
                       ball does. During stoppages players often knock the ball to each
                       other at walking pace (to the throw-in taker, to the free-kick spot),
                       which defeats rule a). But in live play the players do not all stay
                       below walking pace for that long.
                    "Calm" means the rolling CALM_WINDOW median of the players' mean speed is
                    below CALM_SPEED.
                    Exception, the goalkeeper holding the ball (idea taken from the team's
                    earlier analysis/ pipeline, its "goalie hands" rule): when a keeper has the
                    ball in their hands, everyone walks and the ball does not move, which looks
                    exactly like a stoppage. But the ball is in play and the keeper's team has
                    it. So STOPPAGE is not set on frames where the last known ball position (at
                    most KEEPER_BALL_MAX_GAP_S old) is inside a penalty box, the nearest player
                    to it is a goalkeeper (from the ATD metadata) within KEEPER_RADIUS, and at
                    most KEEPER_MAX_NEAR players are within KEEPER_CROWD_RADIUS. Only STOPPAGE
                    is lifted: a goal kick still counts as dead through BALL_OUT or TAG_WINDOW.
                    A STOPPAGE also ends at the first initial point inside it: players are
                    still walking for a few seconds after a restart, but play has resumed.
 64  BALL_OUT_PRE   The ball-missing frames directly before a BALL_OUT stretch. The ball usually
                    crosses the line while the tracker has lost it, so the first frame it is
                    seen outside comes late. All the missing frames back to the last frame the
                    ball was seen (or an initial point, or the start of the half) are dead too.
                    There is no length cap.
 32  SHORT_LIVE     A live stretch shorter than MIN_LIVE_S (2 s) between two dead stretches,
                    with no initial point in it. That is detection noise inside one stoppage,
                    not a real spell of play; left alone it would become a sub-2-second
                    possession.

How CALM_SPEED was chosen: algorithmic labels only, no ground truth. On frames that
initial_points.py marks dead (tag window before the restart), the players' mean speed has
median 0.92 m/s and 75th percentile 1.31 m/s. On frames control.py marks controlled (live),
the median is 1.93 m/s and the 25th percentile 1.42 m/s. 1.2 m/s sits between the two.
LONG_CALM_S was checked on the same proxies: calm runs of 6 s or more cover 63% of the
tag-dead frames and 9.5% of the controlled frames. Players also control the ball during
stoppages, so not all of that 9.5% is wrong.

When does play resume after the ball goes out (BALL_OUT)?
At the first frame after the ball left the pitch where one of these holds:
  * an initial point (a tagged restart taken), or
  * the ball is inside the pitch and is kicked/thrown: speed >= KICK_SPEED, confirmed by the
    median of the next KICK_CONFIRM frames (the same rule initial_points.py uses), or
  * the ball is inside the pitch, a player controls it, and the players are not calm. A
    player walking the ball back to the line is calm, so this does not end the stoppage.
If none happens within MAX_DEAD_S, only the frames with the ball actually outside count as
dead, and the stretch is flagged "no_resume" in the CSV.

Initial points are never restart points: those frames are removed from `restart` at the end.

Outputs
-------
cache/restart.npz   restart (n,) bool  ball out of play, excluding initial points
                       reason  (n,) uint8 bitmask of the sources above
out/restart.csv     one row per dead stretch inside the graded halves: start, end,
                       duration, reasons, and how it ended

Run from the KH folder (after initial_points.py and control.py):

    python restart.py
"""

import csv
import os

import numpy as np
import pandas as pd

import control as C
import initial_points as IP
import tracking as T

# --------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------
OUT_MARGIN = 1.0          # m beyond the line. A ball on the line is still in play, plus tracking slack.
OUT_MIN_FRAMES = 3        # frames. Ignore single-frame excursions (tracking jitter).
MAX_DEAD_S = 90.0         # s. Longest we look ahead for play to resume after the ball goes out.
KICK_SPEED = IP.KICK_SPEED       # 5 m/s, shared with initial_points.py
KICK_CONFIRM = IP.KICK_CONFIRM   # 5 frames
CARRY_SPEED = IP.CARRY_SPEED     # 3 m/s
CALM_SPEED = 1.2          # m/s. Players' mean speed during stoppages (see module doc).
CALM_WINDOW = 50          # frames (2 s). Rolling median window for calmness.
STOP_MIN_S = 3.0          # s. A calm, idle stretch must last this long to be a stoppage.
LONG_CALM_S = 6.0         # s. A calm stretch this long is a stoppage whatever the ball does.
KEEPER_RADIUS = 2.0       # m. Goalkeeper this close to the ball = ball in the keeper's hands.
KEEPER_CROWD_RADIUS = 3.0 # m. analysis/ "goalie hands" rule: few players near the ball...
KEEPER_MAX_NEAR = 2       # ...at most this many (the keeper plus one).
KEEPER_BALL_MAX_GAP_S = 3.0  # s. How stale the last known ball position may be.
BOX_DEPTH, BOX_WIDTH = 16.5, 40.32   # m. Penalty box (laws of the game).

MIN_LIVE_S = 2.0          # s. Shorter live stretches between two dead ones are noise.

OUTSIDE_HALF, PRE_KICKOFF, TAG_WINDOW, BALL_OUT, STOPPAGE, SHORT_LIVE, BALL_OUT_PRE = \
    1, 2, 4, 8, 16, 32, 64
REASON_NAMES = {OUTSIDE_HALF: "outside_half", PRE_KICKOFF: "pre_kickoff",
                TAG_WINDOW: "tag_window", BALL_OUT: "ball_out", STOPPAGE: "stoppage",
                SHORT_LIVE: "short_live", BALL_OUT_PRE: "ball_out_pre"}

CACHE_FILE = os.path.join(T.CACHE_DIR, "restart.npz")
OUT_CSV = os.path.join(T.OUT_DIR, "restart.csv")


# --------------------------------------------------------------------------------------------
# Per-frame signals
# --------------------------------------------------------------------------------------------
def players_calm(data):
    """True where the players' mean speed (2 s rolling median) is below CALM_SPEED."""
    speed = np.linalg.norm(data["player_vel"], axis=2)                    # (n, 46)
    ok = ~np.isnan(speed)
    mean = np.where(ok.any(axis=1),
                    np.where(ok, speed, 0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)
    smooth = pd.Series(mean).rolling(CALM_WINDOW, center=True, min_periods=CALM_WINDOW // 5).median()
    return smooth.to_numpy() < CALM_SPEED


def keeper_holding(data):
    """True where a goalkeeper has the ball in their own penalty box (see module doc)."""
    gk = T.goalkeeper_slots()
    limit = int(KEEPER_BALL_MAX_GAP_S * T.FPS)
    last = pd.DataFrame(data["ball"]).ffill(limit=limit).to_numpy()      # last known ball
    x, y = last[:, 0], last[:, 1]
    with np.errstate(invalid="ignore"):
        in_box = (np.abs(y - T.PITCH_WIDTH / 2) <= BOX_WIDTH / 2) & \
                 ((x <= BOX_DEPTH) | (x >= T.PITCH_LENGTH - BOX_DEPTH))
    dist = np.linalg.norm(data["players"] - last[:, None, :].astype(np.float32), axis=2)
    filled = np.where(np.isnan(dist), np.inf, dist)
    nearest = np.argmin(filled, axis=1)
    nearest_d = filled[np.arange(len(filled)), nearest]
    crowd = (filled <= KEEPER_CROWD_RADIUS).sum(axis=1)
    return in_box & gk[nearest] & (nearest_d <= KEEPER_RADIUS) & (crowd <= KEEPER_MAX_NEAR)


def ball_outside(ball):
    """True where the tracked ball is more than OUT_MARGIN beyond a pitch line."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return ((x < -OUT_MARGIN) | (x > T.PITCH_LENGTH + OUT_MARGIN)
                | (y < -OUT_MARGIN) | (y > T.PITCH_WIDTH + OUT_MARGIN))


def ball_inside(ball):
    """True where the tracked ball is on the pitch (lines included)."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x >= 0) & (x <= T.PITCH_LENGTH) & (y >= 0) & (y <= T.PITCH_WIDTH)


def kicked(ball_speed):
    """True where the ball is kicked: fast now and over the next KICK_CONFIRM frames (median)."""
    s = pd.Series(ball_speed)
    ahead = s[::-1].rolling(KICK_CONFIRM, min_periods=1).median()[::-1].to_numpy()
    return (np.nan_to_num(ball_speed, nan=0.0) >= KICK_SPEED) & (np.nan_to_num(ahead, nan=0.0) >= KICK_SPEED)


def _runs(mask):
    """(start, end_inclusive) of True runs."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


# --------------------------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------------------------
def kickoff_frames():
    """{half: frame} of the 'Kick Off (Start)' initial points, read from initial_points.csv."""
    out = {}
    if not os.path.exists(IP.OUT_CSV):
        return out
    for row in csv.DictReader(open(IP.OUT_CSV)):
        if (row["type"] or "").lower() == "kick off (start)" and row["restart_frame"] and row["half"]:
            out[int(row["half"])] = int(row["restart_frame"])
    return out


def compute_restart(data=None, ctrl=None):
    """Return (restart bool array, reason uint8 array, list of BALL_OUT stretch endings)."""
    data = data or T.load_tracking()
    ctrl = ctrl or C.load_control()
    ip = np.load(IP.OUT_NPZ)
    ip_label, dead_from = ip["label"], ip["dead_from"]

    n = len(data["ball"])
    ball = data["ball"].astype(float)
    speed = data["ball_speed"].astype(float)
    half = data["half"]
    reason = np.zeros(n, dtype=np.uint8)

    # 1. outside the graded halves
    reason[half == 0] |= OUTSIDE_HALF

    # 2. inside a half, before its kick-off
    for h, kick in kickoff_frames().items():
        start = T.GRADED_HALVES[h][0]
        if kick > start:
            reason[start:kick] |= PRE_KICKOFF

    # 3. tagged set-piece windows before the restart
    reason[dead_from] |= TAG_WINDOW

    calm = players_calm(data)
    visible = ~np.isnan(ball[:, 0])
    inside = ball_inside(ball)
    kick = kicked(speed)
    is_ip = ip_label >= 0
    resume = is_ip | (inside & kick) | (inside & ctrl["control"] & ~calm)

    # 4. ball out of the pitch, until play resumes
    endings = []
    max_dead = int(MAX_DEAD_S * T.FPS)
    out_starts, out_ends = _runs(ball_outside(ball) & (half > 0))
    for s, e in zip(out_starts, out_ends):
        if e - s + 1 < OUT_MIN_FRAMES:
            continue
        ahead = np.flatnonzero(resume[e + 1:min(n, e + 1 + max_dead)])
        if len(ahead):
            stop = e + 1 + ahead[0]                     # first frame of play again
            how = "initial_point" if is_ip[stop] else ("kick" if kick[stop] else "control")
        else:
            stop, how = e + 1, "no_resume"
        reason[s:stop] |= BALL_OUT
        endings.append((s, stop, how))
        # The ball usually goes out while the tracker has lost it, so the first frame we SEE
        # it outside is late. Every ball-missing frame directly before it (back to the last
        # frame the ball was seen, an initial point, or the start of the half) is dead too.
        half_start = T.GRADED_HALVES[int(half[s])][0]
        k = s - 1
        while k >= half_start and not visible[k] and not is_ip[k]:
            k -= 1
        reason[k + 1:s] |= BALL_OUT_PRE

    # 5. untagged stoppages: calm players and nobody playing the ball
    idle = np.isnan(speed) | (speed < CARRY_SPEED)
    idle &= ~ctrl["control"]
    st_starts, st_ends = _runs(calm & idle & (half > 0))
    ip_frames = np.flatnonzero(is_ip)

    def mark_stoppage(s, e):
        """Flag [s, e] as STOPPAGE, but end it at the first initial point inside it."""
        i = np.searchsorted(ip_frames, s)
        if i < len(ip_frames) and ip_frames[i] <= e:
            e = ip_frames[i] - 1
        if e >= s:
            reason[s:e + 1] |= STOPPAGE

    for s, e in zip(st_starts, st_ends):                 # rule a)
        if (e - s + 1) >= STOP_MIN_S * T.FPS:
            mark_stoppage(s, e)
    for s, e in zip(*_runs(calm & (half > 0))):          # rule b)
        if (e - s + 1) >= LONG_CALM_S * T.FPS:
            mark_stoppage(s, e)
    keeper = keeper_holding(data)                         # exception: keeper has the ball
    reason[keeper] &= np.uint8(~STOPPAGE & 0xFF)

    # 6. live slivers: a live stretch shorter than MIN_LIVE_S between two dead stretches,
    #    with no initial point in it, is detection noise inside one stoppage
    dead = reason > 0
    for h, (a, b) in T.GRADED_HALVES.items():
        for s, e in zip(*_runs(~dead[a:b + 1])):
            s, e = s + a, e + a
            if s > a and e < b and (e - s + 1) < MIN_LIVE_S * T.FPS and not is_ip[s:e + 1].any():
                reason[s:e + 1] |= SHORT_LIVE

    restart = reason > 0
    restart[is_ip] = False                               # initial points are never restarts
    return restart, reason, endings


def _reason_text(bits):
    return "+".join(name for bit, name in REASON_NAMES.items() if bits & bit)


def main():
    restart, reason, endings = compute_restart()
    np.savez_compressed(CACHE_FILE, restart=restart, reason=reason)

    half = T.half_of_frame(len(restart))
    graded = half > 0
    end_how = {s: how for s, _, how in endings}
    rows = []
    for s, e in zip(*_runs(restart & graded)):
        bits = int(np.bitwise_or.reduce(reason[s:e + 1]))
        how = next((end_how[k] for k in end_how if s <= k <= e), "")
        rows.append({"start": s, "end": e, "duration_s": round((e - s + 1) / T.FPS, 2),
                     "half": int(half[s]), "reasons": _reason_text(bits), "ball_out_ended_by": how})
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start", "end", "duration_s", "half",
                                                    "reasons", "ball_out_ended_by"])
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary -------------------------------------------------------------------------
    g = graded
    print("graded frames {}: restart (dead) {} = {:.1f}%".format(
        int(g.sum()), int((restart & g).sum()), 100.0 * (restart & g).sum() / g.sum()))
    print("\nframes per source (graded; a frame can have several)")
    for bit, name in REASON_NAMES.items():
        if bit == OUTSIDE_HALF:
            continue
        print("  {:<12} {:>7}".format(name, int(((reason & bit) > 0)[g].sum())))
    from collections import Counter
    print("\nball-out stretches: {}   ended by: {}".format(
        len(endings), dict(Counter(how for _, _, how in endings))))
    durs = [r["duration_s"] for r in rows]
    if durs:
        print("dead stretches in graded halves: {}   median {:.1f} s   total {:.1f} min".format(
            len(durs), np.median(durs), sum(durs) / 60))
    print("\nwrote {}\n      {}".format(T.rel(CACHE_FILE), T.rel(OUT_CSV)))


if __name__ == "__main__":
    main()
