# Parameters and scenarios

Every threshold the pipeline uses, where it lives, and the situations it is meant to catch. Values
are copied from the code; if you change one, change it in the named file (and here).

Units: distances in **metres**, speeds in **m/s**, times in **seconds** (the video runs at
**25 frames per second**, so 1 frame = 0.04 s, 25 frames = 1 s). All positions are on the
105.3 × 68.0 m pitch.

None of these values were tuned on the organisers' public ground truth. They come from the laws
of the game, from football common sense, from databallpy's published defaults, or from our own
algorithmic labels (noted where so).

---

## 1. Pitch and match constants

| Parameter | Value | File | Meaning |
|---|---|---|---|
| `FPS` | 25 | tracking.py | video frame rate |
| `PITCH_LENGTH` × `PITCH_WIDTH` | 105.3 × 68.0 m | tracking.py | pitch size from the organisers' `games.py` (the ATD metadata says 105 × 68) |
| `GRADED_HALVES` | 11185–90651, 110808–190197 | tracking.py | the frames that are scored (published in DATA.md) |
| home / away slots | 1–24 / 25–46 | tracking.py | tracker slots per team (slots 24 and 46 are the goalkeepers) |
| `BOX_DEPTH` × `BOX_WIDTH` | 16.5 × 40.32 m | restart.py, tracking.py | penalty box (laws of the game) |
| goal area | 5.5 × 18.32 m | initial_points.py | six-yard box, used for goal kicks |

### Pep Guardiola's 20 zones (tracking.py, used by the classifier)

5 lanes across the width × 4 bands along the length, always seen from Barcelona's attacking
direction (the 2nd half is rotated 180°):

| | own box (0–16.5) | own half (16.5–52.65) | opp. half (52.65–88.8) | opp. box (88.8–105.3) |
|---|---|---|---|---|
| wing (0–13.84) | 1 | 6 | 11 | 16 |
| half-space (13.84–24.84) | 2 | 7 | 12 | 17 |
| centre (24.84–43.16) | 3 | 8 | 13 | 18 |
| half-space (43.16–54.16) | 4 | 9 | 14 | 19 |
| wing (54.16–68.0) | 5 | 10 | 15 | 20 |

Lane lines are the widths of the penalty box (40.32 m) and the six-yard box (18.32 m).

---

## 2. Data cleaning (tracking.py)

| Parameter | Value | Meaning |
|---|---|---|
| `PLAYER_MAX_SPEED` | 12 m/s | a player "faster" than this is a tracker identity jump; velocity set to missing |
| `BALL_MAX_SPEED` | 40 m/s | a ball faster than this (the hardest shots are ~35) is a tracking jump; velocity set to missing |

Velocities are computed from positions, (p[t+1] − p[t−1]) × 25 / 2, because the ball's own speed
channel is mostly empty.

---

## 3. Speeds

| Parameter | Value | File | Meaning |
|---|---|---|---|
| `CALM_SPEED` | 1.2 m/s | restart.py | players' mean speed below this = everyone walking (stoppage). Chosen between dead-ball frames (median 0.92) and controlled frames (median 1.93), both from our own labels |
| `CARRY_SPEED` | 3 m/s | initial_points.py (shared) | ball slower than this = lying still or carried in hands |
| `REL_SPEED_MAX` | 3 m/s | control.py | ball and player velocities differ by at most this = ball moving with the player |
| `PASS_MIN_SPEED` | 4 m/s | pass_sprint.py | slowest a pass can be |
| `DROP_ABS` | 4 m/s | control.py | minimum ball-speed drop for a reception (first touch) |
| `KICK_SPEED` | 5 m/s | initial_points.py (shared) | a kicked or thrown ball moves at least this fast |
| `BV_THRESHOLD` | 5 m/s | control.py | databallpy default: ball speed change that validates a gain of possession |
| `SPRINT_MIN_SPEED` | 5 m/s (~18 km/h) | pass_sprint.py | running speed of a player carrying the ball |
| `SPRINT_SPEED` | 7 m/s (~25 km/h) | classifier.py | standard sprint threshold, for the "players sprinting" feature |
| throw-in back-projection | ≥ 0.5 m/s across the pitch | initial_points.py | minimum sideways ball speed to trace a throw back to the touchline |

---

## 4. Distances

| Parameter | Value | File | Meaning |
|---|---|---|---|
| `GUARD_MARGIN` | 0.5 m | control.py | when contested, the controller must be this much closer to the ball than the nearest opponent |
| `OUT_MARGIN` | 1.0 m | restart.py | the ball must be this far beyond a line to count as out (a ball on the line is in play) |
| `PZ_RADIUS` / `PROXIMITY_M` | 1.5 m | control.py, possession_script.py | possession zone: a player this close can have the ball (databallpy default) |
| `SAME_PLAYER_DIST` | 1.5 m | pass_sprint.py | two track slots this close at the same moment = the same person |
| `KEEPER_RADIUS` | 2.0 m | restart.py | goalkeeper this close to the ball = ball in their hands |
| `SPOT_PLAYER_RADIUS` | 2.0 m | initial_points.py | player this close to the centre spot = kicker standing over the ball |
| `CLOSE_CARRY_DIST` | 2.5 m | pass_sprint.py | ball always this close to the same player = slow carry or shielding |
| `PLACE_TOL` | 3.0 m | initial_points.py | slack around each legal restart area (touchline, goal area) |
| `TAKER_RADIUS` | 3.0 m | initial_points.py | nearest player must be this close to the restart spot to be the taker |
| `PASS_MIN_DIST` | 3.0 m | pass_sprint.py | a pass moves the ball at least this far |
| `KEEPER_CROWD_RADIUS` | 3.0 m | restart.py | at most `KEEPER_MAX_NEAR` = 2 players this close for "keeper holding the ball" |
| `CORNER_RADIUS` | 4.0 m | initial_points.py | corner kick taken within this distance of a corner flag |
| `KICKOFF_RADIUS` | 4.0 m | initial_points.py | kick-off taken within this distance of the centre spot |
| `CARRY_MAX_DIST` | 5.0 m | pass_sprint.py | during a sprint the ball never gets farther than this from the runner |

---

## 5. Times and frame counts

| Parameter | Value | File | Meaning |
|---|---|---|---|
| `OUT_MIN_FRAMES` | 3 frames (0.12 s) | restart.py | ball must be out this long (ignores jitter) |
| `MIN_FRAMES_PZ` | 0 frames | control.py | databallpy default: minimum time in the zone for a gain |
| `KICK_CONFIRM` | 5 frames (0.2 s) | initial_points.py | a kick must keep its speed this long |
| `OPP_MAX_FRAMES` | 5 frames (0.2 s) | pass_sprint.py | opponent near the ball longer than this inside a gap = possible interception |
| `STEADY_WINDOW` | 7 frames (0.28 s) | control.py | smoothing window for "ball moving with the player" |
| `GUARD_FRAMES` | 10 frames (0.4 s) | control.py | a contested controller must have held the margin this long |
| `RECEPTION_LOOKBACK` | 12 frames (~0.5 s) | control.py | how far back to look for the pass speed before a reception |
| `SHORT_GAP_S` | 0.5 s | pass_sprint.py | gaps this short stay with the team (a heavy touch) |
| `SWITCH_S` | 0.5 s | possession_script.py | control a new team needs before possession switches |
| `DEAD_LOOKBACK` | 1 s (`DEAD_SHARE` 80%) | initial_points.py | before a restart kick, 80% of the last second must look dead |
| `MAX_BACKPROJECT_S` | 1.5 s | initial_points.py | longest throw-in flight traced back to its origin |
| `MAX_EXTEND_FRAMES` | 50 frames (2 s) | control.py | longest gap one player's possession is carried across (our cap; databallpy has none) |
| `CALM_WINDOW` | 50 frames (2 s) | restart.py | smoothing window for the players' mean speed |
| `SHORT_BLANK_S` | 2 s | possession_script.py | ball missing for less than this = keep the last team |
| `MIN_LIVE_S` | 2 s | restart.py | live stretch shorter than this between two stoppages = noise, treated as dead |
| `PENDING_MAX_FRAMES` | 3 s | possession_script.py | the new team's 0.5 s of control must fall within this window |
| `STOP_MIN_S` | 3 s | restart.py | calm players + nobody playing the ball this long = stoppage |
| `KEEPER_BALL_MAX_GAP_S` | 3 s | restart.py | how old the last known ball position may be for the keeper rule |
| `MAX_SPRINT_S` | 4 s | pass_sprint.py | longest carry accepted |
| `MAX_PASS_S` | 5 s | pass_sprint.py | longest pass accepted (covers long balls) |
| `LONG_CALM_S` | 6 s | restart.py | calm players this long = stoppage, whatever the ball does. Covers 63% of tag-dead frames and 9.5% of controlled frames (our own labels) |
| `BALL_STALE_CAP_S` | 10 s | classifier.py | cap on the "seconds since the ball was seen" feature |
| `MAX_DEAD_S` | 90 s | restart.py | longest we wait for play to resume after the ball goes out |

---

## 6. Angles

| Parameter | Value | File | Meaning |
|---|---|---|---|
| `BA_THRESHOLD` | 10° | control.py | databallpy default: ball direction change that validates a gain of possession |

---

## 7. Classifier settings (classifier.py)

| Parameter | Value | Meaning |
|---|---|---|
| decision tree | depth ≤ 12, ≥ 50 frames per leaf, balanced classes | cannot memorise single frames |
| KNN | 25 neighbours, distance-weighted, features standardised | |
| `KNN_MAX_TRAIN` | 40,000 frames | fixed random subsample, keeps KNN fast |
| `BLOCK_S` / `TEST_EVERY` | 60 s / every 5th block | held-out test blocks (avoids neighbouring-frame leakage) |
| `RANDOM_STATE` | 0 | fixed seed, so every run gives the same model |

---

# Scenarios

## Is a player controlling the ball? (control.py)

The ball must be **visible**. Then a player controls it if **any** of A–C holds **and** D holds.

| | Scenario | Rule |
|---|---|---|
| A | **Gain of possession** (Vidal-Codina / databallpy) | the player is the nearest, within 1.5 m, and while the ball is in their zone it changes speed by > 5 m/s or direction by > 10°. They keep possession through their later touches (gaps ≤ 2 s) until another player makes a valid gain |
| B | **Dribbling / ball at feet** | within 1.5 m and the ball's velocity matches the player's within 3 m/s (median over 0.28 s) |
| C | **Reception (first touch)** | within 1.5 m and the ball has slowed by ≥ 4 m/s and to ≤ 50% of its speed in the last 0.5 s. Then control lasts as long as the ball stays in their zone |
| D | **Guarding when contested** | if an opponent is also within 1.5 m, the controller must be ≥ 0.5 m closer than them and have been so for 0.4 s. A 50-50 is not control |

**Not control:** a ball rolling straight past a player at constant speed (no change of speed or
direction), a ball flying over a player, a touch by another player with no valid gain, and a
contested ball where nobody is clearly closer.

## When does possession change? (possession_script.py)

Frames are decided in this order:

| Order | Scenario | Result |
|---|---|---|
| 1 | Frame is part of a completed **pass / sprint / carry / short touch** between the same team | that team |
| 2 | **Ball out of play** (restart frame) | 0 (written as `D`), and the "last team" is forgotten |
| 3 | Ball visible, last frame had a team, **same team controls** | keep the team |
| 3 | Ball visible, last frame had a team, **other team controls** | switch only after **0.5 s of their control within 3 s**, back-dated to their first frame of control |
| 3 | Ball visible, **nobody controls** | 0 (loose ball, written as the last team in the submission) |
| 3 | Ball visible, last frame was 0, **nearest player within 1.5 m controls** | that player's team (again needing 0.5 s if it differs from the last team) |
| 4 | **Initial point** (a set-piece restart) | the restarting team, immediately |
| 5 | **Ball missing < 2 s** | keep the last frame's value |
| 6 | **Ball missing ≥ 2 s** | classifier, one majority label for the whole stretch |

**Never a change of possession:** a touch, a deflection, an aerial duel without control, or a
one-frame control blip (all shorter than 0.5 s of control). A loose ball stays with the last team.

## Completed pass, sprint or carry (pass_sprint.py)

Only for a gap where nobody controls the ball, with the **same team on both sides**, no dead ball
inside, and no opponent near the ball for more than 0.2 s.

| Scenario | Rule |
|---|---|
| **Short touch** | gap ≤ 0.5 s |
| **Sprint** | same player at both ends, running ≥ 5 m/s, ball never > 5 m away, gap ≤ 4 s |
| **Pass** | ball travels ≥ 3 m at ≥ 4 m/s, gap ≤ 5 s (a one-two back to the passer counts) |
| **Carry / shielding** | same player at both ends, ball never > 2.5 m away, any speed, gap ≤ 4 s |

## Ball out of play (restart.py)

| Scenario | Rule |
|---|---|
| **Before kick-off, halftime, after the whistle** | outside the graded halves, or before the half's detected kick-off |
| **Waiting for a set piece** | inside a Smart Tagging set-piece window, before the detected restart |
| **Ball off the pitch** | ball > 1 m past a line for ≥ 3 frames; dead until a tagged restart, a kick from inside the pitch, or control while players are moving (within 90 s) |
| **Ball lost just before going out** | every ball-missing frame directly before a "ball off the pitch" stretch, back to the last frame the ball was seen (or an initial point, or the half's start). No length cap |
| **Untagged stoppage (foul, injury)** | players calm (< 1.2 m/s) and nobody playing the ball for ≥ 3 s, **or** players calm for ≥ 6 s whatever the ball does |
| **Live sliver** | live stretch < 2 s between two stoppages, with no tagged restart in it |

**Not a stoppage:**
- **Goalkeeper holding the ball:** last known ball (≤ 3 s old) in a penalty box, the nearest player is a goalkeeper within 2 m, and at most 2 players within 3 m.
- **After a tagged restart:** a calm-players stoppage ends at the first tagged restart inside it.

## Set-piece restarts (initial_points.py)

| Set piece | Where the restart must be | Restart frame | Team |
|---|---|---|---|
| **Kick-off** | ≤ 4 m from the centre spot | first frame there with the ball moving ≥ 5 m/s; otherwise the first frame there at any speed; otherwise the first frame with a player ≤ 2 m from the spot | after a goal: the team that conceded; at the start of a half: nearest player |
| **Throw-in** | ≤ 3 m from a touchline | dead ≥ 80% of the last 1 s, then kicked ≥ 5 m/s for 0.2 s | nearest player (≤ 3 m) to the throw's origin, traced back along the ball's path ≤ 1.5 s |
| **Corner kick** | ≤ 4 m from a corner flag | as throw-in | Smart Tagging team |
| **Goal kick** | in or ≤ 3 m around a goal area | as throw-in | Smart Tagging team |
| **Free kick** | anywhere | as throw-in | Smart Tagging team |

If no frame meets the place rule, it is dropped; then the dead-before rule is dropped. The CSV
records which rule fired.
