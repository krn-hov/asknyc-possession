# ASKNYC: frame-level ball possession from automatic tracking

**More than a Hack 2027, Phase 1: Ball Possession.** Team ASKNYC.
Code: this repository. Reproduce with `python main.py` (see [Reproduction](#7-reproduction)).

## 1. Approach

We built a **rule-based pipeline with a small learned component**, not an end-to-end model.
There are two reasons. First, the ranking metric (spellF1) rewards getting each possession right
as a whole interval, and most of the difficulty is in a few situations: restarts, contested
balls, passes and a ball the tracker loses. Each of these has a football logic we can write
down and check. Second, the only labels we are allowed on this match are ones our own algorithm
produces, so a large learned model would be trained on our own rules anyway.

The pipeline answers four questions in order, then combines them frame by frame:

1. **When does play restart, and who restarts it?** (`initial_points.py`). These are anchors.
2. **Is a player controlling the ball right now, and which team?** (`control.py`)
3. **Is the ball out of play?** (`restart.py`)
4. **Was a gap with no control a completed pass, sprint or carry by one team?** (`pass_sprint.py`)

A frame-by-frame walk (`possession_script.py`) turns these into a possession state for every
frame. A decision tree (`classifier.py`), trained on the frames the walk decided, fills the long
stretches where the ball is not tracked. All thresholds are listed in
[`parameters.md`](parameters.md).

### 1.1 Restart anchors (initial points)

Every Smart Tagging *set piece* (59 in the match) is a window of about 35 s (5 s for
kick-offs), not an instant. The window starts while the ball is dead and runs on into live play,
so copying its team over the whole window would be wrong. Inside each window we find the
**restart frame**:

- **Kick-offs:** the first frame the ball is within 4 m of the centre spot and moving at
  ≥ 5 m/s. Centre-spot anchoring matters because before one kick-off the tracked ball glides
  smoothly at about 10 m/s for several seconds, so a speed-only rule fires early.
- **Throw-ins, corners, goal kicks and free kicks:** the first frame where the ball is kicked
  (≥ 5 m/s, sustained 0.2 s) after a dead second (≥ 80% of frames missing or slower than
  3 m/s, inside the window), from a legal place for that set piece (touchline, corner flag,
  goal area, anywhere for free kicks; 3 m tolerance). Weaker fallbacks are recorded when the
  strict rule fails.

The team comes from the tag when it has one. For a kick-off after a goal it is the conceding
team, taken from the GOALS tags. Otherwise (throw-ins and kick-offs are tagged `N/A`) it is the
team of the nearest player to the restart's origin. A throw-in's ball is often first tracked in
flight, so it is traced back along its velocity to the touchline. Result: **47 anchors** from 59
tags. 11 throw-ins had no player within 3 m and were left without a team rather than guessed.

### 1.2 Control

A numpy re-implementation of **Vidal-Codina et al. (2022)**, as implemented in databallpy 0.8.1,
with its default parameters. The nearest player within 1.5 m is a candidate. The gain is valid
only if the ball changes speed by > 5 m/s or direction by > 10° while in their zone. The player
keeps possession through their later touches. Our one change: possession is never carried
across a gap longer than 2 s.

We add three checks:

- **Dribbling:** the ball's velocity matches the player's within 3 m/s.
- **Reception:** the ball's speed drops by ≥ 4 m/s, to ≤ 50% of its speed in the previous 0.5 s.
- **Guarding:** when both teams are within 1.5 m, the controller must be ≥ 0.5 m closer than the
  nearest opponent and have stayed so for 0.4 s. A 50-50 is not control.

Result: the ball is controlled on 40.1% of the frames where it is tracked, in 1,108 spells.

### 1.3 Ball out of play

Seven traceable sources, each recorded per frame:

- **Outside the graded halves, and before each half's kick-off.** The kick-off frame is detected,
  not hard-coded.
- **Set-piece windows** before the detected restart.
- **Ball off the pitch:** more than 1 m past a line. The ball is dead until play resumes (a
  tagged restart, a kick from inside the pitch, or control while the players are moving). The
  untracked frames just before the ball is first seen outside are dead too, because the ball
  usually crosses the line while the tracker has lost it.
- **Untagged stoppages** (fouls, injuries, missed throw-ins): the players' mean speed is below
  1.2 m/s and nobody plays the ball for ≥ 3 s, or the players are that calm for ≥ 6 s whatever
  the ball does. A stoppage ends at a tagged restart. **A goalkeeper holding the ball in their
  box is not a stoppage**, even though everyone walks.
- **Live slivers** under 2 s between two stoppages count as dead.

The 1.2 m/s and 6 s values were set from our **own algorithmic labels**, never the ground truth.
Players' mean speed has a median of 0.92 m/s on frames our restart detector marks as waiting
for a set piece, and 1.93 m/s on frames our control step marks as controlled. Result: 39.7% of
graded frames are dead, in 124 stretches.

### 1.4 Passes and carries

Between two moments where the **same team** controls the ball, there is a gap where nobody does,
but possession has not changed. A gap is given to that team if it is:

- **a pass:** the ball travels ≥ 3 m at ≥ 4 m/s, within 5 s;
- **a sprint:** the same player runs at ≥ 5 m/s and the ball never gets more than 5 m away;
- **a carry:** the same player, with the ball within 2.5 m;
- **a short touch:** ≤ 0.5 s.

A gap is rejected if the ball went dead inside it or an opponent was near the ball for more
than 0.2 s. Result: 456 passes, 60 short touches, 17 carries and 7 sprints, covering 11.6% of
graded frames.

### 1.5 The possession walk

For each frame, in priority order:

1. A pass/carry frame takes its team.
2. A dead-ball frame is 0.
3. With the ball visible:
   - continued control keeps the team;
   - control by the other team **switches possession only after 0.5 s of their control within
     3 s**, back-dated to their first controlled frame;
   - no control gives 0.
4. Anchors take their known team.
5. A ball dropout shorter than 2 s keeps the previous value.
6. Longer dropouts go to the classifier.

The 0.5 s confirmation encodes the challenge's rule that a touch, a deflection or an aerial duel
without control does not change possession. With the 2 s dropout rule and the goalkeeper
exception, it cut possessions shorter than 1 s from 51 to 29 over the full match. Ending
stoppages at tagged restarts and removing live slivers then brought it to 10, and the ball-out
back-fill to 9. The final file has 339 possessions with a median of 7.1 s.

### 1.6 Classifier for ball-missing stretches

A decision tree (depth ≤ 12, ≥ 50 frames per leaf, balanced classes, fixed seed) predicts H or A
from 98 features:

- **players per zone:** home and away player counts in each of Pep Guardiola's 20 zones (5
  lanes × 4 bands, oriented to Barcelona's attack);
- **ball zone:** the ball's last known zone, and how long ago it was seen;
- **team movement:** each team's mean velocity and number of sprinting players;
- **tactical phase:** which Smart Tagging phases are active for each team.

It is trained on the 51,361 frames the first walk decided. Tested on held-out 60-second blocks
(to avoid leakage between neighbouring frames), its accuracy is 0.74. Each stretch of missing
ball gets **one majority label**, because a frame-by-frame guess flickers and splits
possessions. A KNN variant is also implemented (`--model knn`).

### 1.7 Writing the submission

- Dead frames become `D`.
- Live frames nobody controls take the last team in possession, because a loose ball stays
  with the last team. Right after a restart they take the next team in control instead.
- Frames outside the halves become `X`.

## 2. Assets used and dropped

| Asset | Used? | How / why not |
|---|---|---|
| Tactical ATD (tracking) | **Yes** | positions and velocities of all players and the ball; team per slot and goalkeeper slots from the metadata |
| Tactical Smart Tagging | **Yes** | set pieces and goals for anchors; phases as classifier features |
| Tactical homography | No | the tracking is already in pitch coordinates; we never needed pixels |
| Tactical video | No | not needed for a tracking-based method |
| Broadcast ATD / homography / video | No | not synchronised to the tactical timeline, with no mapping provided, and a third of its frames have no usable homography |
| Organisers' public 5-minute ground truth | Not in the pipeline | never read by any script; used only to check that the pipeline ran correctly |

We also tried a team-mate's earlier pipeline (nearest team with a 2-second hysteresis, and a
decision tree for ball dropouts). Three of its ideas were adopted: confirming possession
switches, holding the team through short dropouts, and the goalkeeper exception.

## 3. Training data

The only learned component is the classifier. It is trained exclusively on **pseudo-labels
produced by our own rules** (step 1.5, first pass) on the provided match. **No human labelled
anything**, on this match or any other, and no ground truth, event data or corrected tracking
was used. No external data was used.

## 4. Missing and noisy tracking

- **Ball missing on 41% of graded frames.**
  - Dropouts shorter than 2 s keep the previous possession.
  - Longer ones go to the classifier, one label per stretch.
  - Missing frames just before the ball is seen outside the pitch count as dead.
- **No player identities; slots swap.**
  - Every rule works at team level or on short time windows.
  - Player velocities above 12 m/s (a slot jumping to another player) are discarded.
  - In pass/sprint detection, two slots within 1.5 m of each other count as the same player.
- **Noisy ball speed.**
  - The ball's speed channel is mostly empty, so velocities are computed from positions.
  - Ball speeds above 40 m/s are discarded.
  - Kicks must stay above speed for 0.2 s.
  - Ball-out needs 3 consecutive frames.
- **Tracker artefacts.** Before one kick-off the ball glides smoothly with no real kick behind
  it, so kick-offs are anchored to the centre spot.

## 5. Possession rules implemented

- **Loose balls:** a loose or contested ball stays with the last team in possession.
- **Touches, deflections and aerial duels:** a touch, a deflection, an aerial duel without
  control or a contested 50-50 never changes possession. The new team must control the ball for
  0.5 s, and a controller who is not clearly closer than an opponent does not count.
- **Challenges:** a tackle that wins controlled possession switches the team from the moment
  control starts (back-dated).
- **Restarts:** set pieces give possession to the restarting team at the detected restart
  frame. Dead-ball time is always `D`, never possession. Dead-ball ownership is not modelled, as
  the challenge asks.
- **Passes, sprints and carries** between teammates keep possession even though nobody
  controls the ball.

## 6. Limitations

- **Untagged stoppages are the weakest part.** Dead balls are inferred from players slowing
  down. Quick throw-ins and free kicks, where nobody slows down, are missed.
- **Timing of possession changes.** The 0.5 s confirmation removes most false changes but can
  blur the exact moment of a real one. The back-dating helps but doesn't fix it fully.
- **The ball-out back-fill has no cap.** If the tracker loses the ball for a long time before
  it is seen outside, the whole stretch is marked dead, including some live play. A cap of a few
  seconds is the obvious next step.
- **Classifier accuracy is 0.74**, and on ball-missing stretches it carries a lot of weight.
- **11 throw-ins have no inferred team**, because the thrower is often not tracked off the pitch.

**With more time:**
- **Ball trajectory:** model the ball's path through dropouts, e.g. with a Kalman filter over
  the tracked positions.
- **Better stoppage detection:** use the video for stoppages (referee whistle and camera cuts
  are visible), and use the homography to read ball height from the image (lobbed balls and
  aerial duels).
- **Learned transitions:** replace the fixed thresholds with a learned state model over control
  events. That needs labelled data from other matches, which the rules allow.

## 7. Reproduction

Requirements: Python ≥ 3.10, and `pip install -r requirements.txt`, which pins numpy 2.2.6,
pandas 2.3.3 and scikit-learn 1.6.1.

```bash
git clone <this repository> && cd <it>
pip install -r requirements.txt
# copy BAR-ATH_tactical_atd.txt, _atd.xml and _smart_tagging.xml into data/tactical/
python main.py
```

This writes `solution/ASKNYC.csv` in about 30 seconds and checks its format. The run is
deterministic (fixed seeds, no time-dependent logic): repeated runs from scratch produce a
byte-identical file, the one submitted (MD5 `84dcf2353a352a70c4e3a1d7ba1bbacf`).
`out/` keeps every intermediate decision as a CSV:
- every detected restart (`initial_points.csv`);
- every dead-ball stretch and why it is dead (`restart.csv`);
- every pass/carry decision with its measurements (`pass_sprint.csv`).
