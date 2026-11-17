# Ball possession, frame by frame (More than a Hack 2027, Phase 1)

For every one of the 199,319 frames of FC Barcelona vs Athletic Club, this pipeline says who has
the ball: `H` (Barcelona), `A` (Athletic Club), `D` (ball out of play) or `X` (not graded). It uses
the organisers' automatic tracking and Smart Tagging only. Every label comes out of the code below;
nothing is labelled by hand.

Running the pipeline writes the solution to `solution/ASKNYC.csv`. The approach is described in
the technical document, [`ASKNYC.md`](ASKNYC.md).

---

## Run it (3 steps, about 30 seconds)

You need **Python 3.10 or newer**.

**1. Clone and install the three dependencies** (a virtual environment is recommended)

```bash
git clone https://github.com/<owner>/asknyc-possession.git
cd asknyc-possession
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add the match data** (not included, it is licensed). Copy these three files from the
organisers' download into `data/tactical/`. [`data/README.md`](data/README.md) has the details:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

**3. Run**

```bash
python main.py
```

That's it. The solution is written to `solution/ASKNYC.csv`, and its format is checked
automatically (199,319 rows, frames 0..199318 in order, labels `H`/`A`/`D`/`X`). Running it again
gives a byte-identical file.

Optional:

```bash
python main.py --fresh          # wipe cache/ and out/ first (re-parse the tracking too)
python main.py --model knn      # use the KNN classifier instead of the decision tree
python main.py --team MyTeam    # write solution/MyTeam.csv instead
```

To double-check with the organisers' own validator, clone
[their repository](https://github.com/metrica-sports/more-than-a-hack-2027) and run:

```bash
python <path-to>/more-than-a-hack-2027/evaluation/validate_submission.py solution/ASKNYC.csv
```

---

## How it works

Each step is one `.py` file, which explains its own logic at the top. `main.py` runs them in this
order:

| # | Step | File | What it does |
|---|---|---|---|
| 1 | Initial points | `initial_points.py` | Every set piece in the Smart Tagging (kick-off, goal kick, corner, free kick, throw-in) is a ~35 s window. Inside it, find the frame where the ball is put back in play, and which team took it. These frames are certain anchors. |
| 2 | Control | `control.py` | Is a player controlling the ball? A numpy version of the Vidal-Codina et al. (2022) method used by databallpy (1.5 m zone; the ball must change speed by >5 m/s or direction by >10°). Our own checks are added on top: the ball moving with the player, a sharp speed drop (a first touch), and the closest player guarding the ball against an opponent. |
| 3 | Restarts | `restart.py` | Every frame where the ball is out of play: outside the halves, before kick-off, set-piece waits, the ball off the pitch (including the untracked frames just before it is seen outside), and untagged stoppages (players walking, nobody playing the ball). A goalkeeper holding the ball is *not* a stoppage, a stoppage ends at a tagged restart, and live slivers under 2 s between two stoppages are treated as dead. |
| 4 | Pass / sprint | `pass_sprint.py` | Between two moments when the *same* team controls the ball, nobody controls it but possession has not changed. Decide whether the gap was a completed pass, a sprint/carry or a short touch, and give it to that team. |
| 5 | Possession, first pass | `possession_script.py` | Walk frame by frame applying the possession rules (listed at the top of the file). Possession only passes to the other team after it has controlled the ball for 0.5 s (back-dated to the gain), and ball dropouts under 2 s keep the last team. Longer dropouts stay unassigned. |
| 6 | Classifier | `classifier.py` | Train a decision tree (or KNN) on the frames step 5 decided. Features are player counts in Pep Guardiola's 20 zones, the ball's last known zone, team velocities and active Smart Tagging phases. |
| 7 | Possession with classifier | `possession_script.py` | The same walk, with the classifier filling the ball-missing frames (one majority label per stretch, so it doesn't flicker). Its submission file becomes the solution. |

`tracking.py` is shared by all steps: it reads the data files, converts to metres, computes
velocities and defines the 20 pitch zones.

### Turning the frame states into the submission

The possession script works with "team", "0" (nobody has it) and "unassigned". The official
format has only `H`/`A`/`D`/`X`, so:

- frames where the ball is out of play → `D`
- frames nobody controls, while the ball is live → the team that had the ball last (a loose ball
  stays with the last team in possession). Straight after a restart, the team that controls it first
- frames outside the two graded halves → `X`

---

## What is where

```
./
├─ README.md               this file
├─ ASKNYC.md               the technical document
├─ parameters.md           every threshold, and the control / possession scenarios
├─ main.py                 runs everything in order
├─ *.py                    one file per step (+ tracking.py, shared)
├─ requirements.txt        pinned versions (numpy, pandas, scikit-learn)
├─ data/                   put the match data here (empty in git)
├─ solution/               main.py writes the solution here (ASKNYC.csv)
├─ out/                    per-step results, for inspecting decisions (created when run)
│   ├─ initial_points.csv  every set piece: detected restart frame, team, how it was found
│   ├─ restart.csv         every dead-ball stretch and why it is dead
│   ├─ pass_sprint.csv     every gap considered: measurements and the decision
│   └─ final.csv           the final frame states, before conversion to the submission
└─ cache/                  intermediate arrays (created when run, safe to delete)
```

## Rules we follow

- **No hand labels.** Every label is produced by code. The classifier is trained only on labels
  our own rules produced (algorithmic pseudo-labels, allowed by RULES.md).
- **No ground truth in the pipeline.** The organisers' public 5-minute ground truth is never read
  by any script. It was used only to check the pipeline was wired correctly.
- **No data in the repository.** `data/`, `out/` and `cache/` are git-ignored, and so is the
  generated solution file (it is uploaded through the submission form instead).
- **Reproducible.** Fixed random seeds, pinned dependencies, and the same file on every run.
