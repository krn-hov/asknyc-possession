"""The classifier: guess possession on frames the rules cannot decide.

Where it fits
-------------
The possession script first runs without a classifier. Frames it can decide (initial points,
restarts, controlled frames, completed passes/sprints) get a label, and the rest stay
unassigned, mostly frames where the ball is not tracked. Those decided frames are the
training data here. The trained model then labels the leftover frames on the next possession
pass ("else f(t_n) = classifier(t_n)").

Training labels never come from a human or from the ground truth. They are produced by our
own algorithm, which RULES.md allows ("pseudo-labels produced by an algorithm").

Models ("either try a KNN or try training a decision tree")
-----------------------------------------------------------
Both are implemented, chosen with --model:
  tree  DecisionTreeClassifier, depth-limited, with a minimum leaf size so it cannot memorise
        single frames, class-balanced, random_state fixed for reproducibility.
  knn   StandardScaler + KNeighborsClassifier (distance-weighted). Features are scaled first
        because KNN compares raw distances. Training frames are capped at KNN_MAX_TRAIN
        (a fixed random subsample) so prediction stays fast.

Features (one row per video frame)
----------------------------------
All positions are first put in Barcelona's attacking frame (tracking.to_barca_frame):
Barcelona always attack towards x = 105.3, so a zone means the same thing in both halves.

1. Players per zone: 40 features. For each of Pep Guardiola's 20 zones (5 lanes x 4 bands,
   see tracking.pitch_zone), the number of home players and the number of away players in
   it. This follows the plan's "use zone number instead of coordinates" and "how many home
   and away players are in each zone".
2. Ball zone: 22 features. The zone of the last known ball position as a one-hot vector (20),
   the seconds since the ball was last tracked (capped at BALL_STALE_CAP_S), and whether it is
   tracked now. A zone *number* is a category, not a quantity, so it is one-hot encoded:
   otherwise KNN would treat zone 12 as "close to" zone 13 and far from zone 2. The ball is
   missing on most frames we predict, so its last known zone plus how stale it is is the
   honest version of "where the ball is".
3. Velocities: 8 features. For each team, the mean velocity along the pitch (vx: positive =
   towards Athletic's goal) and across it (vy), the mean speed, and the number of players
   sprinting (speed > SPRINT_SPEED). A team that is attacking moves forward as a block.
4. Smart Tagging phases: 28 features. For each of the 14 tag codes, whether a phase of that
   code is active at this frame for Barcelona, and whether one is active for Athletic. Phases
   with Team = N/A are not used here.

Validation
----------
Neighbouring frames are almost identical, so a random train/test split would leak and look
far too good. The labelled frames are instead cut into BLOCK_S-second blocks of match time, and
every TEST_EVERY-th block is held out as the test set. The report gives test accuracy,
balanced accuracy and the confusion matrix. Unless --no-refit is given, the final model is
then refit on all labelled frames.

Usage (run from the KH folder)
--------------------------------
    # train on the possession script's first-pass labels (CSV: frame,label)
    python classifier.py --labels out/possession_pass1.csv --model tree

    # until possession_script.py exists, the labels can come from control.py instead
    # (controlled frames -> team in control). This is a stand-in for testing only.
    python classifier.py --labels-from-control --model both

Other steps use:
    classify(frames, kind="tree")  -> array of "H" / "A" (or "0") for the given video frames
    predict_all(kind="tree")       -> the same for every frame of the match

Outputs: cache/classifier_features.npz (feature matrix) and
cache/classifier_<model>.pkl (the trained model and its metadata).
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

import tracking as T

# --------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------
SPRINT_SPEED = 7.0        # m/s (~25 km/h), the usual sprint threshold in football analytics
BALL_STALE_CAP_S = 10.0   # seconds. Cap on "time since the ball was last tracked".
BLOCK_S = 60              # seconds per validation block
TEST_EVERY = 5            # every 5th block is test -> ~20% held out
RANDOM_STATE = 0          # fixed seed, so a rerun reproduces the same model
KNN_MAX_TRAIN = 40000     # cap on KNN training frames (fixed random subsample)

TREE_PARAMS = dict(max_depth=12, min_samples_leaf=50, class_weight="balanced",
                   random_state=RANDOM_STATE)
KNN_PARAMS = dict(n_neighbors=25, weights="distance")

# The 14 Smart Tagging codes listed in DATA.md (upper case, as in the file). Any other code
# found in the file is appended after these, so a changed export is not silently ignored.
TAG_CODES = [
    "BUILD UP", "PROGRESSION", "CREATING CHANCES",
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "ATTACKING TRANSITION", "DEFENSIVE TRANSITION", "SET PIECES", "GOALS", "LONG BALL",
    "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX",
]

CLASS_NAMES = {0: "H", 1: "A", 2: "0"}   # 2 = "neither team" (only with --include-zero)
FEATURES_FILE = os.path.join(T.CACHE_DIR, "classifier_features.npz")


def model_file(kind):
    return os.path.join(T.CACHE_DIR, "classifier_{}.pkl".format(kind))


# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------
def _zone_counts(data):
    """40 features: home and away player counts in each of the 20 zones."""
    players, team, half = data["players"], data["team"], data["half"]
    zones = T.pitch_zone(players[:, :, 0], players[:, :, 1], half[:, None])   # (n, 46), 0 = missing
    home = (team == T.HOME)[None, :]
    away = (team == T.AWAY)[None, :]
    cols, names = [], []
    for side, mask in (("H", home), ("A", away)):
        for z in range(1, T.N_ZONES + 1):
            cols.append(((zones == z) & mask).sum(axis=1))
            names.append("n{}_zone{:02d}".format(side, z))
    return np.stack(cols, axis=1).astype(np.float32), names


def _ball_features(data):
    """22 features: one-hot zone of the last known ball position, staleness, visible flag."""
    ball, half = data["ball"], data["half"]
    visible = ~np.isnan(ball[:, 0])
    last = pd.DataFrame(ball).ffill().to_numpy()                 # last known position
    zone = T.pitch_zone(last[:, 0], last[:, 1], half)            # 0 before the first sighting
    idx = np.arange(len(ball))
    last_seen = pd.Series(np.where(visible, idx, np.nan)).ffill().to_numpy()
    stale = np.where(np.isnan(last_seen), BALL_STALE_CAP_S, (idx - last_seen) / T.FPS)
    stale = np.minimum(stale, BALL_STALE_CAP_S)
    onehot = (zone[:, None] == np.arange(1, T.N_ZONES + 1)[None, :]).astype(np.float32)
    names = ["ball_zone{:02d}".format(z) for z in range(1, T.N_ZONES + 1)]
    feats = np.column_stack([onehot, stale, visible]).astype(np.float32)
    return feats, names + ["ball_stale_s", "ball_visible"]


def _velocity_features(data):
    """8 features: per team mean vx, mean vy, mean speed, sprinting count (Barcelona frame)."""
    vel, team, half = data["player_vel"], data["team"], data["half"]
    sign = np.where(half == 2, -1.0, 1.0)[:, None]               # 180-degree rotation in half 2
    vx, vy = vel[:, :, 0] * sign, vel[:, :, 1] * sign
    speed = np.hypot(vx, vy)
    cols, names = [], []
    for side, code in (("H", T.HOME), ("A", T.AWAY)):
        m = team == code
        cols += [_nanmean(vx[:, m]), _nanmean(vy[:, m]), _nanmean(speed[:, m]),
                 (np.nan_to_num(speed[:, m], nan=0.0) > SPRINT_SPEED).sum(axis=1)]
        names += ["{}_mean_vx".format(side), "{}_mean_vy".format(side),
                  "{}_mean_speed".format(side), "{}_n_sprinting".format(side)]
    return np.stack(cols, axis=1).astype(np.float32), names


def _nanmean(a):
    """Row mean ignoring NaN, 0 where a row has no values (no warning)."""
    ok = ~np.isnan(a)
    count = ok.sum(axis=1)
    return np.where(count > 0, np.where(ok, a, 0).sum(axis=1) / np.maximum(count, 1), 0.0)


def _tag_features(n, tags):
    """2 features per Smart Tagging code: phase active for Barcelona / for Athletic."""
    codes = TAG_CODES + sorted({t["code"] for t in tags} - set(TAG_CODES))
    cols = np.zeros((n, 2 * len(codes)), dtype=np.float32)
    names = []
    for i, code in enumerate(codes):
        names += ["tag_{}_H".format(code.lower().replace(" ", "_")),
                  "tag_{}_A".format(code.lower().replace(" ", "_"))]
    col_of = {code: i for i, code in enumerate(codes)}
    for t in tags:
        if t["team"] is None:
            continue
        c = 2 * col_of[t["code"]] + t["team"]
        cols[max(t["start_frame"], 0):min(t["end_frame"], n), c] = 1.0
    return cols, names


def build_features(data=None, tags=None, force=False):
    """The full feature matrix (n_frames x n_features) and its column names. Cached."""
    if os.path.exists(FEATURES_FILE) and not force:
        z = np.load(FEATURES_FILE, allow_pickle=False)
        return z["X"], list(z["names"])
    data = data or T.load_tracking()
    tags = tags or T.load_smart_tagging()
    parts = [_zone_counts(data), _ball_features(data), _velocity_features(data),
             _tag_features(len(data["ball"]), tags)]
    X = np.concatenate([p[0] for p in parts], axis=1)
    names = sum((p[1] for p in parts), [])
    np.savez_compressed(FEATURES_FILE, X=X, names=np.array(names))
    return X, names


# --------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------
def labels_from_csv(path, include_zero=False):
    """Per-frame class array from a frame,label CSV: 0 = H, 1 = A, 2 = neither, -1 = unused.

    "H" / "A" are possession. "0", "N" or "D" mean "neither team" and are only used with
    include_zero. Anything else (X, empty) is unlabelled.
    """
    df = pd.read_csv(path, dtype={"label": str})
    y = np.full(T.N_FRAMES, -1, dtype=np.int8)
    lab = df["label"].fillna("").str.strip().str.upper().to_numpy()
    frames = df["frame"].to_numpy()
    y[frames[lab == "H"]] = 0
    y[frames[lab == "A"]] = 1
    if include_zero:
        y[frames[np.isin(lab, ["0", "N", "D"])]] = 2
    return y


def labels_from_control():
    """Stand-in labels until possession_script.py exists: controlled frames -> team."""
    import control as C
    c = C.load_control()
    return np.where(c["control"], c["team"], -1).astype(np.int8)


# --------------------------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------------------------
def make_model(kind):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    if kind == "tree":
        return DecisionTreeClassifier(**TREE_PARAMS)
    if kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**KNN_PARAMS))
    raise ValueError("unknown model kind: {}".format(kind))


def _subsample(idx, kind):
    if kind == "knn" and len(idx) > KNN_MAX_TRAIN:
        rng = np.random.default_rng(RANDOM_STATE)
        return np.sort(rng.choice(idx, KNN_MAX_TRAIN, replace=False))
    return idx


def train(kind, X, y, refit=True):
    """Blocked train/test evaluation, then (optionally) refit on everything. Returns (model, report)."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    half = T.half_of_frame(len(y))
    labelled = np.flatnonzero((y >= 0) & (half > 0))
    if len(labelled) == 0:
        raise ValueError("no labelled graded frames to train on")
    block = labelled // (BLOCK_S * T.FPS)
    test_idx = labelled[block % TEST_EVERY == 0]
    train_idx = _subsample(labelled[block % TEST_EVERY != 0], kind)

    model = make_model(kind)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    classes = sorted(set(y[labelled]))
    report = {
        "kind": kind, "n_train": len(train_idx), "n_test": len(test_idx),
        "accuracy": accuracy_score(y[test_idx], pred),
        "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
        "classes": [CLASS_NAMES[c] for c in classes],
        "confusion": confusion_matrix(y[test_idx], pred, labels=classes),
    }
    if refit:
        all_idx = _subsample(labelled, kind)
        model = make_model(kind)
        model.fit(X[all_idx], y[all_idx])
    return model, report


def save_model(kind, model, names, report, label_source):
    with open(model_file(kind), "wb") as handle:
        pickle.dump({"model": model, "kind": kind, "feature_names": names,
                     "report": report, "label_source": label_source,
                     "params": TREE_PARAMS if kind == "tree" else KNN_PARAMS}, handle)


def load_classifier(kind="tree"):
    with open(model_file(kind), "rb") as handle:
        return pickle.load(handle)


# --------------------------------------------------------------------------------------------
# Prediction API for the possession script
# --------------------------------------------------------------------------------------------
_MODELS = {}


def classify(frames, kind="tree"):
    """Predicted labels ("H", "A", or "0") for the given video frames."""
    if kind not in _MODELS:
        _MODELS[kind] = load_classifier(kind)["model"]
    X, _ = build_features()
    pred = _MODELS[kind].predict(X[np.atleast_1d(frames)])
    return np.array([CLASS_NAMES[int(p)] for p in pred])


def predict_all(kind="tree"):
    """classify() for every frame of the match."""
    return classify(np.arange(T.N_FRAMES), kind)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _print_report(r):
    print("\n[{}] train {} frames, test {} frames (every {}th {}-s block held out)".format(
        r["kind"], r["n_train"], r["n_test"], TEST_EVERY, BLOCK_S))
    print("  accuracy {:.3f}   balanced accuracy {:.3f}".format(r["accuracy"], r["balanced_accuracy"]))
    print("  confusion (rows = label, cols = predicted): {}".format(" ".join(r["classes"])))
    for name, row in zip(r["classes"], r["confusion"]):
        print("    {:>2}  {}".format(name, "  ".join("{:>7}".format(v) for v in row)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", help="CSV frame,label from the possession script's first pass")
    src.add_argument("--labels-from-control", action="store_true",
                     help="stand-in labels from control.py (testing only)")
    ap.add_argument("--model", choices=["tree", "knn", "both"], default="tree")
    ap.add_argument("--include-zero", action="store_true",
                    help="also learn a 'neither team' class from labels 0/N/D")
    ap.add_argument("--no-refit", action="store_true",
                    help="keep the model trained on the train blocks only")
    ap.add_argument("--rebuild-features", action="store_true")
    args = ap.parse_args()

    X, names = build_features(force=args.rebuild_features)
    print("features: {} frames x {} columns".format(*X.shape))
    if args.labels:
        y, source = labels_from_csv(args.labels, args.include_zero), args.labels
    else:
        y, source = labels_from_control(), "control.py (stand-in)"
    print("label source: {}   labelled graded frames: {}".format(
        source, int(((y >= 0) & (T.half_of_frame(len(y)) > 0)).sum())))

    kinds = ["tree", "knn"] if args.model == "both" else [args.model]
    for kind in kinds:
        model, report = train(kind, X, y, refit=not args.no_refit)
        _print_report(report)
        save_model(kind, model, names, report, source)
        print("  saved {}".format(T.rel(model_file(kind))))
        if kind == "tree":
            imp = model.feature_importances_
            top = np.argsort(imp)[::-1][:10]
            print("  top features: " + ", ".join("{} ({:.2f})".format(names[i], imp[i]) for i in top))


if __name__ == "__main__":
    main()
