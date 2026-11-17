"""Shared data layer for the KH pipeline: tracking, velocities, Smart Tagging, pitch zones.

None of the process steps (initial_points, control, classifier, ...) parse files themselves.
They all call into this module so every step sees exactly the same arrays.

What it provides
----------------
load_tracking()        -> dict of numpy arrays for the *tactical* recording (the submission
                          timeline), indexed by video frame 0 .. 199318:

    ball      (n, 2)      ball position in metres, NaN where the tracker lost the ball
    players   (n, 46, 2)  player positions in metres, NaN where the slot is empty
    team      (46,)       0 = home (Barcelona), 1 = away (Athletic Club), per track slot
    ball_vel  (n, 2)      ball velocity in m/s (computed here, see below)
    ball_speed(n,)        |ball_vel|
    player_vel(n, 46, 2)  player velocity in m/s (computed here)
    half      (n,)        1 or 2 inside a graded half, 0 elsewhere

load_smart_tagging()   -> list of tag dicts, frames already converted to video frames.
pitch_zone(x, y, half) -> Pep's 20-zone index (1..20), relative to Barcelona's attack.

Facts about the raw data this relies on (checked against the files in tactical/)
------------------------------------------------------------------------------
* Raw ATD line format:  "<frame_id>:<p1>;<p2>;...;<p46>:<ball>"
  each player is "x,y,speed", the ball is "x,y,speed". Missing values are the text "NaN".
* frame_id runs 1..199319, so video_frame = frame_id - 1 (DATA.md).
* Player channel order in the XML is player1 .. player46. Slots 1-24 belong to Barcelona and
  slots 25-46 to Athletic Club (teamId in the XML). Slots 24 and 46 are the goalkeepers.
* Coordinates are normalised 0..1. x runs along the length, y across the width. We convert to
  metres with the 105.3 x 68.0 m pitch from the organisers' games.py (not the XML's 105 x 68).
* Smart Tagging <start>/<end> are seconds from the first video frame, so
  frame = round(seconds * 25) (verified by the organisers' starter/load_smart_tagging.py).
* Barcelona attack left-to-right (towards x = 105.3) in the 1st half (XML
  attack_direction_first_half) and the other way in the 2nd half.
* The ball's own speed channel is filled on only ~21k of 199k frames, so velocities are
  computed here from positions instead.

Velocities
----------
Central difference over one frame either side, (p[t+1] - p[t-1]) * fps / 2. Where a
neighbour is missing it falls back to a one-sided difference, and it is NaN if both are
missing. Player velocities above PLAYER_MAX_SPEED (an identity switch teleporting a slot) are
set to NaN. Ball velocities above BALL_MAX_SPEED are set to NaN for the same reason (the
fastest shots are ~35 m/s).

Caching
-------
Parsing the 196 MB text file takes a while, so the parsed arrays go to
cache/tracking.npz on first use. Delete that file (or run main.py --fresh) to re-parse.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

# --------------------------------------------------------------------------------------------
# Paths. Everything is relative to this folder (KH), so the pipeline runs from a fresh clone
# with no other repository. The match data goes in KH/data (see data/README.md), or anywhere
# else if KH_DATA_DIR points at it.
# --------------------------------------------------------------------------------------------
KH_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("KH_DATA_DIR", os.path.join(KH_DIR, "data"))
CACHE_DIR = os.path.join(KH_DIR, "cache")
OUT_DIR = os.path.join(KH_DIR, "out")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# The three files the pipeline needs, named exactly as the organisers' download names them.
DATA_FILES = {
    "atd_raw": os.path.join("tactical", "BAR-ATH_tactical_atd.txt"),
    "atd_meta": os.path.join("tactical", "BAR-ATH_tactical_atd.xml"),
    "smart_tagging": os.path.join("tactical", "BAR-ATH_tactical_smart_tagging.xml"),
}


def data_path(key):
    """Absolute path of one input file, with a clear error if it is not there."""
    path = os.path.join(DATA_DIR, DATA_FILES[key])
    if not os.path.exists(path):
        raise FileNotFoundError(
            "missing input file: {}\nPut the match data in {} as described in "
            "data/README.md (or set KH_DATA_DIR).".format(path, DATA_DIR))
    return path


def rel(path):
    """A path relative to the KH folder, for printing."""
    return os.path.relpath(path, KH_DIR)


# --------------------------------------------------------------------------------------------
# Match constants, all published by the organisers in DATA.md and games.py (none are tuned).
# The pitch is 105.3 x 68.0 m as measured from the corrected tracking (games.py), not the
# 105 x 68 m written in the ATD metadata.
# --------------------------------------------------------------------------------------------
FPS = 25
N_FRAMES = 199319
PITCH_LENGTH = 105.3            # metres, x direction
PITCH_WIDTH = 68.0              # metres, y direction
N_SLOTS = 46
HOME_SLOTS = range(0, 24)       # 0-based indices of slots 1..24 -> Barcelona
AWAY_SLOTS = range(24, 46)      # slots 25..46 -> Athletic Club
GRADED_HALVES = {1: (11185, 90651), 2: (110808, 190197)}   # inclusive video frames

HOME, AWAY = 0, 1               # team codes used everywhere in KH
TEAM_LABEL = {HOME: "H", AWAY: "A"}
# Smart Tagging spells the clubs like this (note "Atletic", sic).
TAG_TEAM = {"barcelona": HOME, "atletic club": AWAY}

PLAYER_MAX_SPEED = 12.0         # m/s; faster than any sprint -> a slot identity jump
BALL_MAX_SPEED = 40.0           # m/s; faster than any shot -> a tracking jump

CACHE_FILE = os.path.join(CACHE_DIR, "tracking.npz")


# --------------------------------------------------------------------------------------------
# Raw ATD parsing
# --------------------------------------------------------------------------------------------
def _parse_atd(path):
    """Parse the raw ATD text into (ball_xy, players_xy) in normalised 0..1 units.

    Each line holds 1 frame id + 46 * 3 player values + 3 ball values = 142 numbers. Turning
    every separator (':' ';' ',' and newline) into a comma lets numpy read the whole file as
    one flat array of floats ("NaN" parses as nan), which is far faster than splitting line by line.
    """
    with open(path, "r") as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",").replace("\n", ",").strip(",")
    flat = np.array(text.split(","), dtype=np.float64)
    per_line = 1 + N_SLOTS * 3 + 3
    if flat.size % per_line:
        raise ValueError("unexpected ATD layout: {} values is not a multiple of {}".format(
            flat.size, per_line))
    rows = flat.reshape(-1, per_line)

    frame_id = rows[:, 0].astype(int)
    if not (frame_id[0] == 1 and np.all(np.diff(frame_id) == 1) and len(rows) == N_FRAMES):
        raise ValueError("ATD frame ids are not 1..{} contiguous".format(N_FRAMES))

    players = rows[:, 1:1 + N_SLOTS * 3].reshape(-1, N_SLOTS, 3)[:, :, :2]   # drop speed
    ball = rows[:, 1 + N_SLOTS * 3:1 + N_SLOTS * 3 + 2]
    return ball, players


def _team_per_slot(xml_path):
    """Read the team of each of the 46 track slots from the ATD metadata XML.

    The first <Team> in the file is the local (home) team. Returns an int array of 0/1.
    """
    root = ET.parse(xml_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team = np.array([HOME if s == home_id else AWAY for s in slots], dtype=np.int8)
    if len(team) != N_SLOTS:
        raise ValueError("expected {} slots, found {}".format(N_SLOTS, len(team)))
    return team


def goalkeeper_slots():
    """Bool array (46,): True for the track slots the ATD metadata marks as Goalkeeper.

    In this match those are slots 24 (Barcelona) and 46 (Athletic Club).
    """
    root = ET.parse(data_path("atd_meta")).getroot()
    gk = []
    for player in root.iter("Player"):
        kind = ""
        for param in player.iter("ProviderParameter"):
            if param.findtext("Name") == "position_type":
                kind = (param.findtext("Value") or "").strip().lower()
        gk.append(kind == "goalkeeper")
    return np.array(gk, dtype=bool)


def _velocity(pos, max_speed):
    """Velocity (m/s) of a position track along axis 0, tolerant of missing frames.

    pos has shape (n, ..., 2). Central difference where both neighbours exist, one-sided
    difference where only one does, NaN where neither does. Speeds above max_speed are NaN.
    """
    vel = np.full_like(pos, np.nan)
    fwd = np.full_like(pos, np.nan)
    bwd = np.full_like(pos, np.nan)
    fwd[:-1] = (pos[1:] - pos[:-1]) * FPS          # p[t+1] - p[t]
    bwd[1:] = (pos[1:] - pos[:-1]) * FPS           # p[t] - p[t-1]
    central = (fwd + bwd) / 2.0
    vel = np.where(np.isnan(central), np.where(np.isnan(fwd), bwd, fwd), central)
    speed = np.linalg.norm(vel, axis=-1)
    vel[speed > max_speed] = np.nan
    return vel


def half_of_frame(n=N_FRAMES):
    """1 / 2 for frames inside a graded half, 0 for pre-match, halftime and post-match."""
    half = np.zeros(n, dtype=np.int8)
    for h, (a, b) in GRADED_HALVES.items():
        half[a:b + 1] = h
    return half


def load_tracking(force=False):
    """Load (and cache) the tactical tracking as metre-scaled numpy arrays. See module doc."""
    if os.path.exists(CACHE_FILE) and not force:
        data = dict(np.load(CACHE_FILE))
    else:
        ball, players = _parse_atd(data_path("atd_raw"))
        scale = np.array([PITCH_LENGTH, PITCH_WIDTH])
        ball = ball * scale
        players = players * scale
        data = {
            "ball": ball.astype(np.float32),
            "players": players.astype(np.float32),
            "team": _team_per_slot(data_path("atd_meta")),
        }
        data["ball_vel"] = _velocity(ball, BALL_MAX_SPEED).astype(np.float32)
        data["player_vel"] = _velocity(players, PLAYER_MAX_SPEED).astype(np.float32)
        np.savez_compressed(CACHE_FILE, **data)
    data["ball_speed"] = np.linalg.norm(data["ball_vel"], axis=1)
    data["half"] = half_of_frame(len(data["ball"]))
    return data


# --------------------------------------------------------------------------------------------
# Smart Tagging
# --------------------------------------------------------------------------------------------
def load_smart_tagging():
    """All Smart Tagging instances as dicts, with frames on the video timeline.

    Keys: id, code (upper case), start_frame, end_frame (exclusive), tags (dict), team
    (HOME / AWAY / None for "N/A"). The XML clock is seconds from the first video frame, so
    frame = round(seconds * 25) (verified by the host's load_smart_tagging.py).
    Codes and tag values are compared case-insensitively, as DATA.md advises.
    """
    path = data_path("smart_tagging")
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        team_text = (tags.get("Team") or "").strip().lower()
        out.append({
            "id": int(inst.findtext("ID")),
            "code": (inst.findtext("code") or "").strip().upper(),
            "start_frame": int(round(float(inst.findtext("start")) * FPS)),
            "end_frame": int(round(float(inst.findtext("end")) * FPS)),
            "tags": tags,
            "team": TAG_TEAM.get(team_text),
        })
    return out


# --------------------------------------------------------------------------------------------
# Pep Guardiola's 20 zones (layout confirmed with the team: 5 lanes x 4 bands)
# --------------------------------------------------------------------------------------------
# Lanes across the width, split at the penalty-box width (40.32 m) and the six-yard-box width
# (18.32 m), both centred on the pitch:  wing | half-space | centre | half-space | wing.
_BOX_HALF_W = 40.32 / 2
_SIX_HALF_W = 18.32 / 2
LANE_EDGES_Y = np.array([PITCH_WIDTH / 2 - _BOX_HALF_W, PITCH_WIDTH / 2 - _SIX_HALF_W,
                         PITCH_WIDTH / 2 + _SIX_HALF_W, PITCH_WIDTH / 2 + _BOX_HALF_W])
# Bands along the length, split at both penalty-box lines (16.5 m from each goal line) and
# the halfway line: own box | own half | opponent half | opponent box.
BAND_EDGES_X = np.array([16.5, PITCH_LENGTH / 2, PITCH_LENGTH - 16.5])
N_ZONES = 20


def to_barca_frame(x, y, half):
    """Mirror coordinates in the 2nd half so Barcelona always attack towards x = 105.3.

    A 180-degree rotation (x -> L - x, y -> W - y), not a one-axis flip, so left stays left
    from Barcelona's point of view. Frames outside a half (half == 0) are left as they are.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    flip = np.asarray(half) == 2
    return np.where(flip, PITCH_LENGTH - x, x), np.where(flip, PITCH_WIDTH - y, y)


def pitch_zone(x, y, half):
    """Zone number 1..20 for positions in metres (0 where the position is missing).

    Numbering, in Barcelona's attacking frame (Barcelona attack left to right):

        band:      own box   own half   opp half   opp box
        lane 0         1         6         11        16
        lane 1         2         7         12        17      (half-space)
        lane 2         3         8         13        18      (centre)
        lane 3         4         9         14        19      (half-space)
        lane 4         5        10         15        20

        zone = band * 5 + lane + 1
    """
    bx, by = to_barca_frame(x, y, half)
    band = np.searchsorted(BAND_EDGES_X, bx, side="right")
    lane = np.searchsorted(LANE_EDGES_Y, by, side="right")
    zone = band * 5 + lane + 1
    missing = np.isnan(bx) | np.isnan(by)
    return np.where(missing, 0, zone).astype(np.int8)


if __name__ == "__main__":
    d = load_tracking()
    n = len(d["ball"])
    graded = d["half"] > 0
    print("frames {}  slots {}  home slots {}  away slots {}".format(
        n, d["players"].shape[1], int((d["team"] == HOME).sum()), int((d["team"] == AWAY).sum())))
    print("ball visible on {:.1%} of graded frames".format(
        (~np.isnan(d["ball"][graded, 0])).mean()))
    print("median ball speed while visible {:.2f} m/s".format(np.nanmedian(d["ball_speed"][graded])))
