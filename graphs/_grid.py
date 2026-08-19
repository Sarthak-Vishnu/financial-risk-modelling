import json
import re
from pathlib import Path

GRID_JSON = (Path(__file__).resolve().parent.parent
             / "phase5" / "out" / "stress_grid_val2025_fixed.json")

# The condition the whole grid is scored against. Its row supplies both the reference
# level and, through its own bootstrap interval, the single-year noise scale.
REFERENCE = "struct+tfidf [sparse]"

# The no-text structured baseline, drawn as the second reference line on the forest plot.
STRUCTURED = "structured [ridge]"

# ---- editorial rule 1: what belongs in a figure about encoders ---------------
#
# Excluded on purpose, with the reason. Anything here is a real row of the grid that
# these two figures deliberately do not draw.
EXCLUDED = {
    REFERENCE:                     "the reference itself, drawn as a line",
    STRUCTURED:                    "the no-text baseline, drawn as a line",
    "lagged [hgb]":                "persistence-only anchor, belongs to the anchor table",
    "structured [hgb]":            "anchor table",
    "tfidf+lag [sparse]":          "anchor table",
    "struct+change [hgb]":         "disclosure-change lane, not an encoder condition",
    "struct+tfidf+change [sparse]": "disclosure-change lane, not an encoder condition",
}

# ---- editorial rule 2: what each encoder was trained to do -------------------
#
# The three-way split is the chapter's argument on a y-axis. It was originally drawn as
# general-purpose versus task-aligned, on the reading that only the two encoders trained
# on the volatility target reached parity with the count model. The regenerated grid does
# not support that split: under the tree head the only conditions that fail parity are
# sbert and bge, and every encoder trained on this corpus reaches it whether or not it
# ever saw the volatility label. So the first cut is corpus exposure, and task alignment
# is the second cut inside the corpus-trained group.
OFF_SHELF = "Off the shelf, never trained on this corpus"
ADAPTED = "Contrastively adapted on this corpus"
TARGETED = "Trained on the volatility target"
FUSION = "Dense-projected text and full fusion"

FAMILY_ORDER = [OFF_SHELF, ADAPTED, TARGETED, FUSION]

ENCODER_FAMILY = {
    "sbert":      OFF_SHELF,
    "bge":        OFF_SHELF,
    "dual":       ADAPTED,
    "three":      ADAPTED,
    "three_lora": ADAPTED,
    "volaware":   TARGETED,
    "ftvol":      TARGETED,
}

# Conditions that carry text but no single named encoder.
NON_ENCODER_FAMILY = {
    "struct+tfidf_svd [ridge]": FUSION,
    "struct+tfidf_svd [hgb]":   FUSION,
}

# EVERYTHING rows are grouped as fusion whatever encoder they name, because what defines
# them is that they stack every representation block at once.
#
# The attention-pooling rows sit with the encoder they name, which reads naturally next
# to that encoder's mean-pooled rows. Note what the name is worth, though: both the
# EVERYTHING and the pooling rows pick their encoder by maximising tree-head IC on the
# very rows they are then scored on, so the identity is a within-sample artefact. It
# positions a marker here; it is not evidence about that encoder.
_EVERYTHING = re.compile(r"^EVERYTHING ")
_ENC = re.compile(r"enc\[([A-Za-z0-9_]+)")

def _family_of(name):
    if _EVERYTHING.match(name):
        return FUSION
    if name in NON_ENCODER_FAMILY:
        return NON_ENCODER_FAMILY[name]
    m = _ENC.search(name)
    if m:
        enc = m.group(1)
        if enc not in ENCODER_FAMILY:
            raise ValueError(
                f"condition {name!r} names encoder {enc!r}, which ENCODER_FAMILY in "
                f"graphs/_grid.py does not cover. Add it to a family (or to EXCLUDED) "
                f"rather than leaving the figure to drop the row."
            )
        return ENCODER_FAMILY[enc]
    return None

def _head_of(name):
    m = re.search(r"\[(ridge|hgb|sparse)\]\s*$", name)
    return m.group(1) if m else ""

def load_grid():
    with open(GRID_JSON) as fh:
        doc = json.load(fh)
    out = {}
    for row in doc["rows"]:
        lo, hi = row["ic_ci"]
        out[row["name"]] = {
            "name": row["name"],
            "ic": row["ic"],
            "ci_lo": lo,
            "ci_hi": hi,
            "r2": row["r2"],
            "dm_p": row.get("dm_p_vs_ref"),
            "alpha": row.get("alpha"),
        }
    for required in (REFERENCE, STRUCTURED):
        if required not in out:
            raise ValueError(f"{GRID_JSON} has no {required!r} row")
    return out

def encoder_conditions(grid=None):
    grid = grid if grid is not None else load_grid()
    kept, unplaced = [], []
    for name, row in grid.items():
        if name in EXCLUDED:
            continue
        fam = _family_of(name)
        if fam is None:
            unplaced.append(name)
            continue
        kept.append({**row, "family": fam})
    if unplaced:
        raise ValueError(
            "graphs/_grid.py covers neither a family nor an exclusion for: "
            + ", ".join(repr(n) for n in sorted(unplaced))
            + ". Place each in ENCODER_FAMILY / NON_ENCODER_FAMILY or in EXCLUDED."
        )
    return kept

def by_family(grid=None):
    rows = encoder_conditions(grid)
    groups = []
    for fam in FAMILY_ORDER:
        members = sorted((r for r in rows if r["family"] == fam),
                         key=lambda r: (r["ic"], _head_of(r["name"])))
        if members:
            groups.append((fam, members))
    placed = sum(len(m) for _, m in groups)
    if placed != len(rows):
        raise ValueError(f"FAMILY_ORDER omits a family: placed {placed} of {len(rows)} rows")
    return groups

def noise_halfwidth(grid=None):
    grid = grid if grid is not None else load_grid()
    ref = grid[REFERENCE]
    return (ref["ci_hi"] - ref["ci_lo"]) / 2.0
