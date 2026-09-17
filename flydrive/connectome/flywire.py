"""Load a real FlyWire connectome from Codex CSV exports.

Download the snapshot yourself (registration required, data is CC-BY-4.0):

    https://codex.flywire.ai/api/download

You want two files from the same snapshot -- 783 is the published one:

    connections.csv      pre_root_id, post_root_id, neuropil, syn_count, nt_type
    classification.csv   root_id, super_class, class, cell_type, side, ...

Then:

    from flydrive.connectome.flywire import load_flywire
    conn = load_flywire("data/flywire_783")

Everything downstream is identical to the surrogate -- same ports, same
operator -- so a model trained on the surrogate can be re-fit on real wiring
without touching the training code.
"""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

import numpy as np

from .schema import NT_SIGN, Connectome

# Cell types that make up each functional port. Patterns are matched against
# the FlyWire ``cell_type`` field (falling back to ``class``), case-folded.
PORT_PATTERNS: dict[str, list[str]] = {
    "T4T5": [r"^t[45][a-d]?$"],
    "HS": [r"^hs[ens]?$"],
    "VS": [r"^vs\d*$"],
    "LPLC2": [r"^lplc2$", r"^lc4$"],
    "heading": [r"^epg$", r"^e-pg$"],
    "goal": [r"^fc2[a-c]?$"],
    "Delta7": [r"^delta7$", r"^d7$"],
    "PFL3": [r"^pfl3$"],
    "PFL2": [r"^pfl2$"],
    "context": [r"^[vd]?pn.*", r"^olfactory projection neuron$"],
    "KC": [r"^kc.*"],
    "APL": [r"^apl$"],
    "MBON": [r"^mbon\d+.*"],
    "reward": [r"^ppl1.*", r"^pam\d+.*"],
    "DNa02": [r"^dna02$"],
    "DNa01": [r"^dna01$"],
    "DNp09": [r"^dnp09$"],
}


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return open(path, "rt", newline="")


def _find(root: Path, stem: str) -> Path:
    for cand in (f"{stem}.csv", f"{stem}.csv.gz"):
        p = root / cand
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}.csv[.gz] not found in {root}")


def _match_port(cell_type: str, port: str) -> bool:
    return any(re.match(p, cell_type) for p in PORT_PATTERNS[port])


def load_flywire(
    root: str | Path,
    min_syn: int = 5,
    keep_super_classes: tuple[str, ...] | None = None,
) -> Connectome:
    """Build a :class:`Connectome` from Codex CSV exports in ``root``.

    Parameters
    ----------
    min_syn
        Discard connections with fewer synapses. FlyWire's own default
        threshold is 5, which is what the published analyses use.
    keep_super_classes
        Restrict to these ``super_class`` values (e.g. ``("central",
        "descending")`` to drop the optic lobes and halve the network). ``None``
        keeps everything.
    """
    root = Path(root)

    # ---------------------------------------------------- neuron annotations
    ids: list[int] = []
    cell_types: list[str] = []
    super_classes: list[str] = []
    sides: list[int] = []
    side_map = {"left": -1, "right": 1, "center": 0, "": 0}
    with _open(_find(root, "classification")) as fh:
        for row in csv.DictReader(fh):
            sc = (row.get("super_class") or "").strip().lower()
            if keep_super_classes and sc not in keep_super_classes:
                continue
            ids.append(int(row["root_id"]))
            ct = (row.get("cell_type") or "").strip()
            if not ct:
                ct = (row.get("hemibrain_type") or "").strip()
            if not ct:
                ct = (row.get("class") or "").strip() or sc or "unknown"
            cell_types.append(ct)
            super_classes.append(sc or "unknown")
            sides.append(side_map.get((row.get("side") or "").strip().lower(), 0))

    if not ids:
        raise ValueError(f"no neurons parsed from {root}; check the CSV layout")

    neuron_ids = np.array(ids, dtype=np.int64)
    order = np.argsort(neuron_ids)
    neuron_ids = neuron_ids[order]
    cell_types = [cell_types[i] for i in order]
    super_classes = [super_classes[i] for i in order]
    side = np.array(sides, dtype=np.int8)[order]
    index_of = {int(r): i for i, r in enumerate(neuron_ids)}

    type_names = sorted(set(cell_types))
    tmap = {t: i for i, t in enumerate(type_names)}
    type_ids = np.array([tmap[t] for t in cell_types], dtype=np.int32)
    region_names = sorted(set(super_classes))
    rmap = {r: i for i, r in enumerate(region_names)}
    region_ids = np.array([rmap[r] for r in super_classes], dtype=np.int32)

    # --------------------------------------------------------- connectivity
    pre_l: list[int] = []
    post_l: list[int] = []
    w_l: list[float] = []
    nt_l: list[str] = []
    with _open(_find(root, "connections")) as fh:
        for row in csv.DictReader(fh):
            syn = float(row["syn_count"])
            if syn < min_syn:
                continue
            a = index_of.get(int(row["pre_root_id"]))
            b = index_of.get(int(row["post_root_id"]))
            if a is None or b is None:
                continue
            pre_l.append(a)
            post_l.append(b)
            w_l.append(syn)
            nt_l.append((row.get("nt_type") or "unknown").strip().lower())

    pre = np.array(pre_l, dtype=np.int32)
    post = np.array(post_l, dtype=np.int32)
    weight = np.array(w_l, dtype=np.float32)
    sign = np.array([NT_SIGN.get(nt, 0) for nt in nt_l], dtype=np.int8)

    # A neuron releases one fast transmitter, so take the per-neuron majority
    # rather than trusting each edge's independent prediction.
    n_neurons = len(neuron_ids)
    pos = np.bincount(pre[sign > 0], minlength=n_neurons)
    neg = np.bincount(pre[sign < 0], minlength=n_neurons)
    consensus = np.zeros(n_neurons, dtype=np.int8)
    consensus[pos > neg] = 1
    consensus[neg > pos] = -1
    sign = np.where(consensus[pre] != 0, consensus[pre], sign).astype(np.int8)

    # ---------------------------------------------------------------- ports
    lowered = [t.lower() for t in cell_types]
    ports: dict[str, np.ndarray] = {}
    for port in PORT_PATTERNS:
        hits = np.array(
            [i for i, ct in enumerate(lowered) if _match_port(ct, port)], dtype=np.int32
        )
        if hits.size:
            ports[port] = hits

    # PFL3 is one cell type in FlyWire; the left/right populations that carry
    # the steering signal are separated by hemisphere.
    if "PFL3" in ports:
        p3 = ports["PFL3"]
        ports["PFL3L"] = p3[side[p3] < 0]
        ports["PFL3R"] = p3[side[p3] > 0]
    for src, dst in (("DNa02", "turn"), ("DNa01", "speed"), ("DNp09", "stop")):
        if src not in ports:
            continue
        grp = ports[src]
        if dst == "turn":
            ports["turn_L"] = grp[side[grp] < 0]
            ports["turn_R"] = grp[side[grp] > 0]
        else:
            ports[dst] = grp

    conn = Connectome(
        neuron_ids=neuron_ids,
        type_ids=type_ids,
        type_names=type_names,
        region_ids=region_ids,
        region_names=region_names,
        side=side,
        pre=pre,
        post=post,
        weight=weight,
        sign=sign,
        ports=ports,
        name=f"flywire({root.name})",
    )
    _check_ports(conn)
    return conn


def _check_ports(conn: Connectome) -> None:
    required = ["heading", "goal", "turn_L", "turn_R"]
    missing = [p for p in required if p not in conn.ports or conn.ports[p].size == 0]
    if missing:
        raise ValueError(
            f"connectome is missing required ports {missing}. The cell-type "
            "naming in your snapshot probably differs -- extend PORT_PATTERNS "
            "in flydrive/connectome/flywire.py to match it."
        )
