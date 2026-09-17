"""A structurally faithful surrogate Drosophila connectome.

The real FlyWire data cannot always be fetched (it needs registration, and this
repo must be runnable offline), so this module builds a scaled-down brain with
the *same architectural motifs* as the real one:

  optic lobe    T4/T5 elementary motion detectors -> HS/VS wide-field cells
                (optic flow), LPLC2 (looming).
  central cplx  EPG heading ring stabilised by Delta7 offset inhibition; FC2
                goal ring; PFL3 left/right populations reading EPG with equal
                and opposite anatomical shifts, so that their output difference
                encodes a *steering error*. This is the circuit that makes the
                whole project non-arbitrary -- the fly already has a steering
                controller and we are borrowing it.
  mushroom body PN -> KC sparse expansion coding, APL global inhibition for
                sparseness, KC -> MBON plastic readout, DANs carrying reward.
  premotor      PFL3 -> contralateral LAL -> DNa02 (turning); LPLC2 -> DNp09
                (stopping); DNa01 (walking speed).

Swap in the real connectome via ``flydrive.connectome.flywire.load_flywire``;
the downstream code does not care which one it got.
"""

from __future__ import annotations

import numpy as np

from .schema import NT_SIGN, Connectome

# Cell type -> neurotransmitter, following the published annotations for these
# populations. KC/EPG/PFL3/PN are cholinergic; Delta7 and APL are inhibitory.
TYPE_NT: dict[str, str] = {
    "T4T5": "acetylcholine",
    "HS": "acetylcholine",
    "VS": "acetylcholine",
    "LPLC2": "acetylcholine",
    "EPG": "acetylcholine",
    "Delta7": "gaba",
    "FC2": "acetylcholine",
    "PFL3L": "acetylcholine",
    "PFL3R": "acetylcholine",
    "PFL2": "acetylcholine",
    "PN": "acetylcholine",
    "KC": "acetylcholine",
    "APL": "gaba",
    "MBON": "glutamate",
    "DAN": "dopamine",
    "AN": "acetylcholine",
    "LAL": "acetylcholine",
    "LALi": "gaba",
    "DNa02": "acetylcholine",
    "DNa01": "acetylcholine",
    "DNp09": "acetylcholine",
}

TYPE_REGION: dict[str, str] = {
    "T4T5": "OL", "HS": "OL", "VS": "OL", "LPLC2": "OL",
    "EPG": "CX", "Delta7": "CX", "FC2": "CX",
    "PFL3L": "CX", "PFL3R": "CX", "PFL2": "CX",
    "PN": "MB", "KC": "MB", "APL": "MB", "MBON": "MB", "DAN": "MB",
    "AN": "AN", "LAL": "LAL", "LALi": "LAL",
    "DNa02": "DN", "DNa01": "DN", "DNp09": "DN",
}


class _Builder:
    """Accumulates neurons and edges, then freezes them into a Connectome."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.types: list[str] = []
        self.regions: list[str] = []
        self._tname: list[str] = []
        self._rname: list[str] = []
        self._side: list[int] = []
        self.pre: list[np.ndarray] = []
        self.post: list[np.ndarray] = []
        self.weight: list[np.ndarray] = []
        self.groups: dict[str, np.ndarray] = {}

    def add(self, type_name: str, n: int, side: int = 0, key: str | None = None) -> np.ndarray:
        """Add ``n`` neurons of one type and return their indices."""
        start = len(self._tname)
        idx = np.arange(start, start + n, dtype=np.int32)
        self._tname.extend([type_name] * n)
        self._rname.extend([TYPE_REGION[type_name]] * n)
        self._side.extend([side] * n)
        self.groups[key or type_name] = idx
        return idx

    def connect(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        *,
        p: float = 1.0,
        syn: float = 10.0,
        jitter: float = 0.3,
        pairs: np.ndarray | None = None,
    ) -> None:
        """Connect ``src`` to ``dst``.

        ``pairs`` gives an explicit (E, 2) index array; otherwise every
        src x dst pair is drawn independently with probability ``p``.
        """
        if pairs is None:
            grid_pre = np.repeat(src, len(dst))
            grid_post = np.tile(dst, len(src))
            if p < 1.0:
                keep = self.rng.random(grid_pre.size) < p
                grid_pre, grid_post = grid_pre[keep], grid_post[keep]
        else:
            grid_pre, grid_post = pairs[:, 0], pairs[:, 1]
        if grid_pre.size == 0:
            return
        w = syn * np.exp(self.rng.normal(0.0, jitter, grid_pre.size))
        self.pre.append(grid_pre.astype(np.int32))
        self.post.append(grid_post.astype(np.int32))
        self.weight.append(w.astype(np.float32))

    def freeze(self, ports: dict[str, np.ndarray], name: str) -> Connectome:
        type_names = sorted(set(self._tname))
        region_names = sorted(set(self._rname))
        tmap = {t: i for i, t in enumerate(type_names)}
        rmap = {r: i for i, r in enumerate(region_names)}
        type_ids = np.array([tmap[t] for t in self._tname], dtype=np.int32)
        region_ids = np.array([rmap[r] for r in self._rname], dtype=np.int32)

        pre = np.concatenate(self.pre)
        post = np.concatenate(self.post)
        weight = np.concatenate(self.weight)
        sign = np.array(
            [NT_SIGN[TYPE_NT[self._tname[i]]] for i in pre], dtype=np.int8
        )
        return Connectome(
            neuron_ids=np.arange(len(self._tname), dtype=np.int64),
            type_ids=type_ids,
            type_names=type_names,
            region_ids=region_ids,
            region_names=region_names,
            side=np.array(self._side, dtype=np.int8),
            pre=pre,
            post=post,
            weight=weight,
            sign=sign,
            ports=ports,
            name=name,
        )


def _ring_pairs(src: np.ndarray, dst: np.ndarray, shift: int, spread: int = 1) -> np.ndarray:
    """Topographic ring connection: dst[k] <- src[(k + shift) +/- spread]."""
    n = len(dst)
    m = len(src)
    out = []
    for k in range(n):
        for d in range(-spread, spread + 1):
            out.append((src[(k + shift + d) % m], dst[k]))
    return np.array(out, dtype=np.int32)


def build_surrogate(
    scale: float = 1.0,
    n_wedges: int = 16,
    pfl3_shift: int = 4,
    seed: int = 0,
) -> Connectome:
    """Build the surrogate connectome.

    Parameters
    ----------
    scale
        Multiplies the size of the large populations (optic lobe, Kenyon
        cells). ``scale=1`` gives ~2k neurons, which simulates at several
        thousand steps/second on one core.
    n_wedges
        Angular resolution of the CX rings. The real ellipsoid body has 16
        wedges.
    pfl3_shift
        Anatomical offset, in wedges, between the EPG heading input to the
        left and right PFL3 populations. ``n_wedges // 4`` is the 90 degree
        offset that makes ``PFL3R - PFL3L`` proportional to the sine of the
        heading error.
    """
    rng = np.random.default_rng(seed)
    b = _Builder(rng)

    n_t4t5 = max(8, int(200 * scale))
    n_kc = max(32, int(1500 * scale))
    n_pn = max(8, int(50 * scale))
    # The lateral accessory lobe is a substantial neuropil in the real brain,
    # and it is the sole premotor relay between the central complex and the
    # descending neurons. Model it too narrowly and it -- not the DN population
    # -- becomes the information bottleneck: with 20 cells a side, a linear
    # readout of the DNs recovers only R^2=0.6 of a competent steering policy,
    # while the same readout taken across the whole network recovers 0.96.
    n_lal = max(8, int(60 * scale))

    # ------------------------------------------------------------ optic lobe
    t4t5, hs, vs, lplc2 = {}, {}, {}, {}
    for side, tag in ((-1, "L"), (+1, "R")):
        t4t5[tag] = b.add("T4T5", n_t4t5, side, key=f"T4T5_{tag}")
        hs[tag] = b.add("HS", 3, side, key=f"HS_{tag}")     # HSN, HSE, HSS
        vs[tag] = b.add("VS", 10, side, key=f"VS_{tag}")    # VS1-VS10
        lplc2[tag] = b.add("LPLC2", 6, side, key=f"LPLC2_{tag}")

    # ---------------------------------------------------------- central cplx
    epg = b.add("EPG", n_wedges)
    d7 = b.add("Delta7", n_wedges // 2)
    fc2 = b.add("FC2", n_wedges)
    pfl3l = b.add("PFL3L", n_wedges)
    pfl3r = b.add("PFL3R", n_wedges)
    pfl2 = b.add("PFL2", n_wedges)

    # --------------------------------------------------------- mushroom body
    pn = b.add("PN", n_pn)
    kc = b.add("KC", n_kc)
    apl = b.add("APL", 1)
    mbon = b.add("MBON", 34)
    dan = b.add("DAN", 20)

    # --------------------------------------------------------------- premotor
    # Ascending neurons. Several hundred real ANs carry self-motion and leg
    # state from the ventral nerve cord up to the LAL and central complex.
    # They are the fly's proprioceptive channel, and without an equivalent
    # every scrap of information about speed and slip has to squeeze through
    # three HS cells per side, which is nowhere near enough to drive on.
    an = {}
    lal, lali = {}, {}
    dna02, dna01 = {}, {}
    for side, tag in ((-1, "L"), (+1, "R")):
        an[tag] = b.add("AN", max(8, int(40 * scale)), side, key=f"AN_{tag}")
        lal[tag] = b.add("LAL", n_lal, side, key=f"LAL_{tag}")
        lali[tag] = b.add("LALi", n_lal // 2, side, key=f"LALi_{tag}")
        # The real fly has on the order of 1,300 descending neurons. Modelling
        # the steering/speed/stop channels as a handful each makes the motor
        # bus a far narrower bottleneck than the animal's, and the readout
        # cannot represent a driving policy through it.
        dna02[tag] = b.add("DNa02", max(4, int(12 * scale)), side, key=f"DNa02_{tag}")
        dna01[tag] = b.add("DNa01", max(4, int(12 * scale)), side, key=f"DNa01_{tag}")
    dnp09 = b.add("DNp09", max(4, int(12 * scale)))

    # === wiring =========================================================

    # Optic lobe hierarchy: T4/T5 pool onto wide-field cells on the same side.
    for tag in ("L", "R"):
        b.connect(t4t5[tag], hs[tag], p=0.3, syn=8.0)
        b.connect(t4t5[tag], vs[tag], p=0.3, syn=8.0)
        b.connect(t4t5[tag], lplc2[tag], p=0.15, syn=6.0)

    # Ring attractor. Local excitation between neighbouring EPG wedges, and
    # Delta7 inhibition offset by half a ring -- together these hold a single
    # stable heading bump.
    #
    # These connections are built with almost no weight jitter, and that is a
    # load-bearing detail rather than a convenience. A ring attractor only
    # reports heading *relative to goal* if it is translation invariant around
    # the ring. Give the wedges randomly unequal gains and the PFL3 difference
    # starts depending on absolute heading, which sweeps a full turn every lap
    # and completely swamps the few degrees of steering error you wanted to
    # read. The real central complex is correspondingly stereotyped.
    ring_jitter = 0.05
    b.connect(epg, epg, pairs=_ring_pairs(epg, epg, 0, spread=1), syn=14.0, jitter=ring_jitter)
    b.connect(epg, d7, pairs=np.array(
        [(epg[i], d7[i % len(d7)]) for i in range(len(epg))], dtype=np.int32),
        syn=12.0, jitter=ring_jitter)
    far = []
    for j in range(len(d7)):
        centre = j * len(epg) / len(d7)
        for k in range(len(epg)):
            ang = abs(((k - centre + len(epg) / 2) % len(epg)) - len(epg) / 2)
            if ang > len(epg) / 4:
                far.append((d7[j], epg[k]))
    b.connect(d7, epg, pairs=np.array(far, dtype=np.int32), syn=9.0, jitter=ring_jitter)

    # The steering computation. Both PFL3 populations read the goal ring
    # straight, but sample the heading ring with opposite shifts.
    # Sign convention. Three separate lateralities have to compose here: which
    # PFL3 population is recruited by a given heading error, the fact that PFL3
    # projects to the *contralateral* LAL, and which way a DNa02 turns the
    # animal. Only their product is observable, and the product has to be
    # negative feedback -- a heading error left of goal must command a turn to
    # the right. The shift sign below is what closes that loop; if you change
    # the PFL3 -> LAL projection or the decoder laterality, this flips too, and
    # test_steering_output_opposes_heading_error is what catches it.
    b.connect(epg, pfl3r, pairs=_ring_pairs(epg, pfl3r, -pfl3_shift), syn=12.0, jitter=ring_jitter)
    b.connect(epg, pfl3l, pairs=_ring_pairs(epg, pfl3l, +pfl3_shift), syn=12.0, jitter=ring_jitter)
    b.connect(fc2, pfl3r, pairs=_ring_pairs(fc2, pfl3r, 0), syn=12.0, jitter=ring_jitter)
    b.connect(fc2, pfl3l, pairs=_ring_pairs(fc2, pfl3l, 0), syn=12.0, jitter=ring_jitter)
    # PFL2 reads heading vs goal symmetrically and modulates forward drive.
    b.connect(epg, pfl2, pairs=_ring_pairs(epg, pfl2, len(epg) // 2), syn=10.0, jitter=ring_jitter)
    b.connect(fc2, pfl2, pairs=_ring_pairs(fc2, pfl2, 0), syn=10.0, jitter=ring_jitter)

    # PFL3 -> LAL must also be symmetric between the hemispheres, for the same
    # reason: an asymmetric projection biases the steering readout.
    pfl3_to_lal_jitter = 0.05

    # PFL3 projects to the *contralateral* LAL, which is what converts a
    # population difference into a left/right asymmetry at the DNs.
    b.connect(pfl3r, lal["L"], p=1.0, syn=10.0, jitter=pfl3_to_lal_jitter)
    b.connect(pfl3l, lal["R"], p=1.0, syn=10.0, jitter=pfl3_to_lal_jitter)
    for tag, other in (("L", "R"), ("R", "L")):
        b.connect(lal[tag], dna02[tag], p=1.0, syn=16.0)
        b.connect(lal[tag], lali[tag], p=0.6, syn=8.0)
        b.connect(lali[tag], lal[other], p=0.6, syn=8.0)   # mutual inhibition
        b.connect(hs[tag], lal[tag], p=1.0, syn=10.0)      # optomotor feedback
        b.connect(vs[tag], lal[tag], p=0.6, syn=7.0)
        b.connect(vs[tag], dna01[tag], p=1.0, syn=6.0)
        b.connect(lplc2[tag], dnp09, p=1.0, syn=14.0)      # looming -> stop
        b.connect(lplc2[tag], lal[tag], p=0.5, syn=7.0)
        b.connect(pfl2, dna01[tag], p=0.4, syn=8.0)
        # Ascending self-motion input to premotor and central complex.
        b.connect(an[tag], lal[tag], p=0.5, syn=12.0)
        b.connect(an[tag], dna01[tag], p=0.5, syn=10.0)
        b.connect(an[tag], dna02[tag], p=0.4, syn=8.0)
        b.connect(an[tag], dnp09, p=0.4, syn=9.0)
        b.connect(an[tag], pfl2, p=0.3, syn=6.0)

    # Mushroom body. Each Kenyon cell samples a handful of PNs at random --
    # the sparse expansion that makes contexts linearly separable.
    claws = 6
    src = pn[rng.integers(0, len(pn), size=(len(kc), claws))]
    pairs = np.stack([src.ravel(), np.repeat(kc, claws)], axis=1).astype(np.int32)
    b.connect(pn, kc, pairs=pairs, syn=12.0)
    b.connect(kc, apl, p=1.0, syn=1.0)
    b.connect(apl, kc, p=1.0, syn=3.0)     # global gain control -> sparseness
    b.connect(kc, mbon, p=1.0, syn=4.0)    # plastic; owned by learn.mb
    b.connect(dan, mbon, p=0.3, syn=4.0)
    b.connect(mbon, dan, p=0.2, syn=4.0)
    # MBON output steers by biasing the LAL, which is how a learned valence
    # turns into a motor bias.
    for tag in ("L", "R"):
        b.connect(mbon, lal[tag], p=0.5, syn=6.0)
        b.connect(mbon, dna01[tag], p=0.3, syn=5.0)

    ports = dict(b.groups)
    ports.update(
        {
            "heading": epg,
            "goal": fc2,
            "context": pn,
            "reward": dan,
            "T4T5": np.concatenate([t4t5["L"], t4t5["R"]]),
            "LPLC2": np.concatenate([lplc2["L"], lplc2["R"]]),
            "AN": np.concatenate([an["L"], an["R"]]),
            "KC": kc,
            "MBON": mbon,
            "turn_L": dna02["L"],
            "turn_R": dna02["R"],
            "speed": np.concatenate([dna01["L"], dna01["R"]]),
            "stop": dnp09,
        }
    )
    return b.freeze(ports, name=f"surrogate(scale={scale})")
