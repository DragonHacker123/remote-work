"""Closed race tracks with Frenet-frame projection.

A track is a centreline resampled at uniform arc length, plus a half-width.
Everything the controller sees about the road (lateral offset, heading error,
curvature preview) is derived from projecting the car onto that centreline.

The same representation is used for the F1 25 bridge: record a centreline from
the game's own telemetry once, and the observation vector the network sees in
the game is computed exactly as it is here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _catmull_rom(points: np.ndarray, per_segment: int = 40) -> np.ndarray:
    """Closed Catmull-Rom spline through ``points`` (M, 2)."""
    m = len(points)
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(m):
        p0, p1, p2, p3 = (points[(i + k - 1) % m] for k in range(4))
        out.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
        )
    return np.concatenate(out, axis=0)


@dataclass
class Track:
    """A closed circuit sampled at uniform arc length ``ds``."""

    xy: np.ndarray        # (N, 2) centreline points
    psi: np.ndarray       # (N,) tangent heading
    kappa: np.ndarray     # (N,) signed curvature, 1/m
    s: np.ndarray         # (N,) arc length
    ds: float
    half_width: float
    name: str = "track"

    @property
    def n(self) -> int:
        return len(self.s)

    @property
    def length(self) -> float:
        return float(self.n * self.ds)

    # ------------------------------------------------------------- geometry

    def project(self, x: np.ndarray, y: np.ndarray, hint: np.ndarray, window: int = 24):
        """Project cars onto the centreline, searching near ``hint``.

        Cars move continuously, so a local search around the previous index is
        both correct and O(1) in track length.

        Returns ``(idx, s, e_y, e_psi_ref)`` where ``e_y`` is signed lateral
        offset (positive = left of the centreline) and ``e_psi_ref`` is the
        centreline tangent at the projection.
        """
        offs = np.arange(-window, window + 1)
        cand = (hint[:, None] + offs[None, :]) % self.n
        dx = self.xy[cand, 0] - x[:, None]
        dy = self.xy[cand, 1] - y[:, None]
        j = np.argmin(dx * dx + dy * dy, axis=1)
        rows = np.arange(len(x))
        idx = cand[rows, j]

        psi_ref = self.psi[idx]
        tx, ty = np.cos(psi_ref), np.sin(psi_ref)
        vx, vy = x - self.xy[idx, 0], y - self.xy[idx, 1]
        along = vx * tx + vy * ty            # refine s within the segment
        e_y = -vx * ty + vy * tx             # left-positive cross product
        s = self.s[idx] + along
        return idx, s, e_y, psi_ref

    def preview(self, idx: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """Curvature at several lookahead distances ahead of ``idx``.

        Returns (B, len(distances)).
        """
        steps = np.round(distances / self.ds).astype(np.int64)
        look = (idx[:, None] + steps[None, :]) % self.n
        return self.kappa[look]

    def point_at(self, idx: np.ndarray, distance: float) -> np.ndarray:
        """Centreline point a fixed distance ahead. Used as the CX goal."""
        step = int(round(distance / self.ds))
        return self.xy[(idx + step) % self.n]

    # ------------------------------------------------------------ factories

    @classmethod
    def from_controls(
        cls, controls: np.ndarray, half_width: float = 6.0, ds: float = 1.0, name: str = "track"
    ) -> "Track":
        dense = _catmull_rom(np.asarray(controls, dtype=np.float64))
        seg = np.linalg.norm(np.diff(dense, axis=0, append=dense[:1]), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        total = cum[-1] + seg[-1]

        n = int(np.floor(total / ds))
        s = np.arange(n) * ds
        xs = np.interp(s, cum, dense[:, 0], period=total)
        ys = np.interp(s, cum, dense[:, 1], period=total)
        xy = np.stack([xs, ys], axis=1)

        # Derivatives on a closed curve, via centred differences on the ring.
        d1 = (np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)) / (2 * ds)
        d2 = (np.roll(xy, -1, axis=0) - 2 * xy + np.roll(xy, 1, axis=0)) / (ds * ds)
        psi = np.arctan2(d1[:, 1], d1[:, 0])
        speed2 = (d1 * d1).sum(axis=1)
        kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed2, 1e-9) ** 1.5
        # Light circular smoothing: raw finite-difference curvature is noisy
        # and the preview channels would otherwise be mostly aliasing.
        k = 9
        kernel = np.ones(k) / k
        kappa = np.convolve(np.concatenate([kappa[-k:], kappa, kappa[:k]]), kernel, "same")[
            k:-k
        ]
        return cls(xy=xy, psi=psi, kappa=kappa, s=s, ds=ds, half_width=half_width, name=name)


def _close_curvature(kappa: np.ndarray, ds: float, iters: int = 40) -> np.ndarray:
    """Nudge a curvature profile until the path it generates closes on itself.

    Heading closure is exact after scaling so the lap turns through 2*pi. For
    position closure, add a single sine/cosine pair -- which integrates to zero
    over the lap and so cannot break heading closure -- and solve for its two
    coefficients by Newton iteration. The correction is spread over the whole
    lap, so designed corner radii survive almost untouched.
    """
    n = len(kappa)
    s = np.arange(n) * ds
    total = n * ds
    basis = np.stack([np.sin(2 * np.pi * s / total), np.cos(2 * np.pi * s / total)])

    def endpoint(k: np.ndarray) -> np.ndarray:
        psi = np.cumsum(k) * ds
        return np.array([np.sum(np.cos(psi)) * ds, np.sum(np.sin(psi)) * ds])

    kappa = kappa * (2.0 * np.pi / (np.sum(kappa) * ds))
    coef = np.zeros(2)
    for _ in range(iters):
        res = endpoint(kappa + coef @ basis)
        if np.linalg.norm(res) < 1e-6 * total:
            break
        eps = 1e-6
        jac = np.stack(
            [
                (endpoint(kappa + (coef + eps * e) @ basis) - res) / eps
                for e in np.eye(2)
            ],
            axis=1,
        )
        try:
            coef = coef - np.linalg.solve(jac, res)
        except np.linalg.LinAlgError:
            break
    return kappa + coef @ basis


def from_layout(
    layout: list[tuple],
    half_width: float = 6.0,
    ds: float = 1.0,
    name: str = "layout",
) -> Track:
    """Build a circuit from an explicit sequence of straights and arcs.

    ``layout`` entries are ``("straight", length_m)`` or
    ``("corner", radius_m, angle_deg)`` with positive angles turning left.
    Designing in the curvature domain is what gives a real spread of corner
    speeds -- with this car anything above about a 150 m radius is flat out,
    so a circuit needs radii in the 25-120 m band to have corners at all.
    """
    pieces = []
    for entry in layout:
        if entry[0] == "straight":
            pieces.append(np.zeros(max(1, int(round(entry[1] / ds)))))
        elif entry[0] == "corner":
            _, radius, angle = entry
            arc_len = abs(np.deg2rad(angle)) * radius
            k = np.sign(angle) / radius
            pieces.append(np.full(max(1, int(round(arc_len / ds))), k))
        else:
            raise ValueError(f"bad layout entry {entry!r}")
    kappa = np.concatenate(pieces)

    # Soften the instantaneous curvature steps into transitions a car can
    # actually steer through; real circuits use clothoid entries for exactly
    # this reason.
    win = max(3, int(round(18.0 / ds)))
    kernel = np.ones(win) / win
    pad = np.concatenate([kappa[-win:], kappa, kappa[:win]])
    kappa = np.convolve(pad, kernel, "same")[win:-win]

    kappa = _close_curvature(kappa, ds)
    psi = np.cumsum(kappa) * ds
    xy = np.stack([np.cumsum(np.cos(psi)) * ds, np.cumsum(np.sin(psi)) * ds], axis=1)
    xy = xy - xy[0]
    return Track(
        xy=xy,
        psi=psi,
        kappa=kappa,
        s=np.arange(len(kappa)) * ds,
        ds=ds,
        half_width=half_width,
        name=name,
    )


# Circuit layouts. Radii are chosen across the 25-120 m band so that corner
# speeds span roughly 75 to 320 km/h rather than clustering at flat-out.
LAYOUTS: dict[str, list[tuple]] = {
    "national": [
        ("straight", 420), ("corner", 55, 95), ("straight", 150),
        ("corner", 90, 70), ("straight", 240), ("corner", 32, -110),
        ("straight", 180), ("corner", 70, 120), ("straight", 320),
        ("corner", 110, 85), ("straight", 140), ("corner", 45, -75),
        ("straight", 200), ("corner", 60, 100), ("straight", 260),
        ("corner", 85, 75),
    ],
    "gp": [
        ("straight", 900), ("corner", 40, 120), ("straight", 180),
        ("corner", 100, -60), ("straight", 150), ("corner", 75, 105),
        ("straight", 620), ("corner", 26, 165), ("straight", 240),
        ("corner", 115, 55), ("straight", 300), ("corner", 50, -90),
        ("straight", 160), ("corner", 65, 110), ("straight", 480),
        ("corner", 95, 80), ("straight", 220), ("corner", 35, -70),
        ("straight", 190), ("corner", 80, 90),
    ],
    "technical": [
        ("straight", 260), ("corner", 30, 130), ("straight", 90),
        ("corner", 38, -95), ("straight", 140), ("corner", 55, 115),
        ("straight", 110), ("corner", 28, -85), ("straight", 300),
        ("corner", 70, 100), ("straight", 120), ("corner", 42, 90),
        ("straight", 160), ("corner", 34, -80), ("straight", 210),
        ("corner", 60, 125), ("straight", 130), ("corner", 48, 80),
    ],
}


def oval(length: float = 900.0, width: float = 420.0, **kw) -> Track:
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = np.stack([length / 2 * np.cos(th), width / 2 * np.sin(th)], axis=1)
    return Track.from_controls(pts, name="oval", **kw)


def harmonic(
    seed: int = 0,
    radius: float = 520.0,
    n_controls: int = 48,
    amps: tuple[float, ...] = (0.26, 0.14, 0.08),
    modes: tuple[int, ...] = (2, 3, 5),
    **kw,
) -> Track:
    """A closed circuit built from harmonics on a circle.

    Gives a repeatable layout with genuinely varied corner radii -- slow
    hairpins, fast sweepers and short straights -- without hand-placing
    hundreds of points. Different seeds are different circuits.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, len(modes))
    th = np.linspace(0, 2 * np.pi, n_controls, endpoint=False)
    r = np.ones_like(th)
    for a, mode, ph in zip(amps, modes, phases):
        r = r + a * np.cos(mode * th + ph)
    r = radius * r
    pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    return Track.from_controls(pts, name=f"harmonic{seed}", **kw)


TRACKS = {
    "oval": oval,
    "national": lambda **kw: from_layout(LAYOUTS["national"], name="national", **kw),
    "gp": lambda **kw: from_layout(LAYOUTS["gp"], name="gp", **kw),
    "technical": lambda **kw: from_layout(LAYOUTS["technical"], name="technical", **kw),
    "harmonic": lambda **kw: harmonic(seed=1, **kw),
}


def get_track(name: str, **kw) -> Track:
    if name not in TRACKS:
        raise KeyError(f"unknown track {name!r}; have {sorted(TRACKS)}")
    return TRACKS[name](**kw)
