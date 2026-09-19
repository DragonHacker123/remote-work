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


def ring_normals(xy: np.ndarray) -> np.ndarray:
    """Unit left-hand normals of a closed polyline."""
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


def ring_curvature(xy: np.ndarray, smooth: int = 9) -> np.ndarray:
    """Signed curvature of a closed polyline from centred differences.

    The parametric form is used rather than dividing by ``ds^2`` so that it
    stays correct when the points are not evenly spaced -- which is exactly the
    case for a racing line, whose spacing shrinks on the inside of a corner.
    """
    d1 = (np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)) / 2.0
    d2 = np.roll(xy, -1, axis=0) - 2.0 * xy + np.roll(xy, 1, axis=0)
    speed2 = (d1 * d1).sum(axis=1)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed2, 1e-12) ** 1.5
    return _smooth_ring(kappa, smooth)


def _ring_second_difference(n: int):
    """Circulant second-difference operator, so the path closes on itself."""
    from scipy.sparse import diags

    return diags(
        [np.ones(n - 1), np.full(n, -2.0), np.ones(n - 1), [1.0], [1.0]],
        [-1, 0, 1, n - 1, -(n - 1)],
        shape=(n, n),
        format="csr",
    )


def _bounded_min_curvature(
    xy: np.ndarray,
    normal: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    weight: np.ndarray,
    ridge: float = 1e-9,
    iters: int = 40,
) -> np.ndarray:
    """One weighted, box-constrained minimum-curvature solve.

    Displacing the path by ``alpha`` along the normal makes its second
    difference affine in ``alpha``, so minimising a weighted sum of squared
    second differences subject to ``lo <= alpha <= hi`` is a convex box QP.
    Solved by an active set: clamp the variables that leave their bounds, solve
    the remaining ones exactly, release any clamp whose multiplier has the
    wrong sign, repeat. That is exact at convergence, unlike solving free and
    clipping, which just deletes the constraint's effect on its neighbours.

    The ridge exists only to make the normal equations safely factorisable and
    has to stay tiny: the operator's small eigenvalues go as (2 pi / lambda)^4
    for a feature of wavelength lambda, so a ridge of 1e-6 silently suppresses
    everything longer than about 130 m -- which is exactly the length scale a
    racing line works on. It was set there first, and the solver returned a
    line indistinguishable from the centreline.
    """
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve

    n = len(xy)
    d2 = _ring_second_difference(n)
    w = diags(np.sqrt(weight))
    ax = w @ d2 @ diags(normal[:, 0])
    ay = w @ d2 @ diags(normal[:, 1])
    h = (ax.T @ ax + ay.T @ ay).tocsr()
    g = ax.T @ (w @ (d2 @ xy[:, 0])) + ay.T @ (w @ (d2 @ xy[:, 1]))
    h = (h + ridge * float(h.diagonal().mean()) * eye(n, format="csr")).tocsr()

    alpha = np.zeros(n)
    clamp = np.zeros(n, dtype=np.int8)      # -1 at lo, +1 at hi, 0 free
    for _ in range(iters):
        fixed = np.where(clamp > 0, hi, np.where(clamp < 0, lo, 0.0))
        free = clamp == 0
        if not free.any():
            return fixed
        alpha = fixed.copy()
        alpha[free] = spsolve(h[free][:, free].tocsc(), -(g + h @ fixed)[free])
        grad = h @ alpha + g
        add = np.where(
            free & (alpha > hi + 1e-9), 1, np.where(free & (alpha < lo - 1e-9), -1, 0)
        )
        release = ((clamp > 0) & (grad > 1e-9)) | ((clamp < 0) & (grad < -1e-9))
        if not add.any() and not release.any():
            break
        clamp = np.where(add != 0, add, np.where(release, 0, clamp))
    return np.clip(alpha, lo, hi)


def racing_line_offsets(
    xy: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    margin: float = 0.6,
    power: float = 4.0,
    iters: int = 12,
    kappa_floor: float = 0.004,
) -> np.ndarray:
    """Lateral offsets, left-positive, of a minimum-curvature racing line.

    A circuit's centreline is not a driveable line. Spa's centreline through
    the Bus Stop has a 10.9 m radius and La Source 14.6 m; a driver who enters
    wide and exits wide turns both into something far opener. Tracking the
    centreline is what a lane-keeping controller does, not what a racing driver
    does, and on a circuit with a first-gear hairpin the difference decides
    whether the lap happens at all.

    Two corrections turn the textbook formulation into something that actually
    produces a racing line:

    *Arc length.* The plain objective sums squared second differences with
    respect to the centreline index, which equals ``kappa^2 h^3`` per metre for
    a local spacing ``h``. Pulling the line onto the inside of a corner shrinks
    ``h``, so the solver can lower the objective while raising the real
    curvature -- and it does: the first version tightened La Source from 14.6 m
    to 9.4 m. Reweighting by ``1/h^3`` from the previous iterate restores the
    integral of ``kappa^2`` along the path.

    *Which corners matter.* Minimising the integral of ``kappa^2`` is a
    lap-length average, and it will happily spend the hairpin to buy back a
    little on the fast sweepers. Lap time cares about the slowest corner, so
    the objective is raised to a higher power via iteratively reweighted least
    squares (``weight ~ |kappa|^(power-2)``). At ``power=4`` the minimum radius
    around Spa goes from 11.5 m to 16.6 m; at ``power=2`` it barely moves.

    The reweighting is a fixed point, not a descent, and on some layouts it
    cycles rather than settling. So every iterate is scored on what the routine
    is actually for -- peak curvature, with the integral of ``kappa^2`` as the
    tie-break -- and the best is returned. The centreline is scored too, which
    guarantees this never hands back a line worse than the one it started from:
    on a circuit whose corners all turn the same way it does exactly that and
    returns zeros.
    """
    n = len(xy)
    hi = np.maximum(np.asarray(w_left, dtype=float) - margin, 0.2)
    lo = -np.maximum(np.asarray(w_right, dtype=float) - margin, 0.2)
    normal = ring_normals(xy)

    def measure(offsets: np.ndarray):
        path = xy + offsets[:, None] * normal
        spacing = np.linalg.norm(np.roll(path, -1, 0) - np.roll(path, 1, 0), axis=1) / 2.0
        kappa = np.abs(ring_curvature(path))
        key = (float(kappa.max()), float((kappa * kappa * spacing).sum()))
        return key, spacing, np.abs(ring_curvature(path, smooth=15))

    best = np.zeros(n)
    best_key, _, _ = measure(best)
    weight = np.ones(n)
    for _ in range(iters):
        alpha = _bounded_min_curvature(xy, normal, lo, hi, weight)
        key, spacing, kappa = measure(alpha)
        if key < best_key:
            best_key, best = key, alpha
        fresh = (kappa + kappa_floor) ** (power - 2.0) / np.maximum(
            _smooth_ring(spacing, 9), 0.3
        ) ** 3
        weight = 0.5 * weight + 0.5 * fresh / fresh.mean()
    return best


def _smooth_ring(v: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average."""
    if window < 3:
        return v
    k = np.ones(window) / window
    pad = np.concatenate([v[-window:], v, v[:window]])
    return np.convolve(pad, k, "same")[window:-window]


def _fit_ring(
    xy: np.ndarray,
    ds: float,
    smooth: float,
    extras: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fit a periodic smoothing spline and resample at uniform arc length.

    Curvature must come from the spline's analytic derivatives, not from finite
    differences on a resampled polyline. Survey points ~5 m apart resampled to
    1 m and differentiated produce spikes at every original point: on this data
    that reports a 7 m minimum radius for a corner whose real radius is around
    25 m, which would make the circuit undriveable for reasons that are purely
    numerical.

    ``extras`` are per-input-point quantities (elevation, track widths) carried
    onto the same grid. Returns ``(xy, psi, kappa, s, extras)``.
    """
    from scipy.interpolate import splev, splprep

    xy = np.asarray(xy, dtype=np.float64)
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]

    tck, _u = splprep([xy[:, 0], xy[:, 1]], s=smooth, per=True)

    # Uniform arc length: walk a dense evaluation, then invert.
    dense = np.linspace(0.0, 1.0, 20000, endpoint=False)
    dx, dy = splev(dense, tck)
    step = np.hypot(np.diff(dx, append=dx[0]), np.diff(dy, append=dy[0]))
    arc = np.concatenate([[0.0], np.cumsum(step)[:-1]])
    total = float(arc[-1] + step[-1])

    n = int(np.floor(total / ds))
    want = np.arange(n) * ds
    u_at = np.interp(want, arc, dense)

    px, py = splev(u_at, tck)
    d1x, d1y = splev(u_at, tck, der=1)
    d2x, d2y = splev(u_at, tck, der=2)
    speed2 = d1x * d1x + d1y * d1y
    psi = np.arctan2(d1y, d1x)
    kappa = (d1x * d2y - d1y * d2x) / np.maximum(speed2, 1e-9) ** 1.5

    out = {}
    for key, values in (extras or {}).items():
        values = np.asarray(values, dtype=float)
        src_u = np.linspace(0.0, 1.0, len(values), endpoint=False)
        out[key] = np.interp(u_at, src_u, values, period=1.0)
    return np.stack([px, py], axis=1), psi, kappa, want, out


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
    """A closed circuit sampled at uniform arc length ``ds``.

    Elevation is optional and defaults to flat. Where it exists it matters a
    great deal: a gradient adds a gravity component along the direction of
    travel, and the vertical curvature of a compression or a crest changes the
    load on the tyres, which changes how much grip there is. Driving Spa flat
    removes most of what makes Spa difficult.
    """

    xy: np.ndarray        # (N, 2) centreline points
    psi: np.ndarray       # (N,) tangent heading
    kappa: np.ndarray     # (N,) signed curvature, 1/m
    s: np.ndarray         # (N,) arc length
    ds: float
    half_width: float     # mean, used to normalise the observation
    name: str = "track"
    z: np.ndarray | None = None       # elevation, m
    grade: np.ndarray | None = None   # dz/ds, dimensionless
    vcurv: np.ndarray | None = None   # vertical curvature, 1/m (crest negative)
    hw: np.ndarray | None = None        # per-point half width, m
    hw_left: np.ndarray | None = None   # room to the left of the reference path
    hw_right: np.ndarray | None = None  # room to its right

    def __post_init__(self) -> None:
        n = len(self.s)
        if self.z is None:
            self.z = np.zeros(n)
        if self.grade is None:
            self.grade = np.zeros(n)
        if self.vcurv is None:
            self.vcurv = np.zeros(n)
        if self.hw is None:
            self.hw = np.full(n, self.half_width)
        if self.hw_left is None:
            self.hw_left = self.hw.copy()
        if self.hw_right is None:
            self.hw_right = self.hw.copy()

    @property
    def has_elevation(self) -> bool:
        return bool(np.any(self.z != 0.0))

    def preview_grade(self, idx: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """Gradient at several lookahead distances. Returns (B, len(distances))."""
        steps = np.round(distances / self.ds).astype(np.int64)
        return self.grade[(idx[:, None] + steps[None, :]) % self.n]

    def preview_vcurv(self, idx: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """Vertical curvature ahead -- where the car gets heavy and light."""
        steps = np.round(distances / self.ds).astype(np.int64)
        return self.vcurv[(idx[:, None] + steps[None, :]) % self.n]

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
    def from_centreline(
        cls,
        xy: np.ndarray,
        z: np.ndarray | None = None,
        half_widths: np.ndarray | None = None,
        w_left: np.ndarray | None = None,
        w_right: np.ndarray | None = None,
        ds: float = 1.0,
        name: str = "track",
        smooth: float | None = None,
        elevation_smooth: int = 25,
        racing_line: bool = True,
        margin: float = 0.6,
    ) -> "Track":
        """Build from surveyed points, optionally on the racing line.

        With ``racing_line`` the reference path the car is measured against is
        a minimum-curvature line through the corridor rather than the surveyed
        centreline -- see :func:`racing_line_offsets`. Everything downstream
        then improves for free, because the curvature preview the speed profile
        brakes for is the curvature the car will actually experience. The room
        left either side of that line is asymmetric, which is what ``hw_left``
        and ``hw_right`` are for.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if smooth is None:
            smooth = len(xy) * 0.25          # ~0.5 m RMS deviation from survey

        if w_left is None or w_right is None:
            side = (
                np.full(len(xy), 6.0)
                if half_widths is None
                else np.asarray(half_widths, dtype=float)
            )
            w_left = side if w_left is None else w_left
            w_right = side if w_right is None else w_right

        extras = {"w_left": w_left, "w_right": w_right}
        if z is not None:
            extras["z"] = z
        pts, psi, kappa, s, ex = _fit_ring(xy, ds, smooth, extras)

        if racing_line:
            alpha = racing_line_offsets(pts, ex["w_left"], ex["w_right"], margin=margin)
            shifted = {
                "w_left": ex["w_left"] - alpha,
                "w_right": ex["w_right"] + alpha,
            }
            if "z" in ex:
                # Elevation is a property of the ground, and a few metres of
                # lateral offset does not move it appreciably; carrying it
                # across by station is accurate to a couple of centimetres.
                shifted["z"] = ex["z"]
            pts, psi, kappa, s, ex = _fit_ring(
                pts + alpha[:, None] * ring_normals(pts),
                ds,
                len(pts) * 0.02,             # ~0.14 m RMS: smooth the kinks only
                shifted,
            )

        hw_left = np.maximum(ex["w_left"], 0.2)
        hw_right = np.maximum(ex["w_right"], 0.2)
        hw = 0.5 * (hw_left + hw_right)

        if "z" in ex:
            zz = _smooth_ring(ex["z"], elevation_smooth)
            grade = _smooth_ring((np.roll(zz, -1) - np.roll(zz, 1)) / (2 * ds), elevation_smooth)
            d2z = (np.roll(zz, -1) - 2 * zz + np.roll(zz, 1)) / (ds * ds)
            vcurv = _smooth_ring(d2z, elevation_smooth * 3) / (1.0 + grade**2) ** 1.5
        else:
            zz = grade = vcurv = None

        return cls(
            xy=pts, psi=psi, kappa=kappa, s=s, ds=ds,
            half_width=float(hw.mean()), name=name,
            z=zz, grade=grade, vcurv=vcurv,
            hw=hw, hw_left=hw_left, hw_right=hw_right,
        )

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

    # Close the heading by scaling to a full turn, keeping the direction the
    # layout was designed in. Forcing +2*pi on a clockwise circuit would flip
    # every corner's sign and hand back a mirrored track.
    total_turn = np.sum(kappa) * ds
    kappa = kappa * (np.sign(total_turn) * 2.0 * np.pi / total_turn)
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
    # Spa-Francorchamps in character rather than in survey. Built from the
    # circuit's defining features -- a first-gear hairpin, the long flat-out
    # Kemmel straight, a fast downhill sweeper, and a stop-start chicane -- at
    # roughly the real 7 km. The coordinates are not Spa's: this environment has
    # no network access to fetch a real centreline, and elevation (which is half
    # of what makes Eau Rouge Eau Rouge) is not modelled at all. Treat it as a
    # circuit that drives like Spa, not as Spa.
    "spa": [
        ("straight", 300),
        ("corner", 26, -160),    # La Source, first-gear hairpin
        ("straight", 200),
        ("corner", 70, 42),      # Eau Rouge, left
        ("corner", 80, -70),     # Raidillon, right
        ("corner", 200, -25),
        ("straight", 2000),      # Kemmel -- flat out, the longest on the calendar
        ("corner", 44, -85),     # Les Combes
        ("corner", 40, 60),
        ("corner", 75, -60),     # Malmedy
        ("straight", 250),
        ("corner", 27, -130),    # Rivage
        ("straight", 120),
        ("corner", 70, -45),     # Bruxelles
        ("straight", 350),
        ("corner", 120, 95),     # Pouhon, long fast left
        ("straight", 200),
        ("corner", 52, -75),     # Fagnes
        ("corner", 56, 55),
        ("straight", 180),
        ("corner", 65, -90),     # Stavelot
        ("straight", 500),
        ("corner", 140, 45),     # Paul Frere
        ("straight", 300),
        ("corner", 230, 55),     # Blanchimont, flat out
        ("straight", 650),
        ("corner", 30, -80),     # Bus Stop chicane
        ("corner", 32, 85),
        ("straight", 300),
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


_REAL_CACHE: dict[tuple, Track] = {}


def _load_real(name: str, **kw) -> Track:
    """Load a surveyed circuit shipped with the package.

    Cached: solving the racing line takes a couple of seconds, and a training
    run builds the same circuit in every worker and again for every lap
    evaluation. Tracks are treated as read-only everywhere downstream, so one
    instance can safely be shared.
    """
    from pathlib import Path

    key = (name, tuple(sorted(kw.items())))
    if key in _REAL_CACHE:
        return _REAL_CACHE[key]

    path = Path(__file__).resolve().parent / "data" / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/fetch_spa.py` to build it "
            "from the public TUM centreline, OpenStreetMap geometry and SRTM "
            "elevation."
        )
    d = np.load(path)
    track = Track.from_centreline(
        d["xy"], z=d["z"], w_left=d["w_left"], w_right=d["w_right"], name=name, **kw
    )
    _REAL_CACHE[key] = track
    return track


TRACKS = {
    "oval": oval,
    "spa": lambda **kw: _load_real("spa", **kw),
    "national": lambda **kw: from_layout(LAYOUTS["national"], name="national", **kw),
    "spa_synthetic": lambda **kw: from_layout(
        LAYOUTS["spa"], name="spa_synthetic", **{"half_width": 7.0, **kw}
    ),
    "gp": lambda **kw: from_layout(LAYOUTS["gp"], name="gp", **kw),
    "technical": lambda **kw: from_layout(LAYOUTS["technical"], name="technical", **kw),
    "harmonic": lambda **kw: harmonic(seed=1, **kw),
}


def get_track(name: str, **kw) -> Track:
    if name not in TRACKS:
        raise KeyError(f"unknown track {name!r}; have {sorted(TRACKS)}")
    return TRACKS[name](**kw)
