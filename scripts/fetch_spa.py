"""Build a real Spa-Francorchamps centreline with elevation.

Three sources, none of which has everything:

  TUM racetrack-database   surveyed centreline at ~5 m spacing with measured
                           left and right track widths, but in an arbitrary
                           local metric frame with no georeference.
  bacinger/f1-circuits     the circuit in WGS84 from OpenStreetMap, so it can
                           be georeferenced, but only ~150 points.
  AWS elevation tiles      SRTM 1-arcsecond terrain, which needs lat/lon.

So: align the TUM geometry onto the OSM one (Umeyama, searching cyclic shift
and direction), convert the aligned points back to lat/lon, and sample SRTM
there. The result keeps TUM's resolution and widths and gains real elevation.

    python scripts/fetch_spa.py --out flydrive/sim/data/spa.npz
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

TUM_URL = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/Spa.csv"
OSM_URL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/be-1925.geojson"
SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{lat:02d}/{ns}{lat:02d}{ew}{lon:03d}.hgt.gz"
EARTH_R = 6371000.0


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as r:
        return r.read()


def to_metres(lon, lat, lon0, lat0):
    """Equirectangular projection about a local origin. Fine over 7 km."""
    x = np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.stack([x, y], axis=1)


def to_lonlat(xy, lon0, lat0):
    lon = lon0 + np.degrees(xy[:, 0] / (EARTH_R * math.cos(math.radians(lat0))))
    lat = lat0 + np.degrees(xy[:, 1] / EARTH_R)
    return lon, lat


def resample_closed(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to n points evenly spaced by arc length."""
    loop = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(want, s, loop[:, 0]), np.interp(want, s, loop[:, 1])], axis=1)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (scale, rotation, translation) mapping src onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = (b.T @ a) / len(a)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        sign[1, 1] = -1
    rot = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / (a * a).sum() * len(a))
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def best_alignment(src: np.ndarray, dst: np.ndarray, probes: int = 512):
    """Align two closed loops with unknown start point and direction."""
    n = len(src)
    best = None
    for flip in (1, -1):
        cand = src[::flip]
        for shift in range(0, n, max(1, n // probes)):
            rolled = np.roll(cand, -shift, axis=0)
            scale, rot, trans = umeyama(rolled, dst)
            err = np.linalg.norm((scale * (rot @ rolled.T).T + trans) - dst, axis=1).mean()
            if best is None or err < best[0]:
                best = (err, flip, shift, scale, rot, trans)
    return best


class SRTM:
    """SRTM 1-arcsecond tiles, bilinearly sampled."""

    def __init__(self) -> None:
        self.tiles: dict[tuple[int, int], np.ndarray] = {}

    def tile(self, lat: int, lon: int) -> np.ndarray:
        key = (lat, lon)
        if key not in self.tiles:
            url = SRTM_URL.format(
                ns="N" if lat >= 0 else "S", lat=abs(lat),
                ew="E" if lon >= 0 else "W", lon=abs(lon),
            )
            raw = gzip.decompress(fetch(url))
            side = int(math.sqrt(len(raw) // 2))
            self.tiles[key] = np.frombuffer(raw, dtype=">i2").reshape(side, side)
        return self.tiles[key]

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        out = np.zeros(len(lon))
        for i, (lo, la) in enumerate(zip(lon, lat)):
            base_lat, base_lon = int(math.floor(la)), int(math.floor(lo))
            grid = self.tile(base_lat, base_lon)
            side = grid.shape[0]
            step = side - 1
            # Row 0 is the northern edge of the tile.
            r = (base_lat + 1 - la) * step
            c = (lo - base_lon) * step
            r0, c0 = int(np.clip(r, 0, step - 1)), int(np.clip(c, 0, step - 1))
            fr, fc = r - r0, c - c0
            patch = grid[r0:r0 + 2, c0:c0 + 2].astype(float)
            patch[patch < -1000] = np.nan            # SRTM voids
            out[i] = (
                patch[0, 0] * (1 - fr) * (1 - fc) + patch[0, 1] * (1 - fr) * fc
                + patch[1, 0] * fr * (1 - fc) + patch[1, 1] * fr * fc
            )
        return out


def smooth_closed(v: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average, for SRTM noise."""
    if window < 3:
        return v
    k = np.ones(window) / window
    pad = np.concatenate([v[-window:], v, v[:window]])
    return np.convolve(pad, k, "same")[window:-window]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="flydrive/sim/data/spa.npz")
    ap.add_argument("--smooth", type=int, default=15, help="elevation smoothing window, samples")
    args = ap.parse_args()

    print("fetching TUM centreline ...", flush=True)
    rows = np.loadtxt(io.StringIO(fetch(TUM_URL).decode()), delimiter=",", comments="#")
    tum_xy, w_right, w_left = rows[:, :2], rows[:, 2], rows[:, 3]

    print("fetching OSM geometry ...", flush=True)
    geo = json.loads(fetch(OSM_URL))
    coords = geo["features"][0]["geometry"]["coordinates"]
    while isinstance(coords[0][0], list):
        coords = coords[0]
    coords = np.array(coords, dtype=float)
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    lon0, lat0 = float(coords[:, 0].mean()), float(coords[:, 1].mean())
    osm_xy = to_metres(coords[:, 0], coords[:, 1], lon0, lat0)

    n = 256
    err, flip, shift, scale, rot, trans = best_alignment(
        resample_closed(tum_xy, n), resample_closed(osm_xy, n)
    )
    print(f"alignment: mean error {err:.1f} m, scale {scale:.4f}, "
          f"direction {'reversed' if flip < 0 else 'same'}", flush=True)
    if err > 40.0:
        raise SystemExit(f"alignment failed (mean error {err:.0f} m) -- sources disagree")

    aligned = scale * (rot @ tum_xy.T).T + trans
    lon, lat = to_lonlat(aligned, lon0, lat0)

    print("sampling SRTM elevation ...", flush=True)
    z = SRTM().sample(lon, lat)
    if np.isnan(z).any():
        z = np.interp(np.arange(len(z)), np.flatnonzero(~np.isnan(z)), z[~np.isnan(z)])
    z = smooth_closed(z, args.smooth)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        xy=aligned.astype(np.float64),
        z=z.astype(np.float64),
        w_left=w_left, w_right=w_right,
        lon=lon, lat=lat,
        origin=np.array([lon0, lat0]),
    )
    seg = np.linalg.norm(np.diff(np.vstack([aligned, aligned[:1]]), axis=0), axis=1)
    print(f"\nwrote {out}")
    print(f"  {len(aligned)} points, length {seg.sum():.0f} m (real Spa 7004 m)")
    print(f"  elevation {z.min():.0f}-{z.max():.0f} m, range {z.max()-z.min():.0f} m")
    print(f"  track width {(w_left+w_right).mean():.1f} m mean")
    grade = np.diff(np.concatenate([z, z[:1]])) / np.maximum(seg, 1e-6)
    print(f"  gradient: max climb {grade.max()*100:+.1f}%, max descent {grade.min()*100:+.1f}%")


if __name__ == "__main__":
    main()
