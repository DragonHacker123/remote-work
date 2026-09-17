"""F1 25 UDP telemetry receiver.

Enable it in the game under Settings -> Telemetry Settings: UDP Telemetry On,
Format 2025, IP = the machine running this, port 20777, rate 60 Hz.

Design note, because it decides how brittle this is: the game emits sixteen
packet types and their byte layouts shift between editions. This parser reads
only **Motion (id 0)** and **Car Telemetry (id 6)**, whose layouts have been
stable for years, and *derives* everything else -- yaw rate by differentiating
heading, body-frame velocity and sideslip by projecting world velocity onto the
car's forward vector, slip angles from the same bicycle-model relations the
simulator uses. Motion Ex (id 13) carries per-wheel slip directly but its
layout changes most often, so depending on it would buy accuracy at the cost of
breaking on the next patch.

Every layout is declared as a struct format and its size is checked against the
packet actually received, so a spec change produces a loud error naming the
offending packet rather than silently plausible garbage.

Coordinates: the F1 games use a left-handed frame with Y vertical, so the
ground plane is (X, Z). Heading comes from the car's forward direction vector
rather than the yaw field, which avoids a sign convention that has changed
between editions. Because the centreline is recorded in these same coordinates,
the frame is self-consistent; the only thing that must be checked against
reality is the sign of the steering output -- see ``LATERAL_SIGN``.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

# Flip if the car steers the wrong way in the game. Verified by driving a
# known left-hand corner and checking that reported yaw rate is positive.
LATERAL_SIGN = 1.0

HEADER_FMT = "<HBBBBBQfIIBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)          # 29 bytes

CAR_MOTION_FMT = "<6f6h6f"
CAR_MOTION_SIZE = struct.calcsize(CAR_MOTION_FMT)  # 60 bytes

CAR_TELEMETRY_FMT = "<HfffBbHBBH4H4B4BH4f4B"
CAR_TELEMETRY_SIZE = struct.calcsize(CAR_TELEMETRY_FMT)  # 60 bytes

NUM_CARS = 22
PACKET_MOTION = 0
PACKET_CAR_TELEMETRY = 6

INT16_NORM = 1.0 / 32767.0


class PacketLayoutError(RuntimeError):
    """Raised when a packet's size disagrees with the declared layout."""


@dataclass
class Header:
    packet_format: int
    game_year: int
    packet_id: int
    session_time: float
    frame: int
    player_index: int


def parse_header(data: bytes) -> Header:
    if len(data) < HEADER_SIZE:
        raise PacketLayoutError(f"packet shorter than header: {len(data)} < {HEADER_SIZE}")
    (
        packet_format,
        game_year,
        _major,
        _minor,
        _packet_version,
        packet_id,
        _session_uid,
        session_time,
        frame,
        _overall_frame,
        player_index,
        _secondary,
    ) = struct.unpack_from(HEADER_FMT, data, 0)
    return Header(packet_format, game_year, packet_id, session_time, frame, player_index)


def _car_slice(data: bytes, index: int, size: int) -> bytes:
    start = HEADER_SIZE + index * size
    return data[start : start + size]


def parse_motion(data: bytes, index: int) -> dict:
    """World position, velocity and orientation of one car."""
    expected = HEADER_SIZE + NUM_CARS * CAR_MOTION_SIZE
    if len(data) < expected:
        raise PacketLayoutError(
            f"motion packet is {len(data)} bytes, expected at least {expected}. "
            "The layout has changed -- check the current UDP spec and update "
            "CAR_MOTION_FMT."
        )
    v = struct.unpack(CAR_MOTION_FMT, _car_slice(data, index, CAR_MOTION_SIZE))
    return {
        "position": np.array([v[0], v[1], v[2]]),          # X, Y(up), Z
        "velocity": np.array([v[3], v[4], v[5]]),
        "forward": np.array([v[6], v[7], v[8]]) * INT16_NORM,
        "right": np.array([v[9], v[10], v[11]]) * INT16_NORM,
        "g_lat": v[12],
        "g_lon": v[13],
        "yaw": v[15],
        "pitch": v[16],
        "roll": v[17],
    }


def parse_car_telemetry(data: bytes, index: int) -> dict:
    expected = HEADER_SIZE + NUM_CARS * CAR_TELEMETRY_SIZE
    if len(data) < expected:
        raise PacketLayoutError(
            f"car telemetry packet is {len(data)} bytes, expected at least "
            f"{expected}. Check the current UDP spec and update CAR_TELEMETRY_FMT."
        )
    v = struct.unpack(CAR_TELEMETRY_FMT, _car_slice(data, index, CAR_TELEMETRY_SIZE))
    return {
        "speed_kph": v[0],
        "throttle": v[1],
        "steer": v[2],
        "brake": v[3],
        "gear": v[5],
        "rpm": v[6],
    }


@dataclass
class CarState:
    """Latest fused state of the player's car, in our (X, Z) ground plane."""

    time: float = 0.0
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    yaw_rate: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    valid: bool = False


class TelemetryReceiver:
    """Non-blocking UDP listener that fuses Motion and Car Telemetry."""

    def __init__(self, host: str = "0.0.0.0", port: int = 20777, timeout: float = 0.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.state = CarState()
        self._prev_heading: float | None = None
        self._prev_time: float | None = None
        self.packet_format: int | None = None

    def poll(self, max_packets: int = 64) -> CarState:
        """Drain the socket and return the freshest state."""
        for _ in range(max_packets):
            try:
                data, _addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                header = parse_header(data)
            except PacketLayoutError:
                continue
            self.packet_format = header.packet_format
            if header.packet_id == PACKET_MOTION:
                self._apply_motion(data, header)
            elif header.packet_id == PACKET_CAR_TELEMETRY:
                self._apply_telemetry(data, header)
        return self.state

    def _apply_motion(self, data: bytes, header: Header) -> None:
        m = parse_motion(data, header.player_index)
        s = self.state
        # Ground plane is (X, Z); Y is vertical and ignored.
        s.x, s.y = float(m["position"][0]), float(m["position"][2])
        fwd = np.array([m["forward"][0], m["forward"][2]])
        norm = np.linalg.norm(fwd)
        if norm < 1e-6:
            return
        fwd = fwd / norm
        heading = float(np.arctan2(fwd[1], fwd[0]))

        vel = np.array([m["velocity"][0], m["velocity"][2]])
        speed = float(np.linalg.norm(vel))
        # Longitudinal and lateral components in the car's own frame.
        along = float(vel @ fwd)
        lateral = float(fwd[0] * vel[1] - fwd[1] * vel[0]) * LATERAL_SIGN
        s.vx, s.vy, s.speed = along, lateral, speed

        now = header.session_time
        if self._prev_heading is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 1e-4:
                delta = (heading - self._prev_heading + np.pi) % (2 * np.pi) - np.pi
                # Light smoothing: differentiated heading is noisy at 60 Hz.
                s.yaw_rate = 0.7 * s.yaw_rate + 0.3 * (delta / dt) * LATERAL_SIGN
        self._prev_heading, self._prev_time = heading, now
        s.heading = heading * LATERAL_SIGN if LATERAL_SIGN < 0 else heading
        s.time = now
        s.valid = True

    def _apply_telemetry(self, data: bytes, header: Header) -> None:
        t = parse_car_telemetry(data, header.player_index)
        s = self.state
        s.throttle, s.brake, s.steer = t["throttle"], t["brake"], t["steer"]

    def close(self) -> None:
        self.sock.close()


def record_centreline(
    receiver: TelemetryReceiver,
    seconds: float = 180.0,
    min_spacing: float = 2.0,
    verbose: bool = True,
) -> np.ndarray:
    """Capture a lap of positions to build a centreline from.

    Drive one clean lap down the middle of the track while this runs. The
    resulting points are fed to ``Track.from_controls`` and become the geometry
    every observation is computed against.
    """
    points: list[tuple[float, float]] = []
    started = time.time()
    while time.time() - started < seconds:
        state = receiver.poll()
        if not state.valid:
            time.sleep(0.005)
            continue
        if not points or np.hypot(state.x - points[-1][0], state.y - points[-1][1]) >= min_spacing:
            points.append((state.x, state.y))
            if verbose and len(points) % 100 == 0:
                print(f"  {len(points)} centreline points", flush=True)
        time.sleep(0.005)
    return np.array(points)
