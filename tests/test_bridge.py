"""Tests for the F1 25 telemetry parser.

The game is not available here, so these build byte-exact packets from the
published layout and check that what comes out the other side is what went in.
That catches layout drift and sign errors; it cannot catch a spec change, which
is what the size assertions in the parser are for.
"""

import struct

import numpy as np
import pytest

from flydrive.bridge.f1_udp import (
    CAR_MOTION_FMT,
    CAR_MOTION_SIZE,
    CAR_TELEMETRY_FMT,
    CAR_TELEMETRY_SIZE,
    HEADER_FMT,
    HEADER_SIZE,
    NUM_CARS,
    PACKET_CAR_TELEMETRY,
    PACKET_MOTION,
    PacketLayoutError,
    TelemetryReceiver,
    parse_car_telemetry,
    parse_header,
    parse_motion,
)
from flydrive.bridge.gamepad import NullGamepad


def make_header(packet_id, session_time=0.0, player=0, frame=1):
    return struct.pack(
        HEADER_FMT, 2025, 25, 1, 0, 1, packet_id, 1234567890, session_time, frame, frame, player, 255
    )


def make_motion_packet(player=0, position=(10.0, 0.5, 20.0), velocity=(0.0, 0.0, 0.0),
                       forward=(1.0, 0.0, 0.0), session_time=0.0):
    body = b""
    for car in range(NUM_CARS):
        pos = position if car == player else (0.0, 0.0, 0.0)
        vel = velocity if car == player else (0.0, 0.0, 0.0)
        fwd = forward if car == player else (1.0, 0.0, 0.0)
        body += struct.pack(
            CAR_MOTION_FMT,
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            int(fwd[0] * 32767), int(fwd[1] * 32767), int(fwd[2] * 32767),
            0, 0, 0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        )
    return make_header(PACKET_MOTION, session_time, player) + body


def make_telemetry_packet(player=0, speed=100, throttle=0.5, steer=-0.25, brake=0.1):
    body = b""
    for car in range(NUM_CARS):
        vals = (speed, throttle, steer, brake) if car == player else (0, 0.0, 0.0, 0.0)
        body += struct.pack(
            CAR_TELEMETRY_FMT,
            vals[0], vals[1], vals[2], vals[3],
            0, 4, 11000, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            90,
            1.0, 1.0, 1.0, 1.0,
            0, 0, 0, 0,
        )
    return make_header(PACKET_CAR_TELEMETRY, 0.0, player) + body


def test_layout_sizes_match_the_published_spec():
    assert HEADER_SIZE == 29
    assert CAR_MOTION_SIZE == 60
    assert CAR_TELEMETRY_SIZE == 60
    assert len(make_motion_packet()) == HEADER_SIZE + NUM_CARS * CAR_MOTION_SIZE


def test_header_round_trip():
    header = parse_header(make_motion_packet(player=7, session_time=12.5))
    assert header.packet_format == 2025
    assert header.game_year == 25
    assert header.packet_id == PACKET_MOTION
    assert header.player_index == 7
    assert header.session_time == pytest.approx(12.5)


def test_motion_reads_the_players_car_not_car_zero():
    data = make_motion_packet(player=5, position=(100.0, 1.0, -50.0))
    mine = parse_motion(data, 5)
    other = parse_motion(data, 0)
    assert mine["position"] == pytest.approx([100.0, 1.0, -50.0])
    assert other["position"] == pytest.approx([0.0, 0.0, 0.0])


def test_telemetry_round_trip():
    data = make_telemetry_packet(speed=250, throttle=0.75, steer=-0.5, brake=0.25)
    car = parse_car_telemetry(data, 0)
    assert car["speed_kph"] == 250
    assert car["throttle"] == pytest.approx(0.75)
    assert car["steer"] == pytest.approx(-0.5)
    assert car["brake"] == pytest.approx(0.25)


def test_truncated_packet_raises_a_useful_error():
    with pytest.raises(PacketLayoutError):
        parse_motion(make_motion_packet()[:200], 0)
    with pytest.raises(PacketLayoutError):
        parse_header(b"\x00\x01")


def test_velocity_is_resolved_into_the_car_frame():
    """Forward along +X, moving along +Z: all of it should read as sideslip."""
    receiver = TelemetryReceiver(port=0)
    try:
        receiver._apply_motion(
            make_motion_packet(velocity=(0.0, 0.0, 5.0), forward=(1.0, 0.0, 0.0)),
            parse_header(make_motion_packet()),
        )
        state = receiver.state
        assert state.vx == pytest.approx(0.0, abs=1e-3)
        assert abs(state.vy) == pytest.approx(5.0, abs=1e-2)
        assert state.speed == pytest.approx(5.0, abs=1e-3)
    finally:
        receiver.close()


def test_forward_motion_reads_as_longitudinal_speed():
    receiver = TelemetryReceiver(port=0)
    try:
        data = make_motion_packet(velocity=(30.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0))
        receiver._apply_motion(data, parse_header(data))
        assert receiver.state.vx == pytest.approx(30.0, abs=1e-2)
        assert receiver.state.vy == pytest.approx(0.0, abs=1e-2)
    finally:
        receiver.close()


def test_yaw_rate_is_differentiated_from_heading():
    receiver = TelemetryReceiver(port=0)
    try:
        for t, fwd in ((0.0, (1.0, 0.0, 0.0)), (1.0, (np.cos(0.5), 0.0, np.sin(0.5)))):
            data = make_motion_packet(forward=fwd, session_time=t)
            receiver._apply_motion(data, parse_header(data))
        # One smoothing step at 0.3 gain over a 0.5 rad/s change.
        assert receiver.state.yaw_rate == pytest.approx(0.15, abs=0.02)
    finally:
        receiver.close()


def test_full_bridge_loop_against_a_synthetic_game():
    """Drive the whole bridge with a fake F1 25 emitting real packets.

    Exercises receive -> fuse -> observe -> brain -> gamepad, which is the one
    path that cannot be checked against the actual game from here.
    """
    import socket
    import threading

    from flydrive.agents.net import ConnectomeBrain, driving_subgraph
    from flydrive.bridge.run import BridgeConfig, drive
    from flydrive.connectome import build_surrogate
    from flydrive.sim import get_track

    track = get_track("oval")
    brain = ConnectomeBrain(driving_subgraph(build_surrogate(scale=0.35, seed=0)))

    port = 20999
    stop = threading.Event()

    def fake_game():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        t = 0.0
        i = 0
        while not stop.is_set():
            # Walk the car along the oval's centreline at 40 m/s.
            i = (i + 2) % track.n
            x, y = track.xy[i]
            psi = track.psi[i]
            sock.sendto(
                make_motion_packet(
                    position=(x, 0.0, y),
                    velocity=(40.0 * np.cos(psi), 0.0, 40.0 * np.sin(psi)),
                    forward=(np.cos(psi), 0.0, np.sin(psi)),
                    session_time=t,
                ),
                ("127.0.0.1", port),
            )
            sock.sendto(make_telemetry_packet(speed=144), ("127.0.0.1", port))
            t += 0.02
            stop.wait(0.004)
        sock.close()

    sender = threading.Thread(target=fake_game, daemon=True)
    sender.start()
    try:
        result = drive(
            brain, track, BridgeConfig(port=port, rate_hz=50.0, dry_run=True, max_seconds=1.5)
        )
    finally:
        stop.set()
        sender.join(timeout=2.0)

    assert result["steps"] > 20, f"bridge only ran {result['steps']} steps"


def test_null_gamepad_records_and_clamps():
    pad = NullGamepad()
    pad.send(0.3, 0.8, 0.0)
    assert pad.last == (0.3, 0.8, 0.0)
    pad.release()
    assert pad.last == (0.0, 0.0, 0.0)
