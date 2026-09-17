"""Virtual gamepad output.

Uses ``vgamepad`` (which needs the ViGEmBus driver on Windows) to present a
virtual Xbox controller that F1 25 reads as an ordinary input device. Nothing
is injected into the game process -- we read its own supported telemetry
stream and push back through a normal controller interface.

Keep this to offline sessions. Driving an online session with synthetic input
is not what the game's anti-cheat expects, and is not something this project
needs.
"""

from __future__ import annotations


class GamepadUnavailable(RuntimeError):
    pass


class VirtualGamepad:
    """Steering on the left stick, throttle and brake on the triggers."""

    def __init__(self, deadzone: float = 0.01):
        try:
            import vgamepad  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise GamepadUnavailable(
                "vgamepad is not installed. Install with `pip install flydrive[bridge]` "
                "and install the ViGEmBus driver (Windows only)."
            ) from exc
        self._vg = vgamepad
        self.pad = vgamepad.VX360Gamepad()
        self.deadzone = deadzone

    def send(self, steer: float, throttle: float, brake: float) -> None:
        steer = float(min(max(steer, -1.0), 1.0))
        if abs(steer) < self.deadzone:
            steer = 0.0
        self.pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.pad.right_trigger_float(value_float=float(min(max(throttle, 0.0), 1.0)))
        self.pad.left_trigger_float(value_float=float(min(max(brake, 0.0), 1.0)))
        self.pad.update()

    def release(self) -> None:
        self.pad.reset()
        self.pad.update()


class NullGamepad:
    """Stand-in for dry runs: accepts commands and records the last one."""

    def __init__(self) -> None:
        self.last = (0.0, 0.0, 0.0)
        self.history: list[tuple[float, float, float]] = []

    def send(self, steer: float, throttle: float, brake: float) -> None:
        self.last = (steer, throttle, brake)
        self.history.append(self.last)

    def release(self) -> None:
        self.last = (0.0, 0.0, 0.0)


def open_gamepad(dry_run: bool = False):
    if dry_run:
        return NullGamepad()
    return VirtualGamepad()
