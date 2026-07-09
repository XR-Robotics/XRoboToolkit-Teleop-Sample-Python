"""Local BRT magnetic encoder driver used by data collection."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

try:
    import minimalmodbus
    import minimalmodbus as _mm
except ImportError:
    minimalmodbus = None  # type: ignore[assignment]
    _mm = None

try:
    import serial.tools.list_ports as _list_ports
except ImportError:
    _list_ports = None  # type: ignore[assignment]


USB_SERIAL_RE = re.compile(r"(?:SER|SERIAL)=([^ ]+)", re.IGNORECASE)
# BRT Modbus registers (manual p.12).
REG_SINGLE_TURN = 0x0000   # single-turn value — gripper position
REG_RESET_ZERO = 0x0008    # write 1 -> current position becomes raw 0 (persistent)
REG_SET_MIDPOINT = 0x000E  # write 1 -> current position becomes midpoint (persistent)
# Raw single-turn wraps at this value. Must match the encoder resolution: this
# rig's encoder is 12-bit (0..4095, wraps at 4096); yam_umi's BRT is 10-bit (1024).
RAW_FULL_SCALE = 4096


def usb_serial_from_port_info(port_info) -> str:
    serial_number = getattr(port_info, "serial_number", None)
    if serial_number:
        return str(serial_number)
    match = USB_SERIAL_RE.search(getattr(port_info, "hwid", "") or "")
    return match.group(1) if match else ""


def find_port_by_usb_serial(usb_serial: str) -> str | None:
    if _list_ports is None:
        return None
    for port_info in _list_ports.comports():
        if usb_serial_from_port_info(port_info) == usb_serial:
            return port_info.device
    return None


def find_serial_port(
    *,
    baudrate: int = 9600,
    slave_addr: int = 1,
    probe: bool = True,
) -> str | None:
    if _list_ports is None:
        return None
    ports = _list_ports.comports()
    if not ports:
        return None

    keywords = ("ch340", "ch9344", "cp210", "ftdi", "pl2303", "usb-serial", "usb")
    matched: list[str] = []
    for port_info in ports:
        haystack = " ".join(
            (port_info.description or "", port_info.manufacturer or "", port_info.hwid or "")
        ).lower()
        if any(keyword in haystack for keyword in keywords):
            matched.append(port_info.device)
    ordered = matched if matched else [port_info.device for port_info in ports]

    if not probe:
        return ordered[0]

    for port in ordered:
        try:
            inst = create_instrument(port, slave_addr=slave_addr, baudrate=baudrate)
        except Exception:
            continue
        inst.serial.timeout = 0.3
        try:
            if read_raw(inst) is not None:
                return port
        except Exception:
            pass
        finally:
            try:
                inst.serial.close()
            except Exception:
                pass
    return None


def resolve_serial_port(
    *,
    port: str | None = None,
    usb_serial: str | None = None,
    baudrate: int = 9600,
    slave_addr: int = 1,
    probe: bool = True,
) -> str | None:
    if port:
        return port
    if usb_serial:
        return find_port_by_usb_serial(usb_serial)
    return find_serial_port(baudrate=baudrate, slave_addr=slave_addr, probe=probe)


def create_instrument(
    port: str,
    slave_addr: int = 1,
    baudrate: int = 9600,
) -> "minimalmodbus.Instrument":
    if _mm is None:
        raise ImportError("minimalmodbus is not installed")
    inst = _mm.Instrument(port, slave_addr)
    inst.serial.baudrate = baudrate
    inst.serial.bytesize = 8
    inst.serial.parity = _mm.serial.PARITY_NONE
    inst.serial.stopbits = 1
    inst.serial.timeout = 1.0
    inst.mode = _mm.MODE_RTU
    return inst


def read_raw(inst: "minimalmodbus.Instrument") -> int | None:
    try:
        return inst.read_register(REG_SINGLE_TURN, functioncode=3)
    except Exception:
        return None


def reset_zero(inst: "minimalmodbus.Instrument") -> None:
    """Hardware zero-set: the current position becomes raw 0 (persistent).

    Stored in the encoder itself. Use it when the zero point sits inside the
    gripper travel, which makes raw readings jump/wrap across the 0/RAW_FULL_SCALE
    boundary. Move to an endpoint (fully CLOSED) first, then zero, so the whole
    stroke stays in one non-wrapping region. NOTE: shifts every raw value, so any
    saved raw_open/raw_closed calibration is invalidated and must be re-recorded.
    """
    inst.write_register(REG_RESET_ZERO, 1, functioncode=6)


def set_midpoint(inst: "minimalmodbus.Instrument") -> None:
    """Hardware midpoint-set: current position becomes the midpoint (persistent)."""
    inst.write_register(REG_SET_MIDPOINT, 1, functioncode=6)


def probe_ports(*, baudrate: int = 9600, slave_addr: int = 1) -> list[tuple[str, int | None]]:
    """Probe every serial port, return (port, raw) — raw is None if no response.

    Wiggle one gripper and re-run to see which port's raw changes = that side.
    """
    if _list_ports is None:
        return []
    results: list[tuple[str, int | None]] = []
    for p in _list_ports.comports():
        raw = None
        try:
            inst = create_instrument(p.device, slave_addr=slave_addr, baudrate=baudrate)
            raw = read_raw(inst)
        except Exception:
            raw = None
        results.append((p.device, raw))
    return results


class EncoderCalibration:
    """Linear map from raw encoder value to normalized gripper position."""

    def __init__(
        self,
        raw_closed: int | None = None,
        raw_open: int | None = None,
        stroke_mm: float | None = None,
    ) -> None:
        self.raw_closed = raw_closed
        self.raw_open = raw_open
        self.stroke_mm = stroke_mm

    @property
    def is_ready(self) -> bool:
        return (
            self.raw_closed is not None
            and self.raw_open is not None
            and self.raw_open != self.raw_closed
        )

    def normalise(self, raw: int) -> float:
        if not self.is_ready:
            return 0.0
        raw_value = float(raw)
        raw_open = float(self.raw_open)
        raw_closed = float(self.raw_closed)

        if raw_open <= raw_closed:
            wrap_threshold = (raw_closed + RAW_FULL_SCALE) / 2.0
            if raw_value > wrap_threshold:
                raw_value -= RAW_FULL_SCALE
        elif raw_closed <= raw_open:
            wrap_threshold = raw_closed / 2.0
            if raw_value < wrap_threshold:
                raw_value += RAW_FULL_SCALE

        span = raw_open - raw_closed
        return float(np.clip((raw_value - raw_closed) / span, 0.0, 1.0))

    def metric_m(self, raw: int) -> float:
        if self.stroke_mm is None:
            return float("nan")
        return self.normalise(raw) * (self.stroke_mm / 1000.0)

    @classmethod
    def load(cls, path: Path) -> "EncoderCalibration":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                raw_closed=data.get("raw_closed"),
                raw_open=data.get("raw_open"),
                stroke_mm=data.get("stroke_mm"),
            )
        except Exception:
            return cls()
