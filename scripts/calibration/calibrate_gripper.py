"""Interactive BRT encoder -> gripper calibration for data collection.

Ported from songshuguifish/yam_umi (gripper/calibrate.py), adapted to write the
`calibration` block straight into this repo's encoder_mapping.json, so the next
`collect_session` populates `normalized`/`metric` automatically. Without this,
raw_open/raw_closed are unset -> normalise() returns 0 and metric is NaN (which
is exactly why the earlier sessions have empty gripper width).

Run on the collection host (where the encoders are plugged in), from repo root:

    # 0. which port is which side? wiggle one gripper, watch which raw moves
    python scripts/calibration/calibrate_gripper.py --list

    # 1. (recommended) hardware zero-set so raw never wraps: move gripper FULLY
    #    CLOSED first, then:
    python scripts/calibration/calibrate_gripper.py --side left --zero

    # 2. record endpoints + physical stroke -> writes encoder_mapping.json
    python scripts/calibration/calibrate_gripper.py --side left --stroke-mm 85
    python scripts/calibration/calibrate_gripper.py --side right --stroke-mm 85

    # non-interactive endpoints / live view
    python scripts/calibration/calibrate_gripper.py --side left --open 703 --closed 883 --stroke-mm 85
    python scripts/calibration/calibrate_gripper.py --side left --show

Needs minimalmodbus (installed in the collection venv).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.data_collection.encoder_driver import (  # noqa: E402
    create_instrument,
    find_port_by_usb_serial,
    probe_ports,
    read_raw,
    reset_zero,
    set_midpoint,
)

DEFAULT_MAPPING = _REPO / "scripts" / "data_collection" / "configs" / "encoder_mapping.json"
SIDES = ("left", "right")


def _role(side: str) -> str:
    return f"{side}_encoder"


def _load_mapping(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_port(args, mapping: dict) -> "str | None":
    if args.port:
        return args.port
    if args.side:
        entry = mapping.get("roles", {}).get(_role(args.side), {})
        usb = entry.get("usb_serial")
        if usb:
            port = find_port_by_usb_serial(str(usb))
            if port:
                return port
            print(f"[warn] usb_serial {usb} for {args.side} not found on any port")
    return None


def _sample_stable(inst, n: int = 15, dt: float = 0.05) -> "int | None":
    vals = []
    for _ in range(n):
        v = read_raw(inst)
        if v is not None:
            vals.append(v)
        time.sleep(dt)
    return int(statistics.median(vals)) if vals else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=SIDES, help="which gripper to calibrate")
    p.add_argument("--port", help="serial port, e.g. /dev/ttyACM0 (else resolved from --side usb_serial)")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--slave", type=int, default=1)
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="encoder_mapping.json to update")
    p.add_argument("--stroke-mm", type=float, help="physical jaw stroke closed->open (mm)")
    p.add_argument("--open", type=int, dest="raw_open", help="set raw_open directly (non-interactive)")
    p.add_argument("--closed", type=int, dest="raw_closed", help="set raw_closed directly (non-interactive)")
    p.add_argument("--list", action="store_true", help="probe ports (wiggle a gripper to ID the side)")
    p.add_argument("--show", action="store_true", help="stream live raw values, Ctrl-C to stop")
    p.add_argument("--zero", action="store_true", help="hardware zero-set at current position (persistent)")
    p.add_argument("--midpoint", action="store_true", help="hardware midpoint-set (persistent)")
    p.add_argument("-y", "--yes", action="store_true", help="skip confirm for --zero/--midpoint")
    args = p.parse_args()

    mapping = _load_mapping(args.mapping) if args.mapping.exists() else {"roles": {}}

    if args.list:
        print(f"Probing ports (baudrate={args.baudrate}, slave={args.slave})...")
        for port, raw in probe_ports(baudrate=args.baudrate, slave_addr=args.slave):
            print(f"  {port}: {'raw='+str(raw) if raw is not None else 'no response'}")
        print("Wiggle ONE gripper and re-run --list; the port whose raw changes is that side.")
        return

    port = _resolve_port(args, mapping)
    if port is None:
        raise SystemExit("no port: pass --port /dev/ttyACMx, or --side with a usb_serial in the mapping")
    print(f"Using port {port}" + (f" (side={args.side})" if args.side else ""))
    inst = create_instrument(port, slave_addr=args.slave, baudrate=args.baudrate)

    if args.show:
        try:
            while True:
                print(f"\rraw={read_raw(inst)}   ", end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()
        return

    if args.zero or args.midpoint:
        what = "zero" if args.zero else "midpoint"
        if not args.yes:
            resp = input(f"Hardware {what}-set at the CURRENT position (persistent, "
                         f"invalidates saved calibration). Continue? [y/N] ").strip().lower()
            if resp != "y":
                print("aborted")
                return
        (reset_zero if args.zero else set_midpoint)(inst)
        print(f"{what}-set done. raw now = {read_raw(inst)}. Re-record endpoints next.")
        return

    # --- record endpoints ---
    raw_open, raw_closed = args.raw_open, args.raw_closed
    if raw_open is None:
        input("Move gripper FULLY OPEN, then press Enter...")
        raw_open = _sample_stable(inst)
        print(f"  raw_open  = {raw_open}")
    if raw_closed is None:
        input("Move gripper FULLY CLOSED, then press Enter...")
        raw_closed = _sample_stable(inst)
        print(f"  raw_closed = {raw_closed}")
    if raw_open is None or raw_closed is None:
        raise SystemExit("failed to read a stable endpoint value")
    if raw_open == raw_closed:
        raise SystemExit("raw_open == raw_closed; encoder not moving or not calibrated")

    stroke_mm = args.stroke_mm
    if stroke_mm is None:
        val = input("Physical stroke closed->open (mm) [blank to leave unset]: ").strip()
        stroke_mm = float(val) if val else None

    if not args.side:
        print(f"\nraw_open={raw_open} raw_closed={raw_closed} stroke_mm={stroke_mm}")
        print("(no --side given -> not written to mapping)")
        return

    cal = {"raw_open": raw_open, "raw_closed": raw_closed}
    if stroke_mm is not None:
        cal["stroke_mm"] = stroke_mm
    mapping.setdefault("roles", {}).setdefault(_role(args.side), {})["calibration"] = cal
    args.mapping.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote calibration for {args.side} -> {args.mapping}")
    print(f"  {cal}")


if __name__ == "__main__":
    main()
