"""Standalone UART0 diagnostic for the MaixCAM2-to-STM32 link.

Run on the MaixCAM2 from this directory:
    python uart0_test.py
    python uart0_test.py --count 3 --read-ms 300

By default it writes the same fixed 13-byte binary frame used by the motion
program every 500 ms until Ctrl+C stops it.  The STM32 can verify header sync,
signed little-endian fields, and CRC before it is connected to the motion
mechanism.  Pass ``--count N`` for a fixed number of frames, or
``--receive-only`` to listen without transmitting.
"""

from __future__ import annotations

import os
import struct
import sys
import time

from maix import uart


DEVICE_PATH = "/dev/ttyS0"
BAUDRATE = 115200
# Do not delay the continuous transmit loop unless an STM32 reply is requested.
DEFAULT_READ_MS = 0
DEFAULT_INTERVAL_MS = 500

# Positions are absolute rail pulses; angle is a whole signed degree.
TEST_COMMAND = {
    "pick_x_pulse": 5000,
    "pick_y_pulse": 6000,
    "place_x_pulse": 3000,
    "place_y_pulse": 3000,
    "rotate_deg_clockwise": -90.0,
}


def encode_test_frame(command: dict) -> bytes:
    """Build the production UART frame without importing project modules."""
    fields = (
        int(command["pick_x_pulse"]),
        int(command["pick_y_pulse"]),
        int(command["place_x_pulse"]),
        int(command["place_y_pulse"]),
        int(round(float(command["rotate_deg_clockwise"]))),
    )
    if any(value < -32768 or value > 32767 for value in fields):
        raise ValueError("Test field exceeds signed int16 range")
    payload = struct.pack("<2s5h", b"\xA5\x5A", *fields)
    checksum = 0
    for value in payload:
        checksum ^= value
    return payload + bytes((checksum,))


def parse_arguments(arguments: list[str]) -> tuple[bool, int, int, int]:
    """Parse the deliberately small command line without extra dependencies."""
    send_enabled = True
    # Zero means transmit continuously until the user stops the script.
    count = 0
    read_ms = DEFAULT_READ_MS
    interval_ms = DEFAULT_INTERVAL_MS
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--send":
            send_enabled = True
        elif argument == "--receive-only":
            send_enabled = False
        elif argument in ("--count", "--read-ms", "--interval-ms"):
            if index + 1 >= len(arguments):
                raise ValueError("{} requires an integer value".format(argument))
            index += 1
            value = int(arguments[index])
            if value < 0:
                raise ValueError("{} must not be negative".format(argument))
            if argument == "--count":
                count = value
            elif argument == "--read-ms":
                read_ms = value
            else:
                interval_ms = value
        elif argument in ("-h", "--help"):
            print(__doc__.strip())
            raise SystemExit(0)
        else:
            raise ValueError("Unknown option: {}".format(argument))
        index += 1
    return send_enabled, count, read_ms, interval_ms


def read_for(descriptor: int, duration_ms: int) -> bytes:
    """Collect any STM32 response without blocking the UI or motion process."""
    deadline = time.monotonic() + duration_ms / 1000.0
    received = bytearray()
    while time.monotonic() < deadline:
        try:
            chunk = os.read(descriptor, 256)
        except BlockingIOError:
            chunk = b""
        if chunk:
            received.extend(chunk)
        else:
            time.sleep(0.01)
    return bytes(received)


def main() -> None:
    send_enabled, count, read_ms, interval_ms = parse_arguments(sys.argv[1:])

    # UART0 keeps its default U0T/U0R pin mapping.  This object configures the
    # baud rate, while the non-blocking descriptor below handles raw bytes.
    configured_uart = uart.UART(DEVICE_PATH, BAUDRATE)
    descriptor = os.open(
        DEVICE_PATH,
        os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK,
    )
    frame = encode_test_frame(TEST_COMMAND)
    try:
        print("UART0 ready: {} {}bps".format(DEVICE_PATH, BAUDRATE))
        print("Test frame ({} bytes): {}".format(len(frame), frame.hex().upper()))
        print(
            "Fields: pick=(5000,6000)pulse place=(3000,3000)pulse "
            "angle=-90deg CRC={:02X}".format(
                frame[-1]
            )
        )
        if not send_enabled:
            print("Receive-only mode. No test frame transmitted.")
            received = read_for(descriptor, read_ms)
            print("RX {} bytes: {}".format(len(received), received.hex().upper()))
            return

        sequence = 0
        while count == 0 or sequence < count:
            sequence += 1
            written = os.write(descriptor, frame)
            count_label = "continuous" if count == 0 else str(count)
            print("TX {}/{}: {} bytes".format(sequence, count_label, written))
            received = read_for(descriptor, read_ms)
            print("RX {} bytes: {}".format(len(received), received.hex().upper()))
            if count == 0 or sequence < count:
                time.sleep(interval_ms / 1000.0)
    finally:
        os.close(descriptor)
        # Keep the Maix UART object alive until after the descriptor is closed.
        del configured_uart


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("UART0 test failed: {}".format(error))
        raise
