"""Parse PopNet outputs.

We get two artifacts per run:

* ``popnet.log`` -- one line per wire / credit event::

      \\tFrom Router <sx> <sy> to Router <dx> <dy> Port <p> Virtual Channel <v>

  No cycle stamps in vanilla popnet, so we use **event counts per port** as
  the per-dim activity proxy (port 1+2 -> X, port 3+4 -> Y, ...).

* ``stdout`` -- periodic ``Current time: T Incoming packets: I Finished
  packets: F`` snapshots, plus a final ``total finished`` /
  ``average Delay`` summary.

Together they let us compute A2A latency, average packet delay, and a
per-dim load split that distinguishes DimRotation from direct A2A.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


WIRE_RE = re.compile(
    r"From Router\s+([\d\s]+?)\s+to Router\s+([\d\s]+?)\s+Port\s+(\d+)\s+Virtual Channel\s+(\d+)"
)
CURRENT_TIME_RE = re.compile(
    r"Current time:\s+([\d\.eE+-]+)\s+Incoming packets:\s+(\d+)\s+Finished packets:\s+(\d+)"
)
TOTAL_FINISHED_RE = re.compile(r"total finished:\s+(\d+)")
AVERAGE_DELAY_RE = re.compile(r"average Delay:\s+([-\d\.eE+]+)")
PACKET_COUNT_RE = re.compile(r"Packet count:\s+(\d+)")


@dataclass
class StdoutSummary:
    packet_count: int = 0
    total_finished: int = 0
    average_delay: float = 0.0
    final_time: float = 0.0
    snapshots: list[tuple[float, int, int]] = field(default_factory=list)

    @property
    def all_finished(self) -> bool:
        return self.total_finished > 0 and self.total_finished == self.packet_count

    @property
    def finish_time(self) -> float:
        """The earliest ``Current time:`` snapshot at which Finished == total."""
        if not self.snapshots or self.packet_count == 0:
            return self.final_time
        target = self.packet_count
        for (t, _inc, fin) in self.snapshots:
            if fin >= target:
                return t
        return self.final_time


@dataclass
class LogSummary:
    per_port_events: dict[int, int] = field(default_factory=dict)
    total_events: int = 0

    def per_dim_events(self, dims: int) -> dict[int, int]:
        """Aggregate ports 1+2 -> dim 0, 3+4 -> dim 1, 5+6 -> dim 2, ..."""
        out = {d: 0 for d in range(dims)}
        for port, n in self.per_port_events.items():
            if port <= 0:
                continue
            dim = (port - 1) // 2
            if 0 <= dim < dims:
                out[dim] += n
        return out


def parse_stdout(stdout_path: Path) -> StdoutSummary:
    summary = StdoutSummary()
    with Path(stdout_path).open() as f:
        for line in f:
            m = PACKET_COUNT_RE.search(line)
            if m:
                summary.packet_count = int(m.group(1))
                continue
            m = CURRENT_TIME_RE.search(line)
            if m:
                t = float(m.group(1))
                inc = int(m.group(2))
                fin = int(m.group(3))
                summary.snapshots.append((t, inc, fin))
                summary.final_time = max(summary.final_time, t)
                summary.total_finished = max(summary.total_finished, fin)
                continue
            m = TOTAL_FINISHED_RE.search(line)
            if m:
                summary.total_finished = max(summary.total_finished, int(m.group(1)))
                continue
            m = AVERAGE_DELAY_RE.search(line)
            if m:
                token = m.group(1)
                if token != "-":
                    summary.average_delay = float(token)
                continue
    return summary


def parse_log(log_path: Path) -> LogSummary:
    summary = LogSummary()
    p = Path(log_path)
    if not p.is_file():
        return summary
    with p.open() as f:
        for line in f:
            m = WIRE_RE.search(line)
            if not m:
                continue
            port = int(m.group(3))
            summary.per_port_events[port] = summary.per_port_events.get(port, 0) + 1
            summary.total_events += 1
    return summary
