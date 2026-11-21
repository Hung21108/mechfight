"""
Non-blocking JSONL transport and debug helper for mechfight/1.
Standard library only. Ensures we always process the latest observation
if multiple observations queue up on stdin, and flushes stdout after every line.
"""

from __future__ import annotations

import json
import os
import select
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional


class DebugDraw:
    """Helper to accumulate debug drawings within protocol caps."""

    def __init__(self, caps: Optional[Dict[str, Any]] = None) -> None:
        caps = caps or {}
        self.max_lines = int(caps.get("max_lines", 64))
        self.max_points = int(caps.get("max_points", 64))
        self.max_text = int(caps.get("max_text", 16))
        self.max_text_chars = int(caps.get("max_text_chars", 80))

        self._label: Optional[str] = None
        self._lines: List[Dict[str, Any]] = []
        self._points: List[Dict[str, Any]] = []
        self._text: List[Dict[str, Any]] = []

    def label(self, text: str) -> None:
        self._label = text[: self.max_text_chars]

    def line(
        self,
        a: List[float],
        b: List[float],
        color: Optional[str] = None,
    ) -> None:
        if len(self._lines) >= self.max_lines:
            return
        item: Dict[str, Any] = {
            "a": [round(float(x), 2) for x in a[:3]],
            "b": [round(float(x), 2) for x in b[:3]],
        }
        if color:
            item["color"] = color
        self._lines.append(item)

    def point(
        self,
        at: List[float],
        color: Optional[str] = None,
        size: int = 4,
    ) -> None:
        if len(self._points) >= self.max_points:
            return
        item: Dict[str, Any] = {
            "at": [round(float(x), 2) for x in at[:3]],
            "size": max(1, min(12, int(size))),
        }
        if color:
            item["color"] = color
        self._points.append(item)

    def text(
        self,
        text: str,
        at: Optional[List[float]] = None,
        color: Optional[str] = None,
    ) -> None:
        if len(self._text) >= self.max_text:
            return
        item: Dict[str, Any] = {"text": text[: self.max_text_chars]}
        if at is not None:
            item["at"] = [round(float(x), 2) for x in at[:3]]
        if color:
            item["color"] = color
        self._text.append(item)

    def to_dict(self) -> Optional[Dict[str, Any]]:
        out: Dict[str, Any] = {}
        if self._label is not None:
            out["label"] = self._label
        if self._lines:
            out["lines"] = self._lines
        if self._points:
            out["points"] = self._points
        if self._text:
            out["text"] = self._text
        return out if out else None


class StdinLineReader:
    """
    Unbuffered file descriptor line reader that drains any immediately
    available lines on stdin so the bot can skip stale queued observations.
    """

    def __init__(self, fd: int = 0) -> None:
        self.fd = fd
        self.buf = b""
        self.eof = False

    def _has_data_ready(self) -> bool:
        try:
            r, _, _ = select.select([self.fd], [], [], 0.0)
            return bool(r)
        except (ValueError, OSError):
            return False

    def read_line_blocking(self) -> Optional[bytes]:
        while b"\n" not in self.buf:
            if self.eof:
                if self.buf:
                    line = self.buf
                    self.buf = b""
                    return line
                return None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                self.eof = True
            else:
                self.buf += chunk

        idx = self.buf.index(b"\n")
        line = self.buf[:idx]
        self.buf = self.buf[idx + 1 :]
        return line

    def drain_available_lines(self) -> List[bytes]:
        lines: List[bytes] = []
        while True:
            while b"\n" in self.buf:
                idx = self.buf.index(b"\n")
                lines.append(self.buf[:idx])
                self.buf = self.buf[idx + 1 :]
            if self.eof or not self._has_data_ready():
                break
            chunk = os.read(self.fd, 65536)
            if not chunk:
                self.eof = True
                break
            self.buf += chunk
            while b"\n" in self.buf:
                idx = self.buf.index(b"\n")
                lines.append(self.buf[:idx])
                self.buf = self.buf[idx + 1 :]
        return lines


def send_json(msg: Dict[str, Any]) -> None:
    line = json.dumps(msg, separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def run_bot(
    bot_name: str,
    on_hello: Callable[[Dict[str, Any]], None],
    on_obs: Callable[[Dict[str, Any]], Dict[str, Any]],
    on_end: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    reader = StdinLineReader(sys.stdin.fileno())

    while True:
        raw = reader.read_line_blocking()
        if raw is None:
            break
        raw_s = raw.strip()
        if not raw_s:
            continue

        try:
            msg = json.loads(raw_s)
        except json.JSONDecodeError:
            continue

        # Drain any queued lines so we always respond to the newest observation
        extra_lines = reader.drain_available_lines()
        msgs = [msg]
        for extra in extra_lines:
            es = extra.strip()
            if not es:
                continue
            try:
                msgs.append(json.loads(es))
            except json.JSONDecodeError:
                continue

        latest_obs: Optional[Dict[str, Any]] = None
        accumulated_events: List[Dict[str, Any]] = []
        should_exit = False

        for m in msgs:
            mtype = m.get("type")
            if mtype == "hello":
                try:
                    on_hello(m)
                except Exception as exc:
                    log(f"[error] on_hello exception: {exc}\n{traceback.format_exc()}")
                send_json({"type": "ready", "name": bot_name[:32]})
            elif mtype == "obs":
                evs = m.get("events")
                if isinstance(evs, list) and evs:
                    accumulated_events.extend(evs)
                latest_obs = m
            elif mtype == "end":
                if on_end:
                    try:
                        on_end(m)
                    except Exception as exc:
                        log(f"[error] on_end exception: {exc}")
                should_exit = True
                break

        if should_exit:
            break

        if latest_obs is not None:
            if accumulated_events:
                latest_obs["events"] = accumulated_events
            tick = latest_obs.get("tick", 0)
            try:
                action = on_obs(latest_obs)
                if not isinstance(action, dict):
                    action = {}
            except Exception as exc:
                log(
                    f"[error] on_obs exception at tick {tick}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                action = _make_fallback_action(latest_obs)
            action["tick"] = tick
            send_json(action)


def _make_fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emergency schema-valid fallback action if `on_obs` raises an exception,
    preventing a single recoverable runtime error from terminating the bot process.
    """
    self_data = obs.get("self") or {}
    pos = self_data.get("pos") or [0.0, 0.0, 0.0]
    aim = self_data.get("aim") or [0.0, 0.0]
    zone = obs.get("zone") or {}
    nc = zone.get("next_center") or zone.get("center") or [0.0, 0.0]
    dx = float(nc[0]) - float(pos[0])
    dy = float(nc[1]) - float(pos[1])
    d = (dx * dx + dy * dy) ** 0.5
    mv = [round(dx / d, 4), round(dy / d, 4)] if d > 1.0 else [0.0, 0.0]
    offer = obs.get("offer")
    act: Dict[str, Any] = {
        "tick": obs.get("tick", 0),
        "move": mv,
        "boost": not bool(self_data.get("in_zone", True)),
        "jump": False,
        "aim": [round(float(aim[0]), 4), round(float(aim[1]), 4)],
        "fire": [False, False],
        "reload": False,
    }
    if isinstance(offer, dict) and offer.get("choices"):
        act["pick"] = 0
    return act
