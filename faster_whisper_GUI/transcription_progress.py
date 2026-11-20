# coding:utf-8
"""Duration-based progress and remaining-time estimate for a transcription run.

The bar is task-split across the file list and, inside the current file, follows
how far the transcribed timeline has moved:

- One file, or the active file in a list: ``segment.end / duration``.
  Manual clip ranges use the fraction of those ranges already passed instead of
  the whole file.
- Several files: each file's fraction is weighted by the audio the model still
  has to process. That weight is the clip span when clips are set, otherwise
  ``duration_after_vad`` once faster-whisper reports it (speech left after VAD),
  otherwise the probed media duration.
- A finished or failed file counts as fully consumed so later files are not
  charged for audio that will not be processed. The failure itself stays visible
  through the run state.

Remaining time is ``(total_work - processed_work) / (processed_work / elapsed)``.
It stays unset until every file has a known weight, at least two seconds of wall
time have passed, and the processed audio has reached ``min(5s, max(1s, 2% of
the total))``. Before that the UI should say it is still working, not invent a
number. Cancelling, failing, or finishing the run clears the estimate.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Sequence


def media_basename(path: str) -> str:
    name = str(path).replace("\\", "/").split("/")[-1]
    return name or str(path)


def _as_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _positive(value) -> Optional[float]:
    number = _as_float(value)
    if number is None or number <= 0:
        return None
    return number


def _non_negative(value) -> Optional[float]:
    number = _as_float(value)
    if number is None or number < 0:
        return None
    return number


def clip_timestamps_requested(clip_timestamps) -> bool:
    """True when the user asked to transcribe clip ranges rather than the whole file."""
    if clip_timestamps is None:
        return False
    if isinstance(clip_timestamps, str):
        return clip_timestamps.strip() not in ("", "0")
    if isinstance(clip_timestamps, (list, tuple)):
        if len(clip_timestamps) == 0:
            return False
        try:
            values = [float(item) for item in clip_timestamps]
        except (TypeError, ValueError):
            return True
        return values not in ([0.0], [0])
    return False


def _clip_values_from_text(text: str) -> Optional[list]:
    """Parse ``0,10,20,30`` or ``0.0-10.0;20.0-30.0`` into a flat timestamp list."""
    chunks = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
    if any("-" in chunk for chunk in chunks):
        values = []
        for chunk in chunks:
            left, right = chunk.split("-", 1)
            values.append(float(left))
            values.append(float(right))
        return values
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if not parts:
        return None
    return [float(part) for part in parts]


def parse_clip_ranges(clip_timestamps, duration: Optional[float] = None) -> Optional[list]:
    """Return ``[(start, end), ...]`` in seconds, or None when the whole file is used.

    An odd trailing timestamp is closed at ``duration`` when that duration is known.
    Without it, the ranges are not usable yet and this returns None.
    """
    if not clip_timestamps_requested(clip_timestamps):
        return None

    if isinstance(clip_timestamps, str):
        try:
            values = _clip_values_from_text(clip_timestamps)
        except ValueError:
            return None
        if values is None:
            return None
    else:
        try:
            values = [float(item) for item in clip_timestamps]
        except (TypeError, ValueError):
            return None

    if not values:
        return None
    if len(values) % 2 == 1:
        if duration is None or duration <= 0:
            return None
        values.append(float(duration))

    ranges = []
    for index in range(0, len(values), 2):
        start = values[index]
        end = values[index + 1]
        if end < start:
            start, end = end, start
        if duration is not None and duration > 0:
            start = min(max(0.0, start), duration)
            end = min(max(0.0, end), duration)
        if end > start:
            ranges.append((start, end))
    return ranges or None


def estimate_remaining_seconds(
    processed: Optional[float],
    total: Optional[float],
    elapsed: Optional[float],
    min_elapsed: float = 2.0,
) -> Optional[float]:
    """Seconds of wall time left, or None when a rate would be a guess."""
    if total is None or total <= 0 or processed is None or processed <= 0:
        return None
    if elapsed is None or elapsed < min_elapsed:
        return None
    threshold = min(5.0, max(1.0, total * 0.02))
    if processed < threshold:
        return None
    rate = processed / elapsed
    if rate <= 0:
        return None
    return max(0.0, (total - processed) / rate)


def probe_media_duration(path: str) -> Optional[float]:
    """Best-effort media duration in seconds. None when the file cannot be read."""
    try:
        import av
    except Exception:
        return None
    try:
        try:
            container = av.open(path, metadata_errors="ignore")
        except TypeError:
            container = av.open(path)
        with container:
            if container.duration is not None and container.duration > 0:
                return float(container.duration) / 1_000_000.0
            for stream in container.streams:
                if stream.codec_context.type != "audio":
                    continue
                if stream.duration and stream.time_base:
                    seconds = float(stream.duration * stream.time_base)
                    if seconds > 0:
                        return seconds
    except Exception:
        return None
    return None


class _FileProgress:
    def __init__(self, path: str, index: int, probed_duration, clip_timestamps):
        self.path = path
        self.index = index
        self.name = media_basename(path)
        self.probed_duration = _positive(probed_duration)
        self.duration = None
        self.duration_after_vad = None
        self.clip_timestamps = clip_timestamps
        self.uses_clips = clip_timestamps_requested(clip_timestamps)
        self.clip_ranges = None
        self.fraction = 0.0
        self.last_position = None
        self.state = "pending"
        self._refresh_clips()

    def _timeline(self) -> Optional[float]:
        return self.duration or self.probed_duration

    def _refresh_clips(self) -> None:
        if not self.uses_clips:
            self.clip_ranges = None
            return
        self.clip_ranges = parse_clip_ranges(self.clip_timestamps, self._timeline())

    def work_seconds(self) -> Optional[float]:
        if self.uses_clips:
            if not self.clip_ranges:
                return None
            return sum(end - start for start, end in self.clip_ranges)
        if self.duration_after_vad is not None:
            return self.duration_after_vad
        if self.duration is not None:
            return self.duration
        return self.probed_duration

    def fraction_for_position(self, position: float) -> float:
        position = max(0.0, float(position))
        if self.clip_ranges:
            total = sum(end - start for start, end in self.clip_ranges)
            if total <= 0:
                return 0.0
            covered = 0.0
            for start, end in self.clip_ranges:
                if position >= end:
                    covered += end - start
                elif position > start:
                    covered += position - start
            return min(1.0, covered / total)
        timeline = self._timeline()
        if timeline and timeline > 0:
            return min(1.0, position / timeline)
        return 0.0

    def note_position(self, position: float) -> None:
        self.last_position = max(0.0, float(position))
        self.fraction = max(self.fraction, self.fraction_for_position(self.last_position))

    def reread_position(self) -> None:
        if self.last_position is None or self.state in ("done", "failed"):
            return
        self.fraction = max(self.fraction, self.fraction_for_position(self.last_position))


class TranscriptionProgressTracker:
    """Thread-safe progress for one transcription run. Snapshots are plain dicts."""

    def __init__(
        self,
        files: Sequence[str],
        durations: Optional[Sequence[Optional[float]]] = None,
        clip_timestamps="0",
        clock: Optional[Callable[[], float]] = None,
        min_elapsed: float = 2.0,
    ) -> None:
        self._clock = clock or time.monotonic
        self._min_elapsed = min_elapsed
        self._lock = threading.Lock()
        self._started = None
        self._frozen = False
        self.state = "idle"
        probed = list(durations) if durations is not None else [None] * len(files)
        if len(probed) < len(files):
            probed.extend([None] * (len(files) - len(probed)))
        self.files = [
            _FileProgress(path, index + 1, probed[index], clip_timestamps)
            for index, path in enumerate(files)
        ]

    def start(self) -> None:
        with self._lock:
            self._started = self._clock()
            self.state = "running"
            self._frozen = False

    def begin_file(self, path: str) -> None:
        with self._lock:
            if self._frozen:
                return
            current = self._find(path, ("pending",))
            if current is None:
                return
            current.state = "active"

    def set_media_duration(self, path: str, duration, duration_after_vad) -> None:
        with self._lock:
            if self._frozen:
                return
            current = self._find(path, ("active", "pending"))
            if current is None:
                return
            decoded = _positive(duration)
            if decoded is not None:
                current.duration = decoded
            after_vad = _non_negative(duration_after_vad)
            if after_vad is not None:
                current.duration_after_vad = after_vad
            current._refresh_clips()
            current.reread_position()

    def update_file(self, path: str, position: float) -> None:
        with self._lock:
            if self._frozen or self.state != "running":
                return
            current = self._find(path, ("active", "pending"))
            if current is None:
                return
            if current.state == "pending":
                current.state = "active"
            try:
                current.note_position(float(position))
            except (TypeError, ValueError):
                return

    def complete_file(self, path: str) -> None:
        with self._lock:
            if self._frozen:
                return
            current = self._find(path, ("active", "pending"))
            if current is None:
                return
            current.state = "done"
            current.fraction = 1.0

    def fail_file(self, path: str) -> None:
        with self._lock:
            if self._frozen:
                return
            current = self._find(path, ("active", "pending"))
            if current is None:
                return
            current.state = "failed"
            # The rest of this file will not be transcribed.
            current.fraction = 1.0

    def finish(self) -> None:
        """The file loop ended on its own. Unfinished files count as failures."""
        with self._lock:
            if self._frozen:
                return
            for current in self.files:
                if current.state in ("pending", "active"):
                    current.state = "failed"
            if any(current.state == "failed" for current in self.files):
                self.state = "error"
            else:
                for current in self.files:
                    current.fraction = 1.0
                self.state = "completed"
            self._frozen = True

    def fail_run(self) -> None:
        """The run stopped because of an unexpected error."""
        with self._lock:
            if self.state == "cancelled" or self._frozen and self.state == "completed":
                return
            for current in self.files:
                if current.state in ("pending", "active"):
                    current.state = "failed"
            self.state = "error"
            self._frozen = True

    def cancel(self) -> None:
        with self._lock:
            if self.state in ("completed", "error", "cancelled"):
                return
            self.state = "cancelled"
            self._frozen = True

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_unlocked()

    def _find(self, path: str, states) -> Optional[_FileProgress]:
        for current in self.files:
            if current.path == path and current.state in states:
                return current
        return None

    def _elapsed_unlocked(self) -> float:
        if self._started is None:
            return 0.0
        return max(0.0, self._clock() - self._started)

    def _totals_unlocked(self):
        weights = [current.work_seconds() for current in self.files]
        if any(weight is None for weight in weights):
            processed = 0.0
            for current, weight in zip(self.files, weights):
                if weight is None:
                    continue
                processed += current.fraction * weight
            return processed, None
        total = float(sum(weights))
        processed = float(sum(current.fraction * weight for current, weight in zip(self.files, weights)))
        return processed, total

    def _overall_unlocked(self, processed: float, total: Optional[float]) -> float:
        if self.state == "completed":
            return 1.0
        if not self.files:
            return 0.0
        if total is not None and total > 0:
            return min(1.0, max(0.0, processed / total))
        return min(1.0, sum(current.fraction for current in self.files) / len(self.files))

    def _snapshot_unlocked(self) -> dict:
        processed, total = self._totals_unlocked()
        elapsed = self._elapsed_unlocked()
        eta = None
        if self.state == "running":
            eta = estimate_remaining_seconds(processed, total, elapsed, self._min_elapsed)
        active = [
            {
                "name": current.name,
                "index": current.index,
                "fraction": current.fraction,
            }
            for current in self.files
            if current.state == "active"
        ]
        return {
            "state": self.state,
            "file_count": len(self.files),
            "completed_count": sum(current.state == "done" for current in self.files),
            "failed_count": sum(current.state == "failed" for current in self.files),
            "active": active,
            "overall_fraction": self._overall_unlocked(processed, total),
            "processed_seconds": processed,
            "total_seconds": total,
            "eta_seconds": eta,
            "elapsed_seconds": elapsed,
        }
