# coding:utf-8
"""Progress card on the Transcription page.

The card uses the same rounded parameter-panel surface as the transcribe
settings (``#mainObject`` plus the parameter-item style sheet). Chinese strings
are the source language; English comes from the application translator.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from qfluentwidgets import CaptionLabel, IndeterminateProgressBar, ProgressBar, StrongBodyLabel

from .style_sheet import StyleSheet


class TranscriptionProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("transcriptionProgressWidget")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(480)

        self._ignore_until_next_run = False
        self._snapshot = _idle_snapshot()

        self.mainWidget = QWidget(self)
        self.mainWidget.setObjectName("mainObject")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.mainWidget)

        layout = QVBoxLayout(self.mainWidget)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.titleLabel = StrongBodyLabel(self.tr("转写进度"), self)
        self.titleLabel.setObjectName("progressTitleLabel")
        self.percentLabel = StrongBodyLabel("0%", self)
        self.percentLabel.setObjectName("progressPercentLabel")
        self.percentLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.titleLabel)
        header.addStretch(1)
        header.addWidget(self.percentLabel)

        self.captionLabel = CaptionLabel(
            self.tr("当前文件进度，以及根据已处理音频估算的剩余时间"),
            self,
        )
        self.captionLabel.setWordWrap(True)

        self.fileLabel = CaptionLabel(self.tr("等待开始"), self)
        self.fileLabel.setObjectName("progressFileLabel")
        self.fileLabel.setWordWrap(True)

        self.progressBar = ProgressBar(self)
        self.progressBar.setObjectName("transcriptionProgressBar")
        self.progressBar.setRange(0, 1000)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)

        self.indeterminateBar = IndeterminateProgressBar(self, start=False)
        self.indeterminateBar.setObjectName("transcriptionIndeterminateBar")
        self.indeterminateBar.setVisible(False)

        self.barHost = QWidget(self)
        self.barHost.setFixedHeight(4)
        bar_layout = QVBoxLayout(self.barHost)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(0)
        bar_layout.addWidget(self.progressBar)
        bar_layout.addWidget(self.indeterminateBar)

        self.statusLabel = CaptionLabel(self.tr("尚未开始转写"), self)
        self.statusLabel.setObjectName("progressStatusLabel")
        self.statusLabel.setWordWrap(True)

        layout.addLayout(header)
        layout.addWidget(self.captionLabel)
        layout.addWidget(self.fileLabel)
        layout.addWidget(self.barHost)
        layout.addWidget(self.statusLabel)

        StyleSheet.TRANSCRIBEPARAMITEMWIDGET.apply(self)
        self._render(self._snapshot)

    def show_running_placeholder(self, file_count: int = 0, first_name: str = "") -> None:
        """Show an in-progress card immediately, before the worker reports a rate."""
        self._ignore_until_next_run = False
        active = []
        if file_count > 0 and first_name:
            active = [{"name": first_name, "index": 1, "fraction": None}]
        self.apply_snapshot(
            {
                "state": "running",
                "file_count": file_count,
                "completed_count": 0,
                "failed_count": 0,
                "active": active,
                "overall_fraction": 0.0,
                "processed_seconds": 0.0,
                "total_seconds": None,
                "eta_seconds": None,
                "elapsed_seconds": 0.0,
            }
        )

    def mark_cancelled(self) -> None:
        """Settle the card on cancel without waiting for the worker thread."""
        self._ignore_until_next_run = True
        snapshot = dict(self._snapshot)
        snapshot["state"] = "cancelled"
        snapshot["eta_seconds"] = None
        self._render(snapshot)

    def apply_snapshot(self, snapshot: dict) -> None:
        if snapshot is None:
            return
        state = snapshot.get("state")
        if self._ignore_until_next_run and state != "cancelled":
            return
        self._render(snapshot)

    def summary(self) -> str:
        return f"{self.fileLabel.text()}  {self.statusLabel.text()}"

    def _render(self, snapshot: dict) -> None:
        self._snapshot = dict(snapshot)
        state = snapshot.get("state") or "idle"
        indeterminate = _is_indeterminate(snapshot)
        fraction = _clamp(snapshot.get("overall_fraction") or 0.0)

        self.fileLabel.setText(self._file_line(snapshot, indeterminate))
        self.statusLabel.setText(self._status_line(snapshot, indeterminate))
        self.percentLabel.setText(self._percent_text(state, fraction, indeterminate))

        if indeterminate:
            self.progressBar.setVisible(False)
            self.indeterminateBar.setVisible(True)
            if not self.indeterminateBar.isStarted():
                self.indeterminateBar.start()
        else:
            if self.indeterminateBar.isStarted():
                self.indeterminateBar.stop()
            self.indeterminateBar.setVisible(False)
            self.progressBar.setVisible(True)
            self.progressBar.setValue(1000 if state == "completed" else int(round(fraction * 1000)))

        self._apply_bar_state(state)

    def _apply_bar_state(self, state: str) -> None:
        if state == "error":
            self.progressBar.resume()
            self.progressBar.setError(True)
            return
        if state == "cancelled":
            self.progressBar.setError(False)
            self.progressBar.setPaused(True)
            return
        self.progressBar.resume()

    def _file_line(self, snapshot: dict, indeterminate: bool) -> str:
        state = snapshot.get("state") or "idle"
        file_count = int(snapshot.get("file_count") or 0)
        completed = int(snapshot.get("completed_count") or 0)
        failed = int(snapshot.get("failed_count") or 0)
        active = list(snapshot.get("active") or [])

        if state == "idle":
            return self.tr("等待开始")
        if state == "error" and failed > 0 and file_count > 0:
            return self._fill("%1/%2 个文件失败", str(failed), str(file_count))
        if state in ("completed", "cancelled", "error") and not active:
            if file_count > 0:
                return self._fill("已完成 %1/%2 个文件", str(completed), str(file_count))
            return self.tr("转写失败") if state == "error" else self.tr("转写完成")
        if len(active) > 1:
            parts = []
            for item in active[:4]:
                fraction = item.get("fraction")
                if fraction is None or indeterminate:
                    parts.append(str(item.get("name") or ""))
                else:
                    parts.append(f"{item.get('name')} {int(round(_clamp(fraction) * 100))}%")
            detail = " · ".join(part for part in parts if part)
            if len(active) > 4:
                detail = f"{detail} · …"
            return self._fill("同时转写 %1 个文件 · %2", str(len(active)), detail)
        if len(active) == 1:
            item = active[0]
            index = item.get("index") or 1
            name = str(item.get("name") or "")
            fraction = item.get("fraction")
            if fraction is None or indeterminate:
                return self._fill("文件 %1/%2 · %3", str(index), str(max(file_count, 1)), name)
            percent = f"{int(round(_clamp(fraction) * 100))}%"
            return self._fill(
                "文件 %1/%2 · %3 · %4",
                str(index),
                str(max(file_count, 1)),
                name,
                percent,
            )
        if state == "running" and completed == 0 and failed == 0:
            return self.tr("正在准备音频")
        if file_count > 0:
            return self._fill("已完成 %1/%2 个文件", str(completed), str(file_count))
        return self.tr("正在准备音频")

    def _status_line(self, snapshot: dict, indeterminate: bool) -> str:
        state = snapshot.get("state") or "idle"
        if state == "idle":
            return self.tr("尚未开始转写")
        if state == "cancelled":
            return self.tr("已取消")
        if state == "error":
            failed = int(snapshot.get("failed_count") or 0)
            completed = int(snapshot.get("completed_count") or 0)
            if failed > 0 and completed > 0:
                return self.tr("部分文件转写失败")
            return self.tr("转写失败")
        if state == "completed":
            elapsed = self._format_duration(snapshot.get("elapsed_seconds") or 0)
            return self.tr("转写完成") + " · " + self._fill("用时 %1", elapsed)
        eta = snapshot.get("eta_seconds")
        if indeterminate or eta is None:
            if indeterminate:
                return self.tr("正在准备音频")
            return self.tr("正在估算剩余时间")
        if float(eta) < 1:
            return self.tr("即将完成")
        return self._fill("剩余约 %1", self._format_duration(eta))

    def _percent_text(self, state: str, fraction: float, indeterminate: bool) -> str:
        if indeterminate:
            return "…"
        if state == "completed" or fraction >= 1:
            return "100%"
        percent = int(round(fraction * 100))
        if state == "running":
            percent = min(99, percent)
        return f"{percent}%"

    def _format_duration(self, seconds: float) -> str:
        whole = int(round(max(0.0, float(seconds))))
        if whole < 60:
            return self._fill("%1 秒", str(whole))
        minutes, secs = divmod(whole, 60)
        if minutes < 60:
            return self._fill("%1 分 %2 秒", str(minutes), str(secs))
        hours, minutes = divmod(minutes, 60)
        return self._fill("%1 小时 %2 分", str(hours), str(minutes))

    def _fill(self, pattern: str, *parts: str) -> str:
        text = self.tr(pattern)
        for index, part in enumerate(parts, start=1):
            text = text.replace(f"%{index}", part)
        return text


def _idle_snapshot() -> dict:
    return {
        "state": "idle",
        "file_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "active": [],
        "overall_fraction": 0.0,
        "processed_seconds": 0.0,
        "total_seconds": None,
        "eta_seconds": None,
        "elapsed_seconds": 0.0,
    }


def _is_indeterminate(snapshot: dict) -> bool:
    if snapshot.get("state") != "running":
        return False
    if float(snapshot.get("processed_seconds") or 0) > 0:
        return False
    if int(snapshot.get("completed_count") or 0) > 0:
        return False
    if int(snapshot.get("failed_count") or 0) > 0:
        return False
    return True


def _clamp(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number
