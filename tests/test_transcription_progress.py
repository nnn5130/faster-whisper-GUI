# coding:utf-8
"""Progress math and the Transcription page progress card."""

import os
import struct
import sys
import tempfile
import threading
import types
import unittest
import wave

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The package __init__ imports whisperx, which needs torch. Load the GUI
# modules directly so these tests can run without that stack.
if "faster_whisper_GUI" not in sys.modules:
    package = types.ModuleType("faster_whisper_GUI")
    package.__path__ = [os.path.join(ROOT, "faster_whisper_GUI")]
    package.__package__ = "faster_whisper_GUI"
    sys.modules["faster_whisper_GUI"] = package

from PySide6.QtCore import QTranslator
from PySide6.QtWidgets import QApplication

from faster_whisper_GUI.transcription_progress import (
    TranscriptionProgressTracker,
    estimate_remaining_seconds,
    parse_clip_ranges,
    probe_media_duration,
)
from faster_whisper_GUI.transcriptionProgressWidget import TranscriptionProgressWidget


def _clock(start=0.0):
    state = {"now": start}

    def clock():
        return state["now"]

    clock.state = state
    return clock


class ProgressMathTests(unittest.TestCase):
    def test_single_file_segment_progress_and_eta(self):
        clock = _clock()
        tracker = TranscriptionProgressTracker(["talk.wav"], [100.0], clock=clock)
        tracker.start()
        tracker.begin_file("talk.wav")
        tracker.set_media_duration("talk.wav", 100.0, 100.0)
        tracker.update_file("talk.wav", 25.0)

        early = tracker.snapshot()
        self.assertEqual(early["state"], "running")
        self.assertAlmostEqual(early["overall_fraction"], 0.25)
        self.assertEqual(early["active"][0]["index"], 1)
        self.assertAlmostEqual(early["active"][0]["fraction"], 0.25)
        self.assertIsNone(early["eta_seconds"])

        clock.state["now"] = 1.0
        self.assertIsNone(tracker.snapshot()["eta_seconds"])

        clock.state["now"] = 10.0
        rated = tracker.snapshot()
        # 25s of audio in 10s of wall time, 75s left -> 30s.
        self.assertAlmostEqual(rated["eta_seconds"], 30.0)
        self.assertAlmostEqual(rated["processed_seconds"], 25.0)
        self.assertAlmostEqual(rated["total_seconds"], 100.0)

    def test_eta_waits_for_enough_audio(self):
        self.assertIsNone(estimate_remaining_seconds(0.0, 100.0, 10.0))
        self.assertIsNone(estimate_remaining_seconds(10.0, None, 10.0))
        self.assertIsNone(estimate_remaining_seconds(10.0, 1000.0, 1.5))
        # 2% of 1000s is 20s, capped at 5s. Four seconds is not enough.
        self.assertIsNone(estimate_remaining_seconds(4.0, 1000.0, 3.0))
        self.assertAlmostEqual(estimate_remaining_seconds(5.0, 1000.0, 2.0), 398.0)
        # A short file can estimate once one second of audio is done.
        self.assertAlmostEqual(estimate_remaining_seconds(1.0, 30.0, 2.0), 58.0)

    def test_vad_weight_uses_speech_duration(self):
        clock = _clock()
        tracker = TranscriptionProgressTracker(["talk.wav"], [100.0], clock=clock)
        tracker.start()
        tracker.begin_file("talk.wav")
        tracker.set_media_duration("talk.wav", 100.0, 20.0)
        tracker.update_file("talk.wav", 50.0)
        snap = tracker.snapshot()
        self.assertAlmostEqual(snap["active"][0]["fraction"], 0.5)
        self.assertAlmostEqual(snap["processed_seconds"], 10.0)
        self.assertAlmostEqual(snap["total_seconds"], 20.0)
        self.assertAlmostEqual(snap["overall_fraction"], 0.5)

    def test_file_list_is_weighted_and_reports_the_active_file(self):
        clock = _clock()
        tracker = TranscriptionProgressTracker(
            ["short.wav", "long.wav"],
            [100.0, 300.0],
            clock=clock,
        )
        tracker.start()
        tracker.begin_file("short.wav")
        tracker.complete_file("short.wav")
        tracker.begin_file("long.wav")
        tracker.update_file("long.wav", 150.0)
        snap = tracker.snapshot()
        self.assertEqual(snap["file_count"], 2)
        self.assertEqual(snap["completed_count"], 1)
        self.assertEqual(snap["active"][0]["index"], 2)
        self.assertEqual(snap["active"][0]["name"], "long.wav")
        self.assertAlmostEqual(snap["active"][0]["fraction"], 0.5)
        # 100 of the short file plus 150 of the long file.
        self.assertAlmostEqual(snap["processed_seconds"], 250.0)
        self.assertAlmostEqual(snap["overall_fraction"], 250.0 / 400.0)

    def test_clip_ranges_drive_within_file_progress(self):
        ranges = parse_clip_ranges("0.0-10.0;20.0-30.0")
        self.assertEqual(ranges, [(0.0, 10.0), (20.0, 30.0)])
        self.assertIsNone(parse_clip_ranges("0"))
        self.assertEqual(parse_clip_ranges([0, 10, 30], duration=40), [(0.0, 10.0), (30.0, 40.0)])

        tracker = TranscriptionProgressTracker(
            ["clip.wav"],
            [100.0],
            clip_timestamps=[0, 10, 20, 30],
        )
        tracker.start()
        tracker.begin_file("clip.wav")
        tracker.set_media_duration("clip.wav", 100.0, 100.0)
        tracker.update_file("clip.wav", 25.0)
        snap = tracker.snapshot()
        self.assertAlmostEqual(snap["active"][0]["fraction"], 0.75)
        self.assertAlmostEqual(snap["processed_seconds"], 15.0)
        self.assertAlmostEqual(snap["total_seconds"], 20.0)

    def test_progress_does_not_move_backwards(self):
        tracker = TranscriptionProgressTracker(["a.wav"], [100.0])
        tracker.start()
        tracker.begin_file("a.wav")
        tracker.update_file("a.wav", 40.0)
        tracker.update_file("a.wav", 10.0)
        self.assertAlmostEqual(tracker.snapshot()["active"][0]["fraction"], 0.4)

    def test_unknown_duration_has_no_eta(self):
        clock = _clock()
        tracker = TranscriptionProgressTracker(["a.wav"], [None], clock=clock)
        tracker.start()
        tracker.begin_file("a.wav")
        tracker.update_file("a.wav", 12.0)
        clock.state["now"] = 20.0
        snap = tracker.snapshot()
        self.assertIsNone(snap["total_seconds"])
        self.assertIsNone(snap["eta_seconds"])
        self.assertEqual(snap["overall_fraction"], 0.0)
        tracker.complete_file("a.wav")
        self.assertEqual(tracker.snapshot()["overall_fraction"], 1.0)
        self.assertIsNone(tracker.snapshot()["eta_seconds"])

    def test_complete_cancel_and_error_settle(self):
        clock = _clock()
        done = TranscriptionProgressTracker(["a.wav"], [10.0], clock=clock)
        done.start()
        done.begin_file("a.wav")
        done.complete_file("a.wav")
        done.finish()
        snap = done.snapshot()
        self.assertEqual(snap["state"], "completed")
        self.assertEqual(snap["overall_fraction"], 1.0)
        self.assertIsNone(snap["eta_seconds"])
        done.update_file("a.wav", 1.0)
        self.assertEqual(done.snapshot()["state"], "completed")

        clock.state["now"] = 0.0
        cancelled = TranscriptionProgressTracker(["a.wav", "b.wav"], [10.0, 10.0], clock=clock)
        cancelled.start()
        cancelled.begin_file("a.wav")
        cancelled.update_file("a.wav", 4.0)
        cancelled.cancel()
        cancelled.update_file("a.wav", 9.0)
        cancelled.complete_file("a.wav")
        snap = cancelled.snapshot()
        self.assertEqual(snap["state"], "cancelled")
        self.assertAlmostEqual(snap["active"][0]["fraction"], 0.4)
        self.assertIsNone(snap["eta_seconds"])

        failed = TranscriptionProgressTracker(["a.wav", "b.wav"], [10.0, 10.0])
        failed.start()
        failed.begin_file("a.wav")
        failed.fail_file("a.wav")
        failed.begin_file("b.wav")
        failed.complete_file("b.wav")
        failed.finish()
        snap = failed.snapshot()
        self.assertEqual(snap["state"], "error")
        self.assertEqual(snap["failed_count"], 1)
        self.assertEqual(snap["completed_count"], 1)
        self.assertIsNone(snap["eta_seconds"])

        crashed = TranscriptionProgressTracker(["a.wav"], [10.0])
        crashed.start()
        crashed.fail_run()
        snap = crashed.snapshot()
        self.assertEqual(snap["state"], "error")
        self.assertEqual(snap["failed_count"], 1)
        crashed.finish()
        self.assertEqual(crashed.snapshot()["state"], "error")

    def test_parallel_updates_stay_in_range(self):
        tracker = TranscriptionProgressTracker(["a.wav", "b.wav"], [100.0, 100.0])
        tracker.start()

        def work(name):
            tracker.begin_file(name)
            for second in range(1, 101):
                tracker.update_file(name, second)
            tracker.complete_file(name)

        threads = [threading.Thread(target=work, args=(name,)) for name in ("a.wav", "b.wav")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        tracker.finish()
        snap = tracker.snapshot()
        self.assertEqual(snap["state"], "completed")
        self.assertEqual(snap["completed_count"], 2)
        self.assertAlmostEqual(snap["overall_fraction"], 1.0)
        self.assertAlmostEqual(snap["processed_seconds"], 200.0)

    def test_probe_wav_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "silence.wav")
            rate = 16000
            seconds = 0.5
            count = int(rate * seconds)
            with wave.open(path, "w") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(struct.pack("<" + "h" * count, *([0] * count)))
            duration = probe_media_duration(path)
        self.assertIsNotNone(duration)
        self.assertAlmostEqual(duration, 0.5, delta=0.05)
        self.assertIsNone(probe_media_duration(os.path.join(tempfile.gettempdir(), "missing-audio-xyz.wav")))


class ProgressWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _running_snapshot(self, **overrides):
        snapshot = {
            "state": "running",
            "file_count": 3,
            "completed_count": 1,
            "failed_count": 0,
            "active": [{"name": "b.wav", "index": 2, "fraction": 0.4}],
            "overall_fraction": 0.46,
            "processed_seconds": 140.0,
            "total_seconds": 300.0,
            "eta_seconds": 125.0,
            "elapsed_seconds": 80.0,
        }
        snapshot.update(overrides)
        return snapshot

    def test_chinese_states(self):
        widget = TranscriptionProgressWidget()
        widget.resize(720, 180)
        widget.show()
        self.app.processEvents()
        self.assertIn("转写进度", widget.titleLabel.text())
        self.assertIn("等待开始", widget.fileLabel.text())
        self.assertIn("尚未开始转写", widget.statusLabel.text())
        self.assertEqual(widget.percentLabel.text(), "0%")
        self.assertFalse(widget.indeterminateBar.isVisible())
        self.assertEqual(widget.progressBar.value(), 0)

        widget.show_running_placeholder(2, "a.wav")
        self.assertTrue(widget.indeterminateBar.isVisible())
        self.assertFalse(widget.progressBar.isVisible())
        self.assertIn("正在准备音频", widget.statusLabel.text())
        self.assertIn("文件 1/2", widget.fileLabel.text())
        self.assertIn("a.wav", widget.fileLabel.text())
        self.assertNotIn("剩余约", widget.statusLabel.text())
        self.assertEqual(widget.percentLabel.text(), "…")

        widget.apply_snapshot(self._running_snapshot(eta_seconds=None, processed_seconds=8.0, overall_fraction=0.05))
        self.assertFalse(widget.indeterminateBar.isVisible())
        self.assertIn("正在估算剩余时间", widget.statusLabel.text())
        self.assertNotIn("剩余约", widget.statusLabel.text())
        self.assertIn("文件 2/3", widget.fileLabel.text())
        self.assertIn("40%", widget.fileLabel.text())

        widget.apply_snapshot(self._running_snapshot())
        self.assertIn("文件 2/3", widget.fileLabel.text())
        self.assertIn("b.wav", widget.fileLabel.text())
        self.assertIn("40%", widget.fileLabel.text())
        self.assertIn("剩余约 2 分 5 秒", widget.statusLabel.text())
        self.assertEqual(widget.percentLabel.text(), "46%")
        self.assertEqual(widget.progressBar.value(), 460)
        self.assertFalse(widget.progressBar.isError())

        widget.apply_snapshot(self._running_snapshot(eta_seconds=0.4, overall_fraction=0.99))
        self.assertIn("即将完成", widget.statusLabel.text())
        self.assertNotIn("剩余约", widget.statusLabel.text())

        widget.apply_snapshot(
            {
                "state": "completed",
                "file_count": 3,
                "completed_count": 3,
                "failed_count": 0,
                "active": [],
                "overall_fraction": 1.0,
                "processed_seconds": 300.0,
                "total_seconds": 300.0,
                "eta_seconds": None,
                "elapsed_seconds": 182.0,
            }
        )
        self.assertIn("转写完成", widget.statusLabel.text())
        self.assertIn("用时 3 分 2 秒", widget.statusLabel.text())
        self.assertIn("已完成 3/3", widget.fileLabel.text())
        self.assertEqual(widget.percentLabel.text(), "100%")
        self.assertEqual(widget.progressBar.value(), 1000)
        self.assertFalse(widget.progressBar.isPaused())
        self.assertFalse(widget.progressBar.isError())

        widget.apply_snapshot(self._running_snapshot(overall_fraction=0.2))
        widget.mark_cancelled()
        self.assertIn("已取消", widget.statusLabel.text())
        self.assertNotIn("剩余约", widget.statusLabel.text())
        self.assertTrue(widget.progressBar.isPaused())
        self.assertFalse(widget.indeterminateBar.isVisible())
        ignored = widget.fileLabel.text()
        widget.apply_snapshot(self._running_snapshot())
        self.assertEqual(widget.fileLabel.text(), ignored)
        self.assertEqual(widget.statusLabel.text(), widget.statusLabel.text())
        self.assertIn("已取消", widget.statusLabel.text())

        widget.show_running_placeholder(1, "c.wav")
        widget.apply_snapshot(
            {
                "state": "error",
                "file_count": 2,
                "completed_count": 1,
                "failed_count": 1,
                "active": [],
                "overall_fraction": 1.0,
                "processed_seconds": 20.0,
                "total_seconds": 20.0,
                "eta_seconds": None,
                "elapsed_seconds": 12.0,
            }
        )
        self.assertIn("部分文件转写失败", widget.statusLabel.text())
        self.assertIn("1/2 个文件失败", widget.fileLabel.text())
        self.assertTrue(widget.progressBar.isError())
        self.assertFalse(widget.progressBar.isPaused())

        widget.indeterminateBar.stop()
        widget.deleteLater()

    def test_english_strings(self):
        # Importing the widget loads the compiled Qt resource used by the app.
        import resource  # noqa: F401
        translator = QTranslator()
        self.assertTrue(translator.load(":/resource/Translater/en.qm"))
        self.app.installTranslator(translator)
        try:
            widget = TranscriptionProgressWidget()
            self.assertEqual(widget.titleLabel.text(), "Transcription progress")
            self.assertIn("Waiting to start", widget.fileLabel.text())
            self.assertIn("Transcription has not started", widget.statusLabel.text())
            widget.apply_snapshot(self._running_snapshot())
            self.assertIn("File 2 of 3", widget.fileLabel.text())
            self.assertIn("b.wav", widget.fileLabel.text())
            self.assertIn("40%", widget.fileLabel.text())
            self.assertEqual(widget.statusLabel.text(), "About 2 min 5 sec remaining")
            widget.apply_snapshot(self._running_snapshot(eta_seconds=None, processed_seconds=8.0))
            self.assertEqual(widget.statusLabel.text(), "Estimating remaining time")
            widget.show_running_placeholder(1, "a.wav")
            self.assertEqual(widget.statusLabel.text(), "Preparing audio")
            self.assertNotIn("remaining", widget.statusLabel.text())
            widget.deleteLater()
        finally:
            self.app.removeTranslator(translator)

    def test_transcription_page_hosts_the_card_and_other_pages_build(self):
        from faster_whisper_GUI.banditPageNavigationInterface import BanditPageNavigation
        from faster_whisper_GUI.processPageNavigationInterface import ProcessPageNavigationInterface
        from faster_whisper_GUI.tranccribePageNavigationInterface import TranscribeNavigationInterface
        from faster_whisper_GUI.vadPageNavigationInterface import VADNavigationInterface

        page = ProcessPageNavigationInterface()
        self.assertTrue(hasattr(page, "progressWidget"))
        self.assertIn("转写进度", page.progressWidget.titleLabel.text())
        self.assertEqual(page.toolBar.titleLabel.text(), "Transcription")
        page.progressWidget.apply_snapshot(self._running_snapshot())
        self.assertIn("剩余约 2 分 5 秒", page.progressWidget.statusLabel.text())

        VADNavigationInterface()
        BanditPageNavigation()
        TranscribeNavigationInterface()
        page.progressWidget.indeterminateBar.stop()
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
