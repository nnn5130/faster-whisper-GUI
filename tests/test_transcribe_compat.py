# coding:utf-8
"""WhisperModel.transcribe options stay compatible with faster-whisper 0.10.0."""

import os
import sys
import types
import unittest
from dataclasses import dataclass
from typing import NamedTuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if "faster_whisper_GUI" not in sys.modules:
    package = types.ModuleType("faster_whisper_GUI")
    package.__path__ = [os.path.join(ROOT, "faster_whisper_GUI")]
    package.__package__ = "faster_whisper_GUI"
    sys.modules["faster_whisper_GUI"] = package

from faster_whisper_GUI.transcribe_compat import (
    compatible_transcribe_kwargs,
    compatible_vad_parameters,
)
from faster_whisper_GUI.transcription_progress import TranscriptionProgressTracker


def user_options():
    """Options from the failing run, plus the newer keywords the GUI always sends."""
    return {
        "audio": "sample.mp4",
        "language": "ja",
        "task": "transcribe",
        "log_progress": False,
        "beam_size": 5,
        "best_of": 1,
        "patience": 1.0,
        "length_penalty": 1.0,
        "temperature": [0.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "prefix": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "prompt_reset_on_temperature": 0.5,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": False,
        "max_initial_timestamp": 1.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "multilingual": False,
        "max_new_tokens": None,
        "chunk_length": 30,
        "clip_timestamps": "0",
        "hallucination_silence_threshold": 0.0,
        "hotwords": "",
        "language_detection_threshold": 0.5,
        "language_detection_segments": 1,
        "vad_filter": True,
        "vad_parameters": {"threshold": 0.5},
    }


class Model010:
    """Parameter list matches faster-whisper 0.10.0 WhisperModel.transcribe."""

    def transcribe(
        self,
        audio,
        language=None,
        task="transcribe",
        beam_size=5,
        best_of=5,
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        temperature=0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        prompt_reset_on_temperature=0.5,
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=None,
        without_timestamps=False,
        max_initial_timestamp=1.0,
        word_timestamps=False,
        prepend_punctuations="",
        append_punctuations="",
        vad_filter=False,
        vad_parameters=None,
    ):
        self.received = {
            "audio": audio,
            "language": language,
            "task": task,
            "word_timestamps": word_timestamps,
            "vad_filter": vad_filter,
            "vad_parameters": vad_parameters,
            "chunk_length": "absent",
            "log_progress": "absent",
        }
        segments = [
            types.SimpleNamespace(start=0.0, end=10.0, text="こんにちは"),
            types.SimpleNamespace(start=10.0, end=20.0, text="テスト"),
        ]
        info = types.SimpleNamespace(
            language="ja",
            language_probability=0.99,
            duration=30.0,
            duration_after_vad=20.0,
        )
        return segments, info


class Model110:
    """Keyword set matches a faster-whisper release that accepts log_progress."""

    def transcribe(
        self,
        audio,
        language=None,
        task="transcribe",
        log_progress=False,
        beam_size=5,
        best_of=5,
        patience=1,
        length_penalty=1,
        temperature=0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        initial_prompt=None,
        prefix=None,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        prompt_reset_on_temperature=0.5,
        suppress_blank=True,
        suppress_tokens=None,
        without_timestamps=False,
        max_initial_timestamp=1.0,
        word_timestamps=False,
        prepend_punctuations="",
        append_punctuations="",
        multilingual=False,
        max_new_tokens=None,
        chunk_length=None,
        clip_timestamps="0",
        hallucination_silence_threshold=None,
        hotwords=None,
        language_detection_threshold=None,
        language_detection_segments=1,
        vad_filter=False,
        vad_parameters=None,
    ):
        self.received = {
            "audio": audio,
            "language": language,
            "log_progress": log_progress,
            "word_timestamps": word_timestamps,
            "chunk_length": chunk_length,
            "vad_filter": vad_filter,
            "multilingual": multilingual,
        }
        return [], types.SimpleNamespace(duration=1.0, duration_after_vad=1.0, language="ja")


class TranscribeCompatTests(unittest.TestCase):
    def test_010_call_drops_log_progress_and_still_feeds_progress(self):
        model = Model010()
        raw = user_options()
        # The unfiltered call is the TypeError from the user's log.
        with self.assertRaises(TypeError):
            model.transcribe(**raw)

        kwargs, dropped = compatible_transcribe_kwargs(model.transcribe, raw)
        self.assertIn("log_progress", dropped)
        self.assertIn("chunk_length", dropped)
        self.assertIn("multilingual", dropped)
        self.assertNotIn("language", dropped)
        self.assertNotIn("vad_filter", dropped)
        self.assertNotIn("word_timestamps", dropped)

        segments, info = model.transcribe(**kwargs)
        self.assertEqual(model.received["language"], "ja")
        self.assertTrue(model.received["vad_filter"])
        self.assertFalse(model.received["word_timestamps"])
        self.assertEqual(model.received["log_progress"], "absent")
        self.assertEqual(model.received["audio"], "sample.mp4")
        self.assertEqual(info.duration, 30.0)
        self.assertEqual(info.duration_after_vad, 20.0)

        tracker = TranscriptionProgressTracker(["sample.mp4"], [30.0])
        tracker.start()
        tracker.begin_file("sample.mp4")
        tracker.set_media_duration("sample.mp4", info.duration, info.duration_after_vad)
        for segment in segments:
            tracker.update_file("sample.mp4", segment.end)
        snap = tracker.snapshot()
        self.assertAlmostEqual(snap["active"][0]["fraction"], 20.0 / 30.0)
        self.assertAlmostEqual(snap["processed_seconds"], 20.0 * (20.0 / 30.0))
        self.assertIsNotNone(snap["overall_fraction"])
        self.assertGreater(snap["overall_fraction"], 0.0)

    def test_newer_faster_whisper_keeps_log_progress_and_chunk_length(self):
        model = Model110()
        kwargs, dropped = compatible_transcribe_kwargs(model.transcribe, user_options())
        self.assertEqual(dropped, [])
        model.transcribe(**kwargs)
        self.assertFalse(model.received["log_progress"])
        self.assertEqual(model.received["chunk_length"], 30)
        self.assertEqual(model.received["language"], "ja")
        self.assertTrue(model.received["vad_filter"])
        self.assertFalse(model.received["word_timestamps"])


class VadOptions010(NamedTuple):
    """faster-whisper 0.10.0 VadOptions."""

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    window_size_samples: int = 1024
    speech_pad_ms: int = 400


@dataclass
class VadOptions11:
    """faster-whisper 1.1.0 VadOptions."""

    onset: float = 0.5
    offset: float = 0.35
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


def user_vad_parameters():
    return {
        "onset": 0.2,
        "min_speech_duration_ms": 0,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }


class VadOptionsCompatTests(unittest.TestCase):
    def test_010_maps_onset_to_threshold_and_keeps_vad_running(self):
        raw = user_vad_parameters()
        with self.assertRaises(TypeError):
            VadOptions010(**raw)

        adapted, notes = compatible_vad_parameters(raw, VadOptions010)
        self.assertIn("onset -> threshold", notes)
        self.assertNotIn("onset", adapted)
        self.assertEqual(adapted["threshold"], 0.2)
        self.assertEqual(adapted["min_speech_duration_ms"], 0)
        self.assertEqual(adapted["min_silence_duration_ms"], 2000)
        self.assertEqual(adapted["speech_pad_ms"], 400)
        self.assertEqual(adapted["max_speech_duration_s"], float("inf"))

        class Model:
            def transcribe(self, audio, language=None, vad_filter=False, vad_parameters=None, word_timestamps=False):
                if vad_filter:
                    if vad_parameters is None:
                        vad_parameters = VadOptions010()
                    else:
                        vad_parameters = VadOptions010(**vad_parameters)
                self.vad = vad_parameters
                self.vad_filter = vad_filter
                segments = [types.SimpleNamespace(start=0.0, end=12.0, text="こんにちは")]
                info = types.SimpleNamespace(duration=30.0, duration_after_vad=18.0, language="ja")
                return segments, info

        model = Model()
        options = user_options()
        options["vad_parameters"] = adapted
        kwargs, dropped = compatible_transcribe_kwargs(model.transcribe, options)
        self.assertNotIn("log_progress", kwargs)
        self.assertTrue(kwargs["vad_filter"])
        segments, info = model.transcribe(**kwargs)
        self.assertTrue(model.vad_filter)
        self.assertEqual(model.vad.threshold, 0.2)
        self.assertEqual(model.vad.min_speech_duration_ms, 0)
        self.assertEqual(model.vad.speech_pad_ms, 400)

        tracker = TranscriptionProgressTracker(["sample.mp4"], [30.0])
        tracker.start()
        tracker.begin_file("sample.mp4")
        tracker.set_media_duration("sample.mp4", info.duration, info.duration_after_vad)
        for segment in segments:
            tracker.update_file("sample.mp4", segment.end)
        self.assertGreater(tracker.snapshot()["overall_fraction"], 0.0)

    def test_newer_vad_keeps_onset(self):
        adapted, notes = compatible_vad_parameters(user_vad_parameters(), VadOptions11)
        self.assertEqual(notes, [])
        self.assertEqual(adapted["onset"], 0.2)
        self.assertNotIn("threshold", adapted)
        options = VadOptions11(**adapted)
        self.assertEqual(options.onset, 0.2)
        self.assertEqual(options.min_speech_duration_ms, 0)
        self.assertEqual(options.min_silence_duration_ms, 2000)
        self.assertEqual(options.speech_pad_ms, 400)


if __name__ == "__main__":
    unittest.main()
