# coding:utf-8
import gc
import os

import numpy as np
import soundfile
import torch
from PySide6.QtCore import QThread, Signal
from faster_whisper import decode_audio

from .bandit.inference import (
    SAMPLE_RATE,
    channel_first_audio,
    load_bandit_plus,
    separate_mixture,
    waveforms_for_track_index,
)
from .config import STEMS, normalize_bandit_track_index


class BanditWorker(QThread):
    """Background BandIt Plus separation.

    Signals match the previous separation worker: ``signal_vr_over`` is
    emitted when a run finishes, and ``file_process_status`` reports
    load / resample / separate / save progress.
    """

    signal_vr_over = Signal(bool)
    file_process_status = Signal(dict)

    def __init__(
        self,
        parent,
        audio: list,
        stems: int,
        model_path: str,
        *,
        segment: float = 6.0,
        overlap: float = 0.25,
        sample_rate: int = SAMPLE_RATE,
        output_path: str = "",
    ) -> None:
        super().__init__(parent)
        self.is_running = False
        self.model_path = model_path
        self.model = None
        self.audio = audio
        self.sampleRate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.stems = stems
        self.output_path = output_path

    def run(self) -> None:
        self.is_running = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        if not self.is_running:
            return

        self.file_process_status.emit({"file": "", "status": False, "task": "load model"})
        try:
            self.model = load_bandit_plus(
                self.model_path,
                device=device,
                progress=self._download_progress,
            )
        except Exception as exc:
            print(f"load model error: \n{exc}")
            self.signal_vr_over.emit(False)
            self.stop()
            return

        for audio in self.audio:
            if not self.is_running:
                self._release_cuda()
                print("exit")
                return

            self.file_process_status.emit({"file": audio, "status": False, "task": "reasmple audio"})
            print(f"current task: {audio}")
            print("resample audio...")
            try:
                samples = self.reSampleAudio(audio, self.sampleRate)
                print("samples shape: ", samples.shape)
            except Exception as exc:
                print(f"resample audio error:\n{exc}")
                self.signal_vr_over.emit(False)
                self.stop()
                return

            if not self.is_running:
                self._release_cuda()
                print("exit")
                return

            print("separate sources...")
            self.file_process_status.emit({"file": audio, "status": False, "task": "separate sources"})
            try:
                mixture = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
                separated = separate_mixture(
                    self.model,
                    mixture,
                    segment=self.segment,
                    overlap=self.overlap,
                    sample_rate=self.sampleRate,
                    device=device,
                    should_continue=lambda: self.is_running,
                )
            except Exception as exc:
                print(f"\nseparate audio sources error:\n    {exc}")
                self.signal_vr_over.emit(False)
                self.stop()
                return

            if separated is None or not self.is_running:
                print("exit")
                self._release_cuda()
                return

            self.file_process_status.emit({"file": audio, "status": False, "task": "save files"})
            print("save files...")
            try:
                self.saveResult(audio, separated)
            except Exception as exc:
                print(f"save audio file error:\n{exc}")
                self.signal_vr_over.emit(False)
                self.stop()
                return

            self.file_process_status.emit({"file": audio, "status": False, "task": "file over"})
            self._release_cuda()

        self.signal_vr_over.emit(True)
        print("over!")
        del self.model
        self.model = None
        self._release_cuda()
        gc.collect()
        self.requestInterruption()
        self.stop()

    def stop(self):
        self.is_running = False

    def _download_progress(self, downloaded, total):
        if downloaded == 0 or (total and downloaded >= total) or downloaded % (32 * 1024 * 1024) < 1024 * 1024:
            self.file_process_status.emit({"file": "", "status": False, "task": "download model"})
            if total:
                print(f"download model: {downloaded}/{total}")
            else:
                print(f"download model: {downloaded}")

    def _release_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reSampleAudio(self, audio, sample_rate) -> np.ndarray:
        file_path = os.path.abspath(audio)
        split_stereo = False
        import av
        with av.open(file_path) as av_file:
            stream_ = next(s for s in av_file.streams if s.codec_context.type == "audio")
            if stream_.channels >= 2:
                print("multi-channel audio")
                split_stereo = True
            else:
                print("single-channel audio")

        print("resample audio data")
        samples = decode_audio(file_path, sample_rate, split_stereo)
        return channel_first_audio(samples)

    def saveResult(self, file_path: str, separated: dict):
        data_dir, file_name = os.path.split(file_path)
        file_output = ".".join(file_name.split(".")[:-1]) or file_name
        track_index = normalize_bandit_track_index(self.stems)
        chosen = waveforms_for_track_index(separated, track_index, len(STEMS))
        print(f"output stems: {[name for name, _ in chosen]}")

        if not self.output_path:
            output_path = os.path.join(data_dir, file_output)
        else:
            output_path = os.path.join(self.output_path, file_output)

        if not os.path.exists(output_path):
            print(f"create output folder: {output_path}")
            os.makedirs(output_path, exist_ok=True)

        for stem, audio in chosen:
            waveform = audio[0].detach().cpu().numpy()
            output_file_name = os.path.join(output_path, f"{file_output}_{stem}.wav")
            print(f"save file: {output_file_name}")
            soundfile.write(output_file_name, np.ascontiguousarray(waveform.T), self.sampleRate)
