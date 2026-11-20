# coding:utf-8
"""Load the official BandIt Plus checkpoint and separate a mixture.

The checkpoint is the author's BandIt Plus model
(``dnr-3s-mus64-l1snr-plus``), confirmed to use the musical 64-band
configuration in ``configs/model/bsrnn-small-dnr-multi-mus64.yaml``.
It writes three cinematic stems: speech (dialogue), music, and effects.
"""

import hashlib
import os
import urllib.request

import numpy as np
import torch

from .model import MultiMaskMultiSourceBandSplitRNN


SAMPLE_RATE = 44100
MODEL_STEMS = ("speech", "music", "effects")
CHECKPOINT_FILENAME = "dnr-3s-mus64-l1snr-plus.ckpt"
CHECKPOINT_URL = (
    "https://zenodo.org/records/10160601/files/"
    "dnr-3s-mus64-l1snr-plus.ckpt?download=1"
)
CHECKPOINT_MD5 = "aac7e202a91de2c354a18606236a8038"
DEFAULT_CHECKPOINT_PATH = os.path.join("cache", CHECKPOINT_FILENAME)

# Kwargs from the official bsrnn-small-dnr-multi-mus64.yaml.
# use_freq_weights stays at the constructor default (True); the yaml
# does not override it.
BANDIT_PLUS_KWARGS = dict(
    in_channel=1,
    stems=["speech", "music", "effects"],
    band_specs="musical",
    n_bands=64,
    fs=SAMPLE_RATE,
    require_no_overlap=False,
    require_no_gap=True,
    normalize_channel_independently=False,
    treat_channel_as_feature=True,
    n_sqm_modules=8,
    emb_dim=128,
    rnn_dim=256,
    bidirectional=True,
    rnn_type="GRU",
    mlp_dim=512,
    hidden_activation="Tanh",
    hidden_activation_kwargs=None,
    complex_mask=True,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    window_fn="hann_window",
    wkwargs=None,
    power=None,
    center=True,
    normalized=True,
    pad_mode="constant",
    onesided=True,
)


def resolve_checkpoint_path(path=None) -> str:
    """Prefer ``BANDIT_PLUS_CHECKPOINT`` when it is set."""
    env_path = os.environ.get("BANDIT_PLUS_CHECKPOINT", "").strip()
    if env_path:
        return os.path.abspath(env_path)
    if path:
        return os.path.abspath(path)
    return os.path.abspath(DEFAULT_CHECKPOINT_PATH)


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint(path: str, progress=None) -> str:
    """Download the official Zenodo checkpoint when it is missing.

    ``progress`` is an optional callable ``progress(downloaded, total)``.
    A file that is already present and matches the published md5 is kept.
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        actual = _md5(path)
        if actual != CHECKPOINT_MD5:
            raise RuntimeError(
                f"{path} is not the official BandIt Plus checkpoint "
                f"(md5 {actual}, expected {CHECKPOINT_MD5}). "
                "Delete it to download the Zenodo file, or point "
                "BANDIT_PLUS_CHECKPOINT at a correct copy."
            )
        return path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    partial = path + ".part"
    request = urllib.request.Request(
        CHECKPOINT_URL,
        headers={"User-Agent": "faster-whisper-GUI"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        with open(partial, "wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if progress is not None:
                    progress(downloaded, total)

    os.replace(partial, path)
    actual = _md5(path)
    if actual != CHECKPOINT_MD5:
        raise RuntimeError(
            f"BandIt Plus checkpoint md5 mismatch: expected {CHECKPOINT_MD5}, got {actual}"
        )
    return path


def build_model():
    return MultiMaskMultiSourceBandSplitRNN(**BANDIT_PLUS_KWARGS)


def _state_dict_from_checkpoint(checkpoint_path: str):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise RuntimeError("BandIt Plus checkpoint does not contain a state_dict")

    stripped = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("model."):
            stripped[key[len("model."):]] = value
        elif key.startswith(("loss.", "metrics.", "test_metrics.", "fader.")):
            continue
        else:
            stripped[key] = value
    return stripped


def load_bandit_plus(checkpoint_path: str, device, progress=None):
    checkpoint_path = ensure_checkpoint(
        resolve_checkpoint_path(checkpoint_path),
        progress=progress,
    )
    model = build_model()
    state = _state_dict_from_checkpoint(checkpoint_path)
    incompatible = model.load_state_dict(state, strict=False)
    parameter_names = {name for name, _ in model.named_parameters()}
    missing_parameters = [
        key for key in incompatible.missing_keys if key in parameter_names
    ]
    if missing_parameters:
        preview = ", ".join(missing_parameters[:8])
        raise RuntimeError(
            "BandIt Plus checkpoint does not match this model "
            f"({len(missing_parameters)} missing parameters: {preview})"
        )
    model.to(device)
    model.eval()
    return model


def _chunk_window(valid, overlap_frames, is_first, is_last):
    window = torch.ones(valid)
    fade = int(overlap_frames)
    if fade <= 0 or (is_first and is_last):
        return window
    fade = min(fade, valid)
    if not is_first:
        window[:fade] = torch.linspace(0, 1, fade)
    if not is_last:
        # Complementary to the next chunk's fade-in, so the overlap sums to 1.
        window[-fade:] = torch.linspace(1, 0, fade)
    return window


def separate_mixture(
    model,
    mixture,
    segment=6.0,
    overlap=0.25,
    sample_rate=SAMPLE_RATE,
    device=None,
    should_continue=None,
):
    """Overlap-add BandIt Plus over ``mixture``.

    ``mixture`` is a float tensor shaped ``(batch, channels, samples)``.
    ``segment`` is the chunk length in seconds. ``overlap`` is the fraction
    of each chunk that overlaps the next chunk (0.25 is a 25% crossfade).

    Returns a dict of stem name to a CPU float tensor shaped
    ``(batch, channels, samples)``, or ``None`` when ``should_continue``
    reports that the caller asked to stop.
    """
    if device is None:
        device = mixture.device
    if mixture.ndim != 3:
        raise ValueError(
            f"expected mixture shaped (batch, channels, samples), got {tuple(mixture.shape)}"
        )

    batch, channels, length = mixture.shape
    chunk_len = max(int(float(segment) * sample_rate), sample_rate // 10)
    overlap = float(overlap)
    if overlap < 0:
        overlap = 0.0
    # Linear crossfades stay complementary only up to a 50% overlap.
    if overlap > 0.5:
        overlap = 0.5
    overlap_frames = int(round(chunk_len * overlap))
    if overlap_frames >= chunk_len:
        overlap_frames = chunk_len - 1
    hop = max(chunk_len - overlap_frames, 1)

    stems = list(model.stems)
    separated = {
        stem: torch.zeros(batch, channels, length, dtype=torch.float32)
        for stem in stems
    }
    weight = torch.zeros(length, dtype=torch.float32)

    start = 0
    while start < length:
        if should_continue is not None and not should_continue():
            return None

        end = min(start + chunk_len, length)
        valid = end - start
        is_first = start == 0
        is_last = end == length
        chunk = mixture[:, :, start:end]
        if valid < chunk_len:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_len - valid))

        chunk = chunk.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            _, output = model({"audio": {"mixture": chunk}})

        window = _chunk_window(valid, overlap_frames, is_first, is_last)
        for stem in stems:
            audio = output["audio"][stem][:, :, :valid].detach().to("cpu", torch.float32)
            separated[stem][:, :, start:end] += audio * window
        weight[start:end] += window

        del chunk
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_last:
            break
        start += hop

    weight = weight.clamp_min(1e-8)
    for stem in stems:
        separated[stem] /= weight
    return separated


def waveforms_for_track_index(separated, track_index: int, stem_count: int):
    """Pick the wav stems for a combo-box index.

    0 writes speech, music, and effects.
    1, 2, 3 write speech, music, or effects alone.
    The last index writes speech and a background stem (music + effects),
    which replaces the old vocals/accompaniment split.
    ``track_index`` must already be normalized into ``[0, stem_count)``.
    """
    speech = separated["speech"]
    music = separated["music"]
    effects = separated["effects"]

    if track_index <= 0:
        return [("speech", speech), ("music", music), ("effects", effects)]
    if track_index >= stem_count - 1:
        return [("speech", speech), ("background", music + effects)]
    return [(MODEL_STEMS[track_index - 1], separated[MODEL_STEMS[track_index - 1]])]


def channel_first_audio(samples) -> np.ndarray:
    """Normalize decode_audio output to float32 ``(channels, samples)``."""
    if isinstance(samples, tuple):
        channels = [np.asarray(channel, dtype=np.float32).reshape(-1) for channel in samples]
        return np.stack(channels, axis=0)

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio.reshape(1, -1)
    if audio.ndim != 2:
        raise ValueError(f"unexpected audio rank {audio.ndim}")
    if audio.shape[0] <= 2 and audio.shape[1] > 2:
        return np.ascontiguousarray(audio)
    if audio.shape[1] <= 2 and audio.shape[0] > 2:
        return np.ascontiguousarray(audio.T)
    return np.ascontiguousarray(audio)
