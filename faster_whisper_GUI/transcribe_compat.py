# coding:utf-8
"""Call WhisperModel.transcribe with only the arguments that build accepts.

faster-whisper 0.10.0 rejects ``log_progress``. Later releases accept it, and
also accept options such as ``chunk_length`` and ``multilingual`` that 0.10.0
does not. Passing the whole GUI option set therefore raises TypeError on
0.10.0 at the first unknown name. Filtering by the installed signature keeps
0.10.0 working and still forwards ``log_progress`` where it exists. Progress
and the remaining-time estimate do not depend on that flag: they use the
returned duration and each segment's end time.
"""

from __future__ import annotations

import inspect
from typing import Callable, Tuple


# The GUI stores the speech threshold as ``onset``. faster-whisper 1.x names
# that field ``onset``; 0.10.0 still names it ``threshold``.
_THRESHOLD_ALIASES = ("onset", "threshold")


def _vad_field_names(vad_options_type):
    """Field names ``VadOptions(**params)`` will accept, or None to keep every key."""
    dataclass_fields = getattr(vad_options_type, "__dataclass_fields__", None)
    if dataclass_fields:
        return set(dataclass_fields)
    named_fields = getattr(vad_options_type, "_fields", None)
    if named_fields:
        return set(named_fields)
    try:
        signature = inspect.signature(vad_options_type)
    except (TypeError, ValueError):
        return None
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return None
    return {
        name
        for name, param in signature.parameters.items()
        if name not in ("self", "cls")
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def compatible_vad_parameters(vad_parameters, vad_options_type=None):
    """Return VAD parameters this faster-whisper build can construct.

    ``vad_filter`` stays the caller's choice. This only rewrites the parameter
    dict so ``VadOptions(**params)`` does not raise. The speech threshold is
    copied from ``onset`` to ``threshold`` (or the other way) when that is the
    name the installed class uses. Other known fields are passed through.
    """
    if not isinstance(vad_parameters, dict) or not vad_parameters:
        return vad_parameters, []

    if vad_options_type is None:
        try:
            from faster_whisper.vad import VadOptions
        except Exception:
            return dict(vad_parameters), []
        vad_options_type = VadOptions

    accepted = _vad_field_names(vad_options_type)
    if accepted is None:
        return dict(vad_parameters), []

    options = dict(vad_parameters)
    notes = []
    provided_threshold = next((options[name] for name in _THRESHOLD_ALIASES if name in options), None)
    if provided_threshold is not None:
        for name in _THRESHOLD_ALIASES:
            if name in accepted and name not in options:
                source = next(alias for alias in _THRESHOLD_ALIASES if alias in options)
                options[name] = provided_threshold
                notes.append(f"{source} -> {name}")
                break

    adapted = {}
    dropped = []
    for key, value in options.items():
        if key in accepted:
            adapted[key] = value
        elif not any(note.startswith(f"{key} ->") for note in notes):
            dropped.append(key)
    if dropped:
        notes.append("skip " + ", ".join(dropped))
    return adapted, notes


def compatible_transcribe_kwargs(transcribe: Callable, options: dict) -> Tuple[dict, list]:
    """Return ``(accepted, dropped_names)`` for one ``transcribe`` callable.

    If the signature cannot be read, or it accepts ``**kwargs``, every option
    is kept. Otherwise names the callable does not declare are omitted.
    """
    try:
        signature = inspect.signature(transcribe)
    except (TypeError, ValueError):
        return dict(options), []

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(options), []

    accepted_names = set(parameters)
    accepted = {}
    dropped = []
    for key, value in options.items():
        if key in accepted_names:
            accepted[key] = value
        else:
            dropped.append(key)
    return accepted, dropped


def ensure_ctranslate2_float32_features() -> None:
    """Cast float64 model inputs to float32 before CTranslate2 sees them.

    CTranslate2 4.9 ``StorageView.from_array`` rejects numpy float64. Its error
    is ``Unsupported type: <f8`` (``<f8`` is the little-endian float64 type
    string). That shows up on the first encoded window, after VAD and language
    info have already been printed. Mel features are documented as float32;
    casting float64 down to float32 keeps those values and lets encoding run.
    float32 arrays are left unchanged. VAD itself is not modified.
    """
    try:
        import numpy as np
        import faster_whisper.transcribe as fw_transcribe
    except Exception:
        return

    current = fw_transcribe.get_ctranslate2_storage
    if getattr(current, "_fwgui_float32", False):
        return

    def get_ctranslate2_storage(segment):
        if isinstance(segment, np.ndarray) and segment.dtype == np.float64:
            segment = np.ascontiguousarray(segment, dtype=np.float32)
        return current(segment)

    get_ctranslate2_storage._fwgui_float32 = True
    fw_transcribe.get_ctranslate2_storage = get_ctranslate2_storage
