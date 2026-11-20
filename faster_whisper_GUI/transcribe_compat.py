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
