"""Importable compatibility shim for the original ODISE helper.

The original file is deliberately preserved under its historical Chinese filename.
This shim makes its ``OdiseRefiner`` class importable by ``run_zerors.py`` without
duplicating the implementation.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_LEGACY_FILE = Path(__file__).with_name("step2_5_utils(最终需要改,有关odise).py")
_SPEC = spec_from_file_location("zerors_legacy_odise", _LEGACY_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load legacy ODISE helper: {_LEGACY_FILE}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

OdiseRefiner = _MODULE.OdiseRefiner

__all__ = ["OdiseRefiner"]
