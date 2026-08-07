"""
Compatibility shim for pandas 1.x pickles loaded under pandas 2.x.

pandas 2.0 removed pandas.core.indexes.numeric (Int64Index, Float64Index,
UInt64Index were merged into pandas.Index). The ASOS GraphReturns pickle
files were created with pandas 1.x and reference these removed modules,
causing ModuleNotFoundError on unpickling under pandas 2.x.

This is the single implementation of the shim. Import this module before
unpickling any raw dataset file:

    import ml.data.compat  # noqa: F401 — must precede pickle.load()
"""

import sys
import types

import pandas

if "pandas.core.indexes.numeric" not in sys.modules:
    _numeric = types.ModuleType("pandas.core.indexes.numeric")
    _numeric.Int64Index = pandas.Index
    _numeric.Float64Index = pandas.Index
    _numeric.UInt64Index = pandas.Index
    sys.modules["pandas.core.indexes.numeric"] = _numeric

if "pandas.core.indexes.frozen" not in sys.modules:
    try:
        import pandas.core.indexes.frozen  # noqa: F401
    except ImportError:
        _frozen = types.ModuleType("pandas.core.indexes.frozen")
        _frozen.FrozenList = list
        sys.modules["pandas.core.indexes.frozen"] = _frozen
