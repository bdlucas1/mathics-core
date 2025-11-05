# -*- coding: utf-8 -*-
"""NumericArray backed by NumPy arrays."""

from __future__ import annotations

from typing import Optional, Tuple

try:  # pragma: no cover - numpy is optional at runtime
    import numpy
except ImportError:  # pragma: no cover - handled via requires attribute
    numpy = None

from mathics.core.atoms import (
    NumericArrayAtom,
    NUMERIC_ARRAY_TYPE_MAP,
    String,
    numeric_array_summary_string,
)
from mathics.core.builtin import Builtin
from mathics.core.expression import Expression
from mathics.core.symbols import Symbol, strip_context
from mathics.core.systemsymbols import SymbolAutomatic, SymbolFailed, SymbolNumericArray


def _is_numeric_dtype(dtype: numpy.dtype) -> bool:
    if numpy is None:
        raise ImportError("numpy is required for NumericArray")
    return (
        numpy.issubdtype(dtype, numpy.integer)
        or numpy.issubdtype(dtype, numpy.floating)
        or numpy.issubdtype(dtype, numpy.complexfloating)
    )


class NumericArray(Builtin):
    """Implementation of Mathematica's NumericArray based on NumPy."""

    requires = ("numpy",)
    rules = {"NumericArray[data_]": "NumericArray[data, Automatic]"}
    summary_text = "array of numbers stored efficiently"
    messages = {
        "type": "The type specification `1` is not supported in NumericArray.",
        "data": "Numeric data expected at position 1 in NumericArray[`1`].",
    }

    def eval(self, data, typespec, evaluation):
        "NumericArray[data_, typespec_]"

        if numpy is None:
            raise ImportError("numpy is required for NumericArray")

        dtype, valid = self._resolve_dtype(typespec, evaluation)
        if not valid:
            return SymbolFailed

        array = self._coerce_to_array(data, dtype)
        if array is None:
            evaluation.message("NumericArray", "data", data)
            return SymbolFailed

        atom = NumericArrayAtom(array, dtype)
        return Expression(
            SymbolNumericArray,
            atom,
            literal_values=atom.value,
        )

    def eval_normal(self, array_atom, evaluation):
        "System`Normal[NumericArray[array_NumericArrayAtom]]"

        from mathics.core.convert.python import from_python

        return from_python(array_atom.value.tolist())

    def eval_to_string(self, array_atom, evaluation):
        "ToString[NumericArray[array_NumericArrayAtom]]"

        summary = numeric_array_summary_string(array_atom.value)
        return String(f"NumericArray[<{summary}>]")

    def _resolve_dtype(
        self, typespec, evaluation
    ) -> Tuple[Optional[numpy.dtype], bool]:
        if isinstance(typespec, Symbol):
            if typespec is SymbolAutomatic:
                return None, True
            key = strip_context(typespec.get_name())
        elif isinstance(typespec, String):
            if typespec.value == "Automatic":
                return None, True
            key = typespec.value
        else:
            try:
                key = typespec.to_python(string_quotes=False)
            except Exception:
                key = None
            if key == "Automatic":
                return None, True

        if not key:
            evaluation.message("NumericArray", "type", typespec)
            return None, False

        dtype = NUMERIC_ARRAY_TYPE_MAP.get(key)
        if dtype is None:
            try:
                dtype = numpy.dtype(key)
            except (TypeError, ValueError):
                evaluation.message("NumericArray", "type", typespec)
                return None, False

        if not _is_numeric_dtype(dtype):
            evaluation.message("NumericArray", "type", typespec)
            return None, False

        return dtype, True

    def _coerce_to_array(
        self, data, dtype: Optional[numpy.dtype]
    ) -> Optional[numpy.ndarray]:
        if numpy is None:
            raise ImportError("numpy is required for NumericArray")
        if isinstance(data, NumericArrayAtom):
            array = data.value
        elif getattr(data, "get_head_name", None) and data.get_head_name() == "System`NumericArray":
            if hasattr(data, "value") and isinstance(data.value, numpy.ndarray):
                array = data.value
            elif data.elements and isinstance(data.elements[0], NumericArrayAtom):
                array = data.elements[0].value
            else:
                return None
        elif hasattr(data, "is_literal") and data.is_literal:
            array = numpy.asarray(data.value, dtype=dtype)
        else:
            try:
                python_value = data.to_python(string_quotes=False)
            except Exception:
                return None
            try:
                array = numpy.asarray(python_value, dtype=dtype)
            except Exception:
                return None

        if dtype is not None and array.dtype != numpy.dtype(dtype):
            array = array.astype(dtype)

        if array.dtype == numpy.dtype(object):
            return None

        if not _is_numeric_dtype(array.dtype):
            return None

        return array

