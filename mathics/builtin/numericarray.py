# -*- coding: utf-8 -*-
"""Rules for working with NumericArray atoms."""

from typing import Optional, Tuple

try:  # pragma: no cover - numpy is optional at runtime
    import numpy
except ImportError:  # pragma: no cover - handled via requires attribute
    numpy = None

from mathics.core.atoms import NumericArray as NumericArrayAtom, NUMERIC_ARRAY_TYPE_MAP, String
from mathics.core.builtin import Builtin
from mathics.core.convert.python import from_python
from mathics.core.symbols import Symbol, strip_context
from mathics.core.systemsymbols import SymbolAutomatic, SymbolFailed, SymbolNumericArray


# name modeled on Complex_ to avoid collision with NumericArray atom
class NumericArray_(Builtin):

    summary_text = "head for NumericArray"
    name = "NumericArray"
    rules = {
        "NumericArray[list_List]": "NumericArray[list, Automatic]"
    }
    messages = {
        "type": "The type specification `1` is not supported in NumericArray.",
    }

    # rule to convert NumericArray[...nested list...] expression to NumericArray atom
    def eval_list(self, data, typespec, evaluation):
        "System`NumericArray[data_List, typespec_]"

        # get a string key from the typespec
        if isinstance(typespec, Symbol):
            key = strip_context(typespec.get_name())
        elif isinstance(typespec, String):
            key = typespec.value
        else:
            evaluation.message("NumericArray", "type", typespec)            
            return SymbolFailed

        # compute numpy dtype from key
        if key == "Automatic":
            dtype = None
        else:
            dtype = NUMERIC_ARRAY_TYPE_MAP.get(key, None)
            if not dtype:
                evaluation.message("NumericArray", "type", typespec)
                return SymbolFailed
                
        # compute array from data and dtype and wrap it in a NumericArray atom
        python_value = data.to_python()
        array = numpy.array(python_value, dtype=dtype)
        atom = NumericArrayAtom(array, dtype)

        print("xxx eval_list returning", type(atom), atom)
        return atom

    def eval_normal(self, array, evaluation):
        "System`Normal[array_NumericArray]"
        print("xxx eval_normal", type(array))
        return from_python(array.value.tolist())

    # TODO: is this needed or does it happen by default?
    #def eval_to_string(self, array, evaluation):
    #    "ToString[array_NumericArray]"
    #    return String(str(array))

