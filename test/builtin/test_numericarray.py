import numpy as np

from mathics.builtin.numericarray import NumericArray as NumericArrayBuiltin
from mathics.core.atoms import Integer, NumericArray, String
from mathics.core.convert.python import from_python
from mathics.core.expression import Expression
from mathics.core.list import ListExpression
from mathics.core.systemsymbols import SymbolAutomatic, SymbolNumericArray


def test_numericarray_atom_preserves_array_reference():
    arr = np.array([1, 2, 3], dtype=np.int64)
    atom = NumericArray(arr)
    assert atom.value is arr
    np.testing.assert_array_equal(atom.value, arr)


def test_numericarray_expression_from_python_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    expr = from_python(arr)
    assert expr.head is SymbolNumericArray
    atom = expr.elements[0]
    np.testing.assert_array_equal(atom.value, arr)
    assert expr.value is atom.value


class DummyEvaluation:
    def message(self, *args, **kwargs):
        raise AssertionError(f"Unexpected message: {args!r}")


def test_numericarray_builtin_default_type():
    evaluation = DummyEvaluation()
    builtin = NumericArrayBuiltin(expression=False)
    data = from_python([1, 2, 3])
    result = builtin.eval(data, SymbolAutomatic, evaluation)
    assert result.head is SymbolNumericArray
    atom = result.elements[0]
    np.testing.assert_array_equal(atom.value, np.array([1, 2, 3]))


def test_numericarray_builtin_type_conversion():
    evaluation = DummyEvaluation()
    builtin = NumericArrayBuiltin(expression=False)
    data = from_python([1, 2, 3])
    type_spec = String("UnsignedInteger16")
    result = builtin.eval(data, type_spec, evaluation)
    atom = result.elements[0]
    assert atom.value.dtype == np.uint16


def test_numericarray_normal_returns_list_expression():
    evaluation = DummyEvaluation()
    builtin = NumericArrayBuiltin(expression=False)
    arr = np.array([1, 2], dtype=np.int64)
    atom = NumericArray(arr)
    normal_expr = builtin.eval_normal(atom, evaluation)
    assert normal_expr.value == (1, 2)


def test_numericarray_from_expression_nested_lists():
    list_expr = ListExpression(
        ListExpression(Integer(1), Integer(2)),
        ListExpression(Integer(3), Integer(4)),
    )
    expr = Expression(SymbolNumericArray, list_expr)
    atom = NumericArray.from_expression(expr)
    np.testing.assert_array_equal(atom.value, np.array([[1, 2], [3, 4]]))
