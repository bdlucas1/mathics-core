import numpy as np

from mathics.core.list import NumpyArrayListExpression

def test():

    # compare array a with (nested) list l
    def eq(a, l):
        if isinstance(a, np.ndarray):
            return all(eq(aa, ll) for aa, ll in zip(a, l))
        else:
            return a.item() == l.to_python()

    def test(a, s):
        l = NumpyArrayListExpression(a)
        assert not l.is_instantiated, "should not be instantiated immediately after construction"
        assert str(l) == s
        assert eq(a, l), "numpy array and instantiated list should be equal"
        assert l.is_instantiated, "should be instantiated after any use that needs .elements"

    # int, float, complex; empty, 1-d, 2-d
    test(np.array([]),                          "{}")
    test(np.array([1, 2, 3]),                   "{1,2,3}")
    test(np.array([[1, 2], [3, 4]]),            "{{1,2},{3,4}}")
    test(np.array([[17.5, 18.5], [2.2, 3.3]]),  "{{17.5,18.5},{2.2,3.3}}")
    test(np.array([17j+3]),                     "{3.0 + 17.0*I}")

    # unhandled type should raise an exception
    try:
        a = np.array([True, False, False])
        l = NumpyArrayListExpression(a)
        assert False, "expected failure for unhandled numpy type"
    except TypeError:
        pass
