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
        a = np.array(a)
        l = NumpyArrayListExpression(a)
        assert not l.is_instantiated, "should not be instantiated immediately after construction"
        assert eq(a, l.elements), "numpy array and instantiated list should be equal"
        assert l.is_instantiated, "should be instantiated after any use that needs .elements"
        assert str(l) == s

    # int, float, complex, bool, mixed; empty, 1-d, 2-d
    test([],                          "{}")
    test([1, 2, 3],                   "{1,2,3}")
    test([[1, 2], [3, 4]],            "{{1,2},{3,4}}")
    test([[17.5, 18.5], [2.2, 3.3]],  "{{17.5,18.5},{2.2,3.3}}")
    test([17j+3],                     "{3.0 + 17.0*I}")
    test([True, False],               "{System`True,System`False}")
    test([True, False, 17, 2.2],      "{1.0,0.0,17.0,2.2}")
