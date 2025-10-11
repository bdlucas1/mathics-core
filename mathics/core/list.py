# -*- coding: utf-8 -*-
"""
Module containing ListExpression, LazyListExpression, and NumpyArrayListExpression
"""

import abc
import reprlib
from typing import Any, Optional, Sequence, Tuple, cast

import numpy as np

from mathics.core.atoms import Complex, Integer, Real
from mathics.core.convert.python import from_python
from mathics.core.element import BaseElement, ElementsProperties
from mathics.core.evaluation import Evaluation
from mathics.core.expression import Expression
from mathics.core.symbols import EvalMixin, Symbol, SymbolList


class ListExpression(Expression):
    """
    A Mathics3 List-Expression.

    A Mathics3 List is a specialization of Expression where the head is SymbolList.

    positional Arguments:

    - ``*elements`` - optional: the remaining elements

    Keyword Arguments:

    - ``elements_properties`` -- properties of the collection of elements
    - ``literal_values`` -- if this is not ``None``, then it is a Sequence of Python values and the expression is a literal.
    """

    _is_literal: bool
    _sympy: Optional[Any]

    # immutable, iterable, indexable
    # pyright seems to infer that it is not None from __init__, but mypy and pyrefly do not
    value: Sequence

    def __init__(
        self,
        *elements,
        elements_properties: Optional[ElementsProperties] = None,
        literal_values: Optional[Sequence] = None,
    ):
        self.options = None
        self.pattern_sequence = False
        self._head = SymbolList
        self._sympy = None

        # For debugging:

        # if literal_values is not None:
        #     import inspect

        #     curframe = inspect.currentframe()
        #     call_frame = inspect.getouterframes(curframe, 2)
        #     print("caller name:", call_frame[1][3])

        self._elements = elements

        # When self.value is not None it a Python Sequence (not Python
        # list) type that is the Python equivalent value for the Mathics3 list.

        # Check for literalness if it is not known
        if literal_values is not None:
            self._is_literal = True
            self.value = literal_values
        else:
            self._is_literal = True
            values = []
            for element in elements:
                if element.is_literal:
                    values.append(element.value)
                else:
                    self._is_literal = False
                    break
            if self._is_literal:
                self.value = tuple(values)

        self.elements_properties = elements_properties

        # FIXME: get rid of this junk
        self._sequences = None
        self._cache = None

    def __getitem__(self, index: int):
        """
        Allows ListExpression elements to accessed via [], e.g.
        ListExpression[Integer1, Integer0][0] == Integer1
        """
        return self._elements[index]

    def __repr__(self) -> str:
        """(reprlib.repr)-limited display or ListExpression"""
        list_data = reprlib.repr(self._elements)
        return f"<ListExpression: {list_data}>"

    def __str__(self) -> str:
        """str() representation of ListExpression. May be longer than repr()"""
        return "{" + ",".join(str(e) for e in self.elements) + "}"

    # @timeit
    def evaluate_elements(self, evaluation: Evaluation) -> Expression:
        """
        return a new expression with the same head, and the
        evaluable elements evaluated.
        """
        elements_changed = False
        # Make tuple self._elements mutable by turning it into a list.
        elements = list(self._elements)
        for index, element in enumerate(self._elements):
            if not element.has_form("Unevaluated", 1):
                if isinstance(element, EvalMixin):
                    new_value = element.evaluate(evaluation)
                    # We need id() because != by itself is too permissive
                    if new_value is not None and id(element) != id(new_value):
                        elements_changed = True
                        elements[index] = new_value

        if not elements_changed:
            return self

        new_list = ListExpression(*elements)
        new_list._build_elements_properties()
        return new_list

    @property
    def is_literal(self) -> bool:
        """
        True if the value can't change, i.e. a value is set and it does not
        depend on definition bindings. That is why, in contrast to
        `is_uncertain_final_definitions()` we don't need a `definitions`
        parameter.
        """
        return self._is_literal

    def rewrite_apply_eval_step(self, evaluation) -> Tuple[Expression, bool]:
        """
        Perform a single rewrite/apply/eval step of the bigger
        Expression.evaluate() process.

        We return the ListExpression as well as a Boolean False which indicates
        to the caller `evaluate()` that it should consider this a "fixed-point"
        evaluation and not have to iterate calling this routine using the
        returned results.

        Note that this is a recursive process: we may call something
        that may call our parent: evaluate() which calls us again.

        In general, this step is time consuming, complicated, and involved.
        However for a ListExpression, things are much much simpler and faster because
        we don't need the rewrite/apply phases of evaluation.

        """

        if self.elements_properties is None:
            self._build_elements_properties()
        assert self.elements_properties is not None
        if not self.elements_properties.elements_fully_evaluated:
            new = self.shallow_copy().evaluate_elements(evaluation)
            return new, False
        return self, False

    def set_head(self, head: Symbol):
        """
        Change the Head of an Expression.
        Unless this is a ListExpression, this is forbidden here.
        """
        if head != SymbolList:
            raise TypeError("Attempt to modify the Head of a ListExpression")

    def shallow_copy(self) -> "ListExpression":
        """
        For an Expression this does something with its cache.
        Here this does not need that complication.
        """
        return ListExpression(
            *self._elements, elements_properties=self.elements_properties
        )

    def copy(self, reevaluate=False) -> "Expression":
        expr = ListExpression(self._head.copy(reevaluate))
        expr._elements = tuple(element.copy(reevaluate) for element in self._elements)
        expr.options = self.options
        expr.original = self
        expr._sequences = self._sequences
        return expr


#
# Lazily evaluated list expressions
#


class LazyListExpression(ListExpression, abc.ABC):

    """
    A ListExpression with a supplied value that represents a list in
    some way, but no supplied elements. The elements will be
    instantiated on demand from the value. This allows the value to be
    stored and used efficiently but at the same to present a normal
    ListExpression view if required.
    """

    def __init__(self, value: Sequence) -> None:
        # we are a ListExpression with no .elements but a .value
        super().__init__(literal_values=value)

        # will be lazily computed if requested
        # subclass must provide self._make_elements for this purpose
        self.__elements: Tuple[BaseElement, ...] | None = None

    @abc.abstractmethod
    def _make_elements(self) -> Tuple[BaseElement, ...]:
        pass

    # TODO: pyrefly gives us side-eye here becase we're overriding a member attribute with a property
    # (but mypy and pyright do not complain...)
    # the justification I turned up with a little googling was that properties can violate the substitution principle
    # because they can change performance or semantics (e.g. by raising exceptions)
    # personally I don't buy it because the same is true of any overriden method, and if that is a concern of the type
    # system then exceptions and performance characteristics should be part of the type signatures of attributes and methods
    # and the ability to slide in a property to replace an attribute is kind of a core benefit of properties

    @property
    def _elements(self) -> Tuple[BaseElement, ...]:
        if not self.__elements:
            self.__elements = self._make_elements()
        return self.__elements

    @_elements.setter
    def _elements(self, e) -> None:
        self.__elements = e

    # this is primarily for testing
    @property
    def is_instantiated(self) -> bool:
        return self.__elements is not None


class NumpyArrayListExpression(LazyListExpression):

    """
    A lazily instantiated ListExpression backed by a numpy array.
    This allows data to be efficiently stored, transmitted, and
    accessed efficiently as a numpy array by a collaborating source
    and recipient, only instantiating the inefficent elements
    representation if required by some third party.
    """

    def __init__(self, value: np.ndarray):
        # TODO: np.ndarray is not assignable to Sequence because nominal vs structural, but I think this is safe
        # also since .value is declared Sequence typechecker will complain if user tries to modify the array
        # even though arrays allow it; I guess this is a good thing
        super().__init__(cast(Sequence, value))

    # lazy computation of elements from numpy array
    def _make_elements(self) -> Tuple[BaseElement, ...]:
        def np_to_m(v) -> BaseElement | NumpyArrayListExpression:
            if isinstance(v, np.ndarray):
                return NumpyArrayListExpression(v)
            else:
                return from_python(v.item())

        return tuple(np_to_m(v) for v in self.value)
