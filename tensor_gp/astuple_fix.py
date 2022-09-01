"""The builtin dataclasses.astuple forces a deep copy. Tensors don't support deep copy in graph
mode. This module provides a replacement for astuple which offers a flag to turn off the deep copy
behavior. The code in this file is copied verbatim from dataclasses.py, and minimal modifications
are applied to add support for the deepcopy flag."""

import copy
from dataclasses import fields


# The name of an attribute on the class where we store the Field
# objects.  Also used to check if a class is a Data Class.
_FIELDS = '__dataclass_fields__'


def _is_dataclass_instance(obj):
    """Returns True if obj is an instance of a dataclass."""
    return hasattr(type(obj), _FIELDS)


def astuple(obj, *, tuple_factory=tuple, deepcopy=True):
    """Return the fields of a dataclass instance as a new tuple of field values.

    Example usage::

      @dataclass
      class C:
          x: int
          y: int

    c = C(1, 2)
    assert astuple(c) == (1, 2)

    If given, 'tuple_factory' will be used instead of built-in tuple.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts.
    """

    if not _is_dataclass_instance(obj):
        raise TypeError("astuple() should be called on dataclass instances")
    return _astuple_inner(obj, tuple_factory, deepcopy=deepcopy)


def _astuple_inner(obj, tuple_factory, deepcopy):
    if _is_dataclass_instance(obj):
        result = []
        for f in fields(obj):
            value = _astuple_inner(getattr(obj, f.name), tuple_factory, deepcopy)
            result.append(value)
        return tuple_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).
        return type(obj)(*[_astuple_inner(v, tuple_factory, deepcopy) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_astuple_inner(v, tuple_factory, deepcopy) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_astuple_inner(k, tuple_factory, deepcopy),
                          _astuple_inner(v, tuple_factory, deepcopy))
                         for k, v in obj.items())
    elif deepcopy:
        return copy.deepcopy(obj)
    else:
        return copy.copy(obj)
