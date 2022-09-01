from dataclasses import fields
from functools import wraps
from typing import Callable, Tuple

import tensorflow as tf
from tensorflow.python.types.core import TensorLike

from tensor_gp.astuple_fix import astuple
from tensor_gp.data_types import Population, ExecutionContext, PopulationConfig


# Every instruction should be decorated with @instr and have the type signature
# Callable[[Population, ExecutionContext], Population].


def instr(f: Callable[[Population, ExecutionContext], Population]) \
        -> Callable[[Tuple[TensorLike, ...], Tuple[TensorLike, ...]], Tuple[TensorLike, ...]]:
    f_name = f.__name__
    assert f_name

    @tf.function
    @wraps(f)
    def wrapper(population: Tuple[TensorLike, ...],
                context: Tuple[TensorLike, ...],
                config: PopulationConfig) -> Tuple[TensorLike, ...]:
        result = f(Population(config, *population), ExecutionContext(*context))
        assert isinstance(result, Population)
        result.check_shapes()
        result = astuple(result, deepcopy=False)
        return tuple(tf.identity(value, name=f_name + '/' + field.name)
                     for field, value in zip(fields(Population), result))[1:]

    # noinspection PyTypeChecker
    return wrapper


@tf.function
def _push_or_write(stacks: TensorLike, pointers: TensorLike,
                   values: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    stacks = tf.convert_to_tensor(stacks)
    pointers = tf.convert_to_tensor(pointers)
    values = tf.convert_to_tensor(values)

    empty = tf.size(stacks) == 0
    if empty:
        # It should be fine to return early here, but tensorflow seems to have a bug:
        #   https://github.com/tensorflow/tensorflow/issues/57492
        # So instead, stacks is temporarily modified to not be empty.
        stacks = tf.concat([stacks, tf.ones([tf.shape(stacks)[0], 1], dtype=stacks.dtype)], axis=-1)

    tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(values))
    tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(pointers))

    indices = pointers % tf.shape(stacks)[-1]
    stacks = tf.tensor_scatter_nd_update(
        stacks,
        tf.concat([tf.range(tf.shape(stacks)[0])[:, tf.newaxis], indices[:, tf.newaxis]],
                  axis=-1),
        values
    )
    pointers = indices + 1

    tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(pointers))

    if empty:
        # See above for an explanation.
        stacks = stacks[..., :-1]

    return stacks, pointers


# @tf.function
# def _push_or_write(stacks: TensorLike, pointers: TensorLike,
#                    values: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
#     stacks = tf.convert_to_tensor(stacks)
#     pointers = tf.convert_to_tensor(pointers)
#     values = tf.convert_to_tensor(values)
#
#     # No-op for empty stacks/arrays
#     if tf.size(stacks) == 0:
#         return stacks, pointers
#
#     tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(values))
#     tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(pointers))
#
#     indices = pointers % tf.shape(stacks)[-1]
#     stacks = tf.tensor_scatter_nd_update(
#         stacks,
#         tf.concat([tf.range(tf.shape(stacks)[0])[:, tf.newaxis], indices[:, tf.newaxis]],
#                   axis=-1),
#         values
#     )
#     pointers = indices + 1
#
#     tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(pointers))
#     return stacks, pointers


def _pop(stacks: TensorLike, pointers: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    stacks = tf.convert_to_tensor(stacks)
    pointers = tf.convert_to_tensor(pointers)
    tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(pointers))

    indices = (pointers - 1) % tf.shape(stacks)[-1]
    values = tf.gather(stacks, indices, batch_dims=1)
    pointers = indices

    tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(values))
    tf.assert_equal(tf.shape(stacks)[:-1], tf.shape(pointers))
    return values, pointers


def _read(array: TensorLike, pointers: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    array = tf.convert_to_tensor(array)
    pointers = tf.convert_to_tensor(pointers)

    if tf.size(array) == 0:
        values = tf.zeros_like(pointers, dtype=array.dtype)
        return values, pointers

    tf.assert_rank(array, tf.rank(pointers) + 1)
    array = tf.broadcast_to(array, tf.concat([tf.shape(pointers), tf.shape(array)[-1:]], axis=-1))
    tf.assert_equal(tf.shape(array)[:-1], tf.shape(pointers))

    indices = pointers % tf.shape(array)[-1]
    values = tf.gather(array, indices, batch_dims=1)

    pointers = indices + 1

    tf.assert_equal(tf.shape(array)[:-1], tf.shape(values))
    tf.assert_equal(tf.shape(array)[:-1], tf.shape(pointers))
    return values, pointers
