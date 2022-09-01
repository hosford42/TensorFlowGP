import math
from dataclasses import replace

import tensorflow as tf

from tensor_gp.data_types import Population, ExecutionContext
from tensor_gp.instr_support import instr, _read, _push_or_write, _pop


@instr
def f_read_input(population: Population, context: ExecutionContext) -> Population:
    """Read a floating point value from the input, incrementing the read pointer, and push the
    value on the stack."""
    input_values, input_pointers = _read(context.float_inputs[tf.newaxis],
                                         population.float_input_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks,
                                            population.float_stack_pointers,
                                            input_values)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers,
                   float_input_pointers=input_pointers)


@instr
def f_reset_input(population: Population, _context: ExecutionContext) -> Population:
    input_pointers = tf.convert_to_tensor(population.float_input_pointers)
    input_pointers = tf.zeros_like(input_pointers)
    return replace(population, float_input_pointers=input_pointers)


@instr
def f_prev_input(population: Population, context: ExecutionContext) -> Population:
    input_size = context.float_input_size
    input_pointers = (tf.convert_to_tensor(population.float_input_pointers) - 1) % input_size
    return replace(population, float_input_pointers=input_pointers)


@instr
def f_next_input(population: Population, context: ExecutionContext) -> Population:
    input_size = context.float_input_size
    input_pointers = (tf.convert_to_tensor(population.float_input_pointers) + 1) % input_size
    return replace(population, float_input_pointers=input_pointers)


@instr
def f_get_input_index(population: Population, _context: ExecutionContext) -> Population:
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            population.float_input_pointers)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def f_set_input_index(population: Population, _context: ExecutionContext) -> Population:
    input_pointers, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    return replace(population, float_stack_pointers=stack_pointers,
                   float_input_pointers=input_pointers)


@instr
def f_read_output(population: Population, _context: ExecutionContext) -> Population:
    output_values, output_pointers = _read(population.float_outputs,
                                           population.float_output_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks,
                                            population.float_stack_pointers,
                                            output_values)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers,
                   float_output_pointers=output_pointers)


@instr
def f_write_output(population: Population, _context: ExecutionContext) -> Population:
    output_values, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    outputs, output_pointers = _push_or_write(population.float_outputs,
                                              population.float_output_pointers,
                                              output_values)
    return replace(population, float_stack_pointers=stack_pointers, float_outputs=outputs,
                   float_output_pointers=output_pointers)


@instr
def f_reset_output(population: Population, _context: ExecutionContext) -> Population:
    output_pointers = tf.convert_to_tensor(population.float_output_pointers)
    output_pointers = tf.zeros_like(output_pointers)
    return replace(population, float_output_pointers=output_pointers)


@instr
def f_prev_output(population: Population, _context: ExecutionContext) -> Population:
    output_size = population.config.float_output_size
    output_pointers = (tf.convert_to_tensor(population.float_output_pointers) - 1) % output_size
    return replace(population, float_output_pointers=output_pointers)


@instr
def f_next_output(population: Population, _context: ExecutionContext) -> Population:
    output_size = population.config.float_output_size
    output_pointers = (tf.convert_to_tensor(population.float_output_pointers) + 1) % output_size
    return replace(population, float_output_pointers=output_pointers)


@instr
def f_get_output_index(population: Population, _context: ExecutionContext) -> Population:
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            population.float_output_pointers)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def f_set_output_index(population: Population, _context: ExecutionContext) -> Population:
    output_pointers, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    return replace(population, float_stack_pointers=stack_pointers,
                   float_output_pointers=output_pointers)


@instr
def f_read_memory(population: Population, _context: ExecutionContext) -> Population:
    memory_values, memory_pointers = _read(population.float_memories,
                                           population.float_memory_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks,
                                            population.float_stack_pointers,
                                            memory_values)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers,
                   float_memory_pointers=memory_pointers)


@instr
def f_write_memory(population: Population, _context: ExecutionContext) -> Population:
    values, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    memories, memory_pointers = _push_or_write(population.float_memories,
                                               population.float_memory_pointers,
                                               values)
    return replace(population, float_stack_pointers=stack_pointers, float_memories=memories,
                   float_memory_pointers=memory_pointers)


@instr
def f_reset_memory(population: Population, _context: ExecutionContext) -> Population:
    memory_pointers = tf.convert_to_tensor(population.float_memory_pointers)
    memory_pointers = tf.zeros_like(memory_pointers)
    return replace(population, float_memory_pointers=memory_pointers)


@instr
def f_prev_memory(population: Population, _context: ExecutionContext) -> Population:
    memory_size = population.config.memory_size
    memory_pointers = (tf.convert_to_tensor(population.float_memory_pointers) - 1) % memory_size
    return replace(population, float_memory_pointers=memory_pointers)


@instr
def f_next_memory(population: Population, _context: ExecutionContext) -> Population:
    memory_size = population.config.memory_size
    memory_pointers = (tf.convert_to_tensor(population.float_memory_pointers) + 1) % memory_size
    return replace(population, float_memory_pointers=memory_pointers)


@instr
def f_get_memory_index(population: Population, _context: ExecutionContext) -> Population:
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            population.float_memory_pointers)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def f_set_memory_index(population: Population, _context: ExecutionContext) -> Population:
    memory_pointers, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    return replace(population, float_stack_pointers=stack_pointers,
                   float_memory_pointers=memory_pointers)


@instr
def f_cast_from_int(population: Population, _context: ExecutionContext) -> Population:
    arg, i_stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    f_stacks, f_stack_pointers = _push_or_write(population.float_stacks,
                                                population.float_stack_pointers,
                                                tf.cast(arg, population.float_stacks.dtype))
    return replace(population, float_stacks=f_stacks, float_stack_pointers=f_stack_pointers,
                   int_stack_pointers=i_stack_pointers)


@instr
def f_round(population: Population, _context: ExecutionContext) -> Population:
    arg, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, tf.round(arg))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_floor(population: Population, _context: ExecutionContext) -> Population:
    arg, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers,
                                            tf.math.floor(arg))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_ceil(population: Population, _context: ExecutionContext) -> Population:
    arg, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers,
                                            tf.math.ceil(arg))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_add(population: Population, _context: ExecutionContext) -> Population:
    """Pop two floating point value from the stack and push their sum"""
    arg1, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    arg2, stack_pointers = _pop(population.float_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, arg1 + arg2)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_neg(population: Population, _context: ExecutionContext) -> Population:
    """Pop a floating point value from the stack and push its negation"""
    arg, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, -arg)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_abs(population: Population, _context: ExecutionContext) -> Population:
    """Pop a floating point value from the stack and push its absolute value"""
    arg, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, tf.abs(arg))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_sign(population: Population, _context: ExecutionContext) -> Population:
    """Pop a floating point value from the stack and push its sign"""
    arg, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, tf.sign(arg))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_mul(population: Population, _context: ExecutionContext) -> Population:
    """Pop two floating point value from the stack and push their product"""
    arg1, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    arg2, stack_pointers = _pop(population.float_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, arg1 * arg2)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_div(population: Population, _context: ExecutionContext) -> Population:
    """Pop two floating point values from the stack and push their quotient"""
    arg1, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    arg2, stack_pointers = _pop(population.float_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, arg1 / arg2)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_mod(population: Population, _context: ExecutionContext) -> Population:
    """Pop two floating point values and push arg1 % arg2."""
    arg1, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    arg2, stack_pointers = _pop(population.float_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, arg1 % arg2)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_pow(population: Population, _context: ExecutionContext) -> Population:
    """Pop two floating point values from the stack and push arg1 ** arg2, where arg1 is the first
    value popped and arg2 is the second value popped."""
    arg1, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    arg2, stack_pointers = _pop(population.float_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers,
                                            tf.pow(arg1, arg2))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_log(population: Population, _context: ExecutionContext) -> Population:
    """Pop a floating point values from the stack and push its natural logarithm."""
    arg, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers,
                                            tf.math.log(arg))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_exp(population: Population, _context: ExecutionContext) -> Population:
    """Pop a floating point values from the stack and push its natural exponent."""
    arg, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers,
                                            tf.math.exp(arg))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_is_pos(population: Population, _context: ExecutionContext) -> Population:
    """Pop a floating point value and push whether it is positive, as an integer."""
    arg, f_stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    # noinspection PyTypeChecker
    i_stacks, i_stack_pointers = _push_or_write(population.int_stacks,
                                                population.int_stack_pointers,
                                                tf.cast(arg > 0.0, tf.int32))
    return replace(population, float_stack_pointers=f_stack_pointers, int_stacks=i_stacks,
                   int_stack_pointers=i_stack_pointers)


@instr
def f_is_zero(population: Population, _context: ExecutionContext) -> Population:
    """Pop a floating point value and push whether it is zero, as an integer."""
    arg, f_stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    i_stacks, i_stack_pointers = _push_or_write(population.int_stacks,
                                                population.int_stack_pointers,
                                                tf.cast(arg == 0.0, tf.int32))
    return replace(population, float_stack_pointers=f_stack_pointers, int_stacks=i_stacks,
                   int_stack_pointers=i_stack_pointers)


@instr
def f_is_finite(population: Population, _context: ExecutionContext) -> Population:
    """Pop a floating point value and push whether it is finite, as an integer."""
    arg, f_stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    i_stacks, i_stack_pointers = _push_or_write(population.int_stacks,
                                                population.int_stack_pointers,
                                                tf.cast(tf.math.is_finite(arg), tf.int32))
    return replace(population, float_stack_pointers=f_stack_pointers, int_stacks=i_stacks,
                   int_stack_pointers=i_stack_pointers)


@instr
def f_pop(population: Population, _context: ExecutionContext) -> Population:
    """Discard the topmost value on the stack."""
    _values, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    return replace(population, float_stack_pointers=stack_pointers)


@instr
def f_dup(population: Population, _context: ExecutionContext) -> Population:
    """Duplicate the topmost value on the stack."""
    values, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, values)
    stacks, stack_pointers = _push_or_write(stacks, stack_pointers, values)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_swap(population: Population, _context: ExecutionContext) -> Population:
    """Swap the topmost values on the stack."""
    arg1, stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    arg2, stack_pointers = _pop(population.float_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.float_stacks, stack_pointers, arg1)
    stacks, stack_pointers = _push_or_write(stacks, stack_pointers, arg2)
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_pi(population: Population, _context: ExecutionContext) -> Population:
    """Swap the topmost values on the stack."""
    stacks, stack_pointers = _push_or_write(population.float_stacks,
                                            population.float_stack_pointers,
                                            tf.repeat(tf.constant(math.pi)[tf.newaxis],
                                                      tf.shape(population.float_stacks)[0],
                                                      axis=0))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


@instr
def f_e(population: Population, _context: ExecutionContext) -> Population:
    """Swap the topmost values on the stack."""
    stacks, stack_pointers = _push_or_write(population.float_stacks,
                                            population.float_stack_pointers,
                                            tf.repeat(tf.constant(math.e)[tf.newaxis],
                                                      tf.shape(population.float_stacks)[0],
                                                      axis=0))
    return replace(population, float_stacks=stacks, float_stack_pointers=stack_pointers)


FLOATING_POINT_INSTRUCTIONS = (
    f_read_input,
    f_reset_input,
    f_prev_input,
    f_next_input,
    f_get_input_index,
    f_set_input_index,

    f_read_output,
    f_write_output,
    f_prev_output,
    f_next_output,
    f_get_output_index,
    f_set_output_index,

    f_read_memory,
    f_write_memory,
    f_prev_memory,
    f_next_memory,
    f_get_memory_index,
    f_set_memory_index,

    f_cast_from_int,
    f_round,
    f_floor,
    f_ceil,

    f_add,
    f_neg,
    f_abs,
    f_sign,

    f_mul,
    f_div,
    f_mod,

    f_pow,
    f_log,
    f_exp,

    f_is_pos,
    f_is_zero,
    f_is_finite,

    f_pop,
    f_dup,
    f_swap,

    f_pi,
    f_e,
)
