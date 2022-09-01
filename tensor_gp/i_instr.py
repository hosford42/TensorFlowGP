from dataclasses import replace

import tensorflow as tf

from tensor_gp.data_types import Population, ExecutionContext
from tensor_gp.instr_support import instr, _read, _push_or_write, _pop


@instr
def i_read_input(population: Population, context: ExecutionContext) -> Population:
    """Read an integer value from the input, incrementing the read pointer, and push the
    value on the stack."""
    input_values, input_pointers = _read(context.int_inputs[tf.newaxis],
                                         population.int_input_pointers)
    # print(input_values.shape, input_pointers.shape, population.int_stacks.shape,
    #       population.int_stack_pointers.shape)
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            input_values)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers,
                   int_input_pointers=input_pointers)


@instr
def i_reset_input(population: Population, _context: ExecutionContext) -> Population:
    input_pointers = tf.convert_to_tensor(population.int_input_pointers)
    input_pointers = tf.zeros_like(input_pointers)
    return replace(population, int_input_pointers=input_pointers)


@instr
def i_prev_input(population: Population, context: ExecutionContext) -> Population:
    input_size = context.int_input_size
    if input_size == 0:
        return population
    input_pointers = (tf.convert_to_tensor(population.int_input_pointers) - 1) % input_size
    return replace(population, int_input_pointers=input_pointers)


@instr
def i_next_input(population: Population, context: ExecutionContext) -> Population:
    input_size = context.int_input_size
    if input_size == 0:
        return population
    input_pointers = (tf.convert_to_tensor(population.int_input_pointers) + 1) % input_size
    return replace(population, int_input_pointers=input_pointers)


@instr
def i_get_input_index(population: Population, _context: ExecutionContext) -> Population:
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            population.int_input_pointers)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_set_input_index(population: Population, _context: ExecutionContext) -> Population:
    input_pointers, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    return replace(population, int_stack_pointers=stack_pointers,
                   int_input_pointers=input_pointers)


@instr
def i_read_output(population: Population, _context: ExecutionContext) -> Population:
    output_values, output_pointers = _read(population.int_outputs,
                                           population.int_output_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            output_values)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers,
                   int_output_pointers=output_pointers)


@instr
def i_write_output(population: Population, _context: ExecutionContext) -> Population:
    output_values, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    outputs, output_pointers = _push_or_write(population.int_outputs,
                                              population.int_output_pointers,
                                              output_values)
    return replace(population, int_stack_pointers=stack_pointers, int_outputs=outputs,
                   int_output_pointers=output_pointers)


@instr
def i_reset_output(population: Population, _context: ExecutionContext) -> Population:
    output_pointers = tf.convert_to_tensor(population.int_output_pointers)
    output_pointers = tf.zeros_like(output_pointers)
    return replace(population, int_output_pointers=output_pointers)


@instr
def i_prev_output(population: Population, _context: ExecutionContext) -> Population:
    output_size = population.config.int_output_size
    if output_size == 0:
        return population
    output_pointers = (tf.convert_to_tensor(population.int_output_pointers) - 1) % output_size
    return replace(population, int_output_pointers=output_pointers)


@instr
def i_next_output(population: Population, _context: ExecutionContext) -> Population:
    output_size = population.config.int_output_size
    if output_size == 0:
        return population
    output_pointers = (tf.convert_to_tensor(population.int_output_pointers) + 1) % output_size
    return replace(population, int_output_pointers=output_pointers)


@instr
def i_get_output_index(population: Population, _context: ExecutionContext) -> Population:
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            population.int_output_pointers)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_set_output_index(population: Population, _context: ExecutionContext) -> Population:
    output_pointers, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    return replace(population, int_stack_pointers=stack_pointers,
                   int_output_pointers=output_pointers)


@instr
def i_read_memory(population: Population, _context: ExecutionContext) -> Population:
    memory_values, memory_pointers = _read(population.int_memories,
                                           population.int_memory_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            memory_values)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers,
                   int_memory_pointers=memory_pointers)


@instr
def i_write_memory(population: Population, _context: ExecutionContext) -> Population:
    values, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    memories, memory_pointers = _push_or_write(population.int_memories,
                                               population.int_memory_pointers,
                                               values)
    return replace(population, int_stack_pointers=stack_pointers, int_memories=memories,
                   int_memory_pointers=memory_pointers)


@instr
def i_reset_memory(population: Population, _context: ExecutionContext) -> Population:
    memory_pointers = tf.convert_to_tensor(population.int_memory_pointers)
    memory_pointers = tf.zeros_like(memory_pointers)
    return replace(population, int_memory_pointers=memory_pointers)


@instr
def i_prev_memory(population: Population, _context: ExecutionContext) -> Population:
    memory_size = population.config.memory_size
    memory_pointers = (tf.convert_to_tensor(population.int_memory_pointers) - 1) % memory_size
    return replace(population, int_memory_pointers=memory_pointers)


@instr
def i_next_memory(population: Population, _context: ExecutionContext) -> Population:
    memory_size = population.config.memory_size
    memory_pointers = (tf.convert_to_tensor(population.int_memory_pointers) + 1) % memory_size
    return replace(population, int_memory_pointers=memory_pointers)


@instr
def i_get_memory_index(population: Population, _context: ExecutionContext) -> Population:
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            population.int_memory_pointers)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_set_memory_index(population: Population, _context: ExecutionContext) -> Population:
    memory_pointers, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    return replace(population, int_stack_pointers=stack_pointers,
                   int_memory_pointers=memory_pointers)


@instr
def i_cast_from_float(population: Population, _context: ExecutionContext) -> Population:
    arg, f_stack_pointers = _pop(population.float_stacks, population.float_stack_pointers)
    i_stacks, i_stack_pointers = _push_or_write(population.int_stacks,
                                                population.int_stack_pointers,
                                                tf.cast(arg, population.int_stacks.dtype))
    return replace(population, float_stack_pointers=f_stack_pointers, int_stacks=i_stacks,
                   int_stack_pointers=i_stack_pointers)


@instr
def i_add(population: Population, _context: ExecutionContext) -> Population:
    """Pop two integer value from the stack and push their sum"""
    arg1, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    arg2, stack_pointers = _pop(population.int_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers, arg1 + arg2)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_neg(population: Population, _context: ExecutionContext) -> Population:
    """Pop an integer value from the stack and push its negation"""
    arg, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers, -arg)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_abs(population: Population, _context: ExecutionContext) -> Population:
    """Pop an integer value from the stack and push its absolute value"""
    arg, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers, tf.abs(arg))
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_sign(population: Population, _context: ExecutionContext) -> Population:
    """Pop an integer value from the stack and push its sign"""
    arg, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers, tf.sign(arg))
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_mul(population: Population, _context: ExecutionContext) -> Population:
    """Pop two integer value from the stack and push their product"""
    arg1, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    arg2, stack_pointers = _pop(population.int_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers, arg1 * arg2)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_div(population: Population, _context: ExecutionContext) -> Population:
    """Pop two integer value from the stack and push their integer quotient"""
    arg1, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    arg2, stack_pointers = _pop(population.int_stacks, stack_pointers)
    zero_div = arg2 == 0
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers,
                                            (arg1 // tf.where(zero_div, 1, arg2)))
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_mod(population: Population, _context: ExecutionContext) -> Population:
    """Pop two integer values and push arg1 % arg2."""
    arg1, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    arg2, stack_pointers = _pop(population.int_stacks, stack_pointers)
    zero_div = arg2 == 0
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers,
                                            arg1 % tf.where(zero_div, 1, arg2))
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_pow(population: Population, _context: ExecutionContext) -> Population:
    """Pop two integer values from the stack and push arg1 ** abs(arg2), where arg1 is the first
    value popped and arg2 is the second value popped."""
    arg1, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    arg2, stack_pointers = _pop(population.int_stacks, stack_pointers)
    # Surprisingly, tf.abs(tf.int32.min) just returns tf.int32.min, a negative value. This is
    # because of how int32 is represented, with no way to represent -tf.int32.min. So as an awkward
    # workaround, we just make sure tf.int32.min gets mapped to the closest representable positive
    # value, which is -(tf.int32.min + 1).
    arg2 = tf.abs(tf.maximum(arg2, arg2.dtype.min + 1))
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers,
                                            tf.pow(arg1, arg2))
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_is_pos(population: Population, _context: ExecutionContext) -> Population:
    """Pop an integer value and push whether it is positive, as an integer."""
    arg, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    # noinspection PyTypeChecker
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            stack_pointers,
                                            tf.cast(arg > 0, tf.int32))
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_is_zero(population: Population, _context: ExecutionContext) -> Population:
    """Pop an integer value and push whether it is zero, as an integer."""
    arg, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            stack_pointers,
                                            tf.cast(arg == 0, tf.int32))
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_pop(population: Population, _context: ExecutionContext) -> Population:
    """Discard the topmost value on the stack."""
    _values, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    return replace(population, int_stack_pointers=stack_pointers)


@instr
def i_dup(population: Population, _context: ExecutionContext) -> Population:
    """Duplicate the topmost value on the stack."""
    values, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers, values)
    stacks, stack_pointers = _push_or_write(stacks, stack_pointers, values)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


@instr
def i_swap(population: Population, _context: ExecutionContext) -> Population:
    """Swap the topmost values on the stack."""
    arg1, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    arg2, stack_pointers = _pop(population.int_stacks, stack_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks, stack_pointers, arg1)
    stacks, stack_pointers = _push_or_write(stacks, stack_pointers, arg2)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers)


INTEGER_INSTRUCTIONS = (
    i_read_input,
    i_reset_input,
    i_prev_input,
    i_next_input,
    i_get_input_index,
    i_set_input_index,

    i_read_output,
    i_write_output,
    i_prev_output,
    i_next_output,
    i_get_output_index,
    i_set_output_index,

    i_read_memory,
    i_write_memory,
    i_prev_memory,
    i_next_memory,
    i_get_memory_index,
    i_set_memory_index,

    i_cast_from_float,

    i_add,
    i_neg,
    i_abs,
    i_sign,

    i_mul,
    i_div,
    i_mod,

    i_pow,

    i_is_pos,
    i_is_zero,

    i_pop,
    i_dup,
    i_swap,
)
