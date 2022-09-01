"""Control flow-related instructions"""
from dataclasses import replace

import tensorflow as tf

from tensor_gp.data_types import Population, ExecutionContext
from tensor_gp.instr_support import instr, _pop, _read, _push_or_write


@instr
def c_no_op(population: Population, _context: ExecutionContext) -> Population:
    return population


@instr
def c_constant(population: Population, _context: ExecutionContext) -> Population:
    """Load an integer constant onto the stack."""
    values, instruction_pointers = _read(population.instructions, population.instruction_pointers)
    stacks, stack_pointers = _push_or_write(population.int_stacks,
                                            population.int_stack_pointers,
                                            values)
    return replace(population, int_stacks=stacks, int_stack_pointers=stack_pointers,
                   instruction_pointers=instruction_pointers)


@instr
def c_if(population: Population, _context: ExecutionContext) -> Population:
    """Pop an integer value from the stack and skip the next instruction if the popped value is
    zero or negative."""
    instruction_pointers = tf.convert_to_tensor(population.instruction_pointers)
    conditions, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    # noinspection PyTypeChecker
    instruction_pointers = instruction_pointers + tf.cast(conditions <= 0, tf.int32)
    return replace(population, int_stack_pointers=stack_pointers,
                   instruction_pointers=instruction_pointers)


@instr
def c_jump_rel(population: Population, _context: ExecutionContext) -> Population:
    """Pop an integer value from the stack and move the instruction pointer by that amount."""
    instruction_pointers = tf.convert_to_tensor(population.instruction_pointers)
    jump_offsets, stack_pointers = _pop(population.int_stacks, population.int_stack_pointers)
    instruction_pointers = instruction_pointers + jump_offsets
    return replace(population, int_stack_pointers=stack_pointers,
                   instruction_pointers=instruction_pointers)


@instr
def c_jump_abs(population: Population, _context: ExecutionContext) -> Population:
    """Pop an integer value from the stack and move the instruction pointer to that location."""
    instruction_pointers, stack_pointers = _pop(population.int_stacks,
                                                population.int_stack_pointers)
    return replace(population, int_stack_pointers=stack_pointers,
                   instruction_pointers=instruction_pointers)


# Specified in this order, instruction 0 maps to c_no_op.
CONTROL_INSTRUCTIONS = (
    c_no_op,
    c_constant,
    c_if,
    c_jump_rel,
    c_jump_abs,
)
