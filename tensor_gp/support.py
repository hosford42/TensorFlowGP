from typing import Sequence, Callable

import tensorflow as tf

from tensor_gp.c_instr import c_constant
from tensor_gp.data_types import Population, PopulationConfig


def new_population(config: PopulationConfig) -> Population:
    # The code of each organism.
    instructions = tf.Variable(
        initial_value=tf.random.uniform((config.population_size, config.organism_size),
                                        tf.int32.min, tf.int32.max, dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='instructions'
    )

    # The instruction pointer of each organism.
    instruction_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='instruction_pointers'
    )

    # A stack of stored floating point values for each organism.
    float_stacks = tf.Variable(
        initial_value=tf.zeros((config.population_size, config.stack_size)),
        trainable=False,
        dtype=tf.float32,
        name='float_stacks'
    )

    # A pointer into the floating point stack for each organism.
    float_stack_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='float_stack_pointers'
    )

    # A stack of stored integer values for each organism.
    int_stacks = tf.Variable(
        initial_value=tf.zeros((config.population_size, config.stack_size), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='int_stacks'
    )

    # A pointer into the integer stack for each organism.
    int_stack_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='int_stack_pointers'
    )

    # A pointer into the floating point input array for each organism.
    float_input_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='float_input_pointers'
    )

    # A pointer into the integer input array for each organism.
    int_input_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='int_input_pointers'
    )

    # A writeable output array of floating point values for each organism.
    float_outputs = tf.Variable(
        initial_value=tf.zeros((config.population_size, config.float_output_size)),
        trainable=False,
        dtype=tf.float32,
        name='float_outputs'
    )

    # A pointer into the float output array for each organism.
    float_output_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='float_output_pointers'
    )

    # A writeable output array of integer values for each organism.
    int_outputs = tf.Variable(
        initial_value=tf.zeros((config.population_size, config.int_output_size), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='int_outputs'
    )

    # A pointer into the int output array for each organism.
    int_output_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='int_output_pointers'
    )

    # A writeable memory array of floating point values for each organism.
    float_memories = tf.Variable(
        initial_value=tf.zeros((config.population_size, config.memory_size)),
        trainable=False,
        dtype=tf.float32,
        name='float_memories'
    )

    # A pointer into the float memory array for each organism.
    float_memory_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='float_memory_pointers'
    )

    # A writeable memory array of integer values for each organism.
    int_memories = tf.Variable(
        initial_value=tf.zeros((config.population_size, config.memory_size), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='int_memories'
    )

    # A pointer into the int memory array for each organism.
    int_memory_pointers = tf.Variable(
        initial_value=tf.zeros((config.population_size,), dtype=tf.int32),
        trainable=False,
        dtype=tf.int32,
        name='int_memory_pointers'
    )

    return Population(
        config,
        instructions,
        instruction_pointers,
        float_stacks,
        float_stack_pointers,
        int_stacks,
        int_stack_pointers,
        float_input_pointers,
        int_input_pointers,
        float_outputs,
        float_output_pointers,
        int_outputs,
        int_output_pointers,
        float_memories,
        float_memory_pointers,
        int_memories,
        int_memory_pointers,
    )


def program_to_string(program: Sequence[int], instruction_set: Sequence[Callable]) -> str:
    instructions = []
    constant_next = False
    for address, index in enumerate(program):
        if constant_next:
            value = int(index)
            constant_next = False
        else:
            instruction = instruction_set[index % len(instruction_set)]
            value = instruction.__name__
            if instruction == c_constant:
                constant_next = True
        instructions.append('%s: %s' % (address, value))
    return '\n'.join(instructions)
