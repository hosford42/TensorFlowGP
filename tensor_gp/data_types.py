from dataclasses import dataclass
from typing import Union, NamedTuple

import tensorflow as tf


TensorLike = Union[tf.Tensor, tf.Variable]


# TODO: The stack size and output sizes should go in the execution context, not the population
#       config.
PopulationConfig = NamedTuple(
    'PopulationConfig',
    [('population_size', int),
     ('organism_size', int),
     ('stack_size', int),
     ('float_output_size', int),
     ('int_output_size', int),
     ('memory_size', int)]
)


# TODO: Everything but the config and the instructions should go in the execution context, not the
#       population.
@dataclass
class Population:
    config: PopulationConfig

    # The code of each organism.
    instructions: TensorLike  # (population-size, organism_size), int32

    # The instruction pointer of each organism.
    instruction_pointers: TensorLike  # (population_size,), int32

    # A stack of stored floating point values for each organism.
    float_stacks: TensorLike  # (population_size, stack_size), float32

    # A pointer into the floating point stack for each organism.
    float_stack_pointers: TensorLike  # (population_size,), int32

    # A stack of stored integer values for each organism.
    int_stacks: TensorLike  # (population_size, stack_size), int32

    # A pointer into the integer stack for each organism.
    int_stack_pointers: TensorLike  # (population_size,), int32

    # A pointer into the floating point input array for each organism.
    float_input_pointers: TensorLike  # (population_size,), int32

    # A pointer into the integer input array for each organism.
    int_input_pointers: TensorLike  # (population_size,), int32

    # A writeable output array of floating point values for each organism.
    float_outputs: TensorLike  # (population_size, float_output_size), float32

    # A pointer into the float output array for each organism.
    float_output_pointers: TensorLike  # (population_size,), int32

    # A writeable output array of integer values for each organism.
    int_outputs: TensorLike  # (population_size, int_output_size), int32

    # A pointer into the int output array for each organism.
    int_output_pointers: TensorLike  # (population_size,), int32

    # A writeable memory array of floating point values for each organism.
    float_memories: TensorLike  # (population_size, memory_size), float32

    # A pointer into the float memory array for each organism.
    float_memory_pointers: TensorLike  # (population_size,), int32

    # A writeable memory array of integer values for each organism.
    int_memories: TensorLike  # (population_size, memory_size), int32

    # A pointer into the int memory array for each organism.
    int_memory_pointers: TensorLike  # (population_size,), int32

    def init(self) -> None:
        self.instruction_pointers.assign(tf.zeros_like(self.instruction_pointers))

        self.float_stacks.assign(tf.zeros_like(self.float_stacks))
        self.float_stack_pointers.assign(tf.zeros_like(self.float_stack_pointers))
        self.int_stacks.assign(tf.zeros_like(self.int_stacks))
        self.int_stack_pointers.assign(tf.zeros_like(self.int_stack_pointers))

        self.float_input_pointers.assign(tf.zeros_like(self.float_input_pointers))
        self.int_input_pointers.assign(tf.zeros_like(self.int_input_pointers))

        self.float_outputs.assign(tf.ones_like(self.float_outputs) * float('nan'))
        self.float_output_pointers.assign(tf.zeros_like(self.float_output_pointers))
        self.int_outputs.assign(tf.ones_like(self.int_outputs) * self.int_outputs.dtype.min)
        self.int_output_pointers.assign(tf.zeros_like(self.int_output_pointers))

        self.float_memories.assign(tf.zeros_like(self.float_memories))
        self.float_memory_pointers.assign(tf.zeros_like(self.float_memory_pointers))
        self.int_memories.assign(tf.zeros_like(self.int_memories))
        self.int_memory_pointers.assign(tf.zeros_like(self.int_memory_pointers))

    def assign(self, other: 'Population') -> None:
        self.instructions.assign(other.instructions)
        self.instruction_pointers.assign(other.instruction_pointers)

        self.float_stacks.assign(other.float_stacks)
        self.float_stack_pointers.assign(other.float_stack_pointers)
        self.int_stacks.assign(other.int_stacks)
        self.int_stack_pointers.assign(other.int_stack_pointers)

        self.float_input_pointers.assign(other.float_input_pointers)
        self.int_input_pointers.assign(other.int_input_pointers)

        self.float_outputs.assign(other.float_outputs)
        self.float_output_pointers.assign(other.float_output_pointers)
        self.int_outputs.assign(other.int_outputs)
        self.int_output_pointers.assign(other.int_output_pointers)

        self.float_memories.assign(other.float_memories)
        self.float_memory_pointers.assign(other.float_memory_pointers)
        self.int_memories.assign(other.int_memories)
        self.int_memory_pointers.assign(other.int_memory_pointers)

    def check_shapes(self) -> None:
        tf.assert_equal(tf.shape(self.instructions),
                        [self.config.population_size, self.config.organism_size])
        tf.assert_equal(tf.shape(self.instruction_pointers), [self.config.population_size])

        tf.assert_equal(tf.shape(self.float_stacks), [self.config.population_size,
                                                      self.config.stack_size])
        tf.assert_equal(tf.shape(self.float_stack_pointers), [self.config.population_size])
        tf.assert_equal(tf.shape(self.int_stacks), [self.config.population_size,
                                                    self.config.stack_size])
        tf.assert_equal(tf.shape(self.int_stack_pointers), [self.config.population_size])

        tf.assert_equal(tf.shape(self.float_input_pointers), [self.config.population_size])
        tf.assert_equal(tf.shape(self.int_input_pointers), [self.config.population_size])

        tf.assert_equal(tf.shape(self.float_outputs), [self.config.population_size,
                                                       self.config.float_output_size])
        tf.assert_equal(tf.shape(self.float_output_pointers), [self.config.population_size])
        tf.assert_equal(tf.shape(self.int_outputs), [self.config.population_size,
                                                     self.config.int_output_size])
        tf.assert_equal(tf.shape(self.int_output_pointers), [self.config.population_size])

        tf.assert_equal(tf.shape(self.float_memories), [self.config.population_size,
                                                        self.config.memory_size])
        tf.assert_equal(tf.shape(self.float_memory_pointers), [self.config.population_size])
        tf.assert_equal(tf.shape(self.int_memories), [self.config.population_size,
                                                      self.config.memory_size])
        tf.assert_equal(tf.shape(self.int_memory_pointers), [self.config.population_size])

    def fix_shapes(self) -> None:
        self.instructions = tf.reshape(self.instructions, [self.config.population_size,
                                                           self.config.organism_size])
        self.instruction_pointers = tf.reshape(self.instruction_pointers,
                                               [self.config.population_size])

        self.float_stacks = tf.reshape(self.float_stacks, [self.config.population_size,
                                                           self.config.stack_size])
        self.float_stack_pointers = tf.reshape(self.float_stack_pointers,
                                               [self.config.population_size])
        self.int_stacks = tf.reshape(self.int_stacks, [self.config.population_size,
                                                       self.config.stack_size])
        self.int_stack_pointers = tf.reshape(self.int_stack_pointers,
                                             [self.config.population_size])

        self.float_input_pointers = tf.reshape(self.float_input_pointers,
                                               [self.config.population_size])
        self.int_input_pointers = tf.reshape(self.int_input_pointers, [self.config.population_size])

        self.float_outputs = tf.reshape(self.float_outputs, [self.config.population_size,
                                                             self.config.float_output_size])
        self.float_output_pointers = tf.reshape(self.float_output_pointers,
                                                [self.config.population_size])
        self.int_outputs = tf.reshape(self.int_outputs, [self.config.population_size,
                                                         self.config.int_output_size])
        self.int_output_pointers = tf.reshape(self.int_output_pointers,
                                              [self.config.population_size])

        self.float_memories = tf.reshape(self.float_memories, [self.config.population_size,
                                                               self.config.memory_size])
        self.float_memory_pointers = tf.reshape(self.float_memory_pointers,
                                                [self.config.population_size])
        self.int_memories = tf.reshape(self.int_memories, [self.config.population_size,
                                                           self.config.memory_size])
        self.int_memory_pointers = tf.reshape(self.int_memory_pointers,
                                              [self.config.population_size])


@dataclass
class ExecutionContext:
    max_steps: int
    float_input_size: int
    int_input_size: int
    float_inputs: TensorLike  # (float_input_size,), float32
    int_inputs: TensorLike  # (int_input_size,), int32
