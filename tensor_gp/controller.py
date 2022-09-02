from dataclasses import replace
from typing import Sequence, Callable, Tuple

import tensorflow as tf

from tensor_gp.astuple_fix import astuple
from tensor_gp.c_instr import CONTROL_INSTRUCTIONS
from tensor_gp.data_types import ExecutionContext, Population, PopulationConfig, TensorLike
from tensor_gp.f_instr import FLOATING_POINT_INSTRUCTIONS
from tensor_gp.genetic_operators import update_population
from tensor_gp.i_instr import INTEGER_INSTRUCTIONS
from tensor_gp.instr_support import _read

# Specified in this order, instruction 0 maps to c_no_op.
ALL_INSTRUCTIONS = CONTROL_INSTRUCTIONS + INTEGER_INSTRUCTIONS + FLOATING_POINT_INSTRUCTIONS


def make_step_function(config: PopulationConfig, instruction_set: Sequence[Callable]) -> Callable:
    @tf.function
    def step_function(population: Tuple[TensorLike, ...],
                      context: Tuple[TensorLike, ...]) -> Tuple[TensorLike, ...]:
        population_obj = Population(config, *population)
        population_obj.check_shapes()

        instructions = tf.convert_to_tensor(population_obj.instructions)
        instruction_pointers = tf.convert_to_tensor(population_obj.instruction_pointers)
        tf.assert_equal(tf.shape(instructions)[:-1], tf.shape(instruction_pointers))

        current_instructions, instruction_pointers = _read(instructions, instruction_pointers)
        # noinspection PyTypeChecker
        current_instructions = current_instructions % len(instruction_set)
        tf.assert_equal(tf.shape(instruction_pointers), [config.population_size])
        tf.assert_equal(tf.shape(current_instructions), [config.population_size])

        population_obj = replace(population_obj, instruction_pointers=instruction_pointers)
        population_obj.check_shapes()
        population = astuple(population_obj, deepcopy=False)[1:]

        # Perform each instruction
        candidate_updates = [instruction(population, context, config)
                             for instruction in instruction_set]

        # Concatenate all the instructions together
        candidate_updates = [tf.concat([item[:, tf.newaxis] for item in items], axis=1)
                             for items in zip(*candidate_updates)]

        # Use the current instruction to index into each group of instruction results.
        selected_items = []
        for item_index, items in enumerate(candidate_updates):
            tf.assert_greater(tf.rank(items), 1)
            tf.assert_equal(tf.shape(items)[2:], tf.shape(population[item_index])[1:])
            tf.assert_equal(tf.shape(items)[0], tf.shape(population[item_index])[0])
            selected_item = tf.gather(items, current_instructions, batch_dims=1)
            tf.assert_equal(tf.shape(selected_item), tf.shape(population[item_index]))
            selected_items.append(selected_item)

        population = tuple(selected_items)
        Population(config, *population).check_shapes()

        population_obj = Population(config, *population)
        population_obj.fix_shapes()
        population = astuple(population_obj, deepcopy=False)[1:]
        return population

    return step_function


def make_run_function(config: PopulationConfig, step_function):
    @tf.function
    def run_function(population: Tuple[TensorLike, ...],
                     context: Tuple[TensorLike, ...],
                     max_steps: int):
        Population(config, *population).check_shapes()
        shapes = tuple(tf.TensorShape(item.shape) for item in population)
        print(shapes)
        for _step in tf.range(max_steps):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(population, shapes)]
            )
            # tf.print("Step:", step)  # TODO: Use a progress bar.
            population = step_function(population, context)
            Population(config, *population).check_shapes()
            print(tuple(tf.TensorShape(item.shape) for item in population))
        return population

    return run_function


class Controller:

    def __init__(self, population: Population, instruction_set: Sequence[Callable] = None):
        self.population = population
        self.instruction_set = instruction_set or ALL_INSTRUCTIONS
        self.step_function = make_step_function(self.population.config, self.instruction_set)
        self.run_function = make_run_function(self.population.config, self.step_function)

        self._mutation_rate = tf.Variable(
            1.0 / tf.cast(tf.shape(population.instructions)[-1], tf.float32),
            trainable=False,
            name='mutation_rate'
        )
        self._crossover_rate = tf.Variable(
            1.0,
            trainable=False,
            name='crossover_rate'
        )
        self._elite = tf.Variable(
            1,
            trainable=False,
            name='elite'
        )

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate.numpy()

    @mutation_rate.setter
    def mutation_rate(self, value: float) -> None:
        self._mutation_rate.assign(float(value))

    @property
    def crossover_rate(self) -> float:
        return self._crossover_rate.numpy()

    @crossover_rate.setter
    def crossover_rate(self, value: float) -> None:
        self._crossover_rate.assign(float(value))

    @property
    def elite(self) -> int:
        return self._elite.numpy()

    @elite.setter
    def elite(self, value: int) -> None:
        assert value == int(value)
        self._elite.assign(int(value))

    def run(self, context: ExecutionContext) -> None:
        self.population.init()
        max_steps = context.max_steps
        population = astuple(self.population, deepcopy=False)[1:]
        context = astuple(context, deepcopy=False)
        population = self.run_function(population, context, max_steps)
        population = Population(self.population.config, *population)
        population.check_shapes()
        self.population.assign(population)

    def update(self, fitnesses: TensorLike):
        fitnesses = tf.convert_to_tensor(fitnesses)
        tf.assert_rank(fitnesses, 1)
        tf.assert_equal(tf.size(fitnesses), tf.shape(self.population.instructions)[0])
        tf.debugging.assert_non_negative(fitnesses)
        instructions = update_population(
            parents=self.population.instructions,
            fitnesses=fitnesses,
            crossover_rate=self._crossover_rate,
            mutation_rate=self._mutation_rate,
            elite=self._elite
        )
        self.population.instructions.assign(instructions)
