import math

import tensorflow as tf
import tensorflow_probability as tfp

from tensor_gp.controller import Controller
from tensor_gp.data_types import ExecutionContext, PopulationConfig
from tensor_gp.support import new_population, program_to_string


def test_f_squared():
    config = PopulationConfig(
        population_size=10000,
        organism_size=20,
        stack_size=10,
        float_output_size=1,
        int_output_size=0,
        memory_size=10,
    )
    population = new_population(config)
    controller = Controller(population)

    context = ExecutionContext(
        max_steps=20,
        float_input_size=1,
        int_input_size=0,
        float_inputs=tf.Variable([0.0], trainable=False, name='float_inputs', dtype=tf.float32),
        int_inputs=tf.Variable([], trainable=False, name='int_inputs', dtype=tf.int32)
    )

    for epoch in range(1000):
        answer = tf.random.uniform((1,), 0.0, 100000.0)
        value_sign = tf.cast(tf.random.uniform((), 0, 2, dtype=tf.int32) * 2 - 1, tf.float32)
        value = value_sign * tf.sqrt(answer)
        context.float_inputs.assign(value)
        controller.run(context)

        loss = tf.reduce_sum(tf.square(controller.population.float_outputs - answer[tf.newaxis, :]),
                             axis=-1)
        loss = tf.where(tf.math.is_finite(loss), loss, tf.float32.max)  # remove nan & inf

        max_loss = tf.reduce_max(loss)
        min_loss = tf.reduce_min(loss)
        loss_range = max_loss - min_loss
        if loss_range == 0.0:
            loss_range = 1.0
        relative_fitness = (max_loss - loss) / loss_range * 1000000.0

        input_heads_moved = (
                tf.cast(controller.population.int_input_pointers != 0, tf.float32) +
                tf.cast(controller.population.float_input_pointers != 0, tf.float32)
        )

        outputs_written = (
                tf.reduce_sum(tf.cast(controller.population.int_outputs != tf.int32.min,
                                      tf.float32),
                              axis=-1) +
                tf.reduce_sum(tf.cast(tf.math.is_finite(controller.population.float_outputs),
                                      tf.float32),
                              axis=-1)
        )
        output_heads_moved = (
                tf.cast(controller.population.int_output_pointers != 0, tf.float32) +
                tf.cast(controller.population.float_output_pointers != 0, tf.float32)
        )

        relative_fitness = (relative_fitness +
                            input_heads_moved +
                            outputs_written +
                            output_heads_moved)

        best = controller.population.instructions[tf.argmax(relative_fitness)]

        print()
        print(epoch,
              tf.reduce_min(relative_fitness).numpy(),
              tf.reduce_mean(relative_fitness).numpy(),
              tf.reduce_max(relative_fitness).numpy())
        print(value.numpy(), answer.numpy(),
              tf.reduce_min(loss).numpy(),
              tfp.stats.percentile(loss, 50.0, interpolation='midpoint').numpy(),  # median
              tf.reduce_sum(loss / tf.cast(tf.size(loss), loss.dtype)).numpy(),  # mean
              tf.reduce_max(loss).numpy())
        print()

        if epoch % 10 == 0:
            print(program_to_string(best, controller.instruction_set))
            print()

        controller.update(relative_fitness)


def test_f_normal():
    config = PopulationConfig(
        population_size=2 ** 15,
        organism_size=35,
        stack_size=10,
        float_output_size=1,
        int_output_size=0,
        memory_size=10,
    )
    population = new_population(config)
    controller = Controller(population)
    # controller.mutation_rate = 2.0 / 35.0

    context = ExecutionContext(
        max_steps=35,
        float_input_size=3,
        int_input_size=0,
        float_inputs=tf.Variable(tf.zeros((3,)), trainable=False, name='float_inputs',
                                 dtype=tf.float32),
        int_inputs=tf.Variable([], trainable=False, name='int_inputs', dtype=tf.int32)
    )

    for epoch in range(1000000):
        m = tf.random.normal((1,))
        s = tf.exp(tf.random.normal((1,)))
        x = tf.random.normal((1,), m, s)
        value = tf.concat([m, s, x], axis=-1)
        answer = tf.exp(-0.5 * tf.square((x - m) / s)) / (tf.sqrt(2.0 * math.pi) * s)
        context.float_inputs.assign(value)
        controller.run(context)

        loss = tf.reduce_sum(tf.square(controller.population.float_outputs - answer[tf.newaxis, :]),
                             axis=-1)
        loss = tf.where(tf.math.is_finite(loss), loss, tf.float32.max)  # remove nan & inf

        scaled_loss = tf.reduce_sum(
            tf.square((controller.population.float_outputs - answer[tf.newaxis, :]) /
                      (tf.abs(answer[tf.newaxis, :]) + 0.000001)),
            axis=-1
        )
        scaled_loss = tf.where(tf.math.is_finite(scaled_loss),
                               scaled_loss, tf.float32.max)  # remove nan & inf

        max_loss = tf.reduce_max(scaled_loss)
        min_loss = tf.reduce_min(scaled_loss)
        loss_range = max_loss - min_loss
        if loss_range == 0.0:
            loss_range = 1.0
        relative_fitness = (max_loss - scaled_loss) / loss_range * 1000000.0

        input_heads_moved = (
                tf.cast(controller.population.int_input_pointers != 0, tf.float32) +
                tf.cast(controller.population.float_input_pointers != 0, tf.float32)
        )

        outputs_written = (
                tf.reduce_sum(tf.cast(controller.population.int_outputs != tf.int32.min,
                                      tf.float32),
                              axis=-1) +
                tf.reduce_sum(tf.cast(tf.math.is_finite(controller.population.float_outputs),
                                      tf.float32),
                              axis=-1)
        )
        output_heads_moved = (
                tf.cast(controller.population.int_output_pointers != 0, tf.float32) +
                tf.cast(controller.population.float_output_pointers != 0, tf.float32)
        )

        unique_output = (
            sum(1.0 / tf.cast(tf.unique_with_counts(controller.population.int_outputs[i])[-1],
                              tf.float32)
                for i in range(tf.shape(controller.population.int_outputs)[1])) +
            sum(1.0 / tf.cast(tf.unique_with_counts(controller.population.float_outputs[i])[-1],
                              tf.float32)
                for i in range(tf.shape(controller.population.float_outputs)[1]))
        )

        relative_fitness = (relative_fitness +
                            input_heads_moved +
                            outputs_written +
                            output_heads_moved +
                            unique_output)

        best = controller.population.instructions[tf.argmax(relative_fitness)]

        print()
        print(epoch,
              tf.reduce_min(relative_fitness).numpy(),
              tf.reduce_mean(relative_fitness).numpy(),
              tf.reduce_max(relative_fitness).numpy())
        print(value.numpy(), answer.numpy(),
              tf.reduce_min(loss).numpy(),
              tfp.stats.percentile(loss, 50.0, interpolation='midpoint').numpy(),  # median
              tf.reduce_sum(loss / tf.cast(tf.size(loss), loss.dtype)).numpy(),  # mean
              tf.reduce_max(loss).numpy())
        print()

        if epoch % 10 == 0:
            print(program_to_string(best, controller.instruction_set))
            print()

        controller.mutation_rate = 1.0 / (0.01 * epoch + 2.0)
        controller.update(relative_fitness)


if __name__ == '__main__':
    # test_f_squared()
    test_f_normal()
