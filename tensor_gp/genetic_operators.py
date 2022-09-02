from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from tensor_gp.data_types import TensorLike


@tf.function
def mutation(parent: TensorLike, rate: TensorLike, min_val=None, max_val=None) -> tf.Tensor:
    parent = tf.convert_to_tensor(parent)
    if tf.reduce_all(rate <= 0.0):
        return parent
    shape = tf.shape(parent)
    dtype = parent.dtype
    if min_val is None:
        min_val = dtype.min
    if max_val is None:
        max_val = dtype.max
    mutation_selectors = tf.random.uniform(shape)
    child = tf.where(mutation_selectors <= rate,
                     tf.random.uniform(shape, min_val, max_val, dtype=dtype),
                     parent)
    return child


@tf.function
def lossless_crossover(parent1: TensorLike, parent2: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    parent1 = tf.convert_to_tensor(parent1)
    parent2 = tf.convert_to_tensor(parent2)
    tf.assert_equal(tf.shape(parent1), tf.shape(parent2))
    batch_shape = tf.shape(parent1)[:-1]
    org_size = tf.shape(parent1)[-1]
    crossover_points = tf.random.uniform(batch_shape, 0, org_size + 1, dtype=tf.int32)
    indices = tf.reshape(tf.range(org_size),
                         tf.concat([tf.ones_like(batch_shape), [org_size]], axis=-1))
    # noinspection PyTypeChecker
    selectors: tf.Tensor = indices < crossover_points[:, tf.newaxis]
    child1 = tf.where(selectors, parent1, parent2)
    child2 = tf.where(selectors, parent2, parent1)
    return child1, child2


@tf.function
def normalize_weights(weights: TensorLike) -> tf.Tensor:
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    tf.debugging.assert_non_negative(weights)
    weight_total = tf.reduce_sum(weights)
    tf.debugging.assert_all_finite(weight_total, "Weight total is undefined.")
    if weight_total <= 0.0:
        weights = weights + 1.0
        weight_total = weight_total + tf.cast(tf.size(weights), weights.dtype)
    return weights / weight_total


@tf.function
def stochastic_universal_sampling(choices: TensorLike, weights: TensorLike, n=None) -> tf.Tensor:
    """AKA fitness-proportionate selection."""
    choices = tf.convert_to_tensor(choices)
    tf.debugging.assert_rank_at_least(choices, 1)
    weights = normalize_weights(weights)
    tf.assert_rank(weights, 1)
    tf.assert_equal(tf.shape(choices)[:1], tf.shape(weights))
    if n is None:
        n = tf.shape(choices)[0]
    tf.assert_greater(tf.shape(choices)[0] + 1, n)
    interval = 1.0 / tf.cast(n, dtype=tf.float32)
    offset = tf.random.uniform((), 0.0, interval)
    pointers = (offset + interval * tf.range(n, dtype=tf.float32))[:, tf.newaxis]
    pointers = tf.identity(pointers, name='pointers')
    weight_sums = tf.cumsum(weights)[tf.newaxis, :]
    weight_sums = tf.identity(weight_sums, name='weight_sums')
    indices = tf.argmin(pointers >= weight_sums, axis=1)
    return tf.gather(choices, indices)


@tf.function
def elite_selection(choices: TensorLike, weights: TensorLike, count: int) -> tf.Tensor:
    choices = tf.convert_to_tensor(choices)
    weights = tf.convert_to_tensor(weights)
    _, protected_indices = tf.nn.top_k(weights, count, sorted=False)
    return tf.gather(choices, protected_indices)


@tf.function
def update_population(parents: TensorLike, fitnesses: TensorLike, crossover_rate: TensorLike = 0.0,
                      mutation_rate: TensorLike = 0.0, elite: TensorLike = 0, min_val=None,
                      max_val=None) -> tf.Tensor:
    parents = tf.convert_to_tensor(parents)
    fitnesses = tf.convert_to_tensor(fitnesses)
    tf.assert_equal(tf.shape(parents)[:1], tf.shape(fitnesses))

    parent_count = tf.shape(parents)[0]
    tf.assert_greater(parent_count, elite)

    elite_parents = elite_selection(parents, fitnesses, elite)

    selected_parents = stochastic_universal_sampling(parents, fitnesses, n=parent_count - elite)
    selected_parents = tf.random.shuffle(selected_parents)

    crossover_count = tfp.distributions.Binomial(tf.cast(tf.shape(selected_parents)[0], tf.float32),
                                                 probs=crossover_rate).sample()
    # Round down to the nearest even number.
    crossover_count = tf.cast(crossover_count * 0.5, tf.int32) * 2
    crossover_parents = selected_parents[:crossover_count]
    non_crossover_parents = selected_parents[crossover_count:]
    parents1 = crossover_parents[:crossover_count // 2]
    parents2 = crossover_parents[crossover_count // 2:]
    children1, children2 = lossless_crossover(parents1, parents2)
    crossed_over = tf.concat([children1, children2, non_crossover_parents], axis=0)

    mutated = mutation(crossed_over, mutation_rate, min_val, max_val)

    children = tf.concat([elite_parents, mutated], axis=0)
    tf.assert_equal(tf.shape(children), tf.shape(parents))

    return children
