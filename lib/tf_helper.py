import tensorflow as tf


def length(vector):
    return tf.sqrt(tf.reduce_sum(tf.square(vector), reduction_indices=0))


def normalize_vector(vector):
    return vector / tf.sqrt(tf.reduce_sum(tf.square(vector), reduction_indices=0))


def vector_fill(shape, vector, dtype=None):
    return tf.stack([
        tf.fill(shape, vector[0]),
        tf.fill(shape, vector[1]),
        tf.fill(shape, vector[2]),
    ])