import numpy as np
import tensorflow as tf
import math


def tr(x, y, z):
    return tf.constant([[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0.0, 0.0, 1]])


def rt(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return tf.constant(np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                                 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                                 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                                 [0, 0, 0, 1]]), dtype=tf.float32)