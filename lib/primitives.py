import numpy as np

import tensorflow as tf
import tf_helper as tfh
from lib import operation
from lib.build_zernike_coefficients import build_coefs
from lib.operation import i


def sphere(r):
    def s(x, y, z):
        return tf.sqrt(tf.reduce_sum(tf.pow(tf.stack([x, y, z]), 2), axis=0)) - float(r)

    return s


def torus(t1, t2):
    def f(x, y, z):
        q = tf.stack([
            tfh.length(tf.stack([x, z])) - t1,
            y
        ])
        return tfh.length(q) - t2

    return f


def cylinder(r):
    def c(x, y, z):
        return tf.sqrt(tf.reduce_sum(tf.pow(tf.stack([x, y]), 2), axis=0)) - float(r)

    return c


def zernike_surface(zsi, r=4.0):
    cy = cylinder(r)
    zsi = np.pad(zsi, (0, 105 - len(zsi)), "constant", constant_values=(0, 0))
    zs = tf.convert_to_tensor(zsi, dtype=tf.float32)

    def z(x, y, z):
        cfs = tf.convert_to_tensor(build_coefs(x, y, (400, 400)))
        t = tf.einsum("i,ijm->jm", zs, cfs)
        # t = cfs * zs
        return z - t

    return operation.i(z, cy)

