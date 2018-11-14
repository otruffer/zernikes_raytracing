from lib import operation
from lib.build_zernike_coefficients import build_coefs
from lib.primitives import cylinder
import numpy as np
import tensorflow as tf


cy = cylinder(1.0)
zsi = [1.0, 1.0]
zsi = np.pad(zsi, (0, 105 - len(zsi)), "constant", constant_values=(0, 0))
zs = tf.convert_to_tensor(zsi, dtype=tf.float32)

def z(x, y, z):
    cfs = tf.convert_to_tensor(build_coefs(x, y, (400, 400)))
    t = tf.einsum("i,ijm->jm", zs, cfs)
    # t = cfs * zs
    return z - t

scene = operation.i(z, cy)
x = tf.ones((400, 400)) * 1
y = tf.ones((400, 400)) * 2
z = tf.ones((400, 400)) * 3

world = scene(x, y, z)

session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(world)


