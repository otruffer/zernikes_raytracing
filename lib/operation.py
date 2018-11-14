import tensorflow as tf


def tx(f1, tx):
    def f(x, y, z):
        M = tf.matrix_inverse(tx)
        v = tf.stack([x, y, z, tf.ones(x.shape)])
        Mt = tf.matrix_transpose(M)
        t = tf.einsum("imk,ij->jmk", v, Mt)

        return f1(t[0], t[1], t[2])

    return f


def u(f1, f2):
    def f(x, y, z):
        return tf.minimum(f1(x, y, z), f2(x, y, z))

    return f


def i(f1, f2):
    def f(x, y, z):
        return tf.maximum(f1(x, y, z), f2(x, y, z))

    return f


def blend(f1, f2, k=0.1):
    def f(x, y, z):
        a = f1(x, y, z)
        b = f2(x, y, z)
        h = tf.maximum(k - tf.abs(a - b), 0.0)
        return tf.minimum(a, b) - h * h * 0.25 / k

    return f
