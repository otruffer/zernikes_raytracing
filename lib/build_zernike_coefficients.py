import numpy as np

import tensorflow as tf
import numpy as np

def build_coefs(x, y, shape):
    cfs = [None] * 105
    cfs[0] = 1.0 * tf.ones(shape, dtype=tf.float32)
    cfs[1] = 2.0 * y
    cfs[2] = 2.0 * x
    cfs[3] = 2.449489742783178 * (2.0 * x * y)
    cfs[4] = 1.7320508075688772 * (-1 + 2 * (tf.pow(x, 2) + tf.pow(y, 2)))
    cfs[5] = 2.449489742783178 * (tf.pow(x, 2) - tf.pow(y, 2))
    cfs[6] = 2.8284271247461903 * (3 * tf.pow(x, 2) * y - tf.pow(y, 3))
    cfs[7] = 2.8284271247461903 * (-2 * y + 3 * (tf.pow(x, 2) + tf.pow(y, 2)) * y)
    cfs[8] = 2.8284271247461903 * (-2 * x + 3 * (tf.pow(x, 2) + tf.pow(y, 2)) * x)
    cfs[9] = 2.8284271247461903 * (tf.pow(x, 3) - 3 * x * tf.pow(y, 2))
    cfs[10] = 3.1622776601683795 * (4 * tf.pow(x, 3) * y - 4 * x * tf.pow(y, 3))
    cfs[11] = 3.1622776601683795 * (-6 * x * y + 8 * (tf.pow(x, 2) + tf.pow(y, 2)) * x * y)
    cfs[12] = 2.23606797749979 * (
            1 - 6 * (tf.pow(x, 2) + tf.pow(y, 2)) + 6 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2))
    cfs[13] = 3.1622776601683795 * (
            -3 * tf.pow(x, 2) + 4 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) + 3 * tf.pow(y, 2) - 4 * (
            tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 2))
    cfs[14] = 3.1622776601683795 * (tf.pow(x, 4) - 6 * tf.pow(x, 2) * tf.pow(y, 2) + tf.pow(y, 4))
    cfs[15] = 3.4641016151377544 * (5 * tf.pow(x, 4) * y - 10 * tf.pow(x, 2) * tf.pow(y, 3) + tf.pow(y, 5))
    cfs[16] = 3.4641016151377544 * (
            -12 * tf.pow(x, 2) * y + 15 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * y + 4 * tf.pow(y,
                                                                                                        3) - 5 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 3))
    cfs[17] = 3.4641016151377544 * (
            3 * y - 12 * (tf.pow(x, 2) + tf.pow(y, 2)) * y + 10 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2) * y)
    cfs[18] = 3.4641016151377544 * (
            3 * x - 12 * (tf.pow(x, 2) + tf.pow(y, 2)) * x + 10 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2) * x)
    cfs[19] = 3.4641016151377544 * (
            -4 * tf.pow(x, 3) + 5 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) + 12 * x * tf.pow(y, 2) - 15 * (
            tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 2))
    cfs[20] = 3.4641016151377544 * (tf.pow(x, 5) - 10 * tf.pow(x, 3) * tf.pow(y, 2) + 5 * x * tf.pow(y, 4))
    cfs[21] = 3.7416573867739413 * (6 * tf.pow(x, 5) * y - 20 * tf.pow(x, 3) * tf.pow(y, 3) + 6 * x * tf.pow(y, 5))
    cfs[22] = 3.7416573867739413 * (
            -20 * tf.pow(x, 3) * y + 24 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * y + 20 * x * tf.pow(y,
                                                                                                             3) - 24 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 3))
    cfs[23] = 3.7416573867739413 * (
            12 * x * y - 40 * (tf.pow(x, 2) + tf.pow(y, 2)) * x * y + 30 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                  2) * x * y)
    cfs[24] = 2.6457513110645907 * (
            -1 + 12 * (tf.pow(x, 2) + tf.pow(y, 2)) - 30 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2) + 20 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3))
    cfs[25] = 3.7416573867739413 * (
            6 * tf.pow(x, 2) - 20 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) + 15 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) - 6 * tf.pow(y, 2) + 20 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(
        y, 2) - 15 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(y, 2))
    cfs[26] = 3.7416573867739413 * (
            -5 * tf.pow(x, 4) + 6 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) + 30 * tf.pow(x, 2) * tf.pow(y,
                                                                                                              2) - 36 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 2) - 5 * tf.pow(y, 4) + 6 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 4))
    cfs[27] = 3.7416573867739413 * (
            tf.pow(x, 6) - 15 * tf.pow(x, 4) * tf.pow(y, 2) + 15 * tf.pow(x, 2) * tf.pow(y, 4) - tf.pow(y, 6))
    cfs[28] = 4.0 * (
            7 * tf.pow(x, 6) * y - 35 * tf.pow(x, 4) * tf.pow(y, 3) + 21 * tf.pow(x, 2) * tf.pow(y, 5) - tf.pow(y,
                                                                                                                7))
    cfs[29] = 4.0 * (-30 * tf.pow(x, 4) * y + 35 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * y + 60 * tf.pow(x,
                                                                                                                  2) * tf.pow(
        y, 3) - 70 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 3) - 6 * tf.pow(y, 5) + 7 * (
                             tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 5))
    cfs[30] = 4.0 * (30 * tf.pow(x, 2) * y - 90 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * y + 63 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * y - 10 * tf.pow(y, 3) + 30 * (
                             tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 3) - 21 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                       2) * tf.pow(y, 3))
    cfs[31] = 4.0 * (-4 * y + 30 * (tf.pow(x, 2) + tf.pow(y, 2)) * y - 60 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                   2) * y + 35 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * y)
    cfs[32] = 4.0 * (-4 * x + 30 * (tf.pow(x, 2) + tf.pow(y, 2)) * x - 60 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                   2) * x + 35 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x)
    cfs[33] = 4.0 * (10 * tf.pow(x, 3) - 30 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) + 21 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) - 30 * x * tf.pow(y, 2) + 90 * (
                             tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 2) - 63 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * x * tf.pow(y, 2))
    cfs[34] = 4.0 * (
            -6 * tf.pow(x, 5) + 7 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) + 60 * tf.pow(x, 3) * tf.pow(y,
                                                                                                              2) - 70 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 2) - 30 * x * tf.pow(y, 4) + 35 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 4))
    cfs[35] = 4.0 * (
            tf.pow(x, 7) - 21 * tf.pow(x, 5) * tf.pow(y, 2) + 35 * tf.pow(x, 3) * tf.pow(y, 4) - 7 * x * tf.pow(y,
                                                                                                                6))
    cfs[36] = 4.242640687119285 * (
            8 * tf.pow(x, 7) * y - 56 * tf.pow(x, 5) * tf.pow(y, 3) + 56 * tf.pow(x, 3) * tf.pow(y,
                                                                                                 5) - 8 * x * tf.pow(
        y, 7))
    cfs[37] = 4.242640687119285 * (
            -42 * tf.pow(x, 5) * y + 48 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * y + 140 * tf.pow(x,
                                                                                                          3) * tf.pow(
        y, 3) - 160 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 3) - 42 * x * tf.pow(y, 5) + 48 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 5))
    cfs[38] = 4.242640687119285 * (
            60 * tf.pow(x, 3) * y - 168 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * y + 112 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * y - 60 * x * tf.pow(y, 3) + 168 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 3) - 112 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                   2) * x * tf.pow(y, 3))
    cfs[39] = 4.242640687119285 * (
            -20 * x * y + 120 * (tf.pow(x, 2) + tf.pow(y, 2)) * x * y - 210 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                     2) * x * y + 112 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x * y)
    cfs[40] = 3.0 * (
            1 - 20 * (tf.pow(x, 2) + tf.pow(y, 2)) + 90 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2) - 140 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) + 70 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4))
    cfs[41] = 4.242640687119285 * (
            -10 * tf.pow(x, 2) + 60 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) - 105 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) + 56 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x,
                                                                                                              2) + 10 * tf.pow(
        y, 2) - 60 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 2) + 105 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                 2) * tf.pow(y, 2) - 56 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(y, 2))
    cfs[42] = 4.242640687119285 * (
            15 * tf.pow(x, 4) - 42 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) + 28 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) - 90 * tf.pow(x, 2) * tf.pow(y, 2) + 252 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y,
                                                                         2) - 168 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 2) + 15 * tf.pow(y, 4) - 42 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 4) + 28 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(y, 4))
    cfs[43] = 4.242640687119285 * (
            -7 * tf.pow(x, 6) + 8 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) + 105 * tf.pow(x, 4) * tf.pow(y,
                                                                                                               2) - 120 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 2) - 105 * tf.pow(x, 2) * tf.pow(y,
                                                                                                             4) + 120 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 4) + 7 * tf.pow(y, 6) - 8 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 6))
    cfs[44] = 4.242640687119285 * (
            tf.pow(x, 8) - 28 * tf.pow(x, 6) * tf.pow(y, 2) + 70 * tf.pow(x, 4) * tf.pow(y, 4) - 28 * tf.pow(x,
                                                                                                             2) * tf.pow(
        y, 6) + tf.pow(y, 8))
    cfs[45] = 4.47213595499958 * (
            9 * tf.pow(x, 8) * y - 84 * tf.pow(x, 6) * tf.pow(y, 3) + 126 * tf.pow(x, 4) * tf.pow(y,
                                                                                                  5) - 36 * tf.pow(
        x, 2) * tf.pow(y, 7) + tf.pow(y, 9))
    cfs[46] = 4.47213595499958 * (
            -56 * tf.pow(x, 6) * y + 63 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) * y + 280 * tf.pow(x,
                                                                                                          4) * tf.pow(
        y, 3) - 315 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 3) - 168 * tf.pow(x, 2) * tf.pow(y,
                                                                                                                5) + 189 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 5) + 8 * tf.pow(y, 7) - 9 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 7))
    cfs[47] = 4.47213595499958 * (
            105 * tf.pow(x, 4) * y - 280 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * y + 180 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) * y - 210 * tf.pow(x, 2) * tf.pow(y, 3) + 560 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 3) - 360 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 3) + 21 * tf.pow(y, 5) - 56 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 5) + 36 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                              2) * tf.pow(y, 5))
    cfs[48] = 4.47213595499958 * (
            -60 * tf.pow(x, 2) * y + 315 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * y - 504 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * y + 252 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(
        x, 2) * y + 20 * tf.pow(y, 3) - 105 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 3) + 168 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(y, 3) - 84 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(y, 3))
    cfs[49] = 4.47213595499958 * (
            5 * y - 60 * (tf.pow(x, 2) + tf.pow(y, 2)) * y + 210 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                          2) * y - 280 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * y + 126 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * y)
    cfs[50] = 4.47213595499958 * (
            5 * x - 60 * (tf.pow(x, 2) + tf.pow(y, 2)) * x + 210 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                          2) * x - 280 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x + 126 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * x)
    cfs[51] = 4.47213595499958 * (
            -20 * tf.pow(x, 3) + 105 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) - 168 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) + 84 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x,
                                                                                                              3) + 60 * x * tf.pow(
        y, 2) - 315 * (tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 2) + 504 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                      2) * x * tf.pow(y,
                                                                                                      2) - 252 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x * tf.pow(y, 2))
    cfs[52] = 4.47213595499958 * (
            21 * tf.pow(x, 5) - 56 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) + 36 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 5) - 210 * tf.pow(x, 3) * tf.pow(y, 2) + 560 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 2) - 360 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * tf.pow(y, 2) + 105 * x * tf.pow(y, 4) - 280 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 4) + 180 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * x * tf.pow(y, 4))
    cfs[53] = 4.47213595499958 * (
            -8 * tf.pow(x, 7) + 9 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 7) + 168 * tf.pow(x, 5) * tf.pow(y,
                                                                                                               2) - 189 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * tf.pow(y, 2) - 280 * tf.pow(x, 3) * tf.pow(y,
                                                                                                             4) + 315 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 4) + 56 * x * tf.pow(y, 6) - 63 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 6))
    cfs[54] = 4.47213595499958 * (
            tf.pow(x, 9) - 36 * tf.pow(x, 7) * tf.pow(y, 2) + 126 * tf.pow(x, 5) * tf.pow(y, 4) - 84 * tf.pow(x,
                                                                                                              3) * tf.pow(
        y, 6) + 9 * x * tf.pow(y, 8))
    cfs[55] = 4.69041575982343 * (
            10 * tf.pow(x, 9) * y - 120 * tf.pow(x, 7) * tf.pow(y, 3) + 252 * tf.pow(x, 5) * tf.pow(y,
                                                                                                    5) - 120 * tf.pow(
        x, 3) * tf.pow(y, 7) + 10 * x * tf.pow(y, 9))
    cfs[56] = 4.69041575982343 * (
            -72 * tf.pow(x, 7) * y + 80 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 7) * y + 504 * tf.pow(x,
                                                                                                          5) * tf.pow(
        y, 3) - 560 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * tf.pow(y, 3) - 504 * tf.pow(x, 3) * tf.pow(y,
                                                                                                                5) + 560 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 5) + 72 * x * tf.pow(y, 7) - 80 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 7))
    cfs[57] = 4.69041575982343 * (
            168 * tf.pow(x, 5) * y - 432 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * y + 270 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 5) * y - 560 * tf.pow(x, 3) * tf.pow(y, 3) + 1440 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 3) - 900 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * tf.pow(y, 3) + 168 * x * tf.pow(y, 5) - 432 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 5) + 270 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                   2) * x * tf.pow(y, 5))
    cfs[58] = 4.69041575982343 * (
            -140 * tf.pow(x, 3) * y + 672 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * y - 1008 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * y + 480 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(
        x, 3) * y + 140 * x * tf.pow(y, 3) - 672 * (tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 3) + 1008 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * x * tf.pow(y, 3) - 480 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                          3) * x * tf.pow(y, 3))
    cfs[59] = 4.69041575982343 * (
            30 * x * y - 280 * (tf.pow(x, 2) + tf.pow(y, 2)) * x * y + 840 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                    2) * x * y - 1008 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x * y + 420 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * x * y)
    cfs[60] = 3.3166247903554 * (
            -1 + 30 * (tf.pow(x, 2) + tf.pow(y, 2)) - 210 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2) + 560 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) - 630 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) + 252 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 5))
    cfs[61] = 4.69041575982343 * (
            15 * tf.pow(x, 2) - 140 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) + 420 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) - 504 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(
        x,
        2) + 210 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(x, 2) - 15 * tf.pow(y, 2) + 140 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 2) - 420 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(y, 2) + 504 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(
        y,
        2) - 210 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(y, 2))
    cfs[62] = 4.69041575982343 * (
            -35 * tf.pow(x, 4) + 168 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) - 252 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) + 120 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x,
                                                                                                               4) + 210 * tf.pow(
        x, 2) * tf.pow(y, 2) - 1008 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 2) + 1512 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 2) - 720 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                     3) * tf.pow(x, 2) * tf.pow(y,
                                                                                                                2) - 35 * tf.pow(
        y, 4) + 168 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 4) - 252 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                  2) * tf.pow(y, 4) + 120 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(y, 4))
    cfs[63] = 4.69041575982343 * (
            28 * tf.pow(x, 6) - 72 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) + 45 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 6) - 420 * tf.pow(x, 4) * tf.pow(y, 2) + 1080 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 2) - 675 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) * tf.pow(y, 2) + 420 * tf.pow(x, 2) * tf.pow(y,
                                                                                                    4) - 1080 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 4) + 675 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 4) - 28 * tf.pow(y, 6) + 72 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 6) - 45 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(y, 6))
    cfs[64] = 4.69041575982343 * (
            -9 * tf.pow(x, 8) + 10 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 8) + 252 * tf.pow(x, 6) * tf.pow(y,
                                                                                                                2) - 280 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) * tf.pow(y, 2) - 630 * tf.pow(x, 4) * tf.pow(y,
                                                                                                             4) + 700 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 4) + 252 * tf.pow(x, 2) * tf.pow(y,
                                                                                                             6) - 280 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 6) - 9 * tf.pow(y, 8) + 10 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 8))
    cfs[65] = 4.69041575982343 * (
            tf.pow(x, 10) - tf.pow(y, 10) - 45 * tf.pow(x, 8) * tf.pow(y, 2) + 210 * tf.pow(x, 6) * tf.pow(y,
                                                                                                           4) - 210 * tf.pow(
        x, 4) * tf.pow(y, 6) + 45 * tf.pow(x, 2) * tf.pow(y, 8))
    cfs[66] = 4.898979485566356 * (
            11 * tf.pow(x, 10) * y - tf.pow(y, 11) - 165 * tf.pow(x, 8) * tf.pow(y, 3) + 462 * tf.pow(x,
                                                                                                      6) * tf.pow(y,
                                                                                                                  5) - 330 * tf.pow(
        x, 4) * tf.pow(y, 7) + 55 * tf.pow(x, 2) * tf.pow(y, 9))
    cfs[67] = 4.898979485566356 * (
            -90 * tf.pow(x, 8) * y + 99 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 8) * y + 840 * tf.pow(x,
                                                                                                          6) * tf.pow(
        y, 3) - 924 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) * tf.pow(y, 3) - 1260 * tf.pow(x, 4) * tf.pow(y,
                                                                                                                 5) + 1386 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 5) + 360 * tf.pow(x, 2) * tf.pow(y,
                                                                                                             7) - 396 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 7) - 10 * tf.pow(y, 9) + 11 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 9))
    cfs[68] = 4.898979485566356 * (
            252 * tf.pow(x, 6) * y - 630 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) * y + 385 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 6) * y - 1260 * tf.pow(x, 4) * tf.pow(y, 3) + 3150 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 3) - 1925 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) * tf.pow(y, 3) + 756 * tf.pow(x, 2) * tf.pow(y, 5) - 1890 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 5) + 1155 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 5) - 36 * tf.pow(y, 7) + 90 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 7) - 55 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                              2) * tf.pow(y, 7))
    cfs[69] = 4.898979485566356 * (
            -280 * tf.pow(x, 4) * y + 1260 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * y - 1800 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) * y + 825 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(
        x, 4) * y + 560 * tf.pow(x, 2) * tf.pow(y, 3) - 2520 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x,
                                                                                                      2) * tf.pow(y,
                                                                                                                  3) + 3600 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 3) - 1650 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                      3) * tf.pow(x, 2) * tf.pow(y,
                                                                                                                 3) - 56 * tf.pow(
        y, 5) + 252 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 5) - 360 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                  2) * tf.pow(y, 5) + 165 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(y, 5))
    cfs[70] = 4.898979485566356 * (
            105 * tf.pow(x, 2) * y - 840 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * y + 2268 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * y - 2520 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(
        x, 2) * y + 990 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(x, 2) * y - 35 * tf.pow(y, 3) + 280 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 3) - 756 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                               2) * tf.pow(y, 3) + 840 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(y, 3) - 330 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(y,
                                                                                                               3))
    cfs[71] = 4.898979485566356 * (
            -6 * y + 105 * (tf.pow(x, 2) + tf.pow(y, 2)) * y - 560 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                            2) * y + 1260 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * y - 1260 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * y + 462 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 5) * y)
    cfs[72] = 4.898979485566356 * (
            -6 * x + 105 * (tf.pow(x, 2) + tf.pow(y, 2)) * x - 560 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                            2) * x + 1260 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x - 1260 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * x + 462 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 5) * x)
    cfs[73] = 4.898979485566356 * (
            35 * tf.pow(x, 3) - 280 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) + 756 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) - 840 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x,
                                                                                                               3) + 330 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(x, 3) - 105 * x * tf.pow(y, 2) + 840 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 2) - 2268 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                    2) * x * tf.pow(y,
                                                                                                    2) + 2520 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x * tf.pow(y, 2) - 990 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                          4) * x * tf.pow(y, 2))
    cfs[74] = 4.898979485566356 * (
            -56 * tf.pow(x, 5) + 252 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) - 360 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 5) + 165 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x,
                                                                                                               5) + 560 * tf.pow(
        x, 3) * tf.pow(y, 2) - 2520 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 2) + 3600 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * tf.pow(y, 2) - 1650 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                      3) * tf.pow(x, 3) * tf.pow(y,
                                                                                                                 2) - 280 * x * tf.pow(
        y, 4) + 1260 * (tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 4) - 1800 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                        2) * x * tf.pow(y,
                                                                                                        4) + 825 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x * tf.pow(y, 4))
    cfs[75] = 4.898979485566356 * (
            36 * tf.pow(x, 7) - 90 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 7) + 55 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 7) - 756 * tf.pow(x, 5) * tf.pow(y, 2) + 1890 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * tf.pow(y,
                                                                         2) - 1155 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 5) * tf.pow(y, 2) + 1260 * tf.pow(x, 3) * tf.pow(y,
                                                                                                     4) - 3150 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y,
                                                                         4) + 1925 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * tf.pow(y, 4) - 252 * x * tf.pow(y, 6) + 630 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 6) - 385 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * x * tf.pow(y, 6))
    cfs[76] = 4.898979485566356 * (
            -10 * tf.pow(x, 9) + 11 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 9) + 360 * tf.pow(x, 7) * tf.pow(y,
                                                                                                                 2) - 396 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 7) * tf.pow(y, 2) - 1260 * tf.pow(x, 5) * tf.pow(y,
                                                                                                              4) + 1386 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * tf.pow(y, 4) + 840 * tf.pow(x, 3) * tf.pow(y,
                                                                                                             6) - 924 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 6) - 90 * x * tf.pow(y, 8) + 99 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 8))
    cfs[77] = 4.898979485566356 * (
            tf.pow(x, 11) - 11 * x * tf.pow(y, 10) - 55 * tf.pow(x, 9) * tf.pow(y, 2) + 330 * tf.pow(x, 7) * tf.pow(
        y, 4) - 462 * tf.pow(x, 5) * tf.pow(y, 6) + 165 * tf.pow(x, 3) * tf.pow(y, 8))
    cfs[78] = 5.0990195135927845 * (
            12 * tf.pow(x, 11) * y - 12 * x * tf.pow(y, 11) - 220 * tf.pow(x, 9) * tf.pow(y, 3) + 792 * tf.pow(x,
                                                                                                               7) * tf.pow(
        y, 5) - 792 * tf.pow(x, 5) * tf.pow(y, 7) + 220 * tf.pow(x, 3) * tf.pow(y, 9))
    cfs[79] = 5.0990195135927845 * (
            -110 * tf.pow(x, 9) * y + 120 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 9) * y + 1320 * tf.pow(x,
                                                                                                             7) * tf.pow(
        y, 3) - 1440 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 7) * tf.pow(y, 3) - 2772 * tf.pow(x, 5) * tf.pow(y,
                                                                                                                  5) + 3024 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * tf.pow(y, 5) + 1320 * tf.pow(x, 3) * tf.pow(y,
                                                                                                              7) - 1440 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 7) - 110 * x * tf.pow(y,
                                                                                                  9) + 120 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 9))
    cfs[80] = 5.0990195135927845 * (
            360 * tf.pow(x, 7) * y - 880 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 7) * y + 528 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 7) * y - 2520 * tf.pow(x, 5) * tf.pow(y, 3) + 6160 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * tf.pow(y, 3) - 3696 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 5) * tf.pow(y, 3) + 2520 * tf.pow(x, 3) * tf.pow(y,
                                                                                                     5) - 6160 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * tf.pow(y, 5) + 3696 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * tf.pow(y, 5) - 360 * x * tf.pow(y, 7) + 880 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 7) - 528 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                   2) * x * tf.pow(y, 7))
    cfs[81] = 5.0990195135927845 * (
            -504 * tf.pow(x, 5) * y + 2160 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 5) * y - 2970 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 5) * y + 1320 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(
        x, 5) * y + 1680 * tf.pow(x, 3) * tf.pow(y, 3) - 7200 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x,
                                                                                                       3) * tf.pow(
        y, 3) + 9900 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * tf.pow(y, 3) - 4400 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x, 3) * tf.pow(y, 3) - 504 * x * tf.pow(y, 5) + 2160 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 5) - 2970 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                    2) * x * tf.pow(y,
                                                                                                    5) + 1320 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x * tf.pow(y, 5))
    cfs[82] = 5.0990195135927845 * (
            280 * tf.pow(x, 3) * y - 2016 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 3) * y + 5040 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 3) * y - 5280 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(
        x, 3) * y + 1980 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(x, 3) * y - 280 * x * tf.pow(y,
                                                                                                        3) + 2016 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * x * tf.pow(y, 3) - 5040 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                    2) * x * tf.pow(y,
                                                                                                    3) + 5280 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x * tf.pow(y, 3) - 1980 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                           4) * x * tf.pow(y, 3))
    cfs[83] = 5.0990195135927845 * (
            -42 * x * y + 560 * (tf.pow(x, 2) + tf.pow(y, 2)) * x * y - 2520 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                      2) * x * y + 5040 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * x * y - 4620 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                4) * x * y + 1584 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 5) * x * y)
    cfs[84] = 3.605551275463989 * (
            1 - 42 * (tf.pow(x, 2) + tf.pow(y, 2)) + 420 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 2) - 1680 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) + 3150 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) - 2772 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 5) + 924 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 6))
    cfs[85] = 5.0990195135927845 * (
            -21 * tf.pow(x, 2) + 280 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) - 1260 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) + 2520 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x,
                                                                                                                2) - 2310 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(x, 2) + 792 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 5) * tf.pow(x,
                                                                                                               2) + 21 * tf.pow(
        y, 2) - 280 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 2) + 1260 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                   2) * tf.pow(y,
                                                                                               2) - 2520 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(y, 2) + 2310 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(y,
                                                                                                                2) - 792 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 5) * tf.pow(y, 2))
    cfs[86] = 5.0990195135927845 * (
            70 * tf.pow(x, 4) - 504 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) + 1260 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) - 1320 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x,
                                                                                                                4) + 495 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(x, 4) - 420 * tf.pow(x, 2) * tf.pow(y, 2) + 3024 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 2) - 7560 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 2) + 7920 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                      3) * tf.pow(x, 2) * tf.pow(y,
                                                                                                                 2) - 2970 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(x, 2) * tf.pow(y, 2) + 70 * tf.pow(y, 4) - 504 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 4) + 1260 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                2) * tf.pow(y,
                                                                                            4) - 1320 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(y, 4) + 495 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 4) * tf.pow(y,
                                                                                                               4))
    cfs[87] = 5.0990195135927845 * (
            -84 * tf.pow(x, 6) + 360 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) - 495 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 6) + 220 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(x,
                                                                                                               6) + 1260 * tf.pow(
        x, 4) * tf.pow(y, 2) - 5400 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 2) + 7425 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) * tf.pow(y, 2) - 3300 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                      3) * tf.pow(x, 4) * tf.pow(y,
                                                                                                                 2) - 1260 * tf.pow(
        x, 2) * tf.pow(y, 4) + 5400 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 4) - 7425 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 4) + 3300 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                      3) * tf.pow(x, 2) * tf.pow(y,
                                                                                                                 4) + 84 * tf.pow(
        y, 6) - 360 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 6) + 495 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                                  2) * tf.pow(y, 6) - 220 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 3) * tf.pow(y, 6))
    cfs[88] = 5.0990195135927845 * (
            45 * tf.pow(x, 8) - 110 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 8) + 66 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 8) - 1260 * tf.pow(x, 6) * tf.pow(y, 2) + 3080 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) * tf.pow(y, 2) - 1848 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 6) * tf.pow(y, 2) + 3150 * tf.pow(x, 4) * tf.pow(y,
                                                                                                     4) - 7700 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 4) + 4620 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 4) * tf.pow(y, 4) - 1260 * tf.pow(x, 2) * tf.pow(y,
                                                                                                     6) + 3080 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 6) - 1848 * tf.pow(
        tf.pow(x, 2) + tf.pow(y, 2), 2) * tf.pow(x, 2) * tf.pow(y, 6) + 45 * tf.pow(y, 8) - 110 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 8) + 66 * tf.pow(tf.pow(x, 2) + tf.pow(y, 2),
                                                                              2) * tf.pow(y, 8))
    cfs[89] = 5.0990195135927845 * (
            -11 * tf.pow(x, 10) + 12 * (tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 10) + 11 * tf.pow(y, 10) - 12 * (
            tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(y, 10) + 495 * tf.pow(x, 8) * tf.pow(y, 2) - 540 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 8) * tf.pow(y, 2) - 2310 * tf.pow(x, 6) * tf.pow(y,
                                                                                                              4) + 2520 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 6) * tf.pow(y, 4) + 2310 * tf.pow(x, 4) * tf.pow(y,
                                                                                                              6) - 2520 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 4) * tf.pow(y, 6) - 495 * tf.pow(x, 2) * tf.pow(y,
                                                                                                             8) + 540 * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * tf.pow(x, 2) * tf.pow(y, 8))
    cfs[90] = 5.0990195135927845 * (
            tf.pow(x, 12) - 66 * tf.pow(x, 10) * tf.pow(y, 2) + 495 * tf.pow(x, 8) * tf.pow(y, 4) - 924 * tf.pow(x,
                                                                                                                 6) * tf.pow(
        y, 6) + 495 * tf.pow(x, 4) * tf.pow(y, 8) - 66 * tf.pow(x, 2) * tf.pow(y, 10) + tf.pow(y, 12))
    cfs[91] = 5.291502622129181 * (
            13 * tf.pow(x, 12) * y - 286 * tf.pow(x, 10) * tf.pow(y, 3) + 1287 * tf.pow(x, 8) * tf.pow(y,
                                                                                                       5) - 1716 * tf.pow(
        x, 6) * tf.pow(y, 7) + 715 * tf.pow(x, 4) * tf.pow(y, 9) - 78 * tf.pow(x, 2) * tf.pow(y, 11) + 1 * tf.pow(y,
                                                                                                                  13))
    cfs[92] = 5.291502622129181 * (
            -132 * tf.pow(x, 10) * y + 1980 * tf.pow(x, 8) * tf.pow(y, 3) - 5544 * tf.pow(x, 6) * tf.pow(y,
                                                                                                         5) + 3960 * tf.pow(
        x, 4) * tf.pow(y, 7) - 660 * tf.pow(x, 2) * tf.pow(y, 9) + 12 * tf.pow(y, 11) + 143 * tf.pow(x, 10) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 2145 * tf.pow(x, 8) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 6006 * tf.pow(x, 6) * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 4290 * tf.pow(x, 4) * tf.pow(y, 7) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 715 * tf.pow(x, 2) * tf.pow(y, 9) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 13 * tf.pow(y, 11) * (tf.pow(x, 2) + tf.pow(y, 2)))
    cfs[93] = 5.291502622129181 * (
            495 * tf.pow(x, 8) * y - 4620 * tf.pow(x, 6) * tf.pow(y, 3) + 6930 * tf.pow(x, 4) * tf.pow(y,
                                                                                                       5) - 1980 * tf.pow(
        x, 2) * tf.pow(y, 7) + 55 * tf.pow(y, 9) - 1188 * tf.pow(x, 8) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 11088 * tf.pow(x, 6) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 16632 * tf.pow(x, 4) * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 4752 * tf.pow(x, 2) * tf.pow(y, 7) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 132 * tf.pow(y, 9) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 702 * tf.pow(x, 8) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 6552 * tf.pow(x, 6) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 9828 * tf.pow(x, 4) * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 2808 * tf.pow(x, 2) * tf.pow(y, 7) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 78 * tf.pow(y, 9) * (tf.pow(x, 2) + tf.pow(y, 2)) * 2)
    cfs[94] = 5.291502622129181 * (
            -840 * tf.pow(x, 6) * y + 4200 * tf.pow(x, 4) * tf.pow(y, 3) - 2520 * tf.pow(x, 2) * tf.pow(y,
                                                                                                        5) + 120 * tf.pow(
        y, 7) + 3465 * tf.pow(x, 6) * y * (tf.pow(x, 2) + tf.pow(y, 2)) - 17325 * tf.pow(x, 4) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 10395 * tf.pow(x, 2) * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 495 * tf.pow(y, 7) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 4620 * tf.pow(x, 6) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 23100 * tf.pow(x, 4) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 13860 * tf.pow(x, 2) * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 660 * tf.pow(y, 7) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 2002 * tf.pow(x, 6) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 10010 * tf.pow(x, 4) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 + 6006 * tf.pow(x, 2) * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 286 * tf.pow(y, 7) * (tf.pow(x, 2) + tf.pow(y, 2)) * 3)
    cfs[95] = 5.291502622129181 * (
            630 * tf.pow(x, 4) * y - 1260 * tf.pow(x, 2) * tf.pow(y, 3) + 126 * tf.pow(y, 5) - 4200 * tf.pow(x,
                                                                                                             4) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 8400 * tf.pow(x, 2) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 840 * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 9900 * tf.pow(x, 4) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 19800 * tf.pow(x, 2) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 1980 * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 9900 * tf.pow(x, 4) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 + 19800 * tf.pow(x, 2) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 1980 * tf.pow(y, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 + 3575 * tf.pow(x, 4) * y * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 4 - 7150 * tf.pow(x, 2) * tf.pow(y, 3) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 4 + 715 * tf.pow(y, 5) * (tf.pow(x, 2) + tf.pow(y, 2)) * 4)
    cfs[96] = 5.291502622129181 * (-168 * tf.pow(x, 2) * y + 56 * tf.pow(y, 3) + 1890 * tf.pow(x, 2) * y * (
            tf.pow(x, 2) + tf.pow(y, 2)) - 630 * tf.pow(y, 3) * (tf.pow(x, 2) + tf.pow(y, 2)) - 7560 * tf.pow(x,
                                                                                                              2) * y * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 2520 * tf.pow(y, 3) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 13860 * tf.pow(x, 2) * y * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 4620 * tf.pow(y, 3) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 11880 * tf.pow(x, 2) * y * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 4 + 3960 * tf.pow(y, 3) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 4 + 3861 * tf.pow(x, 2) * y * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 5 - 1287 * tf.pow(y, 3) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 5)
    cfs[97] = 5.291502622129181 * (7 * y - 168 * y * (tf.pow(x, 2) + tf.pow(y, 2)) + 1260 * y * (
            tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 4200 * y * (tf.pow(x, 2) + tf.pow(y, 2)) * 3 + 6930 * y * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 4 - 5544 * y * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 5 + 1716 * y * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 6)
    cfs[98] = 5.291502622129181 * (7 * x - 168 * y * (tf.pow(x, 2) + tf.pow(y, 2)) + 1260 * x * (
            tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 4200 * x * (tf.pow(x, 2) + tf.pow(y, 2)) * 3 + 6930 * x * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 4 - 5544 * x * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 5 + 1716 * x * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 6)
    cfs[99] = 5.291502622129181 * (-56 * tf.pow(x, 3) + 168 * x * tf.pow(y, 2) + 630 * tf.pow(x, 3) * (
            tf.pow(x, 2) + tf.pow(y, 2)) - 1890 * x * tf.pow(y, 2) * (tf.pow(x, 2) + tf.pow(y, 2)) - 2520 * tf.pow(
        x, 3) * (tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 7560 * x * tf.pow(y, 2) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 4620 * tf.pow(x, 3) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 13860 * x * tf.pow(y, 2) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 3960 * tf.pow(x, 3) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 4 + 11880 * x * tf.pow(y, 2) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 4 + 1287 * tf.pow(x, 3) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 5 - 3861 * x * tf.pow(y, 2) * (
                                           tf.pow(x, 2) + tf.pow(y, 2)) * 5)
    cfs[100] = 5.291502622129181 * (
            126 * tf.pow(x, 5) - 1260 * tf.pow(x, 3) * tf.pow(y, 2) + 630 * x * tf.pow(y, 4) - 840 * tf.pow(x,
                                                                                                            5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 8400 * tf.pow(x, 3) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 4200 * x * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 1980 * tf.pow(x, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 19800 * tf.pow(x, 3) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 9900 * x * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 1980 * tf.pow(x, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 + 19800 * tf.pow(x, 3) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 9900 * x * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 + 715 * tf.pow(x, 5) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 4 - 7150 * tf.pow(x, 3) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 4 + 3575 * x * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 4)
    cfs[101] = 5.291502622129181 * (
            -120 * tf.pow(x, 7) + 2520 * tf.pow(x, 5) * tf.pow(y, 2) - 4200 * tf.pow(x, 3) * tf.pow(y,
                                                                                                    4) + 840 * x * tf.pow(
        y, 6) + 495 * tf.pow(x, 7) * (tf.pow(x, 2) + tf.pow(y, 2)) - 10395 * tf.pow(x, 5) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 17325 * tf.pow(x, 3) * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 3465 * x * tf.pow(y, 6) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 660 * tf.pow(x, 7) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 13860 * tf.pow(x, 5) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 23100 * tf.pow(x, 3) * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 4620 * x * tf.pow(y, 6) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 286 * tf.pow(x, 7) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 6006 * tf.pow(x, 5) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 + 10010 * tf.pow(x, 3) * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3 - 2002 * x * tf.pow(y, 6) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 3)
    cfs[102] = 5.291502622129181 * (
            55 * tf.pow(x, 9) - 1980 * tf.pow(x, 7) * tf.pow(y, 2) + 6930 * tf.pow(x, 5) * tf.pow(y,
                                                                                                  4) - 4620 * tf.pow(
        x, 3) * tf.pow(y, 6) + 495 * x * tf.pow(y, 8) - 132 * tf.pow(x, 9) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 4752 * tf.pow(x, 7) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 16632 * tf.pow(x, 5) * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 11088 * tf.pow(x, 3) * tf.pow(y, 6) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 1188 * x * tf.pow(y, 8) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 78 * tf.pow(x, 9) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 2808 * tf.pow(x, 7) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 9828 * tf.pow(x, 5) * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 - 6552 * tf.pow(x, 3) * tf.pow(y, 6) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2 + 702 * x * tf.pow(y, 8) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) * 2)
    cfs[103] = 5.291502622129181 * (
            -12 * tf.pow(x, 11) + 660 * tf.pow(x, 9) * tf.pow(y, 2) - 3960 * tf.pow(x, 7) * tf.pow(y,
                                                                                                   4) + 5544 * tf.pow(
        x, 5) * tf.pow(y, 6) - 1980 * tf.pow(x, 3) * tf.pow(y, 8) + 132 * x * tf.pow(y, 10) + 13 * tf.pow(x, 11) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 715 * tf.pow(x, 9) * tf.pow(y, 2) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 4290 * tf.pow(x, 7) * tf.pow(y, 4) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 6006 * tf.pow(x, 5) * tf.pow(y, 6) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) + 2145 * tf.pow(x, 3) * tf.pow(y, 8) * (
                    tf.pow(x, 2) + tf.pow(y, 2)) - 143 * x * tf.pow(y, 10) * (tf.pow(x, 2) + tf.pow(y, 2)))
    cfs[104] = 5.291502622129181 * (
            1 * tf.pow(x, 13) - 78 * tf.pow(x, 11) * tf.pow(y, 2) + 715 * tf.pow(x, 9) * tf.pow(y,
                                                                                                4) - 1716 * tf.pow(
        x, 7) * tf.pow(y, 6) + 1287 * tf.pow(x, 5) * tf.pow(y, 8) - 286 * tf.pow(x, 3) * tf.pow(y,
                                                                                                10) + 13 * x * tf.pow(
        y, 12))
    return cfs