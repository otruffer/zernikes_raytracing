import math
import numpy as np
from scipy.misc import comb

def N(n, m):
    dm = 1.0 if m == 0 else 0.0
    result = math.sqrt(2.0 * (n + 1.0) / (1.0 + dm))
    if result % 1 == 0:
        return result
    else:
        return "np.sqrt(%s)" % (2.0 * (n + 1.0) / (1.0 + dm))

def Z(m, n):
    M = int(n/2.0 - math.fabs(m - n / 2.0))
    d = int(n - 2.0 * m)
    s = int(np.sign(d))
    p = int(math.fabs(s) / 2 * (s + 1))
    q = int((d - s * n % 2) * s / 2)
    sum = ""
    step_q = np.sign(q) if q != 0 else 1
    step_M = np.sign(M) if M != 0 else 1
    for i in range(0, int(q) + step_q, step_q):
        for j in range(0, int(M) + step_M, step_M):
            step_Mj = np.sign(M - j) if (M - j) != 0 else 1
            for k in range(0, int(M - j) + step_Mj, step_Mj):
                try:
                    eps = 2 * (i + k) + p
                    nu = n - 2 * (i + j + k) - p
                    bef = math.pow(-1, i + j) * comb(n - 2 * M, 2 * i + p) * comb(M - j, k) * math.factorial(n - j) / (math.factorial(j) * math.factorial(M - j) * math.factorial(n - M - j))
                    x = ""
                    y = ""
                    if eps > 0:
                        x = " * x^%s" % eps
                    if nu > 0:
                        y = " * y^%s" % nu
                    if bef > 0:
                        sum += "%s%s%s" % (bef, x, y)
                except Exception as e:
                    return e.message
    return sum

for n in range(0, 14):
    for m in range(-n, n + 1, 2):
        print("N(%s, %s) = %s * %s" % (n, m, N(n, m), Z(m, n)))
