import numpy as np


def crossprod(u, v):
    # produto vetorial u x v (Definicao 6.4 do livro)
    w = np.zeros(3)
    w[0] = u[1] * v[2] - u[2] * v[1]
    w[1] = u[2] * v[0] - u[0] * v[2]
    w[2] = u[0] * v[1] - u[1] * v[0]
    return w


# vetores do problema 6.22 a)
u = np.array([1.0, -1.0, 3.0])
v = np.array([-7.0, 12.0, 1.0])

uxv = crossprod(u, v)
vxu = crossprod(v, u)

print("u x v      =", uxv)
print("v x u      =", vxu)
print("<u x v, u> =", np.dot(uxv, u))
print("<v x u, v> =", np.dot(vxu, v))
print("np.cross   =", np.cross(u, v))

# teste com vetores aleatorios
a = np.random.rand(3)
b = np.random.rand(3)
axb = crossprod(a, b)
bxa = crossprod(b, a)

print()
print("a x b      =", axb)
print("b x a      =", bxa)
print("<a x b, a> =", np.dot(axb, a))
print("<b x a, b> =", np.dot(bxa, b))
print("np.cross   =", np.cross(a, b))
