import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (a) funcao que monta a matriz bidiagonal de Wilkinson n x n
def wilkinson_bidiag(n):
    """
    Diagonal principal: n, n-1, ..., 1
    Diagonal de cima:   n repetido n-1 vezes
    """
    principal = np.arange(n, 0, -1, dtype=float)
    de_cima = n * np.ones(n - 1)
    return np.diag(principal) + np.diag(de_cima, 1)


print("Matriz para n = 4:")
print(wilkinson_bidiag(4))

# (b) numero de condicao para n = 1, ..., 15
ordens = np.arange(1, 16)
conds = []
for n in ordens:
    A = wilkinson_bidiag(n)
    conds.append(np.linalg.cond(A))    # norma-2, igual ao cond do MATLAB

print()
print(" n   cond(A)")
for n, c in zip(ordens, conds):
    print(f"{n:2d}   {c:.4e}")

plt.figure(figsize=(6, 4))
plt.semilogy(ordens, conds, "o-")
plt.xlabel("ordem n")
plt.ylabel("cond(A)  (escala log)")
plt.title("Numero de condicao da bidiagonal de Wilkinson")
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.savefig("cond_wilkinson.png", dpi=150)

# (c) autovalores da matriz 20 x 20 antes e depois da perturbacao
A = wilkinson_bidiag(20)
E1 = np.linalg.eigvals(A)

B = A.copy()
B[19, 0] = B[19, 0] + 1e-10    # A(20,1) do MATLAB = B[19,0] no Python
E2 = np.linalg.eigvals(B)

# ordena pela parte real, do maior para o menor
E1 = E1[np.argsort(-E1.real)]
E2 = E2[np.argsort(-E2.real)]

print()
print("Autovalores de A e da matriz perturbada:")
for x, y in zip(E1, E2):
    print(f"{x.real:8.4f}    {y.real:9.4f} {y.imag:+9.4f}i")

plt.figure(figsize=(6, 4))
plt.plot(E1.real, E1.imag, "o", label="original")
plt.plot(E2.real, E2.imag, "x", label="perturbada")
plt.xlabel("parte real")
plt.ylabel("parte imaginaria")
plt.title("Autovalores da Wilkinson 20 x 20")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("autovalores_wilkinson.png", dpi=150)
