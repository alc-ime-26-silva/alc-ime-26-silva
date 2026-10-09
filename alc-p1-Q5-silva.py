# EE 600200 - Algebra Linear Computacional (IME, 2026)
# Prova 1 - Questao 5
# Solucao de A x = b pela decomposicao LU, sem pivoteamento
# Aluno: Itair Silva Rodrigues

import numpy as np

TOL = 1e-12  # abaixo disso o pivo eh considerado nulo


def substituicao_progressiva(L, b):
    """Resolve L y = b, com L triangular inferior (de cima para baixo)."""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma = soma + L[i, j] * y[j]
        y[i] = (b[i] - soma) / L[i, i]  # aqui L[i, i] = 1
    return y


def substituicao_regressiva(U, y):
    """Resolve U x = y, com U triangular superior (de baixo para cima)."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma = soma + U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]
    return x


def resolve_lu(A, b):
    """
    Resolve A x = b pela decomposicao A = L U, sem pivoteamento.
    Retorna, nesta ordem, as matrizes L e U e o vetor solucao x.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    if b.ndim == 2:  # se b vier como vetor coluna (n x 1), passa para 1D
        b = np.array([b[i, 0] for i in range(n)])

    L = np.eye(n)   # comeca como identidade: diagonal ja vale 1
    U = np.array(A)  # copia de A, que vai virar triangular superior

    for k in range(n):
        # pivo da coluna k
        if abs(U[k, k]) < TOL:
            raise Exception(
                f"Pivo nulo na posicao ({k}, {k}). A decomposicao LU sem "
                f"pivoteamento nao pode continuar. Utilize uma funcao "
                f"alternativa, como a eliminacao de Gauss com pivoteamento."
            )
        # zera os elementos abaixo do pivo
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]   # multiplicador da eliminacao
            L[i, k] = m             # o multiplicador vai para L
            for j in range(k, n):
                U[i, j] = U[i, j] - m * U[k, j]

    y = substituicao_progressiva(L, b)   # L y = b
    x = substituicao_regressiva(U, y)    # U x = y
    return L, U, x


if __name__ == "__main__":
    A = [[2, 1, 1],
         [4, 3, 3],
         [8, 7, 9]]
    b = [4, 10, 24]

    L, U, x = resolve_lu(A, b)
    print("L =")
    print(L)
    print("U =")
    print(U)
    print("x =", x)   # esperado: [1. 1. 1.]

    # exemplo com pivo nulo: deve lancar a excecao
    try:
        resolve_lu([[0, 1], [1, 1]], [1, 2])
    except Exception as e:
        print("Excecao:", e)
