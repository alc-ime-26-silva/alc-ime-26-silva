# EE 600200 - Algebra Linear Computacional (IME, 2026)
# Lista de Exercicios #2 - Questao 5 (Exercicios 6.38 e 6.39 de [FORD])
# Teste de ortogonalidade de matrizes pela definicao e pelos vetores coluna

import numpy as np

# As matrizes do livro estao arredondadas (5 algarismos), entao
# P^T P nao da exatamente a identidade. Por isso uso uma tolerancia.
TOL = 1e-4


def is_orthogonal_by_definition(P):
    """
    Testa se P eh ortogonal pela definicao: P^T P = I.
    Recebe a matriz quadrada P e retorna True ou False.
    """
    P = np.array(P, dtype=float)
    n, m = P.shape
    if n != m:
        return False
    D = P.T @ P - np.eye(n)
    return bool(np.max(np.abs(D)) < TOL)


def is_orthogonal_by_vectors(P):
    """
    Testa se P eh ortogonal olhando as colunas: cada coluna tem
    que ter norma 1 e ser ortogonal a todas as outras.
    Recebe a matriz quadrada P e retorna True ou False.
    """
    P = np.array(P, dtype=float)
    n, m = P.shape
    if n != m:
        return False
    for i in range(n):
        # norma da coluna i
        if abs(np.linalg.norm(P[:, i]) - 1) > TOL:
            return False
        # produto interno da coluna i com as colunas seguintes
        for j in range(i + 1, n):
            if abs(np.dot(P[:, i], P[:, j])) > TOL:
                return False
    return True


# Exercicio 6.38
A1 = [[-0.40825, 0.43644, 0.80178],
      [-0.8165, 0.21822, -0.53452],
      [-0.40825, -0.87287, 0.26726]]

A2 = [[-0.51450, 0.48507, 0.70711],
      [-0.68599, -0.72761, 0.0000],
      [0.51450, -0.48507, 0.70711]]

# Exercicio 6.39
B1 = [[-0.58835, 0.70206, 0.40119],
      [-0.78446, -0.37524, -0.49377],
      [-0.19612, -0.60523, 0.77152]]

B2 = [[-0.47624, -0.4264, 0.30151],
      [0.087932, 0.86603, -0.40825],
      [-0.87491, -0.26112, 0.86164]]

testes = [("6.38 a)", A1), ("6.38 b)", A2),
          ("6.39 a)", B1), ("6.39 b)", B2)]

for nome, M in testes:
    r1 = is_orthogonal_by_definition(M)
    r2 = is_orthogonal_by_vectors(M)
    print(nome, " definicao:", r1, " vetores:", r2)

# para entender o caso que deu False
B2 = np.array(B2)
print("P^T P da matriz 6.39 b):")
print(np.round(B2.T @ B2, 5))
