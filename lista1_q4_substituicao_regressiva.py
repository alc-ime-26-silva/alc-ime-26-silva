# EE 600200 - Algebra Linear Computacional (IME, 2026)
# Lista de Exercicios #1 - Questao 4
# Substituicao regressiva para resolver U x = b (U triangular superior)

import numpy as np


def substituicao_regressiva(U, b):
    """
    Resolve o sistema linear U x = b, onde U eh uma matriz
    triangular superior de ordem n e b eh um vetor coluna de
    tamanho n.

    Parametros
    ----------
    U : matriz triangular superior (n x n)
    b : vetor coluna (tamanho n)

    Retorna
    -------
    x : vetor solucao (tamanho n)

    Levanta
    -------
    ValueError se as dimensoes forem incompativeis ou se algum
    elemento da diagonal principal de U for nulo.
    """
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float).flatten()

    n, m = U.shape
    if n != m:
        raise ValueError("A matriz U deve ser quadrada.")
    if b.size != n:
        raise ValueError("O vetor b deve ter o mesmo tamanho de U.")

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError(
                f"Elemento nulo na diagonal principal (linha {i}). "
                f"O sistema nao possui solucao unica."
            )
        soma = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (b[i] - soma) / U[i, i]

    return x


if __name__ == "__main__":
    # Exemplo de uso
    U = [[2, 1, -1],
         [0, 3, 2],
         [0, 0, 4]]
    b = [3, 13, 8]

    x = substituicao_regressiva(U, b)
    print("Solucao x =", x)  # esperado: [1. 3. 2.]

    # Exemplo com diagonal nula (deve levantar excecao)
    try:
        substituicao_regressiva([[0, 1], [0, 2]], [1, 2])
    except ValueError as e:
        print("Excecao capturada como esperado:", e)
