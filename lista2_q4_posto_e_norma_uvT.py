# EE 600200 - Algebra Linear Computacional (IME, 2026)
# Lista de Exercicios #2 - Questao 4 (Exercicio 7.32 de [FORD])
# Posto e norma-2 da matriz A = u v^T para vetores aleatorios u e v

import numpy as np

# semente fixa so para os numeros do relatorio nao mudarem
rng = np.random.default_rng(2026)

print("  n  posto   ||u||*||v||     ||u v^T||_2   diferenca")

for n in (5, 15, 25):
    # equivalente ao rand(n,1) do MATLAB
    u = rng.random((n, 1))
    v = rng.random((n, 1))

    A = u @ v.T  # matriz n x n

    posto = np.linalg.matrix_rank(A)
    prod_normas = np.linalg.norm(u) * np.linalg.norm(v)
    norma_A = np.linalg.norm(A, 2)  # norma-2 de matriz

    dif = abs(prod_normas - norma_A)
    print(f"{n:3d} {posto:6d} {prod_normas:13.8f} {norma_A:15.8f}"
          f" {dif:11.1e}")
