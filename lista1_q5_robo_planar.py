# EE 600200 - Algebra Linear Computacional (IME, 2026)
# Lista de Exercicios #1 - Questao 5
# Posicao do efetuador de um robo planar de dois elos (L1 = 20 cm, L2 = 15 cm)

import math


def posicao_efetuador(theta1, theta2, L1=20.0, L2=15.0):
    """
    Recebe os angulos theta1 e theta2 (em radianos) e retorna a
    posicao (X_U, Y_U), em cm, do efetuador final do robo planar
    de dois elos, com precisao de 1 casa decimal.
    """
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return (round(x, 1), round(y, 1))


if __name__ == "__main__":
    # Exemplo: theta1 = 10 graus, theta2 = 20 graus
    t1 = math.radians(10)
    t2 = math.radians(20)
    print(posicao_efetuador(t1, t2))  # -> (32.7, 11.0)
