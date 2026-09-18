import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FUNÇÕES DE ATIVAÇÃO
# ============================================================

def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def sigmoide_derivada(a):
    return a * (1 - a)


def tanh(z):
    return np.tanh(z)


def tanh_derivada(a):
    return 1 - a ** 2


# ============================================================
# FORWARD
# ============================================================

def forward(X, W1, b1, W2, b2, ativacao="sigmoide"):

    z1 = np.dot(X, W1) + b1

    if ativacao == "tanh":
        h = tanh(z1)
    else:
        h = sigmoide(z1)

    z2 = np.dot(h, W2) + b2

    y_previsto = sigmoide(z2)

    return h, y_previsto


# ============================================================
# FUNÇÃO DE PERDA
# ============================================================

def calcular_perda(y_real, y_previsto):
    return np.mean((y_real - y_previsto) ** 2)


# ============================================================
# BACKWARD
# ============================================================

def backward(
    X,
    y,
    h,
    y_previsto,
    W2,
    ativacao="sigmoide"
):

    # Erro da saída
    erro_saida = (
        y_previsto - y
    ) * sigmoide_derivada(y_previsto)

    # Gradientes da saída
    dW2 = np.dot(h.T, erro_saida)

    db2 = np.sum(
        erro_saida,
        axis=0,
        keepdims=True
    )

    # Derivada da camada oculta
    if ativacao == "tanh":
        derivada_oculta = tanh_derivada(h)
    else:
        derivada_oculta = sigmoide_derivada(h)

    # Erro da camada oculta
    erro_oculta = (
        np.dot(erro_saida, W2.T)
        * derivada_oculta
    )

    # Gradientes da camada oculta
    dW1 = np.dot(
        X.T,
        erro_oculta
    )

    db1 = np.sum(
        erro_oculta,
        axis=0,
        keepdims=True
    )

    return dW1, db1, dW2, db2


# ============================================================
# FUNÇÃO PARA TREINAR A REDE
# ============================================================

def treinar_rede(
    X,
    y,
    n_oculta,
    ativacao="sigmoide",
    epocas=20000,
    taxa_aprendizagem=0.5
):

    np.random.seed(1)

    n_entrada = X.shape[1]
    n_saida = 1

    # Pesos da camada de entrada -> oculta
    W1 = np.random.randn(
        n_entrada,
        n_oculta
    )

    b1 = np.zeros(
        (1, n_oculta)
    )

    # Pesos da camada oculta -> saída
    W2 = np.random.randn(
        n_oculta,
        n_saida
    )

    b2 = np.zeros(
        (1, n_saida)
    )

    historico_perda = []

    # ========================================================
    # TREINAMENTO
    # ========================================================

    for epoca in range(epocas):

        # Forward
        h, y_previsto = forward(
            X,
            W1,
            b1,
            W2,
            b2,
            ativacao
        )

        # Calcula a perda
        perda = calcular_perda(
            y,
            y_previsto
        )

        historico_perda.append(perda)

        # Backward
        dW1, db1, dW2, db2 = backward(
            X,
            y,
            h,
            y_previsto,
            W2,
            ativacao
        )

        # Atualiza os pesos
        W2 -= taxa_aprendizagem * dW2
        b2 -= taxa_aprendizagem * db2

        W1 -= taxa_aprendizagem * dW1
        b1 -= taxa_aprendizagem * db1

    return (
        W1,
        b1,
        W2,
        b2,
        historico_perda
    )


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

def mostrar_resultados(
    nome,
    X,
    y,
    W1,
    b1,
    W2,
    b2,
    ativacao
):

    print("\n========================================")
    print(nome)
    print("========================================")

    _, y_final = forward(
        X,
        W1,
        b1,
        W2,
        b2,
        ativacao
    )

    acertos = 0

    for entrada, previsto, real in zip(
        X,
        y_final,
        y
    ):

        if previsto[0] >= 0.5:
            classe = "ZUMBI"
            classe_numero = 1
        else:
            classe = "HUMANO"
            classe_numero = 0

        if classe_numero == int(real[0]):
            acertos += 1

        print(
            f"{entrada} -> "
            f"{previsto[0]:.3f} "
            f"({classe}) | "
            f"real: {int(real[0])}"
        )

    acuracia = acertos / len(y) * 100

    print(
        f"\nAcertos: {acertos}/{len(y)}"
    )

    print(
        f"Acurácia: {acuracia:.1f}%"
    )