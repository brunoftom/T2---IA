from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.datasets import load_iris

# 1- Instancia e Divisão da base de dados
class IrisDB:
    def __init__(self) -> None:
        # Carrega o conjunto de dados Iris
        self.db = load_iris()

        # Inicializa os conjuntos de dados X e Y para as classes A, B e C usando o método _extract_data
        self.XA, self.YA = self._extract_data(0)  # Dados da classe A
        self.XB, self.YB = self._extract_data(1)  # Dados da classe B
        self.XC, self.YC = self._extract_data(2)  # Dados da classe C

        # Obtém os nomes das características e das classes alvo
        self.indice = self.db['feature_names']  # Nomes das características
        self.metas = self.db['target_names']    # Nomes das classes alvo

    def _extract_data(self, start_index: int) -> Tuple[List[List[float]], List[str]]:
        # Método privado para extrair dados da base de dados Iris
        X, Y = [], []

        # Itera sobre os índices de start_index até 149 (com um passo de 3)
        for k in range(start_index, 149, 3):
            # Adiciona os dados e as classes alvo aos conjuntos X e Y
            X.append(self.db['data'][k])
            Y.append(self.db['target'][k])

        # Retorna os conjuntos de dados X e Y
        return X, Y

# Instancia a classe IrisDB, criando uma instância chamada 'iris'
iris = IrisDB()

# 2- Gráfico de Barras
# Define a largura das barras
largura_barra = 0.20

# Configura o tamanho da figura
fig, ax = plt.subplots(figsize=(8, 4))

# Define as posições das barras no eixo X
br1 = np.arange(3)
br2 = [x + largura_barra for x in br1]
br3 = [x + largura_barra for x in br2]

# Cores personalizadas para cada classe
cores = ['y', 'g', 'b']

# Cria o gráfico de barras
for i in range(3):
    plt.bar(br1 + largura_barra * i, [iris.YA.count(i), iris.YB.count(i), iris.YC.count(i)], color=cores[i],
            width=largura_barra, edgecolor='black', label=iris.metas[i])

# Adiciona rótulos no eixo X
plt.xlabel('Nome segmento', fontweight='bold', fontsize=14) 
plt.ylabel('Nro de amostras', fontweight='bold', fontsize=12) 
plt.xticks([r + largura_barra for r in range(3)], ['A', 'B', 'C'])
plt.yticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

# Adiciona a legenda
plt.legend()

# Exibe o gráfico
plt.show()

# 3- Árvore de Decisão
# 3.1- Treinamento (A+B) e teste(C)

# Cria uma instância do classificador de árvore de decisão
clf1 = DecisionTreeClassifier()

# Combina os conjuntos de treinamento A e B para treinar o classificador
X_train = iris.XA + iris.XB
Y_train = iris.YA + iris.YB

# Treina o classificador usando os dados de treinamento combinados
clf1.fit(X_train, Y_train)

# Faz a previsão para o conjunto de teste C
YC_pred = clf1.predict(iris.XC)

# 3.2- Métricas (primeiro treinamento)
# Calcula as métricas
acuracia = metrics.accuracy_score(iris.YC, YC_pred)
matriz_confusao = metrics.multilabel_confusion_matrix(iris.YC, YC_pred)
tn = matriz_confusao[:, 0, 0]
tp = matriz_confusao[:, 1, 1]
fn = matriz_confusao[:, 1, 0]
fp = matriz_confusao[:, 0, 1]
sensibilidade = tp / (tp + fn)
especificidade = tn / (tn + fp)
precisao = metrics.precision_score(iris.YC, YC_pred, average=None)

# Exibe os resultados
print("Acurácia:", acuracia)
print("\nSensitividade:")
for i in range(3):
    print(f'{iris.metas[i]}: {sensibilidade[i]:.8f}')
print("\nEspecificidade:")
for i in range(3):
    print(f'{iris.metas[i]}: {especificidade[i]:.8f}')
print("\nPrecisão:")
for i in range(3):
    print(f'{iris.metas[i]}: {precisao[i]:.8f}')

# 3.3- Configuração da visualização da árvore
plt.figure(figsize=(8, 8))
plot_tree(clf1, filled=True, rounded=True, feature_names=iris.indice, class_names=iris.metas, fontsize=10)

# 3.4- Exibe a árvore de decisão
plt.show()