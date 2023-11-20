from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

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

"""Comentar KNN
#3- KNN

#3.1- Treinamento (A+B) e teste (C)
# Criar e treinar o classificador KNN
knn1 = KNeighborsClassifier(n_neighbors=9)
knn1.fit(np.concatenate((iris.XA, iris.XB)), np.concatenate((iris.YA, iris.YB)))

# Fazer previsões no conjunto de teste (YC)
YC_pred = knn1.predict(iris.XC)

# Calcula as métricas
acuracia1 = metrics.accuracy_score(iris.YC, YC_pred)
matriz_confusao = metrics.multilabel_confusion_matrix(iris.YC, YC_pred)
tn = matriz_confusao[:, 0, 0]
tp = matriz_confusao[:, 1, 1]
fn = matriz_confusao[:, 1, 0]
fp = matriz_confusao[:, 0, 1]
sensibilidade1 = tp / (tp + fn)
especificidade1 = tn / (tn + fp)
precisao1 = metrics.precision_score(iris.YC, YC_pred, average=None)
media_sens = np.mean(sensibilidade1)
media_espe = np.mean(especificidade1)
media_prec = np.mean(precisao1)

# Exibe os resultados
print("Primeiro treinamento: (A+B) e teste(C)")
print("Acurácia:", acuracia1)
print("\nSensitividade:")
for i in range(3):
    print(f'{iris.metas[i]}: {sensibilidade1[i]:.8f}')
print(f'Media: {media_sens:.8f}')
print("\nEspecificidade:")
for i in range(3):
    print(f'{iris.metas[i]}: {especificidade1[i]:.8f}')
print(f'Media: {media_espe:.8f}')
print("\nPrecisão:")
for i in range(3):
    print(f'{iris.metas[i]}: {precisao1[i]:.8f}')
print(f'Media: {media_prec:.8f}')

#3.2-  Treinamento (A+C) e teste (B)
# Criar e treinar o classificador KNN
knn1 = KNeighborsClassifier(n_neighbors=9)
knn1.fit(np.concatenate((iris.XA, iris.XC)), np.concatenate((iris.YA, iris.YC)))

# Fazer previsões no conjunto de teste (YB)
YB_pred = knn1.predict(iris.XB)

# Calcula as métricas
acuracia2 = metrics.accuracy_score(iris.YB, YB_pred)
matriz_confusao = metrics.multilabel_confusion_matrix(iris.YB, YB_pred)
tn = matriz_confusao[:, 0, 0]
tp = matriz_confusao[:, 1, 1]
fn = matriz_confusao[:, 1, 0]
fp = matriz_confusao[:, 0, 1]
sensibilidade2 = tp / (tp + fn)
especificidade2 = tn / (tn + fp)
precisao2 = metrics.precision_score(iris.YB, YB_pred, average=None)
media_sens = np.mean(sensibilidade2)
media_espe = np.mean(especificidade2)
media_prec = np.mean(precisao2)

# Exibe os resultados
print("Segundo treinamento: (A+C) e teste(B)")
print("Acurácia:", acuracia2)
print("\nSensitividade:")
for i in range(3):
    print(f'{iris.metas[i]}: {sensibilidade2[i]:.8f}')
print(f'Media: {media_sens:.8f}')
print("\nEspecificidade:")
for i in range(3):
    print(f'{iris.metas[i]}: {especificidade2[i]:.8f}')
print(f'Media: {media_espe:.8f}')
print("\nPrecisão:")
for i in range(3):
    print(f'{iris.metas[i]}: {precisao2[i]:.8f}')
print(f'Media: {media_prec:.8f}')

#3.3- Treinamento (C+B) e teste(A)
# Criar e treinar o classificador KNN
knn1 = KNeighborsClassifier(n_neighbors=9)
knn1.fit(np.concatenate((iris.XC, iris.XB)), np.concatenate((iris.YC, iris.YB)))

# Fazer previsões no conjunto de teste (YA)
YA_pred = knn1.predict(iris.XA)

# Calcula as métricas
acuracia3 = metrics.accuracy_score(iris.YA, YA_pred)
matriz_confusao = metrics.multilabel_confusion_matrix(iris.YA, YA_pred)
tn = matriz_confusao[:, 0, 0]
tp = matriz_confusao[:, 1, 1]
fn = matriz_confusao[:, 1, 0]
fp = matriz_confusao[:, 0, 1]
sensibilidade3 = tp / (tp + fn)
especificidade3 = tn / (tn + fp)
precisao3 = metrics.precision_score(iris.YA, YA_pred, average=None)
media_sens = np.mean(sensibilidade3)
media_espe = np.mean(especificidade3)
media_prec = np.mean(precisao3)

# Exibe os resultados
print("Terceiro treinamento: (C+B) e teste(A)")
print("Acurácia:", acuracia3)
print("\nSensitividade:")
for i in range(3):
    print(f'{iris.metas[i]}: {sensibilidade3[i]:.8f}')
print(f'Media: {media_sens:.8f}')
print("\nEspecificidade:")
for i in range(3):
    print(f'{iris.metas[i]}: {especificidade3[i]:.8f}')
print(f'Media: {media_espe:.8f}')
print("\nPrecisão:")
for i in range(3):
    print(f'{iris.metas[i]}: {precisao3[i]:.8f}')
print(f'Media: {media_prec:.8f}')

#3.4 - Valor Medio das Metricas - KNN 

mediasknn = [np.mean([acuracia1, acuracia2, acuracia3]),
np.mean([np.mean(sensibilidade1), np.mean(sensibilidade2), np.mean(sensibilidade3)]),
np.mean([np.mean(especificidade1), np.mean(especificidade2), np.mean(especificidade3)]),
np.mean([np.mean(precisao1), np.mean(precisao2), np.mean(precisao3)])]

fig, ax = plt.subplots(figsize = (7,15))
met = ['Acuracia', 'Sensibilidade', 'Especificidade', 'Precisao']
bar_colors = ['tab:pink', 'tab:blue', 'tab:orange', 'tab:green']
ax.barh(met, mediasknn, color=bar_colors)

for index, value in enumerate(mediasknn):
    plt.text(value, index, str(value))

ax.set_xlabel('Valor medio de cd. metrica')
ax.set_title('Valor medio das metricas p/ knn')

plt.show()


print('\n\nMedias KNN: \n', mediasknn[0])
print('Acuracia:\n', mediasknn[0])
print('Sensitividade:\n', mediasknn[1])
print('Especificidade:\n', mediasknn[2])
print('Especificidade:\n', mediasknn[3])
#"""

"""Comentar AD
# 4- Árvore de Decisão
# 4.1- Treinamento (A+B) e teste(C)

# Cria uma instância do classificador de árvore de decisão
clf1 = DecisionTreeClassifier()

# Combina os conjuntos de treinamento A e B para treinar o classificador
X_train = iris.XA + iris.XB
Y_train = iris.YA + iris.YB

# Treina o classificador usando os dados de treinamento combinados
clf1.fit(X_train, Y_train)

# Faz a previsão para o conjunto de teste C
YC_pred = clf1.predict(iris.XC)

# 4.1.1- Métricas (primeiro treinamento)
# Calcula as métricas
acuraciaB = metrics.accuracy_score(iris.YC, YC_pred)
matriz_confusao = metrics.multilabel_confusion_matrix(iris.YC, YC_pred)
tn = matriz_confusao[:, 0, 0]
tp = matriz_confusao[:, 1, 1]
fn = matriz_confusao[:, 1, 0]
fp = matriz_confusao[:, 0, 1]
sensibilidadeB = tp / (tp + fn)
especificidadeB = tn / (tn + fp)
precisaoB = metrics.precision_score(iris.YC, YC_pred, average=None)
media_sens = np.mean(sensibilidadeB)
media_espe = np.mean(especificidadeB)
media_prec = np.mean(precisaoB)

# Exibe os resultados
print("Primeiro treinamento: (A+B) e teste(C)")
print("\nAcurácia:", acuraciaB)
print("\nSensitividade:")
for i in range(3):
    print(f'{iris.metas[i]}: {sensibilidadeB[i]:.8f}')
print(f'Media: {media_sens:.8f}')
print("\nEspecificidade:")
for i in range(3):
    print(f'{iris.metas[i]}: {especificidadeB[i]:.8f}')
print(f'Media: {media_espe:.8f}')
print("\nPrecisão:")
for i in range(3):
    print(f'{iris.metas[i]}: {precisaoB[i]:.8f}')
print(f'Media: {media_prec:.8f}')

# 4.1.2- Configuração da visualização da árvore
plt.figure(figsize=(8, 8))
plot_tree(clf1, filled=True, rounded=True, feature_names=iris.indice, class_names=iris.metas, fontsize=10)

# 4.1.3- Exibe a árvore de decisão
plt.show()

# 4.2- Treinamento (A+C) e teste(B)
# Cria uma instância do classificador de árvore de decisão
clf2 = DecisionTreeClassifier()

# Combina os conjuntos de treinamento A e B para treinar o classificador
X_train = iris.XA + iris.XC
Y_train = iris.YA + iris.YC

# Treina o classificador usando os dados de treinamento combinados
clf2.fit(X_train, Y_train)

# Faz a previsão para o conjunto de teste B
YB_pred = clf2.predict(iris.XB)

# 4.2.1- Métricas (segundo treinamento)
# Calcula as métricas
acuraciaC = metrics.accuracy_score(iris.YB, YB_pred)
matriz_confusao = metrics.multilabel_confusion_matrix(iris.YB, YB_pred)
tn = matriz_confusao[:, 0, 0]
tp = matriz_confusao[:, 1, 1]
fn = matriz_confusao[:, 1, 0]
fp = matriz_confusao[:, 0, 1]
sensibilidadeC = tp / (tp + fn)
especificidadeC = tn / (tn + fp)
precisaoC = metrics.precision_score(iris.YB, YB_pred, average=None)
media_sens = np.mean(sensibilidadeC)
media_espe = np.mean(especificidadeC)
media_prec = np.mean(precisaoC)

# Exibe os resultados
print("\n\nSegundo treinamento: (A+C) e teste(B)")
print("Acurácia:", acuraciaC)
print("\nSensitividade:")
for i in range(3):
    print(f'{iris.metas[i]}: {sensibilidadeC[i]:.8f}')
print(f'Media: {media_sens:.8f}')
print("\nEspecificidade:")
for i in range(3):
    print(f'{iris.metas[i]}: {especificidadeC[i]:.8f}')
print(f'Media: {media_espe:.8f}')
print("\nPrecisão:")
for i in range(3):
    print(f'{iris.metas[i]}: {precisaoC[i]:.8f}')
print(f'Media: {media_prec:.8f}')

# 4.2.2- Configuração da visualização da árvore
plt.figure(figsize=(8, 8))
plot_tree(clf2, filled=True, rounded=True, feature_names=iris.indice, class_names=iris.metas, fontsize=10)

# 4.2.3- Exibe a árvore de decisão
plt.show()

# 4.3- Treinamento (C+B) e teste(A)
# Cria uma instância do classificador de árvore de decisão
clf3 = DecisionTreeClassifier()

# Combina os conjuntos de treinamento A e B para treinar o classificador
X_train = iris.XC + iris.XB
Y_train = iris.YC + iris.YB

# Treina o classificador usando os dados de treinamento combinados
clf3.fit(X_train, Y_train)

# Faz a previsão para o conjunto de teste A
YA_pred = clf3.predict(iris.XA)

# 4.3.1- Métricas (terceiro treinamento)
# Calcula as métricas
acuraciaD = metrics.accuracy_score(iris.YA, YA_pred)
matriz_confusao = metrics.multilabel_confusion_matrix(iris.YA, YA_pred)
tn = matriz_confusao[:, 0, 0]
tp = matriz_confusao[:, 1, 1]
fn = matriz_confusao[:, 1, 0]
fp = matriz_confusao[:, 0, 1]
sensibilidadeD = tp / (tp + fn)
especificidadeD = tn / (tn + fp)
precisaoD = metrics.precision_score(iris.YA, YA_pred, average=None)
media_sens = np.mean(sensibilidadeD)
media_espe = np.mean(especificidadeD)
media_prec = np.mean(precisaoD)

# Exibe os resultados
print("\n\nTerceiro treinamento: (C+B) e teste(A)")
print("Acurácia:", acuraciaD)
print("\nSensitividade:")
for i in range(3):
    print(f'{iris.metas[i]}: {sensibilidadeD[i]:.8f}')
print(f'Media: {media_sens:.8f}')
print("\nEspecificidade:")
for i in range(3):
    print(f'{iris.metas[i]}: {especificidadeD[i]:.8f}')
print(f'Media: {media_espe:.8f}')
print("\nPrecisão:")
for i in range(3):
    print(f'{iris.metas[i]}: {precisaoD[i]:.8f}')
print(f'Media: {media_prec:.8f}')

# 4.3.2- Configuração da visualização da árvore
plt.figure(figsize=(5, 5))
plot_tree(clf3, filled=True, rounded=True, feature_names=iris.indice, class_names=iris.metas, fontsize=10)

# 4.3.3- Exibe a árvore de decisão
plt.show()

#4.3.4- Valor Medio das Metricas - AD
mediasad = [np.mean([acuraciaB, acuraciaC, acuraciaD]),
np.mean([np.mean(sensibilidadeB), np.mean(sensibilidadeC), np.mean(sensibilidadeD)]),
np.mean([np.mean(especificidadeB), np.mean(especificidadeC), np.mean(especificidadeD)]),
np.mean([np.mean(precisaoB), np.mean(precisaoC), np.mean(precisaoD)])]
print('\n\nMedias AD: \n')
print('Acuracia:\n', mediasad[0])
print('Sensitividade:\n', mediasad[1])
print('Especificidade:\n', mediasad[2])
print('Precisao:\n', mediasad[3])


fig, ax = plt.subplots(figsize = (7,15))
met = ['Acuracia', 'Sensitividade', 'Especificidade', 'Precisao']
bar_colors = ['tab:pink', 'tab:blue', 'tab:orange', 'tab:green']
ax.barh(met, mediasad, color=bar_colors)

for index, value in enumerate(mediasad):
    plt.text(value, index, str(value))

ax.set_xlabel('Valor medio de cd. metrica')
ax.set_title('Valor medio das metricas p/ AD')

plt.show()

#FIM AD
"""


#5- KNN # AD
mediasAD = [0.9129251700680272, 0.9129901960784313, 0.9567179144385028, 0.9129901960784313]
mediasKNN = [0.9663945578231292, 0.9669117647058824, 0.9833630421865717, 0.9691358024691358]

# Defina a largura da linha
lineWidth = 2

# Configuração do gráfico de linhas
fig, ax = plt.subplots(figsize=(9, 6))

# Linha para Árvore de Decisão
plt.plot(['Acuracia', 'Sensibilidade', 'Especificidade', 'Precisao'], mediasAD, marker='o', color='r', linewidth=lineWidth, label='Árvore de Decisão')

# Linha para K-Nearest Neighbours
plt.plot(['Acuracia', 'Sensibilidade', 'Especificidade', 'Precisao'], mediasKNN, marker='o', color='b', linewidth=lineWidth, label='K-Nearest Neighbours')

plt.minorticks_on()

plt.ylabel('Valor médio de cada métrica', fontweight='bold', fontsize=12)
plt.yticks([0.0, 0.3, 0.6, 0.9, 1.2])

plt.title('Comparativo AD e KNN')

plt.legend()
plt.show()