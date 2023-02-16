# Nome: Luan Mendes Gonçalves Freitas
# Disciplina: Fundamentos de Sistemas Inteligentes - Turma A
# Projeto 1 - Classificadores de Manuscritos
# Módulo da superclasse algoritmo e das subclasses knn e lda

import collections
import numpy as np
import pandas as pd
from mnist import MNIST
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Algoritmo:
    
    def __init__ (self):
        self.imagens = None
        self.labels = None
        self.mnistImagens = MNIST('imagens')
        
    # Método para processamento de teste aprendizado
    def processoTeste(self):
        self.imagens, self.labels = self.mnistImagens.load_training()
    
        # Método de fábrica para criar subclasses de tupla com campos nomeados
        teste = collections.namedtuple('teste', ['imagens', 'labels'])
    
        return teste(self.imagens, self.labels)
    
    # Método para processamento de treino aprendizado
    def processoTreino(self):
        self.imagens, self.labels = self.mnistImagens.load_testing()
    
        # Método de fábrica para criar subclasses de tupla com campos nomeados
        treino = collections.namedtuple('treino', ['imagens', 'labels'])
    
        return treino(self.imagens, self.labels)

# Classe do Algoritmo de Aprendizagem Supervisionada K-NN (Classificador K-Vizinhos)
class KNN (Algoritmo):
    
    def __init__ (self):
        super().__init__()
    
    def processamento(self):
        print('Algoritmo KNN')
        
        treino = self.processoTreino() 
        
        # Crie uma matriz
        treinoImagem = np.array(treino.imagens)
        treinoLabels = np.array(treino.labels)
        print('Matriz da Soma das linhas do arquivo de treino: ' + str(treinoLabels))
        
        teste = self.processoTeste()
    
        # Crie uma matriz
        testeImagem = np.array(teste.imagens)
        testeLabels = np.array(teste.labels)    
        print('Matriz da Soma das linhas do arquivo de teste: ' + str(testeLabels))
    
        n_neighbors = 3
        print('Classificando Vizinhos k = ' + str(n_neighbors))
        
        # Encontra os Vizinhos mais proximos ao valor de K
        classificadorVizinhos = KNeighborsClassifier(n_neighbors)
        
        # Ajustar o modelo usando o paramêtro 1 como dados de treinamento e 
        # paramêtro 2 como valores de destino
        classificadorVizinhos.fit(treinoImagem, treinoLabels)
        
        # Prever os rótulos de classe para os dados fornecidos
        precisao = classificadorVizinhos.predict(testeImagem)
        
        # Pontuação de classificação de precisão.
        print('Precisão: ' + str(accuracy_score(testeLabels, precisao) * 100) + '%')
        
        print("Matriz de Confusão KNN")
        # Compute a matriz de confusão para avaliar a precisão de uma classificação
        print(pd.crosstab(testeLabels, precisao, rownames=['Real'], colnames=['Predito'], margins=True))

# Classe do Algoritmo de Aprendizagem Supervisionada LDA (Análise Discriminante Linear)    
class LDA(Algoritmo):

    def __init__(self):
        super().__init__()
        
    def processamento(self):
        print('Algoritmo LDA')
        treino = self.processoTreino() 
        
        # Crie uma matriz
        treinoImagem = np.array(treino.imagens)
        treinoLabels = np.array(treino.labels)
        print('Soma das linhas do arquivo de treino: ' + str(treinoLabels))
        
        teste = self.processoTeste()
        
        # Crie uma matriz
        testeImagem = np.array(teste.imagens)
        testeLabels = np.array(teste.labels)
        print('Soma das linhas do arquivo de teste: ' + str(testeLabels))
    
        # Um classificador com um limite de decisão linear, 
        # gerado pela montagem de densidades condicionais de classe 
        # aos dados e usando a regra de Bayes.
        AnalisadorDiscriminante = LinearDiscriminantAnalysis()
        
        # Ajuste o modelo LinearDiscriminantAnalysis de acordo com os dados e parâmetros de treinamento fornecidos.
        AnalisadorDiscriminante.fit(treinoImagem, treinoLabels)
    
        # Prever os rótulos de classe para os dados fornecidos
        precisao = AnalisadorDiscriminante.predict(testeImagem)
        print('Precisão: ' + str(accuracy_score(testeLabels, precisao) * 100) + '%')
        
        print("Matriz de Confusao LDA")
        # Compute a matriz de confusão para avaliar a precisão de uma classificação        
        print(pd.crosstab(testeLabels, precisao, rownames=['Real'], colnames=['Predito'], margins=True))
