# Nome: Luan Mendes Gonçalves Freitas
# Disciplina: Fundamentos de Sistemas Inteligentes - Turma A
# Projeto 1 - Classificadores de Manuscritos
# Módulo main

import os
from kna_lda import KNN, LDA

def main():
    opcao = 0
    while opcao < 4:
        print('1 - Executar Algoritmo LDA')
        print('2 - Executar Algoritmo KNN')
        print('3 - Executar Algoritmos LDA e KNN')
        print('4 - Sair')
        opcao = int(input('Digite uma opção '))
        if opcao == 1:
            os.system('clear') or None
            lda = LDA()
            lda.processamento()
        elif opcao == 2:
            os.system('clear') or None
            knn = KNN()
            knn.processamento()
        elif opcao == 3:
            os.system('clear') or None
            lda = LDA()
            lda.processamento()
            knn = KNN()
            knn.processamento()
        elif opcao != 4:
            os.system('clear') or None
            print('Invalido opção digite novamente ')
            opcao = 0
    print('fim')

    
if __name__ == '__main__':
    main()
