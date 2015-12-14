import numpy as np
import os
import csv
import pandas as pd

def cramersV(x, y):
    """
    Calc Cramer's V.

    Parameters
    ----------
    x : {numpy.ndarray, pandas.Series}
    y : {numpy.ndarray, pandas.Series}
    """
    table = np.array(pd.crosstab(x, y)).astype(np.float32)
    n = table.sum()
    colsum = table.sum(axis=0)
    rowsum = table.sum(axis=1)
    expect = np.outer(rowsum, colsum) / n
    chisq = np.sum((table - expect) ** 2 / expect)
    return np.sqrt(chisq / (n * (np.min(table.shape) - 1)))

if __name__ == "__main__":
    # Carregando os dados
    with open("MRMR_data.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        headers = next(readCSV)

        cabecalho = []
        dados = []

        for i in headers:
            cabecalho.append(i)

        for i in readCSV:
            dados.append(np.array(i))

        dados = np.asarray(dados)
        print("Total de notas:",len(dados))

        # Removendo todas as linhas que tenha algum '-1'
        ind_manter = []
        for i in range(1, len(dados)):
            if 'NA' not in dados[i, 8:19]:
                ind_manter.append(i)

        dados = dados[ind_manter]
        print("Total de notas validas:",len(dados))

        # Testando o cramer's v
        for i in range(8, 19):
            for j in range(i + 1, 20):
                print(i, cabecalho[i], j, cabecalho[j], "->", cramersV(dados[:, i], dados[:, j]))
