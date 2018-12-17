import numpy as np



def k_fold(matriz,k=10):
    Lista = []
    
    for i in range(k):
        Aux = []
        Lista.append(Aux)
        
    Ax = np.shape(matriz)[0] 
    for i in range(Ax):
        Lista[i%k].append(matriz[i])
    print(np.shape(Lista),Lista)
    return Lista




matriz = np.zeros(100)

lista = k_fold(matriz)