import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
import os
from skimage import io,color
import random
import time
K.set_image_dim_ordering('th')


# define the larger model
def larger_model():
	# create model
    model = Sequential()
    model.add(Conv2D(128, (5, 5),input_shape = (1,80,80), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
def k_fold(matriz,matriz2,k=10):
    Lista = []
    Lista2 = []
    for i in range(k):
        Aux = []
        Lista.append(Aux.copy())
        Lista2.append(Aux.copy())
        
    Ax = 3059
    print(np.shape(matriz[0]),np.shape(matriz2))
    for i in range(Ax):
        Lista[i%k].append(matriz[i])
        Lista2[i%k].append(matriz2[i])
#    print(np.shape(Lista),Lista)
    return Lista,Lista2
def redefine(Y):
    Y = np.array(Y)
    Y[np.where(Y==5)] = 0
    Y[np.where(Y==10)] = 1
    Y[np.where(Y==25)] = 2
    Y[np.where(Y==50)] = 3
    Y[np.where(Y==100)] = 4
    Y = list(Y)
    return Y
def org (X_all,Y_all,i,k=10):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for j in range(k):
        Aux = X_all[j]
        Aux2 = Y_all[j]
        if(j==i):
            X_test = X_all[j]
            Y_test = Y_all[j]
        else:
            for s0 in Aux:
                X_train.append(s0)
            for s0 in Aux2:
                Y_train.append(s0)
    X_train = np.array(X_train,dtype='float32')
    X_test = np.array(X_test,dtype='float32')
    Y_train = np.array(Y_train,dtype='float32')
    Y_test = np.array(Y_test,dtype='float32')
    return X_train,Y_train,X_test,Y_test
path = r"C:\Users\gbz_2\Desktop\UTFPR\PI\picado"
imgs = os.listdir(path)
X_all = []
Y_all = []
print('Etapa 1')
i=0
for file in imgs:
#    print(i)
#    i+=1
    if(file.endswith('jpg')):
        im = io.imread(path+'/'+file)
        im = color.rgb2gray(im)
        im = np.resize(im,(1,80,80))
#        print(np.shape(im))
        X_all.append(im)
        nome = file.split('_')
        Y_all.append(int(nome[0]))

X_fold,Y_fold = k_fold(X_all,Y_all)
#print(np.shape(X_fold),np.shape(Y_fold))

print('Etapa 2')
seed = 7
np.random.seed(seed)
#X_fold  = np.array(X_fold,dtype='float32')

## one hot encode outputs
for i in range(10):
    Y_fold[i] = redefine(Y_fold[i])
    Y_fold[i] = np_utils.to_categorical(Y_fold[i])

num_classes = 5
print('Etapa 3')
l_scores = []
start_time = time.time()
for i in range(10):
    X_train,y_train,X_test,y_test = org(X_fold,Y_fold,i)
#    print(np.shape(X_train),np.shape(X_test))

   
    # build the model
    model = larger_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=8,verbose=2, batch_size=100)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    l_scores.append(scores[1])
    print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

end_time = time.time()
print("Tempo gasto na rede: ",end_time-start_time,'s')
print('Taxas de acerto: ',l_scores)
print('Média de acerto de ',np.mean(l_scores),'com desvio padrão de ',np.std(l_scores))

