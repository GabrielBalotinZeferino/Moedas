import os
from skimage import io,color
import numpy as np


def divide (X_all,Y_all):
    Q = np.shape(Y_all)[0]
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(Q):
        if(i%5==0):
            X_test.append(X_all[i])
            Y_test.append(Y_all[i])
        else:
            X_train.append(X_all[i])
            Y_train.append(Y_all[i])
    
    print(np.shape(Y_test),np.shape(Y_train))
    return X_train,Y_train,X_test,Y_test

path = r"C:\Users\gbz_2\Desktop\UTFPR\PI\aaaa\SS\all"
imgs = os.listdir(path)
X_all = []
Y_all = []
for file in imgs:
    if(file.endswith('jpg')):
        im = io.imread(path+'/'+file)
        im = color.rgb2gray(im)
        X_all.append(im)
        nome = file.split('_')
        Y_all.append(int(nome[0]))
        
X_train,Y_train,X_test,Y_test = divide(X_all,Y_all)
X_train = np.array(X_train,dtype='float32')
X_test = np.array(X_test,dtype='float32')
Y_train = np.array(Y_train,dtype='int')
Y_test = np.array(Y_test,dtype='int')
np.save('X_train.npy',X_train)
np.save('Y_train.npy',Y_train)
np.save('X_test.npy',X_test)
np.save('y_test.npy',Y_test)