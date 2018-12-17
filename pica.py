#import lib_matting
from matplotlib import pyplot as plt
from skimage import io,color
from skimage.morphology import disk
from skimage import measure
import numpy as np
import os
from scipy.ndimage.morphology import binary_dilation as BD
from scipy.ndimage.morphology import binary_erosion as BE
from scipy.ndimage.morphology import binary_fill_holes as fh
#import scipy.misc
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import gabor_kernel
from skimage.transform import resize
def reg_prop(im):
    label_image = measure.label(im)
    label_image2 = np.zeros(np.shape(label_image))
#    print(np.shape(im))
#    print(np.shape(label_image))
    for region in measure.regionprops(label_image):
        idx = region.coords
        a,b = np.shape(idx)
#        print(np.shape(idx))
        if(region.perimeter>40):
#            print(region.perimeter)
            label_image2[idx[0:a,0],idx[0:a,1]] = 4*region.area/(region.perimeter*region.perimeter)
        else:
            
#            print(region.perimeter)
            label_image2[idx[0:a,0],idx[0:a,1]] = 0
                        
    label_image2 = np.where(label_image2==np.max(label_image2),1,0)
    return label_image2
def recon(im):
    imR = im[:,:,0].copy()
#    imG = im[:,:,1].copy()
    imB = im[:,:,2].copy()
#    print(np.max(imR))
    im_aux = np.zeros(np.shape(imR))
    imR = imR.astype('int')
    imB = imB.astype('int')
    im_aux = imR-imB
#    im_aux = np.where(np.bitwise_and(imR>200,imG<50),1,0)
    im_aux = np.where(im_aux>60,1,0)
    im_aux = BD(im_aux)
    im_aux = fh(im_aux)
    
#    struct = disk(2.5)
###           
#    im_aux = BD(im_aux, iterations=8,structure=struct)
#    im_aux = BE(im_aux, iterations=8,structure=struct)
    return im_aux
def marca_area(im,im_s):
    im_s = reg_prop(im_s)
    X,Y = np.shape(im_s)
    Ax,Ay = np.where(im_s>0)
    Cx = int(Ax.mean())
    Cy = int(Ay.mean())
#    print(Ax.shape)
    if(np.shape(Ax)[0]==0 or np.shape(Ax)[0]>(X*Y)/2):
#        print('Passei aqui')
        result = resize(im,(80,80,3))
        return (result*255).astype('uint8')
    if(Cx<40):
        Cx =40
    if(Cy<40):
        Cy = 40
    if(Cx+40>X):
        Cx = X-41
    if(Cy+40>Y):
        Cy = Y-41
    result = im[Cx-40:Cx+40,Cy-40:Cy+40]
    return result
path = r'C:\Users\gbz_2\Desktop\UTFPR\PI\Resultados\comparação'
path_o = r'C:\Users\gbz_2\Desktop\UTFPR\PI\aaaa\SS\all'
imgs = os.listdir(path_o )

for file in imgs:
    if(file.endswith('jpg')):
        nome = file.split('_')
        print(file)
        im_s = io.imread(path+'/s_'+file)#imagem segmentada
        im_s = recon(im_s)
        im_o = io.imread(path_o+'/'+file)
        im_s = marca_area(im_o,im_s)
#        plt.figure()
#        plt.subplot(121)
#        plt.imshow(im_o,cmap='gray')
#        plt.subplot(122)
#        plt.imshow(im_s,cmap = 'gray')
        io.imsave(r'C:\Users\gbz_2\Desktop\UTFPR\PI\picado'+'/'+file,im_s,quality=100)