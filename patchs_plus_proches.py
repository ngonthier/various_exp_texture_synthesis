#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:48:06 2018

@author: said
"""

#import skimage 
#import skimage.io as skio
import tensorflow as tf
import numpy as np

#%%

def calcul_plus_proches(im1,im2,ps=5,nbpatch=None,sess=None):
    """ renvoie une carte x,y donnant la position du plus proche 
    dans im1 du patch de im2 :
        EXPLICATION: x[30,45]=la coordonnee x du patch de im1 
    qui est le plus proche du patch im2[30:30+ps,45:45+ps]"""
    #"""Construire le reseau. """  
    
    coul=(len(im1.shape)==3)
    if coul:
        nbcoul=im1.shape[2]
    else:
        nbcoul=1
    (ty2,tx2)=im2.shape[0:2]
    (ty1,tx1)=im1.shape[0:2]
    if nbpatch is None:
        # la taille maximale des tensors tensorflow va donner le max
        # de filtres a appliquer en parallele
        nbpatch=min(2*1024**3//((ty2-ps)*(tx2-ps)*4),10000)        

    im1=np.float32(im1.reshape(ty1,tx1,nbcoul))
    im2=np.float32(im2.reshape(ty2,tx2,nbcoul))
    
    nbtotpatchs=(ty1-ps+1)*(tx1-ps+1)
    tabpatchs=np.zeros((ps,ps,nbcoul,nbtotpatchs))
    xcoord=np.zeros((nbtotpatchs,),np.int32)
    ycoord=np.zeros((nbtotpatchs,),np.int32)
    n2patch=np.zeros((nbtotpatchs,),np.float32)
    
    #imconst=np.zeros((ty1-ps+1,tx1-ps+1,1),np.float32)
    #im2c=(im2**2).sum(axis=2,keepdims=True)
    #for dx in range(ps):
    #    for dy in range(ps):
    #       imconst=im2c[dy:dy+ty2-ps+1,dx:dx+tx2-ps+1]
            
    #imconst=-imconst/2
    
    imentree=tf.constant(im2.reshape((1,*im2.shape)))
    #imconsttf=tf.constant(imconst)
    patchs=tf.placeholder(tf.float32,shape=[ps,ps,nbcoul,nbpatch])
    bias=tf.placeholder(tf.float32,[nbpatch])
    conv=tf.nn.conv2d(imentree,patchs,[1,1,1,1],padding='VALID')+bias
    
    
    maxval=tf.reduce_max(conv,axis=3)
    maxpos=tf.argmax(conv,axis=3,output_type=tf.int32)
    
    if sess is None:
        sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    c=0
    for y in range(ty1-ps+1):
        for x in range(tx1-ps+1):
            xcoord[c]=x
            ycoord[c]=y
            tabpatchs[:,:,:,c]=im1[y:y+ps,x:x+ps]
            n2patch[c]=-1/2*(im1[y:y+ps,x:x+ps]**2).sum()
            c+=1
    actualmax=None
    for l in range(0,nbtotpatchs,nbpatch):
        fin=min(nbtotpatchs,l+nbpatch)
        nbr=fin-l
        if nbr<nbpatch:
            var=np.zeros((ps,ps,nbcoul,nbpatch),np.float32)
            bia=np.zeros((nbpatch,),np.float32)
            for mk in range(0,nbpatch,nbr):
                nnbr=min(nbpatch,mk+nbr)-mk
                var[:,:,:,mk:mk+nnbr]=tabpatchs[:,:,:,l:l+nnbr]
                bia[mk:mk+nnbr]=n2patch[l:l+nnbr]
        else:
            var=tabpatchs[:,:,:,l:fin]
            bia=n2patch[l:fin]
        out=sess.run([maxval,maxpos],feed_dict={patchs:var,bias:bia})
        vmax=out[0].reshape((ty2-ps+1,tx2-ps+1))
        vpos=l+out[1].reshape((ty2-ps+1,tx2-ps+1))%nbr
        if actualmax is None:
            actualmax=vmax
            actualpos=vpos
        else:
            mask=vmax>actualmax
            actualpos[mask]=vpos[mask]
            actualmax[mask]=vmax[mask]
      
    sess.close()
    return (xcoord[actualpos],ycoord[actualpos])
#%% 


#im1=skio.imread()
rnd=np.random
N=512
nbcoul=3
im1=rnd.randn(N,N,nbcoul)

im2=im1+0.9*rnd.randn(N,N,nbcoul)

ps=5

#sess=tf.InteractiveSession()
#%% Exemple d'usage
import time
import matplotlib.pyplot as plt

t0=time.time()
(X,Y)=calcul_plus_proches(im1,im2,ps=ps)
print(time.time()-t0)

#avec deux images im2=im1+bruit on s'attend a ce que X soit presque l'identite...

plt.imshow(X)
plt.show()
# si on prefere les deplacements relatifs aux positions absolues
# c'est comme cela qu'on voit si une zone est un copier-coller, 
# le vecteur X-X0,Y-Y0 est constant sur une grande plage. 
# .... X-XO est nul partout ou X est l'identite
X0,Y0=np.meshgrid(np.arange(0,im2.shape[1]-ps+1),np.arange(0,im2.shape[0]-ps+1))
plt.imshow(abs(X-X0))
plt.show()
#%%  VERFICATION
t0=time.time()
for m in range(10):
    x=rnd.randint(0,N-ps+1)
    y=rnd.randint(0,N-ps+1)
    patchim=im2[y:y+ps,x:x+ps]
    actuamax=None
    for k in range(N-ps+1):
        for l in range(N-ps+1):
            pi1=im1[k:k+ps,l:l+ps]
            n2=((pi1-patchim)**2).sum()
            if actuamax is None:
                actuamax=n2
                mieuxX=l
                mieuxY=k
            elif actuamax>n2:
                actuamax=n2
                mieuxX=l
                mieuxY=k
            
    print (mieuxX,mieuxY,X[y,x],Y[y,x])
print((time.time()-t0)/10*N*N)