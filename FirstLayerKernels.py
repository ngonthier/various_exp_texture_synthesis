#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24 oct 2017

The goal of this script is to vizualised the filter of the first layer

@author: nicolas
"""

import scipy
import numpy as np
import tensorflow as tf
import Style_Transfer as st
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

def plot_and_saveinPDF():
    path = 'Results/Filter_Rep/'
    pltname = path+'FirstLayers_Kernels.pdf'
    pp = PdfPages(pltname) 
    vgg_layers = st.get_vgg_layers()
    index_in_vgg = 0
    kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
    #  A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #print(kernels.shape)
    bias = vgg_layers[index_in_vgg][0][0][2][0][1]
    #print(bias.shape)
    input_kernel = kernels.shape[2]
    number_of_kernels = kernels.shape[3]
    alpha=0.6
    cmkernel = plt.get_cmap('hot')
    colorList = ['b','g','r']
    X = np.arange(0, 3, 1)
    Y = np.arange(0, 3, 1)
    X, Y = np.meshgrid(X, Y)
    
    for i in range(number_of_kernels):
        
            kernel =  kernels[:,:,:,i]
            mean_kernel = np.mean(kernel)
            vmin = np.min(kernel)
            vmax = np.max(kernel)
            print("Features",i)
            # For each feature
            fig, axes = plt.subplots(2, 3)
            #print(axes)
            
            list_colors = ['Blues','Greens','Reds']
            
            # Plot the values of each channel of the kernel 
            axes[0,0].matshow(kernel[:,:,0],cmap=plt.get_cmap('Blues'),vmin=vmin, vmax=vmax)
            axes[0,1].matshow(kernel[:,:,1],cmap=plt.get_cmap('Greens'),vmin=vmin, vmax=vmax)
            axes[0,2].matshow(kernel[:,:,2],cmap=plt.get_cmap('Reds'),vmin=vmin, vmax=vmax)
            # rajouter mean 
            for j in range(3):
                title =  '{:.2e}'.format(np.mean(kernel[:,:,j])) + '\n'
                axes[0,j].set_title(title)
            axes[1,0].matshow(kernel[:,:,0],cmap=cmkernel,vmin=vmin, vmax=vmax)
            axes[1,1].matshow(kernel[:,:,1],cmap=cmkernel,vmin=vmin, vmax=vmax)
            axes[1,2].matshow(kernel[:,:,2],cmap=cmkernel,vmin=vmin, vmax=vmax)
            
            titre = 'Kernel {} '.format(i)
            plt.suptitle(titre)
            plt.savefig(pp, format='pdf')
            plt.close()
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # Plot on the same figure 
            for j in range(3):
                ax.plot_surface(X, Y, kernel[:,:,j],cmap=plt.get_cmap(list_colors[j]),alpha=alpha,linewidth=0, antialiased=False)           
            titre = 'Kernel {} in 3D'.format(i)
            plt.suptitle(titre)
            plt.savefig(pp, format='pdf')
            plt.close()
            
            fig = plt.figure()
            kernelsChange = (kernel.transpose(2,0,1).reshape(3,9)).transpose(1,0)
            box = plt.boxplot(kernelsChange, patch_artist=True)
            for patch, color in zip(box['boxes'], colorList):
                patch.set_facecolor(color)
            titre = 'Kernel {} in barplot'.format(i)
            plt.suptitle(titre)
            plt.savefig(pp, format='pdf')
            plt.close()
            
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            for j in range(3):
                data = kernel[:,:,j].flatten()
                ax1.scatter(data,np.zeros_like(data), c=colorList[j],alpha=alpha)
            titre = 'Kernel {} in 1D'.format(i)
            plt.suptitle(titre)
            plt.savefig(pp, format='pdf')
            plt.close()
            
            # Problem of Alpha for plotting in Barplot 3D
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            ## Plot on the same figure 
            #zpos = np.zeros_like(X.flatten('F'))
            #alphabar = 0.5*np.ones_like(X.flatten('F'))
            #alphabar = 0.5
            #print(alphabar)
            #for j in range(3):
                #ax.bar3d(X.flatten('F'), Y.flatten('F'),zpos ,dx=np.ones_like(X.flatten('F')),dy=np.ones_like(X.flatten('F')),dz=kernel[:,:,j].flatten(),alpha=alphabar,color=colorList[j],linewidth=0)           
            #plt.suptitle(titre)
            #plt.savefig(pp, format='pdf')
            #plt.close()
           
    pp.close()
    plt.clf()
    
def modify_FirstLayersWeight(filename='zero_net.mat'):
    VGG19_mat='normalizedvgg.mat'
    vgg_rawnet = scipy.io.loadmat(VGG19_mat)
    vgg_layers = vgg_rawnet['net'][0]['layers'][0][0]
    index_in_vgg = 0
    kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
    bias = vgg_layers[index_in_vgg][0][0][2][0][1]
    number_of_kernels = kernels.shape[3]
    itera = 0
    for i in range(number_of_kernels): 
        kernel =  kernels[:,:,:,i]
        maxB = np.max(kernel[:,:,0])
        maxG = np.max(kernel[:,:,1])
        maxR = np.max(kernel[:,:,2])
        minB = np.min(kernel[:,:,0])
        minG = np.min(kernel[:,:,1])
        minR = np.min(kernel[:,:,2])
        if ((maxB < minG) and (maxB < minR))  or   ((maxG < minG) and (maxG < minR)) or ((maxR < minB) and (maxR < minG)) or ((minB > maxG) and (minB > maxR)) or ((minG > maxB) and (minG > maxR)) or ((minR > maxB) and (minR > maxG)):
            itera += 1
            kernel_zeros = np.zeros_like(kernel)
            for j in range(3):
                kernel_zeros[:,:,j] = np.mean(kernel[:,:,j])*np.ones_like(kernel[:,:,j])
            bias_zeros = bias[i]
            #bias_zeros = np.zeros_like(bias[i])
            vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg][0][0][2][0][0][:,:,:,i] = kernel_zeros
            vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg][0][0][2][0][1][i] = bias_zeros
    print("Number of kernels modifed :",itera)
    scipy.io.savemat(filename,vgg_rawnet)
    
def modify_FirstLayersWeight2(filename='zero_net.mat'):
    VGG19_mat='normalizedvgg.mat'
    vgg_rawnet = scipy.io.loadmat(VGG19_mat)
    vgg_layers = vgg_rawnet['net'][0]['layers'][0][0]
    index_in_vgg = 0
    kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
    bias = vgg_layers[index_in_vgg][0][0][2][0][1]
    number_of_kernels = kernels.shape[3]
    itera = 0
    for i in range(number_of_kernels): 
        kernel =  kernels[:,:,:,i]
        maxB = np.max(kernel[:,:,0])
        maxG = np.max(kernel[:,:,1])
        maxR = np.max(kernel[:,:,2])
        minB = np.min(kernel[:,:,0])
        minG = np.min(kernel[:,:,1])
        minR = np.min(kernel[:,:,2])
        if ((maxB < minG) and (maxB < minR))  or   ((maxG < minG) and (maxG < minR)) or ((maxR < minB) and (maxR < minG)) or ((minB > maxG) and (minB > maxR)) or ((minG > maxB) and (minG > maxR)) or ((minR > maxB) and (minR > maxG)):
            itera += 1
            kernel_zeros = np.zeros_like(kernel)
            # Try to give the kernel a better shape :
            
            for j in range(3):
                corner_mean  = (1./4.)*(kernel[0,0,j]+kernel[0,2,j]+kernel[2,0,j]+kernel[2,2,j])
                kernel_zeros[0,0,j] = corner_mean
                kernel_zeros[0,2,j] = corner_mean
                kernel_zeros[2,0,j] = corner_mean
                kernel_zeros[2,2,j] = corner_mean
                vert_mean = (1./2.)*(kernel[0,1,j]+kernel[2,1,j])
                hor_mean = (1./2.)*(kernel[1,0,j]+kernel[1,2,j])
                kernel_zeros[0,1,j] = vert_mean 
                kernel_zeros[2,1,j] = vert_mean 
                kernel_zeros[1,0,j] = hor_mean 
                kernel_zeros[1,2,j] = hor_mean 
                kernel_zeros[1,1,j] = kernel[1,1,j]  
            bias_zeros = bias[i]
            #bias_zeros = np.zeros_like(bias[i])
            vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg][0][0][2][0][0][:,:,:,i] = kernel_zeros
            vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg][0][0][2][0][1][i] = bias_zeros
    print("Number of kernels modifed :",itera)
    scipy.io.savemat(filename,vgg_rawnet)

def compute_pvalue_regroup(kernelci,kernelcj,ci,cj,identical_canal,alpha,verbose=False):
    """
    On calcul la pvalue entre deux filtres
    """
    
    n = 9
    if n==9 and alpha==0.05:
        critical_value = 6
    else:
        raise(NotImplementedError)
    
    # On calcule la differences entre les canaux 2 a 2 
#    diff_ci_cj = np.abs(kernelci - kernelcj)
#    diff_ci_cj_ravel = np.ravel(diff_ci_cj)
#    # Here we do the certainly wrong assumption that the distribution is
#    # normal
#    # Test if mean of random sample is equal to true mean =0.,
#    # and different mean. We reject the null hypothesis in the second case 
#    # and don’t reject it in the first case.
##    print(ci,cj,diff_ci_cj_ravel)
#    t,p = stats.ttest_1samp(diff_ci_cj_ravel,popmean=0.)
    
    #Wilcoxon test 
    # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.
    diff_ci_cj = np.ravel(kernelci - kernelcj)
    zeros = np.zeros_like(diff_ci_cj)
    t,p =  stats.wilcoxon(diff_ci_cj,zeros)
    if verbose: print(ci,cj,t,p)
    # my obtained value is smaller than the critical value, and so I would conclude
    # that the difference between the two conditions in my study was unlikely to
    # occur by chance
    if t >= critical_value:
       if len(identical_canal)==0:
           identical_canal += [[ci,cj]]
       else:
           # On regroupe les elements ensembles
           new_identical_canal = []
           for tab in identical_canal:
               if ci in tab or cj in tab:
                   stack = tab +[ci,cj]
                   new_tab = list(np.unique(stack))
                   new_identical_canal += [new_tab]
               else:
                  new_identical_canal += [tab] 
           identical_canal =  new_identical_canal
    return(identical_canal)    
    
def modify_kernel(kernel,verbose=False):
    s1,s2, c = kernel.shape # c number of channel
    if verbose: print('size',s1,s2, c )
    n = s1*s2
    alpha =0.05
    kernel_new = kernel.copy()
#    bias_new = bias.copy()
    list_canal_modified = []
    
    
    # 1 : il faut detecter si les kernels se recoupent beaucoup ou pas
    # 2 : Si les kernels ont des valeurs très proches : on les fixes à la moyenne de leur valeurs
    # S'il ne sont pas proches et qu'ils ont des formes similaires on les fixes avec une dynamiques identiques
    # S'ils sont symétriques par rapport au contre on les fixes à une dynamiques identiques mais symétriques
    
    list_canal_modified = []
    identical_canal = []
    for ci in range(c):
       for cj in range(ci+1,c):
#           print(ci,cj,identical_canal)
           kernelci=kernel[:,:,ci]
           kernelcj= kernel[:,:,cj]
           new_identical_canal = compute_pvalue_regroup(kernelci,kernelcj,ci,cj,identical_canal,alpha,verbose=verbose)
           identical_canal =  new_identical_canal
    if verbose:  print('Fin comparaison simple :',identical_canal)
    # On remplace les canaux differents par leur moyenne
    for tab in identical_canal:   
        mean_kernel = np.mean(kernel[:,:,tab],axis=-1)
        for index in tab:
            kernel_new[:,:,index] = mean_kernel
#        bias_new[tab] = np.mean(bias[tab])
        list_canal_modified += tab
#    print('kernel remplace')
#    print('list_canal_modified',list_canal_modified)
    # Savoir s'ils sont proches en forme
    # Pour ce faire on soustrait à chaque canal sa moyenne 
    kernel_minus_mean = kernel - np.mean(kernel,axis=(0,1))
    list_tranformation = ['identity','flipud','fliplr','flipudlr','transpose','transposelr','transposeud']
    list_tranformation = ['identity']
    for transf in list_tranformation:
        identical_canal = []
        for ci in range(c):
           if ci in list_canal_modified:
               continue
           for cj in range(ci+1,c):
               if cj in list_canal_modified:
                   continue

               kernelci=kernel_minus_mean[:,:,ci]
               if transf=='identity':
                   kernelcj= kernel_minus_mean[:,:,cj]
               elif transf=='flipud':  #Flip array in the up/down direction
                   kernelcj= np.flip(kernel_minus_mean[:,:,cj],axis=0)
               elif transf=='fliplr':  #Flip array in the left/right direction
                   kernelcj= np.flip(kernel_minus_mean[:,:,cj],axis=1)
               elif transf=='flipudlr': # Flip array in the up/down and left/right direction
                   kernelcj= np.flip(np.flip(kernel_minus_mean[:,:,cj],axis=0),axis=1)
               elif transf=='transpose':
                   kernelcj= np.transpose(kernel_minus_mean[:,:,cj])
               elif transf=='transposelr': # Transpose then flip lr = Rotation 45°
                   kernelcj= np.flip(np.transpose(kernel_minus_mean[:,:,cj]),axis=1)
               elif transf=='transposeud': # Transpose then flip lr = Rotation -45°
                   kernelcj= np.flip(np.transpose(kernel_minus_mean[:,:,cj]),axis=0)
               new_identical_canal = compute_pvalue_regroup(kernelci,kernelcj,ci,cj,identical_canal,alpha,verbose=verbose)
               identical_canal =  new_identical_canal
                    
        if verbose: print('transf',transf,'identical_canal :',identical_canal)    
        # On remplace les canaux differents par la forme + leur moyenne
        for tab in identical_canal:
            # TODO : faire quelque chose pour gérer les autres transformations
            if transf=='identity':
                ref_kernel = np.mean(kernel_minus_mean[:,:,tab],axis=-1)
            for index in tab:
                kernel_new[:,:,index] = np.mean(kernel,axis=(0,1))[index] + ref_kernel
            list_canal_modified += tab      
        if verbose: print(list_canal_modified)    
    if verbose: print('kernel_new.shape',kernel_new.shape)
    return(kernel_new)
#    
#    stds = np.std(kernal,axis=-1)
#    std_stds = np.std(stds)
#    mean_stds = np.mean(stds)
#    for i in range(s1):
#        for j in range(s2):
#            if np.abs(stds[i,j] - mean_stds) < std_stds:
#                kernel_new[i,j,:] = np.mean(kernel[i,j,:] )
#                
#     
#    median = np.median(data,axis=[0,1])
#    upper_quartile = np.percentile(data, 75,axis=[0,1])
#    lower_quartile = np.percentile(data, 25,axis=[0,1])
#    
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4')

def modify_AllLayersWeight(filename='zero_net.mat',verbose=False):
    VGG19_mat='normalizedvgg.mat'
    vgg_rawnet = scipy.io.loadmat(VGG19_mat)
    vgg_layers = vgg_rawnet['net'][0]['layers'][0][0].copy()
    VGG19_LAYERS_modif =  [VGG19_LAYERS[0]]
    VGG19_LAYERS_modif =  VGG19_LAYERS
    index_in_vgg = 0
    for i, name in enumerate(VGG19_LAYERS_modif):
        kind = name[:4]
        if(kind == 'conv'):
            if verbose: print(name,index_in_vgg,i)
            kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
            bias = vgg_layers[index_in_vgg][0][0][2][0][1]
            number_of_kernels = kernels.shape[3]
            for k in range(number_of_kernels): 
                if verbose: print('kernel number',k)
                kernel =  kernels[:,:,:,k]
                kernel_zeros = modify_kernel(kernel,verbose=verbose)
                bias_zeros = bias[k]
                #bias_zeros = np.zeros_like(bias[i])
#                print(vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg])
#                print(vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg][i])
#                print(vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg][i][0][2][0][0])
                vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg][0][0][2][0][0][:,:,:,k] = kernel_zeros
                vgg_rawnet['net'][0]['layers'][0][0][index_in_vgg][0][0][2][0][1][k] = bias_zeros
        index_in_vgg += 1
 
    scipy.io.savemat(filename,vgg_rawnet)
    
if __name__ == '__main__':
    #plot_and_saveinPDF()
    modify_FirstLayersWeight2()
    
