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
	
if __name__ == '__main__':
	#plot_and_saveinPDF()
	modify_FirstLayersWeight2()
	
