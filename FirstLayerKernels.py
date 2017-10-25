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
		
		if(i<1) or True:
		
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
			plt.suptitle(titre)
			plt.savefig(pp, format='pdf')
			plt.close()
			
			fig = plt.figure()
			kernelsChange = (kernel.transpose(2,0,1).reshape(3,9)).transpose(1,0)
			#print(kernel)
			#print(kernelsChange)
			box = plt.boxplot(kernelsChange, patch_artist=True)
			for patch, color in zip(box['boxes'], colorList):
				patch.set_facecolor(color)
			plt.savefig(pp, format='pdf')
			plt.close()
			
			#
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
	
if __name__ == '__main__':
	plot_and_saveinPDF()
	
