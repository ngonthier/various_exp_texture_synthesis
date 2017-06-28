#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 02 2017

The goal of this script is to vizualised the reponse of the filter of the
different convolution of the network

@author: nicolas
"""

import scipy
import numpy as np
import tensorflow as tf
import Style_Transfer as st
from Arg_Parser import get_parser_args 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
from tensorflow.python.framework import dtypes
import matplotlib.gridspec as gridspec
import math
from skimage import exposure
from PIL import Image

# Name of the 19 first layers of the VGG19
VGG19_LAYERS = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	'relu5_3', 'conv5_4', 'relu5_4'
)

VGG19_LAYERS_INDICES = {'conv1_1' : 0,'conv1_2' : 2,'conv2_1' : 5,'conv2_2' : 7,
	'conv3_1' : 10,'conv3_2' : 12,'conv3_3' : 14,'conv3_4' : 16,'conv4_1' : 19,
	'conv4_2' : 21,'conv4_3' : 23,'conv4_4' : 25,'conv5_1' : 28,'conv5_2' : 30,
	'conv5_3' : 32,'conv5_4' : 34}

VGG19_LAYERS_INTEREST = (
    'conv1_1','conv2_1', 'conv3_1'
)

style_layers_size =  {'input':3,'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}

VGG19_LAYERS_INTEREST = ('conv1_1','pool1','pool2','pool3','pool4')
#VGG19_LAYERS_INTEREST = {'conv1_1'}

def plot_and_save_pdf(Matrix,path,name=''):
	pltname = path+name+'_hist.pdf'
	pltname_rep = path+name+'_img.pdf'
	pp = PdfPages(pltname)

	Matrix = Matrix[0] # Remove first dim
	h,w,channels = Matrix.shape
	df_Matrix = pd.DataFrame(np.reshape(Matrix,(h*w,channels)))
	len_columns = len(df_Matrix.columns)
	for i in range(len_columns):
		df_Matrix.hist(column = i, bins = 128)
		plt.savefig(pp, format='pdf')
		plt.close()
	pp.close()

	plt.clf()
	# Result of the convolution 
	pp_img = PdfPages(pltname_rep)
	for i in range(len_columns):
		plt.imshow(Matrix[:,:,i], cmap='gray')
		plt.savefig(pp_img, format='pdf')
		plt.close()
	pp_img.close()



	

def getGradImg():
	img_folder = 'NoiseOrigin/'
	img_name = 'Pastiche_Gaussian'
	img_ext = '.png'
	denoised_img_name = 'denoised_Gaussian'
	image_path = img_folder + img_name +img_ext
	img = scipy.misc.imread(image_path) 
	image_path = img_folder + denoised_img_name +img_ext
	denoised_img = scipy.misc.imread(image_path)
	grad_img = img - denoised_img
	output_image_path = img_folder + 'grad_Gaussian' +img_ext
	scipy.misc.toimage(grad_img).save(output_image_path)
	print(np.var(img),np.var(denoised_img))
	
def do_pdf_comparison_forNoise():
	directory_path = 'NoiseOrigin/' 
	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	
	# Plot for the original image ! 
	# TODO : plot the image response of the filter the image denoised
	# Compute the 4 first moments of the filters 
	# Compute total variation 
	# Plot the Gram Matrix value ! 

	img_folder = 'NoiseOrigin/'
	ref_img_name = 'BrickSmallBrown0293_1_S'
	img_name = 'Pastiche_Gaussian'
	img_ext = '.png'
	denoised_img_name = 'denoised_Gaussian'
	image_path = img_folder + img_name +img_ext
	gen_img = scipy.misc.imread(image_path).astype('float32')
	image_path = img_folder + denoised_img_name +img_ext
	denoised_img = scipy.misc.imread(image_path).astype('float32')
	grad_img = gen_img - denoised_img
	
	gen_img = st.preprocess(gen_img)
	denoised_img = st.preprocess(denoised_img)
	grad_img = st.preprocess(grad_img)
	
	image_path = img_folder + ref_img_name +img_ext
	ref_img = st.preprocess(scipy.misc.imread(image_path).astype('float32'))
	
	list_imgs = [ref_img,gen_img,denoised_img,grad_img]
	
	sns.set_style("white")
	
	_,image_h_art, image_w_art, _ = ref_img.shape 
	M_dict = st.get_M_dict(image_h_art,image_w_art)
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, ref_img) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=ref_img.shape)
	assign_op = net['input'].assign(placeholder)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	cmImg =  'jet'
	list_name = ['Ref','Gen','Denoised','Grad']

	ext = '_Gaussian'
	for layer in VGG19_LAYERS_INTEREST:
		print(layer)
		list_a = []
		N = style_layers_size[layer[:5]]
		M = M_dict[layer[:5]]
		list_G = []
		for img in list_imgs:
			sess.run(assign_op, {placeholder: img})
			a = net[layer]
			G = st.gram_matrix(a,N,M)
			G_eval = G.eval(session=sess)
			a_eval = a.eval(session=sess)
			a_eval = a_eval[0]
			list_a += [a_eval]
			list_G += [G_eval]
		pltname = directory_path+layer+'_'+ref_img_name+'_comp_denoised'+ext+'.pdf'
		pp = PdfPages(pltname)
		Matrix = list_a[0] # Remove first dim
		h,w,channels = Matrix.shape
			
		max_a_all = np.max(list_a)
		min_a_all = np.min(list_a)
		for i in range(channels): # Features i 
			#print("features",i)
			f, axarr = plt.subplots(2, 2)
			for j in range(4):
				a = list_a[j][:,:,i]
				axarr[j%2,j//2].matshow(a, cmap=cmImg)
				#axarr[j%2,j//2].matshow(a, cmap=cmImg,vmax=max_a_all,vmin=min_a_all)
				#titre = list_name[j] + ' Gram Elt = {:.2e}'.format(list_G[j][i,i])
				titre = list_name[j] 
				axarr[j%2,j//2].set_title(titre)
				axarr[j%2,j//2].axis('off')	
			titre = 'Kernel {}'.format(i)
			plt.suptitle(titre)
			plt.savefig(pp, format='pdf')
			plt.close()
		
		print("Start Gram Representation")
		# Compute Gram Matrix :
		f, axarr = plt.subplots(2, 2)
		max_G = np.max(list_G)
		min_G = np.min(list_G)
		for j in range(4):
			axarr[j%2,j//2].matshow(list_G[j], cmap=cmImg)
			#axarr[j%2,j//2].matshow(list_G[j], cmap=cmImg,vmax=max_G,vmin=min_G)
			diff_Gram = list_G[0] - list_G[j]
			diff_Gram_mean = np.mean(diff_Gram)
			diff_Gram_max = np.max(diff_Gram)
			diff_Gram_std = np.std(diff_Gram)
			diff_Gram_energie = sum(sum(np.power(diff_Gram,2)))
			titre1 = list_name[j] + ' Gram Diff E = {:.2e}, Max = {:.2e}, Mean = {:.2e}, Std = {:.2e}'.format(diff_Gram_energie,diff_Gram_max,diff_Gram_mean,diff_Gram_std)
			print(titre1)
			titre = list_name[j] + ' Gram Diff E = {:.2e},'.format(diff_Gram_energie)
			axarr[j%2,j//2].set_title(titre)
			axarr[j%2,j//2].axis('off')
		plt.suptitle(layer)
		plt.savefig(pp, format='pdf')
		plt.close()
	
		pp.close()
		plt.clf()
			
			

def plot_compare_pdf(vgg_layers,Matrix,path,name):
	number_img_large_tab = {'conv1_1' : 1,'conv1_2' : 4,'conv2_1' : 4,'conv2_2' : 8,
	'conv3_1' : 8,'conv3_2' : 8,'conv3_3' : 16,'conv3_4' : 16,'conv4_1' : 16,
	'conv4_2' : 16,'conv4_3' : 16,'conv4_4' : 16,'conv5_1' : 16,'conv5_2' : 16,
	'conv5_3' : 16,'conv5_4' : 16}
	pltname = path+name+'_comp.pdf'
	pp = PdfPages(pltname)
	Matrix = Matrix[0] # Remove first dim
	h,w,channels = Matrix.shape
	Matrix_reshaped = np.reshape(Matrix,(h*w,channels))
	df_Matrix = pd.DataFrame(Matrix_reshaped)
	len_columns = len(df_Matrix.columns)
	index_in_vgg = VGG19_LAYERS_INDICES[name]
	kernels = vgg_layers[index_in_vgg][0][0][2][0][0]
	#  A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
	print(kernels.shape)
	bias = vgg_layers[index_in_vgg][0][0][2][0][1]
	print(bias.shape)
	#len_columns = 1
	input_kernel = kernels.shape[2]
	alpha=0.75
	#cmkernel = 'gray'
	cmImg =  'jet'
	cmkernel = plt.get_cmap('hot')
	for i in range(len_columns):
		#print("Features",i)
		# For each feature
		f = plt.figure()
		gs0 = gridspec.GridSpec(1,3, width_ratios=[0.05,4,4]) # 2 columns
		axcm = plt.subplot(gs0[0])
		number_img_large = number_img_large_tab[name]
		if(not(name=='conv1_1')):
			gs00 = gridspec.GridSpecFromSubplotSpec(input_kernel//number_img_large, number_img_large, subplot_spec=gs0[1])
			axes = []
			for j in range(input_kernel):
				ax = plt.subplot(gs00[j])
				axes += [ax]
			kernel = kernels[:,:,:,i]
			mean_kernel = np.mean(kernel)
			bias_i = bias[i,0]
			j = 0
			vmin = np.min(kernel)
			vmax = np.max(kernel)
			for ax in axes:
				im = ax.matshow(kernel[:,:,j],cmap=cmkernel,alpha=alpha,vmin=vmin, vmax=vmax)
				ax.axis('off')
				j += 1
		else:
			gs00 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[1])
			axes = []
			for j in range(input_kernel):
				ax = plt.subplot(gs00[j])
				axes += [ax]
			kernel = kernels[:,:,:,i]
			mean_kernel = np.mean(kernel)
			bias_i = bias[i,0]
			j = 0
			vmin = np.min(kernel)
			vmax = np.max(kernel)
			for ax in axes:
				im = ax.matshow(kernel[:,:,j],cmap=cmkernel,alpha=alpha,vmin=vmin, vmax=vmax)
				ax.axis('off')
				j += 1
			ax0 = plt.subplot(gs00[3])
			# bgr to rgb
			img = kernel[...,::-1]
			#img = Image.fromarray(img, 'RGB')
			#img = exposure.rescale_intensity(img, in_range='uint8')
			img -= np.min(img) 
			img /= (np.max(img)/255.)
			img = np.floor(img).astype('uint8')
			ax0.imshow(img)
			ax0.axis('off')
			ax0.set_title('Color Kernel')
		plt.colorbar(im, cax=axcm)
		#plt.colorbar(im, cax=axes[-1])
		#f.subplots_adjust(right=0.8)
		#cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
		#f.colorbar(im, cax=cbar_ax)
		#plt.colorbar(im, cax=axes)
		gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[2],height_ratios=[4,3])
		ax1 = plt.subplot(gs01[1])
		ax2 = plt.subplot(gs01[0])
		samples = Matrix_reshaped[:,i]
		beta, loc, scale = stats.gennorm.fit(samples)
		D,pvalue = stats.kstest(samples, 'gennorm',(beta, loc, scale )) # 1-alph = 0.9 
		#D,pvalue = stats.stats.ks_2samp(samples,stats.gennorm.rvs(beta,loc=loc,scale=scale,size=len(samples) ))
		Dcritial = 1.224/math.sqrt(len(samples))
		#print("pvalue",pvalue)
		df_Matrix.hist(column = i, bins = 128,ax=ax1,normed=True, histtype='stepfilled', alpha=1) 
		x = np.linspace(stats.gennorm.ppf(0.005, beta, loc, scale),
                stats.gennorm.ppf(0.995, beta, loc, scale), 128)
		ax1.plot(x, stats.gennorm.pdf(x, beta, loc, scale ),'r-',alpha=0.4, label='gennorm pdf')
		#ax1.legend(loc='best', frameon=False)
		#textstr = '$\mu=%.2f$\n$\mathrm{scale}=%.2f$\n$beta=%.2f$ \n $\mathrm{D}=%.4f$ \n $\mathrm{Dcri}=%.4f$ '%(loc,scale,beta,D,Dcritial)
		textstr = '$\mu=%.2f$\n$\mathrm{scale}=%.2f$\n$beta=%.2f$\n$\mathrm{pvalue}=%.4f$'%(loc,scale,beta,pvalue)
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=6,
			verticalalignment='top', bbox=props)
		ax2.matshow(Matrix[:,:,i], cmap=cmImg)
		ax2.axis('off')
		titre = 'Kernel {} with mean = {:.2e} in range [{:.2e},{:.2e}] and bias = {:.2e}'.format(i,mean_kernel,vmin,vmax,bias_i)
		plt.suptitle(titre)
		#gs0.tight_layout(f)
		plt.savefig(pp, format='pdf')
		plt.close()
	pp.close()
	plt.clf()
	

def main_plot_commun(name=None):
	"""
	Plot for each layer in VGG Interest the kernels, the response of the 
	kernel but also the histogram fitted
	"""
	parser = get_parser_args()
	if(name==None):
		style_img_name = "StarryNight"
	else:
		style_img_name = name
	parser.set_defaults(style_img_name=style_img_name)
	args = parser.parse_args()
	do_pdf_comparison(args)

if __name__ == '__main__':
	#main_plot_commun('grad_Uniform')
	do_pdf_comparison_forNoise()
