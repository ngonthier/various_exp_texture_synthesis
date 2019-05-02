#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 2017

The goal of this script is to vizualised the Gram Matrix difference between
the differents image

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
from skimage.color import gray2rgb

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
VGG19_LAYERS_INTEREST = (
    'conv1_1','pool1', 'pool2','pool3','pool4'
)

VGG19_LAYERS_INTEREST =(
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
	'conv2_1'
)

def do_pdf_comparison(args):
	sns.set_style("white")
	folder_path ='/home/nicolas/Style-Transfer/Results/ArtifactsComp/'
	pltname = folder_path+'Artifacts.pdf'
	pp = PdfPages(pltname)
	
	image_style_path = folder_path + 'BrickSmallBrown0293_1_S.png' 
	image_reference = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	_,image_h_art, image_w_art, _ = image_reference.shape 
	list_img = [image_reference]
	list_img_name = ['image_reference']
	
	image_png_path = folder_path + 'Pastiche_BRICK_PNG.png' 
	image_png = st.preprocess(scipy.misc.imread(image_png_path).astype('float32')) 
	list_img += [image_png]
	list_img_name += ['image_png']
	image_jpg_path = folder_path + 'Pastiche_BRICK_JPG.jpg' 
	image_jpg_path = folder_path + 'Pastiche_BRICK_Flou.png' 
	#	image_jpg_path = folder_path + 'Pastiche_BRICK_JPG_Apres2000IterDePlus.png' 
	image_jpg = st.preprocess(scipy.misc.imread(image_jpg_path).astype('float32')) 
	list_img += [image_jpg]
	list_img_name += ['image_Modified']
	
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_reference) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=image_reference.shape)
	assign_op = net['input'].assign(placeholder)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	f, ax = plt.subplots(1,3)
	for j,name in enumerate(list_img_name):
		ax[j].set_title(name)
		img = st.postprocess(list_img[j])
		ax[j].imshow(img)
		ax[j].axis('off')
	titre = 'Images Depart'
	plt.suptitle(titre)
	plt.savefig(pp, format='pdf',dpi=600)
	plt.close()

	# Plot the Gram Matrix
	layer = 'conv1_1'
	f, ax = plt.subplots(2,3)
	list_of_Gram = []
	list_reponse = []
	for i in range(3):
		sess.run(assign_op, {placeholder: list_img[i]})
		a = net[layer].eval(session=sess)
		_,h,w,N = a.shape
		M = h*w
		list_reponse += [a[0]]
		G_a = sess.run(st.gram_matrix(a,N,M))
		list_of_Gram += [G_a]
		
	vmax= np.max(list_of_Gram)
	vmin= np.min(list_of_Gram)
	vmax2= np.max((vmax**2,vmin**2))
	vmin2= np.min((vmax**2,vmin**2,0))
	for i in range(3):
		G_a = list_of_Gram[i]
		G_a = np.triu(G_a)
		ax[0,i].set_title(list_img_name[i])
		ax[0,i].imshow(G_a,vmin=vmin, vmax=vmax)
		ax[0,i].axis('off')
		G_a_Square = np.power(G_a,2)
		ax[1,i].imshow(G_a_Square,vmin=vmin2, vmax=vmax2)
		ax[1,i].axis('off')	
	titre = 'Comparison of the Gram Matrix and the squared ones'
	plt.suptitle(titre)
	plt.savefig(pp, format='pdf')
	plt.close()
	
	# Plot the difference between the Gram Matrix
	f, ax = plt.subplots(2,3)
	list_title = ['Ref Minus png','Ref Minus jpg','jpg Minus png']
	list_diff = []
	for i in range(3):	
		if(i==0):
			diff = list_of_Gram[0] - list_of_Gram[1]
		elif(i==1):
			diff = list_of_Gram[0] - list_of_Gram[2]
		else:
			diff = list_of_Gram[2] - list_of_Gram[1]
		list_diff += [diff] # = np.triu(diff)
	
	vmax= np.max(list_diff)
	vmin= np.min(list_diff)
	vmax2= np.max((vmax**2,vmin**2))
	vmin2= np.min((vmax**2,vmin**2,0))
	for i in range(3):	
		diff = list_diff[i]
		diff = np.triu(diff) # triangular upper
		titre_sub1 = list_title[i]
		ax[0,i].set_title(titre_sub1)
		ax[0,i].imshow(diff,vmin=vmin, vmax=vmax)
		ax[0,i].axis('off')
		diff = np.power(diff,2)
		title_sub2 = 'max diff : ' + str(np.max(diff)) + ' min :' + str(np.min(diff))
		ax[1,i].set_title(title_sub2)
		ax[1,i].imshow(diff,vmin=vmin2, vmax=vmax2)
		ax[1,i].axis('off')	
	titre = 'Difference of the Gram Matrixes and the squared difference'
	plt.suptitle(titre)
	plt.savefig(pp, format='pdf')
	plt.close()
	
	# Plot the ratio
	
	diff_ref_x = np.power(list_diff[0],2)
	diff_png_jpg = np.power(list_diff[2],2)
	ratio = diff_png_jpg / diff_ref_x
	im = plt.imshow(ratio,cmap= 'jet')
	plt.colorbar(im)
	plt.suptitle("Ratio Gram Matrix Optim Minus Flou On Ref Minus Optim")
	plt.savefig(pp, format='pdf')
	plt.close()
	print(np.mean(ratio),np.median(ratio),np.min(ratio),np.max(ratio))
	mask = np.where(ratio < np.median(ratio),1,0)
	print("np.sum(mask)",np.sum(mask))
	np.save("mask",mask)
	im = plt.imshow(mask,cmap= 'jet')
	plt.colorbar(im)
	plt.suptitle("Masque")
	plt.savefig(pp, format='pdf')
	plt.close()
	
	
	
	diff_ref_x = np.power(list_diff[0],2)
	diff_ref_jpg = np.power(list_diff[1],2)
	ratio = diff_ref_jpg / diff_ref_x
	im = plt.imshow(ratio,cmap= 'jet')
	plt.colorbar(im)
	plt.suptitle("Ratio Gram Matrix  Ref Minus Flou On Ref Minus Optim")
	plt.savefig(pp, format='pdf')
	plt.close()
	
	diff_ref_jpg =  np.power(list_diff[1],2)
	diagonal = diff_ref_jpg.diagonal()
	diagonal_mean = np.mean(diagonal)
	diagonal_std = np.std(diagonal)
	kernel_big_energy_index = np.where(diagonal > diagonal_mean + diagonal_std)
	kernel_big_energy_index = kernel_big_energy_index[0]
	
	print('kernel_big_energy_index',kernel_big_energy_index)
	# Plot the response of each filter
	a_ref = list_reponse[0]
	a_png = list_reponse[1]
	a_jpg = list_reponse[2]
			
	for j in range(N):
		f, ax = plt.subplots(1,3)
		a_ref_j = a_ref[:,:,j]
		a_png_j = a_png[:,:,j]
		a_jpg_j = a_jpg[:,:,j]
		vmin = np.min((a_ref_j,a_png_j,a_jpg_j))
		vmax = np.max((a_ref_j,a_png_j,a_jpg_j))
		
		for i in range(3):
			if(i==0): a = a_ref_j
			if(i==1): a = a_png_j
			if(i==2): a = a_jpg_j
			ax[i].set_title(list_img_name[i])
			ax[i].imshow(a,vmin=vmin, vmax=vmax)
			ax[i].axis('off')
		titre = 'Kernel ' +str(j)
		if(j in kernel_big_energy_index):
			titre += ' Response different for JPG'
		plt.suptitle(titre)
		plt.savefig(pp, format='pdf')
		plt.close()
	
	# Close the PDF
	pp.close()
	plt.clf()

def do_pdf_comparison_GramOnly(args):
	VGG19_LAYERS_INTEREST = (
    'conv1_1','pool1', 'pool2','pool3','pool4'
	)
	sns.set_style("white")
	folder_path ='/home/nicolas/Style-Transfer/Results/ArtifactsComp/'
	pltname = folder_path+'Artifacts_Gram.pdf'
	pp = PdfPages(pltname)
	
	image_style_path = folder_path + 'BrickSmallBrown0293_1_S.png' 
	image_reference = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	_,image_h_art, image_w_art, _ = image_reference.shape 
	list_img = [image_reference]
	list_img_name = ['image_reference']
	
	image_png_path = folder_path + 'Pastiche_BRICK_PNG.png' 
	image_png = st.preprocess(scipy.misc.imread(image_png_path).astype('float32')) 
	list_img += [image_png]
	list_img_name += ['image_png']
	image_jpg_path = folder_path + 'Pastiche_BRICK_JPG.jpg' 
	image_jpg_path = folder_path + 'Pastiche_BRICK_Flou.png' 
	#	image_jpg_path = folder_path + 'Pastiche_BRICK_JPG_Apres2000IterDePlus.png' 
	image_jpg = st.preprocess(scipy.misc.imread(image_jpg_path).astype('float32')) 
	list_img += [image_jpg]
	list_img_name += ['image_Modified']
	
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_reference) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=image_reference.shape)
	assign_op = net['input'].assign(placeholder)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	f, ax = plt.subplots(1,3)
	for j,name in enumerate(list_img_name):
		ax[j].set_title(name)
		img = st.postprocess(list_img[j])
		ax[j].imshow(img)
		ax[j].axis('off')
	titre = 'Images Depart'
	plt.suptitle(titre)
	plt.savefig(pp, format='pdf',dpi=600)
	plt.close()
	mask_dict = {}
	# Plot the Gram Matrix
	for layer in VGG19_LAYERS_INTEREST:
		print(layer)
		f, ax = plt.subplots(2,3)
		list_of_Gram = []
		list_reponse = []
		for i in range(3):
			sess.run(assign_op, {placeholder: list_img[i]})
			a = net[layer].eval(session=sess)
			_,h,w,N = a.shape
			M = h*w
			list_reponse += [a[0]]
			G_a = sess.run(st.gram_matrix(a,N,M))
			list_of_Gram += [G_a]
			
		vmax= np.max(list_of_Gram)
		vmin= np.min(list_of_Gram)
		vmax2= np.max((vmax**2,vmin**2))
		vmin2= np.min((vmax**2,vmin**2,0))
		for i in range(3):
			G_a = list_of_Gram[i]
			G_a = np.triu(G_a)
			ax[0,i].set_title(list_img_name[i])
			ax[0,i].imshow(G_a,vmin=vmin, vmax=vmax,cmap= 'jet')
			ax[0,i].axis('off')
			G_a_Square = np.power(G_a,2)
			ax[1,i].imshow(G_a_Square,vmin=vmin2, vmax=vmax2,cmap= 'jet')
			ax[1,i].axis('off')	
		titre = 'Comparison of the Gram Matrix and the squared ones'
		plt.suptitle(titre)
		plt.savefig(pp, format='pdf')
		plt.close()
		
		# Plot the difference between the Gram Matrix
		f, ax = plt.subplots(2,3)
		list_title = ['Ref Minus png','Ref Minus jpg','jpg Minus png']
		list_diff = []
		for i in range(3):	
			if(i==0):
				diff = list_of_Gram[0] - list_of_Gram[1]
			elif(i==1):
				diff = list_of_Gram[0] - list_of_Gram[2]
			else:
				diff = list_of_Gram[2] - list_of_Gram[1]
			list_diff += [diff] # = np.triu(diff)
		
		vmax= np.max(list_diff)
		vmin= np.min(list_diff)
		vmax2= np.max((vmax**2,vmin**2))
		vmin2= np.min((vmax**2,vmin**2,0))
		for i in range(3):	
			diff = list_diff[i]
			diff = np.triu(diff) # triangular upper
			titre_sub1 = list_title[i]
			ax[0,i].set_title(titre_sub1)
			ax[0,i].imshow(diff,vmin=vmin, vmax=vmax,cmap= 'jet')
			ax[0,i].axis('off')
			diff = np.power(diff,2)
			title_sub2 = 'max diff : ' + str(np.max(diff)) + ' min :' + str(np.min(diff))
			ax[1,i].set_title(title_sub2)
			ax[1,i].imshow(diff,vmin=vmin2, vmax=vmax2,cmap= 'jet')
			ax[1,i].axis('off')	
		titre = 'Difference of the Gram Matrixes and the squared difference ' + layer 
		plt.suptitle(titre)
		plt.savefig(pp, format='pdf')
		plt.close()
		
		# Plot the ratio
		
		diff_ref_x = np.power(list_diff[0],2)
		inds_diff_ref_x_0 = np.where(diff_ref_x==0)
		inds_diff_ref_x_not_0 = np.where(diff_ref_x>0)
		diff_ref_x[inds_diff_ref_x_0] = np.min(diff_ref_x[inds_diff_ref_x_not_0])
		diff_png_jpg = np.power(list_diff[2],2)
		ratio = np.divide(diff_png_jpg,diff_ref_x)
		for j in range(len(inds_diff_ref_x_0[0])):
			indice1 = inds_diff_ref_x_0[0][j]
			indice2 = inds_diff_ref_x_0[1][j]
			if not(diff_png_jpg[indice1,indice2]==0):
				ratio[indice1,indice2] = np.max(ratio)
				
		f, ax = plt.subplots(1,2)
		im = ax[0].imshow(ratio,cmap= 'jet')
		plt.colorbar(im)
		ax[0].set_title("Ratio Gram Matrix Optim Minus Flou On Ref Minus Optim")
		
		print(np.mean(ratio),np.median(ratio),np.min(ratio),np.max(ratio))
		mask = np.where(ratio < np.median(ratio),1,0)
		print("np.sum(mask)",np.sum(mask))
		mask_dict[layer] = mask
		im = ax[1].imshow(mask,cmap= 'jet')
		ax[1].set_title("Masque 0 1")
		plt.savefig(pp, format='pdf')
		plt.close()
		
	
	with open('mask_dict.pkl', 'wb') as output_pkl:
		pickle.dump(mask_dict,output_pkl)
	# Close the PDF
	pp.close()
	plt.clf()
	
	
def do_pdf_comparison_GramOnly_autreratio(args):
	VGG19_LAYERS_INTEREST = (
    'conv1_1','pool1', 'pool2','pool3','pool4'
	)
	sns.set_style("white")
	folder_path ='/home/nicolas/Style-Transfer/Results/ArtifactsComp/'
	pltname = folder_path+'Artifacts_Gram_autreratio.pdf'
	pp = PdfPages(pltname)
	
	image_style_path = folder_path + 'BrickSmallBrown0293_1_S.png' 
	image_reference = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	_,image_h_art, image_w_art, _ = image_reference.shape 
	list_img = [image_reference]
	list_img_name = ['image_reference']
	
	image_png_path = folder_path + 'Pastiche_BRICK_PNG.png' 
	image_png = st.preprocess(scipy.misc.imread(image_png_path).astype('float32')) 
	list_img += [image_png]
	list_img_name += ['image_png']
	image_jpg_path = folder_path + 'Pastiche_BRICK_JPG.jpg' 
	image_jpg_path = folder_path + 'Pastiche_BRICK_Flou.png' 
	#	image_jpg_path = folder_path + 'Pastiche_BRICK_JPG_Apres2000IterDePlus.png' 
	image_jpg = st.preprocess(scipy.misc.imread(image_jpg_path).astype('float32')) 
	list_img += [image_jpg]
	list_img_name += ['image_Modified']
	
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_reference) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=image_reference.shape)
	assign_op = net['input'].assign(placeholder)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	f, ax = plt.subplots(1,3)
	for j,name in enumerate(list_img_name):
		ax[j].set_title(name)
		img = st.postprocess(list_img[j])
		ax[j].imshow(img)
		ax[j].axis('off')
	titre = 'Images Depart'
	plt.suptitle(titre)
	plt.savefig(pp, format='pdf',dpi=600)
	plt.close()
	mask_dict = {}
	# Plot the Gram Matrix
	for layer in VGG19_LAYERS_INTEREST:
		print(layer)
		f, ax = plt.subplots(2,3)
		list_of_Gram = []
		list_reponse = []
		for i in range(3):
			sess.run(assign_op, {placeholder: list_img[i]})
			a = net[layer].eval(session=sess)
			_,h,w,N = a.shape
			M = h*w
			list_reponse += [a[0]]
			G_a = sess.run(st.gram_matrix(a,N,M))
			list_of_Gram += [G_a]
			
		vmax= np.max(list_of_Gram)
		vmin= np.min(list_of_Gram)
		vmax2= np.max((vmax**2,vmin**2))
		vmin2= np.min((vmax**2,vmin**2,0))
		for i in range(3):
			G_a = list_of_Gram[i]
			G_a = np.triu(G_a)
			ax[0,i].set_title(list_img_name[i])
			ax[0,i].imshow(G_a,vmin=vmin, vmax=vmax,cmap= 'jet')
			ax[0,i].axis('off')
			G_a_Square = np.power(G_a,2)
			ax[1,i].imshow(G_a_Square,vmin=vmin2, vmax=vmax2,cmap= 'jet')
			ax[1,i].axis('off')	
		titre = 'Comparison of the Gram Matrix and the squared ones'
		plt.suptitle(titre)
		plt.savefig(pp, format='pdf')
		plt.close()
		
		# Plot the difference between the Gram Matrix
		f, ax = plt.subplots(2,3)
		list_title = ['Ref Minus png','Ref Minus jpg','jpg Minus png']
		list_diff = []
		for i in range(3):	
			if(i==0):
				diff = list_of_Gram[0] - list_of_Gram[1]
			elif(i==1):
				diff = list_of_Gram[0] - list_of_Gram[2]
			else:
				diff = list_of_Gram[2] - list_of_Gram[1]
			list_diff += [diff] # = np.triu(diff)
		
		vmax= np.max(list_diff)
		vmin= np.min(list_diff)
		vmax2= np.max((vmax**2,vmin**2))
		vmin2= np.min((vmax**2,vmin**2,0))
		for i in range(3):	
			diff = list_diff[i]
			diff = np.triu(diff) # triangular upper
			titre_sub1 = list_title[i]
			ax[0,i].set_title(titre_sub1)
			ax[0,i].imshow(diff,vmin=vmin, vmax=vmax,cmap= 'jet')
			ax[0,i].axis('off')
			diff = np.power(diff,2)
			title_sub2 = 'max diff : ' + str(np.max(diff)) + ' min :' + str(np.min(diff))
			ax[1,i].set_title(title_sub2)
			ax[1,i].imshow(diff,vmin=vmin2, vmax=vmax2,cmap= 'jet')
			ax[1,i].axis('off')	
		titre = 'Difference of the Gram Matrixes and the squared difference ' + layer 
		plt.suptitle(titre)
		plt.savefig(pp, format='pdf')
		plt.close()
		
		# Plot the ratio
		
		inds_zero = np.where(list_of_Gram[2]==0)
		inds_no_zero = np.where(list_of_Gram[2]>0)
		list_of_Gram[2][inds_zero] = np.min(np.abs(list_of_Gram[2][inds_no_zero]))
		ratio = np.divide(np.power(list_of_Gram[1],2),np.power(list_of_Gram[2],2))
		for j in range(len(inds_zero[0])):
			indice1 = inds_zero[0][j]
			indice2 = inds_zero[1][j]
			if not(list_of_Gram[1][indice1,indice2]==0):
				ratio[indice1,indice2] = np.max(ratio)
				
		f, ax = plt.subplots(1,2)
		im = ax[0].imshow(ratio,cmap= 'jet')
		plt.colorbar(im)
		ax[0].set_title("Ratio Gram Matrix Optim  On Optim Flou")
		
		print(np.mean(ratio),np.median(ratio),np.min(ratio),np.max(ratio))
		mask = np.where(ratio < np.mean(ratio) + np.std(ratio),1,0)
		print("np.sum(mask)",np.sum(mask))
		mask_dict[layer] = mask
		im = ax[1].imshow(mask,cmap= 'jet')
		ax[1].set_title("Masque 0 1")
		plt.savefig(pp, format='pdf')
		plt.close()
		
	
	with open('mask_dict.pkl', 'wb') as output_pkl:
		pickle.dump(mask_dict,output_pkl)
	# Close the PDF
	pp.close()
	plt.clf()
	
def pattern_comp_pdf(args):
	style_layers_size =  {'input':3,'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}
	path_origin = '/home/nicolas/Style-Transfer/Results/Patterns/'
	path_origin_img = path_origin + 'img/'
	dirs = os.listdir(path_origin_img)
	dirs = sorted(dirs, key=str.lower)
	list_img = {}
	print(dirs)
	for name in dirs:
		name_get = path_origin_img  + name
		img =scipy.misc.imread(name_get).astype('float32')
		img = gray2rgb(img)
		image = st.preprocess(img) 
		list_img[name] = image
		
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=image.shape)
	assign_op = net['input'].assign(placeholder)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	Data = {}
	for layer in VGG19_LAYERS_INTEREST:
		Data[layer] = {}
		for name in dirs:
			sess.run(assign_op, {placeholder: list_img[name]})
			a = net[layer].eval(session=sess)
			Data[layer][name] = a[0]
			
	for layer in VGG19_LAYERS_INTEREST:
		print("Print PDF",layer)
		pltname = path_origin+ layer+'_Patterns.pdf'
		pp = PdfPages(pltname)
		f, ax = plt.subplots(2,3)
		for j,name in enumerate(dirs):
			ax[j%2,j//2].set_title(name)
			img = st.postprocess(list_img[name])
			ax[j%2,j//2].imshow(img)
			ax[j%2,j//2].axis('off')
		titre = 'Images Depart'
		plt.suptitle(titre)
		plt.savefig(pp, format='pdf',dpi=600)
		plt.close()
	
		for i in range(style_layers_size[layer[:5]]):
			f, ax = plt.subplots(2,3)
			for j,name in enumerate(dirs):
				ax[j%2,j//2].set_title(name)
				ax[j%2,j//2].imshow(Data[layer][name][:,:,i])
				ax[j%2,j//2].axis('off')
			titre = 'Kernel ' +str(i)
			plt.suptitle(titre)
			plt.savefig(pp, format='pdf',dpi=600)
			plt.close()
	
		pp.close()
		plt.clf()


def pattern_comp_img(args):
	style_layers_size =  {'input':3,'conv1' : 64,'relu1' : 64,'pool1': 64,'conv2' : 128,'relu2' : 128,'pool2':128,'conv3' : 256,'relu3' : 256,'pool3':256,'conv4': 512,'relu4' : 512,'pool4':512,'conv5' : 512,'relu5' : 512,'pool5':512}
	path_origin = '/home/nicolas/Style-Transfer/Results/Patterns/'
	path_origin_img = path_origin + 'img/'
	path_output = path_origin + 'img_output/'
	dirs = os.listdir(path_origin_img)
	dirs = sorted(dirs, key=str.lower)
	list_img = {}
	print(dirs)
	for name in dirs:
		name_get = path_origin_img  + name
		img =scipy.misc.imread(name_get).astype('float32')
		img = gray2rgb(img)
		image = st.preprocess(img) 
		list_img[name] = image
		
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image) # net for the style image
	placeholder = tf.placeholder(tf.float32, shape=image.shape)
	assign_op = net['input'].assign(placeholder)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	Data = {}
	for layer in VGG19_LAYERS_INTEREST:
		Data[layer] = {}
		for name in dirs:
			name_wt_ext,_ = name.split('.')
			sess.run(assign_op, {placeholder: list_img[name]})
			a = net[layer].eval(session=sess)
			img = a[0]
			path_output_bis = path_origin + 'img_output/' + name_wt_ext +'/'
			if not(os.path.isdir(path_output_bis)):
				os.mkdir(path_output_bis)
			for i in range(style_layers_size[layer[:5]]):
				name_img = path_output_bis +  name_wt_ext + '_'+layer+'_'+str(i)+'.png'
				image =  img[:,:,i]
				scipy.misc.toimage(image).save(name_img)	
	  
def generation_Damier(output_name = 'DamierBig_Proces.png',size=256):
	damier = np.zeros((size,size,3)).astype('uint8')
	for i in range(size):
		for j in range(size):
			if ((i%2==0) and (j%2==0)) or ((i%2==1) and (j%2==1)):
				for k in range(3):
					damier[i,j,k] = 255
	damier = damier.astype('uint8')
	scipy.misc.toimage(damier).save(output_name)
	return(0)
	
	
	
def main():
	"""
	Plot
	"""
	parser = get_parser_args()
	parser.set_defaults(verbose=True)
	args = parser.parse_args()
	#do_pdf_comparison_GramOnly(args)
	do_pdf_comparison_GramOnly_autreratio(args)
	#pattern_comp_pdf(args)
	#pattern_comp_img(args)
if __name__ == '__main__':
	main()
