#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  evol_Resol.py
#  
#  Copyright 2018 gonthier <gonthier@Morisot>
#  
#  The goal of this script is to visualize the evolution of the differents
#  loss with the resolution of the images
#  

import tensorflow as tf
import Style_Transfer as st
import numpy as np
from Arg_Parser import get_parser_args 
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def plot_annotation(labels,datax, datay):
	for label, x, y in zip(labels, datax, datay):
		plt.annotate(
			label,
			xy=(x, y), xytext=(-20, 20),
			textcoords='offset points', ha='right', va='bottom',
			bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

def main(args):
	path_img = 'dataImages2'
	name_img = 'TexturesCom_TilesOrnate0158_1_seamless_S.jpg'
	#name_img = 'metal_ground_1024.png'
	#name_img = 'CRW_4065_1024.png'
	name_img_complet = path_img + '/'+ name_img
	im = cv2.imread(name_img_complet)
	h,w,_ = im.shape
	list_dim = [1024,512,256]
	parser = get_parser_args()
	args = rgs = parser.parse_args()
	pooling_type = args.pooling_type
	padding = args.padding
	vgg_layers = st.get_vgg_layers(args.vgg_name)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	style_layers = [('conv1_1',1),('pool1',1),('pool2',1),('pool3',1),('pool4',1)]
	
	st_l_tab = []
	sp_l_tab = []
	st_l_tab2 = []
	sp_l_tab2 = []
	labels = []
	labels2 = []
	
	for dim in list_dim:
		if not(dim==h and dim==w):
			resized = cv2.resize(im, (dim,dim), interpolation = cv2.INTER_AREA)
		else:
			resized = im
		print('resized.shape',resized.shape)
		image_style = (resized -np.array([123.68, 116.779,103.939 ]).reshape((1,1,1,3))).astype('float32') # Substract the mean for a BGR image
		M_dict = st.get_M_dict(dim,dim)
		
		dict_gram = st.get_Gram_matrix_wrap(args,vgg_layers,image_style,pooling_type,padding)
		
		net = st.net_preloaded(vgg_layers,image_style,pooling_type,padding)
		init_img = st.get_init_img_wrap(args,'',image_style)
		print('init_img.shape',init_img.shape)

		sess = tf.Session(config=config)
		placeholder = tf.placeholder(tf.float32, shape=init_img.shape)
		assign_op = net['input'].assign(placeholder)
		style_loss =  st.sum_style_losses(sess, net, dict_gram,M_dict,style_layers)
		loss_spectrum = st.loss_spectrum(sess,net,image_style,M_dict,args.beta_spectrum)
		
		sess.run(tf.global_variables_initializer())
		
		# First with a white noise generate the same size than the image
		sess.run(assign_op, {placeholder: init_img})
		style_loss_value = sess.run(style_loss)
		loss_spectrum_value = sess.run(loss_spectrum)
		st_l_tab += [style_loss_value]
		sp_l_tab += [loss_spectrum_value]
		labels += ['R_'+str(dim)]
		print('Initialisation normal compute for ',dim)
		print('style_loss_value = {:.2e}'.format(style_loss_value))
		print('loss_spectrum_value  = {:.2e}'.format(loss_spectrum_value))
		# Second with a white noise from a subsampling 
		if dim==list_dim[0]:
			saved_init = init_img.reshape((dim,dim,3))
		else:
			print(saved_init.shape)
			init_img = cv2.resize(saved_init, (dim,dim), interpolation = cv2.INTER_AREA)
			print(init_img.shape)
			init_img = init_img.reshape((1,dim,dim,3))
			sess.run(assign_op, {placeholder: init_img})
			style_loss_value = sess.run(style_loss)
			loss_spectrum_value = sess.run(loss_spectrum)
			st_l_tab2 += [style_loss_value]
			sp_l_tab2 += [loss_spectrum_value]
			labels2 += ['R_'+str(dim)]
			print('Initialisation subsampled compute for ',dim)

	
		sess.close()
	
	fig = plt.figure()
	ax = plt.gca()
	plt.subplots_adjust(bottom = 0.1)
	ax.scatter(np.arange(len(st_l_tab)),st_l_tab, marker='o', c='r',label='Style Loss')
	ax.scatter(np.arange(len(sp_l_tab)),sp_l_tab, marker='o', c='b',label='Spectrum Loss')
	ax.scatter(np.arange(1,1+len(st_l_tab2)),st_l_tab2, marker='x', c='m',label='Style Loss Initialisation subsampled')
	ax.scatter(np.arange(1,1+len(sp_l_tab2)),sp_l_tab2, marker='x', c='g',label='Spectrum Loss Initialisation subsampled')
	ax.set_yscale('log')
	plot_annotation(labels,np.arange(len(st_l_tab)),st_l_tab)
	plot_annotation(labels,np.arange(len(sp_l_tab)),sp_l_tab)
	plot_annotation(labels2,np.arange(1,1+len(st_l_tab2)),st_l_tab2)
	plot_annotation(labels2,np.arange(1,1+len(sp_l_tab2)),sp_l_tab2)

	plt.legend(loc='best')
	plt.show()	
	#input("wait to close")
	plt.close()
	
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
