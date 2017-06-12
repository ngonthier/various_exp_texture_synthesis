#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 2017

This script have the goal to generate texture with different seetings

@author: nicolas
"""

from Arg_Parser import get_parser_args 
import utils
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
import matplotlib.gridspec as gridspec
import math
from PIL import Image
from os import listdir

path_origin = '/home/nicolas/Images/original/'
path_output = '/home/nicolas/Images/synresult_nico/'

VGG19_LAYERS_INTEREST = (
    'conv1_1','conv2_1', 'conv3_1'
)

def load_img_reshape(args,img_name,size=(256,256)):
	"""
	This function load the image and convert it to a numpy array and do 
	the preprocessing
	"""
	image_path = args.img_folder + img_name +args.img_ext
	new_img_ext = args.img_ext
	try:
		img = scipy.misc.imread(image_path)  # Float between 0 and 255
	except IOError:
		if(args.verbose): print("Exception when we try to open the image, try with a different extension format ",str(args.img_ext))
		if(args.img_ext==".jpg"):
			new_img_ext = ".png"
		elif(args.img_ext==".png"):
			new_img_ext = ".jpg"
		try:
			image_path = args.img_folder + img_name +new_img_ext # Try the new path
			img = scipy.misc.imread(image_path)
			if(args.verbose): print("The image have been sucessfully loaded with a different extension")
		except IOError:
			if(args.verbose): print("Exception when we try to open the image, we already test the 2 differents extension.")
			raise
	img_reshape = scipy.misc.imresize(img, size, interp='bilinear', mode=None)
	img = preprocess(img_reshape.astype('float32'))
	return(img)

def get_list_of_images():
	dirs = os.listdir(path_origin)
	dirs = sorted(dirs, key=str.lower)
	return(dirs)
	
def compute_moments_of_filter(TypeOfComputation='moments',n = 9):
	"""
	Plot the 9th first moments of each of the filter
	n = number of moments
	"""
	img_folder = path_origin
	parser = get_parser_args()
	parser.set_defaults(img_folder=img_folder)
	args = parser.parse_args()
	dirs = get_list_of_images()
	Data =  {}
	print("Computation")
	h_old = 0
	w_old = 0
	vgg_layers = st.get_vgg_layers() 
	Notfirst = False
	for name_img in dirs:
		name_img_wt_ext,_ = name_img.split('.')
		print(name_img_wt_ext)
		image_style = st.load_img(args,name_img_wt_ext) # Sans reshape
		Data[name_img] = {}
		_,h,w,_ =  image_style.shape
		if not((h_old == h) and (w_old==w)) and Notfirst:
			Notfirst = True
			sess.close()
		if not((h_old == h) and (w_old==w)): 
			tf.reset_default_graph()
			net = st.net_preloaded(vgg_layers, image_style) # Need to load a new net
			sess = tf.Session()
		sess.run(net['input'].assign(image_style))
		for layer in VGG19_LAYERS_INTEREST:
			a = net[layer]
			if(TypeOfComputation=='moments'):
				listOfMoments = sess.run(st.compute_n_moments(a,n))
			elif(TypeOfComputation=='Lp'):
				listOfMoments = sess.run(st.compute_Lp_norm(a,n))
			Data[name_img][layer] = listOfMoments
		h_old,w_old = h,w
	sess.close()
	data_path = args.data_folder + "moments_all_textures.pkl"
	with open(data_path, 'wb') as output_pkl:
		pickle.dump(Data,output_pkl)
	print("End")
	
def plot_stats_etc_on_moment(TypeOfComputation='moments'):
	fontsize = 6
	data_folder = 'data/'
	data_path = data_folder + "moments_all_textures.pkl"
	with open(data_path, 'rb') as output_pkl:
		Data = pickle.load(output_pkl)	
	path = 'Results/Filter_Rep/'
	pltname = path +TypeOfComputation+'_textures.pdf'
	pp = PdfPages(pltname)
	keys = Data.keys()
	#keys = ['WoodChips0040_2_S.png','Leather0069_1_S.png']
	Data2 = {}
	for name in keys:
		name_img_wt_ext,_ = name.split('.')
		print(name_img_wt_ext)
		layers = Data[name]
		f = plt.figure()
		gs = gridspec.GridSpec(1,len(layers))
		sorted_keys = sorted(layers.keys(),key=str.lower)
		for i,layer in enumerate(sorted_keys):
			listOfMoments = layers[layer]
			number_of_moments = len(listOfMoments)
			ax = plt.subplot(gs[i])
			ax = sns.boxplot(x=list(range(1,number_of_moments+1)),y=listOfMoments)
			ax.set_title(layer)
			ax.set_xlabel("ordre")
			ax.set_yscale("symlog")
			if not(layer in Data2.keys()):
				Data2[layer] = {}
				for j in range(number_of_moments):
					Data2[layer][j] = listOfMoments[j]
			else:
				for j in range(number_of_moments):
					Data2[layer][j] = np.vstack((Data2[layer][j],listOfMoments[j]))
		plt.suptitle(name_img_wt_ext)
		plt.savefig(pp, format='pdf')
		plt.close()
	pp.close()
	print("End per image")
	pltname = path +TypeOfComputation+'_textures_layers.pdf'
	pp = PdfPages(pltname)	
	f = plt.figure()
	layers = Data2.keys()
	gs = gridspec.GridSpec(1,len(layers))
	for i,layer in enumerate(layers):
		listOfMoments = list()
		for j in range(number_of_moments):
			listOfMoments.append(Data2[layer][j].ravel())
		number_of_moments = len(listOfMoments)
		ax = plt.subplot(gs[i])
		ax = sns.boxplot(x=list(range(1,number_of_moments+1)),y=listOfMoments)
		ax.set_title(layer)
		ax.set_xlabel("ordre")
		ax.set_yscale("symlog")
	titre = "Valeur par "+TypeOfComputation+" sur toutes les images"
	plt.suptitle( titre )
	plt.savefig(pp, format='pdf')
	for i,layer in enumerate(layers):
		listOfMoments = list()
		fig = plt.figure()
		gs = gridspec.GridSpec(3,3)
		#gs.update(hspace=0.05, wspace=0.05)
		for j in range(number_of_moments):
			distrb = np.mean(Data2[layer][j],axis=0) # Mean on column
			ax = plt.subplot(gs[j],label='small')
			ax = sns.distplot(distrb, kde=False, rug=True)
			titre = "ordre " + str(j)
			ax.set_title(titre)
			ax.set_xlabel("Range")
		plt.suptitle(layer)
		gs.tight_layout(fig,rect=[0, 0.03, 1, 0.95]) 
		plt.savefig(pp, format='pdf')
		plt.close()
	pp.close()
	print("End total")
	plt.close()
	plt.clf()
			
def generate_all_texture(args,path_output_mod,pooling_type='avg',padding='VALID'):
	dirs = get_list_of_images()
	
	for name_img in dirs:
		tf.reset_default_graph()
		name_img_wt_ext,_ = name_img.split('.')
		
		if args.verbose:
			tinit = time.time()
			print("Name :",name_img_wt_ext)
		
		output_image_path = path_output_mod + name_img_wt_ext +"_syn" +args.img_ext
		image_style = st.load_img(args,name_img_wt_ext) # Sans reshape
		_,image_h, image_w, number_of_channels = image_style.shape 
		M_dict = st.get_M_dict(image_h,image_w)
			
		# TODO add something that reshape the image 
		t1 = time.time()
		vgg_layers = st.get_vgg_layers()
		
		# Precomputation Phase :
		dict_gram = st.get_Gram_matrix_wrap(args,vgg_layers,image_style,pooling_type,padding)
		
		net = st.net_preloaded(vgg_layers, image_style,pooling_type,padding) # The output image as the same size as the content one
		
		t2 = time.time()
		if(args.verbose): print("net loaded and gram computation after ",t2-t1," s")

		try:
			config = tf.ConfigProto()
			if(args.gpu_frac <= 0.):
				config.gpu_options.allow_growth = True
				if args.verbose: print("Memory Growth")
			elif(args.gpu_frac <= 1.):
				config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
				if args.verbose: print("Becareful args.gpu_frac = ",args.gpu_frac,"It may cause problem if the value is superior to the available memory place.")
			sess = tf.Session(config=config)
			init_img = st.get_init_img_wrap(args,output_image_path,image_style) # Pour avoir la meme taille que l'image de style
			dict_features_repr = None
			loss_total,list_loss,list_loss_name = st.get_losses(args,sess, net, dict_features_repr,M_dict,image_style,dict_gram,pooling_type,padding)				
				
			# Preparation of the assignation operation
			placeholder = tf.placeholder(tf.float32, shape=init_img.shape)
			assign_op = net['input'].assign(placeholder)
			
			# LBFGS seem to require more memory than Adam optimizer
			
			bnds = st.get_lbfgs_bnds(init_img)
			
			nb_iter = args.max_iter  // args.print_iter
			max_iterations_local = args.max_iter // nb_iter
			if(args.verbose): print("Start LBFGS optim with a print each ",max_iterations_local," iterations")
			optimizer_kwargs = {'maxiter': max_iterations_local,'maxcor': args.maxcor}
			optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds,
				method='L-BFGS-B',options=optimizer_kwargs)         
			sess.run(tf.global_variables_initializer())
			sess.run(assign_op, {placeholder: init_img})
						
			sess.graph.finalize() # To test if the graph is correct
			if(args.verbose): print("sess.graph.finalize()") 
			
			if(args.verbose): print("loss before optimization")
			if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
			for i in range(nb_iter):
				t3 =  time.time()
				optimizer.minimize(sess)
				t4 = time.time()
				if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
				if(args.verbose): st.print_loss_tab(sess,list_loss,list_loss_name)
				result_img = sess.run(net['input'])
				if(args.plot): fig = st.plot_image_with_postprocess(args,result_img.copy(),"Intermediate Image",fig)
				result_img_postproc = st.postprocess(result_img)
				scipy.misc.toimage(result_img_postproc).save(output_image_path)
			result_img = sess.run(net['input'])
			if(args.plot): st.plot_image_with_postprocess(args,result_img.copy(),"Final Image",fig)
			result_img_postproc = st.postprocess(result_img)
			scipy.misc.toimage(result_img_postproc).save(output_image_path)     
		except KeyboardInterrupt:
			if(args.verbose): print("Error, in the lbfgs case the image can be strange and incorrect")
			result_img = sess.run(net['input'])
			result_img_postproc = st.postprocess(result_img)
			output_image_path_error = args.img_folder + args.output_img_name+'_error' +args.img_ext
			scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
			# In the case of the lbfgs optimizer we only get the init_img if we did not do a check point before
			raise
		except:
			if(args.verbose): print("Error, in the lbfgs case the image can be strange and incorrect")
			result_img = sess.run(net['input'])
			result_img_postproc = st.postprocess(result_img)
			output_image_path_error = args.img_folder + args.output_img_name+'_error' +args.img_ext
			scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
		finally:
			sess.close()
			tf.reset_default_graph() # It is sub optimal 
			if(args.verbose): 
				print("Close Sess")
				tend = time.time()
				print("Computation total for ",tend-tinit," s")
		# TODO change that and get back to a version without the dictionnary and precomputation
		data_style_path = args.data_folder + "gram_"+args.style_img_name+"_"+str(image_h)+"_"+str(image_w)+"_"+str(pooling_type)+"_"+str(padding)+".pkl"
		os.remove(data_style_path) 
		sess.close()
	
def generation_Texture():	
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 500
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	style_img_name = "temp"
	max_order_nmoments = 5
	min_order_nmoments = 3
	losses_to_test = [['autocorr'],['nmoments'],['texture'],['InterScale'],['Lp'],['texture','nmoments'],['texture','Lp']]
	
	for list_of_loss in losses_to_test:
		print("loss = ",list_of_loss)
		img_folder = path_origin 
		path_output_mod = path_output + "_".join(list_of_loss)
		if('nmoments' in list_of_loss) and (len(list_of_loss)==1):
			for n in range(min_order_nmoments,max_order_nmoments+1,1):
				path_output_mod2 = path_output_mod
				print("n",n)
				path_output_mod2 += "_"+str(n)+'/'
				if not(os.path.isdir(path_output_mod2)):
					os.mkdir(path_output_mod2)
				parser.set_defaults(max_iter=max_iter,print_iter=print_iter,img_folder=img_folder,
					init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,
					optimizer=optimizer,n=n,loss=list_of_loss,style_img_name=style_img_name)
				args = parser.parse_args()
				generate_all_texture(args,path_output_mod2)
		else:
			path_output_mod +='/'
			if not(os.path.isdir(path_output_mod)):
					os.mkdir(path_output_mod)
			parser.set_defaults(max_iter=max_iter,print_iter=print_iter,img_folder=img_folder,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,
				optimizer=optimizer,loss=list_of_loss)
			args = parser.parse_args()
			generate_all_texture(args,path_output_mod)

if __name__ == '__main__':
	generation_Texture()
	#TypeOfComputation='moments'
	#TypeOfComputation='Lp'
	#compute_moments_of_filter(TypeOfComputation)
	#plot_stats_etc_on_moment(TypeOfComputation)
