"""
Created on  Wed 28 June 2017

This script have the goal to generate texture with different loss function

@author: nicolas
"""

import Style_Transfer as st
from Arg_Parser import get_parser_args 
import os
import tensorflow as tf
import os.path
from shutil import copyfile

def get_list_of_images(path_origin):
	dirs = os.listdir(path_origin)
	dirs = sorted(dirs, key=str.lower)
	return(dirs)

def do_mkdir(path):
	if not(os.path.isdir(path)):
		os.mkdir(path)
	return(0)

def genereation_Texture_Padding():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy_output/'
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 2000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	loss =  'texture'
	config_layers = 'GatysConfig'
	
	padding_list = ['Davy','VALID','SAME']
	
	list_img = get_list_of_images(path_origin)
	print(list_img)
	
	for padding in padding_list:
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			args = parser.parse_args()
			st.style_transfer(args)
	
def generation_Texture_LossFct():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImages/'

	#path_origin = '/home/nicolas/random_phase_noise_v1.3/im/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy/output/'
	#path_output = '/home/nicolas/Style-Transfer/LossFct/random_phase_noise_v1.3/'
	#path_output = '/home/nicolas/Style-Transfer/LossFct/tmp/'
	do_mkdir(path_output)
		
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 500
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	n_list = [1,2,4]
	p = 4
	losses_to_test = [['autocorr'],['Lp'],['fft3D'],['bizarre']]
	#losses_to_test = [['texture','HF'],['texture'],['texture','TV'],['texture','HFmany']]
	losses_to_test = [['nmoments'],['Lp'],['texture'],['PhaseAlea']]
	losses_to_test = [['autocorr_rfft'],['autocorr']]
	losses_to_test = [['texture']]
	config_layers_tab = ['PoolConfig','FirstConvs']
	config_layers_tab = ['PoolConfig']
	
	list_img = get_list_of_images(path_origin)
	
	for config_layers in config_layers_tab:
		for loss in losses_to_test:
			if loss[0] == 'nmoments':
				for n in n_list: 
					if(config_layers=='FirstConvs') and (n > 2):
						loss = ['nmoments_reduce']
					for name_img in list_img:
						tf.reset_default_graph() # Necessity to use a new graph !! 
						name_img_wt_ext,_ = name_img.split('.')
						img_folder = path_origin
						img_output_folder = path_origin
						output_img_name = name_img_wt_ext
						for loss_item in loss:
							output_img_name += '_' + loss_item + '_' + str(n)
						parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
							img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
							init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
							optimizer=optimizer,loss=loss,init=init,init_range=init_range,p=p,n=n,clipping_type=clipping_type,
							vgg_name=vgg_name)
						args = parser.parse_args()
						st.style_transfer(args)

			else:
				n = 4
				for name_img in list_img:
					tf.reset_default_graph() # Necessity to use a new graph !! 
					name_img_wt_ext,_ = name_img.split('.')
					img_folder = path_origin
					img_output_folder = path_origin
					output_img_name = name_img_wt_ext
					for loss_item in loss:
						output_img_name += '_' + loss_item
					parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
						img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
						init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
						optimizer=optimizer,loss=loss,init=init,init_range=init_range,p=p,n=n,clipping_type=clipping_type,
						vgg_name=vgg_name)
					args = parser.parse_args()
					st.style_transfer(args)
	
def generation_Texture_LossFctAlphaPhaseAlea():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy_output/'
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 2000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	
	#list_img = get_list_of_images(path_origin)
	list_img = ['TilesOrnate0158_512.png']
	print(list_img)
	
	# Comparison on the loss function !!! 
	loss = ['phaseAlea']
	padding = 'SAME'
	alpha_list = [10**4,10**2,10,1,10**(-1),10**(-2),10**(-4)]
	for alpha in alpha_list:
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding+'_'+str(alpha)
			for loss_item in loss:
				output_img_name += '_' + loss_item
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding,alpha_phaseAlea=alpha)
			args = parser.parse_args()
			st.style_transfer(args)

def generation_Texture_LossFctBetaSpectrum():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy_output/'
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 2000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	
	#list_img = get_list_of_images(path_origin)
	list_img = ['TilesOrnate0158_512.png']
	list_img = ['TexturesCom_TilesOrnate0158_1_seamless_S.png']
	print(list_img)
	
	# Comparison on the loss function !!! 
	loss = ['texture','spectrum']
	padding = 'SAME'
	beta_list = [10**(6),10**(4),10**(5)] # beta spectrum
	for beta in beta_list:
		print("Beta :",str(beta))
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding+'_'+str(beta)
			for loss_item in loss:
				output_img_name += '_' + loss_item
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding,beta_spectrum=beta)
			args = parser.parse_args()
			st.style_transfer(args)
			
def generation_Texture_LossGatysPlusAutocorr():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy_output/GatysPlusAutocorr'
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 1000
	print_iter = 1000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	
	#list_img = get_list_of_images(path_origin)
	list_img = ['TilesOrnate0158_512.png']
	print(list_img)
	
	# Comparison on the loss function !!! 
	loss = ['texture','spectrum']
	padding = 'SAME'
	gamma_list = [10**(4),10**(3),10**(2),10,1,0.1,0.01,0.001,10**(-4),10**(-6),10**(-8)] # beta spectrum
	for gamma in gamma_list:
		print("gamma :",str(gamma))
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding+'_'+str(gamma)
			for loss_item in loss:
				output_img_name += '_' + loss_item
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding,gamma_autocorr=gamma)
			args = parser.parse_args()
			st.style_transfer(args)
			
def generation_Texture_LossFctBetaSpectrum_PhaseAlea():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy_output/'
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 2000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	
	#list_img = get_list_of_images(path_origin)
	list_img = ['TilesOrnate0158_512.png']
	print(list_img)
	
	# Comparison on the loss function !!! 
	loss = ['phaseAlea','spectrum']
	padding = 'SAME'
	gamma_phaseAlea_list = [0.1,1.,10,10**2,10**3,10**4,10**5] #gamma_phaseAlea
	for gamma_phaseAlea in gamma_phaseAlea_list:
		print("Gamma :",str(gamma_phaseAlea))
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding+'_gamma'+str(gamma_phaseAlea)
			for loss_item in loss:
				output_img_name += '_' + loss_item
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding,gamma_phaseAlea=gamma_phaseAlea)
			args = parser.parse_args()
			st.style_transfer(args)
	
def generation_Texture_LossFct3():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImages2/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImages2_output/'
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 2000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	beta_spectrum = 100
	alpha = 0.01
	list_img = get_list_of_images(path_origin)
	DrawAgain = False # Erase already synthesied image
	print(list_img)
	
	# Comparison on the loss function !!! 
	#losses_to_test = [['autocorr'],['phaseAlea'],['texture','spectrum'],['texture','TVronde'],['texture','TV1'],['texture','TV']]
	losses_to_test = [['texture'],['texture','spectrum'],['autocorr'],['phaseAlea']]
	padding = 'SAME'
	for loss in losses_to_test:
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			path_output_tmp = path_output+name_img_wt_ext
			do_mkdir(path_output_tmp)
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding
			for loss_item in loss:
				output_img_name += '_' + loss_item
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			args = parser.parse_args()
			output_img_name_full = path_output + output_img_name + '.png'
			if DrawAgain or not(os.path.isfile(output_img_name_full)):
				st.style_transfer(args)
				src=output_img_name_full
				dst = path_output_tmp+'/'+ output_img_name + '.png'
				copyfile(src, dst)
					
def generation_Texture_LossFct2():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy_output/'
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 2000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	beta_spectrum = 100
	alpha = 0.01
	list_img = get_list_of_images(path_origin)
	DrawAgain = False # Erase already synthesied image
	print(list_img)
	
	# Comparison on the loss function !!! 
	#losses_to_test = [['autocorr'],['phaseAlea'],['texture','spectrum'],['texture','TVronde'],['texture','TV1'],['texture','TV']]
	losses_to_test = [['texture'],['texture','spectrum'],['autocorr'],['phaseAlea']]
	padding = 'SAME'
	for loss in losses_to_test:
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding
			for loss_item in loss:
				output_img_name += '_' + loss_item
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			args = parser.parse_args()
			output_img_name_full = path_output + output_img_name + '.png'
			if DrawAgain or not(os.path.isfile(output_img_name_full)):
				st.style_transfer(args)
	
	losses_to_test = [['autocorr','spectrum']]
	gamma = 10**4
	for loss in losses_to_test:
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding
			for loss_item in loss:
				output_img_name += '_' + loss_item
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,gamma_phaseAlea=gamma,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			args = parser.parse_args()
			output_img_name_full = path_output + output_img_name + '.png'
			if DrawAgain or not(os.path.isfile(output_img_name_full)):
				st.style_transfer(args)
	
			
	#losses_to_test = [['phaseAlea','spectrum']]
	#gamma = 10**4
	#for loss in losses_to_test:
		#for name_img in list_img:
			#name_img_wt_ext,_ = name_img.split('.')
			#tf.reset_default_graph() # Necessity to use a new graph !! 
			#img_folder = path_origin
			#img_output_folder = path_origin
			#output_img_name = name_img_wt_ext + '_'+padding
			#for loss_item in loss:
				#output_img_name += '_' + loss_item
			#parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				#img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				#init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				#optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,gamma_phaseAlea=gamma,
				#vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			#args = parser.parse_args()
			#output_img_name_full = path_output + output_img_name + '.png'
			#if DrawAgain or not(os.path.isfile(output_img_name_full)):
				#st.style_transfer(args)
	
			
	#losses_to_test = [['autocorr','texture']]
	#gamma = 0.1
	#for loss in losses_to_test:
		#for name_img in list_img:
			#name_img_wt_ext,_ = name_img.split('.')
			#tf.reset_default_graph() # Necessity to use a new graph !! 
			#img_folder = path_origin
			#img_output_folder = path_origin
			#output_img_name = name_img_wt_ext + '_'+padding
			#for loss_item in loss:
				#output_img_name += '_' + loss_item
			#parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				#img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				#init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				#optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,gamma_autocorr=gamma,
				#vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			#args = parser.parse_args()
			#output_img_name_full = path_output + output_img_name + '.png'
			#if DrawAgain or not(os.path.isfile(output_img_name_full)):
				#st.style_transfer(args)
				
	#print('Padding')
	#loss =  'texture'
	#padding_list = ['Davy','VALID','SAME']
	## Comparison of the padding influence !!! 
	
	#for padding in padding_list:
		#for name_img in list_img:
			#name_img_wt_ext,_ = name_img.split('.')
			#tf.reset_default_graph() # Necessity to use a new graph !! 
			#img_folder = path_origin
			#img_output_folder = path_origin
			#output_img_name = name_img_wt_ext + '_'+padding + '_' +loss
			#parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				#img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				#init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				#optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				#vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			#args = parser.parse_args()
			#output_img_name_full = path_output + output_img_name + '.png'
			#if DrawAgain or not(os.path.isfile(output_img_name_full)):
				#st.style_transfer(args)
				
	## Deep Corr Config 
	#config_layer_test = ['DCor','DCor_TV']
	#padding = 'SAME'
	#loss = ['']
	#for config_layers in config_layer_test:
		#for name_img in list_img:
			#name_img_wt_ext,_ = name_img.split('.')
			#tf.reset_default_graph() # Necessity to use a new graph !! 
			#img_folder = path_origin
			#img_output_folder = path_origin
			#output_img_name = name_img_wt_ext + '_'+padding
			#for loss_item in loss:
				#output_img_name += '_' + loss_item 
			#output_img_name += config_layers
			#parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				#img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				#init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				#optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				#vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			#args = parser.parse_args()
			#output_img_name_full = path_output + output_img_name + '.png'
			#if DrawAgain or not(os.path.isfile(output_img_name_full)):
				#st.style_transfer(args)
	
def generation_Texture_JustTexture_and_TexturePlusSpectrum():
	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy/'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/dataImagesDavy_output/'
	do_mkdir(path_output)
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 2000
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	maxcor = 20
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	config_layers = 'GatysConfig'
	beta_spectrum = 100
	alpha = 0.01
	list_img = get_list_of_images(path_origin)
	DrawAgain = False # Erase already synthesied image
	print(list_img)
	
	# Comparison on the loss function !!! 
	losses_to_test = [['texture'],['texture','spectrum']]
	padding = 'SAME'
	for loss in losses_to_test:
		for name_img in list_img:
			name_img_wt_ext,_ = name_img.split('.')
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding
			for loss_item in loss:
				output_img_name += '_' + loss_item
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding)
			args = parser.parse_args()
			output_img_name_full = path_output + output_img_name + '.png'
			if DrawAgain or not(os.path.isfile(output_img_name_full)):
				st.style_transfer(args)
			
	

if __name__ == '__main__':
	#generation_Texture_LossFct2()
	#generation_Texture_JustTexture_and_TexturePlusSpectrum()
	#generation_Texture_LossFctAlphaPhaseAlea()
	#generation_Texture_LossFct()
	# python LossFct_Test_Gen.py >> /home/nicolas/Style-Transfer/LossFct/results/output.txt
	#generation_Texture_LossFctBetaSpectrum()
	generation_Texture_LossFct3()
	#generation_Texture_LossFctBetaSpectrum_PhaseAlea()
	#generation_Texture_LossFct2()
	#generation_Texture_LossGatysPlusAutocorr()	
