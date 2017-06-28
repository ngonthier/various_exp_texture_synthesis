#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 

Arguments parser for the function

@author: nicolas
"""

import argparse

def get_parser_args():
	"""
	Parser of the argument of the program
	Becareful there is an order to use the option in the command line 
	terminal :'(
	"""
	desc = "TensorFlow implementation of 'A Neural Algorithm for Artisitc Style'"  
	parser = argparse.ArgumentParser(description=desc)

	# Verbose argument
	parser.add_argument('--verbose',action="store_true",
		help='Boolean flag indicating if statements should be printed to the console.')
		
	# Plot argument
	parser.add_argument('--plot',action="store_true",
		help='Boolean flag indicating if image should be plotted.')
		
	# Name of the Images
	parser.add_argument('--output_img_name', type=str, 
		default='Pastiche',help='Filename of the output image.')
		
	parser.add_argument('--style_img_name',  type=str,default='StarryNight',
		help='Filename of the style image (example: StarryNight). It must be a .jpg image otherwise change the img_ext.')
  
	parser.add_argument('--content_img_name', type=str,default='Louvre',
		help='Filename of the content image (example: Louvre). It must be a .jpg image otherwise change the img_ext.')
		
	# Name of the folders 
	parser.add_argument('--img_folder',  type=str,default='images/',
		help='Name of the images folder')
  
	parser.add_argument('--img_output_folder',  type=str,default='images/',
		help='Name of the images output folder')
  
	parser.add_argument('--data_folder', type=str,default='data/',
		help='Name of the data folder')
	
	# Extension of the image
	parser.add_argument('--img_ext',  type=str,default='.png',
		choices=['.jpg','.png'],help='Extension of the image') #TODO : remove the '.' that cannot be read in command line 
		
	# Infomartion about the optimization
	parser.add_argument('--optimizer',  type=str,default='lbfgs',
		choices=['lbfgs', 'adam', 'GD'],
		help='Loss minimization optimizer. (default|recommended: %(default)s)')
	
	parser.add_argument('--max_iter',  type=int,default=1000,
		help='Number of Iteration Maximum. (default %(default)s)')
	
	parser.add_argument('--print_iter',  type=int,default=100,
		help='Number of iteration between each checkpoint. (default %(default)s)')
		
	parser.add_argument('--maxcor',  type=int,default=10,
		help='The maximum number of variable metric corrections used to define the limited memory matrix in LBFGS method. (default %(default)s)')
		
	parser.add_argument('--learning_rate',  type=float,default=10.,
		help='Learning rate only for adam or GD method. (default %(default)s) We advised to use 10 for Adam and 10**(-10) for GD')	
		
	# Argument for clipping the value in the Adam or GD case
	parser.add_argument('--clip_var',  type=int,default=1,
		help='Clip the values of the variable after each iteration only for adam or GD method. Equal to 1 for true and 0 otherwise (default %(default)s)')	
	
	# Profiling Tensorflow
	parser.add_argument('--tf_profiler',action='store_true',
		help='Profiling Tensorflow operation available only for adam.')
		
	# Info on the style transfer
	parser.add_argument('--content_strengh',  type=float,default=0.001,
		help='Importance give to the content : alpha/beta ratio. (default %(default)s)')
	
	parser.add_argument('--init_noise_ratio',type=float,default=0.1,
		help='Propostion of the initialization image that is noise. (default %(default)s)')
		
	parser.add_argument('--init',type=str,default='Uniform',choices=['Uniform','smooth_grad','Gaussian','Cst'],
		help='Type of initialization for the image variable.')
		
	parser.add_argument('--init_range',type=float,default=127.5,
		help='Range for the initialialisation value')
		
	parser.add_argument('--start_from_noise',type=int,default=0,choices=[0,1],
		help='Start compulsory from the content image noised if = 1 or from the former image with the output name if = 0. (default %(default)s)')
	
	# VGG 19 info
	parser.add_argument('--pooling_type', type=str,default='avg',
    choices=['avg', 'max'],help='Type of pooling in convolutional neural network. (default: %(default)s)')
	
	# Info on the loss function 
	parser.add_argument('--loss',nargs='+',type=str,default='full',
		choices=['full','Gatys','texture','content','4moments','nmoments','InterScale','autocorr','Lp','TV','fft3D','spectrum','phaseAlea','SpectrumOnFeatures','intercorr','bizarre','current'],
		help='Choice the term of the loss function. (default %(default)s)') # TODO need to be allow to get list of str loss
	
	parser.add_argument('--tv',  action='store_true',
		help='Add a Total variation term for regularisation of the noise')	# TODO need to be change 
	
	parser.add_argument('--sampling',  type=str,default='down',
		choices=['down','up'],
		help='Sampling parameter in the inter scale loss function. (default %(default)s)')
		
	parser.add_argument('--n',  type=int,default=4,
		help='Number of moments used in nmoments loss function. (default %(default)s)')
	
	parser.add_argument('--p',  type=int,default=4,
		help='Number of Lp norm compute for the loss function. (default %(default)s)')
	
	parser.add_argument('--type_of_loss',  type=str,default='add',choices=['add','mul','max','Keeney'],
		help='Type of map on the sub loss. (default %(default)s)')
		
	# Info about the Gatys loss function and layer used in the different function
	#parser.add_argument('--content_layers', nargs='+', type=str, 
		#default=['conv4_2'],
		#help='VGG19 layers used for the content image. (default: %(default)s)')
  
	#parser.add_argument('--style_layers', nargs='+', type=str,
		#default=['conv1_1','conv2_1','conv3_1'],
		#help='VGG19 layers used for the style image. (default: %(default)s)')
  
	#parser.add_argument('--content_layer_weights', nargs='+', type=float, 
		#default=[1.0], 
		#help='Contributions (weights) of each content layer to loss. (default: %(default)s)')
  
	#parser.add_argument('--style_layer_weights', nargs='+', type=float, 
		#default=[1.,1.,1.],
		#help='Contributions (weights) of each style layer to loss. (default: %(default)s)')
		
	# GPU Config :
	parser.add_argument('--gpu_frac',  type=float,default=0.,
		help='Fraction of the memory for the GPU process, if <=0. then memoryground = True. And if > 1. then normal behaviour ie 0.95%% of the memory is allocated without error. (default %(default)s)')
	
	return(parser)
