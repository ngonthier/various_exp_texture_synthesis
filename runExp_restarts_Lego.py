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
				

def generation_Texture_LegoTest():

	number_of_restarts = 10

	path_origin = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/images/'
	name_img = 'lego_1024Ref_256.png'
	path_output = '/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/images/LegoSpectrum/'
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
	#beta_spectrum = 100
	#alpha = 0.01
	DrawAgain = False # Erase already synthesied image
	eps_list=[10**(-16),0.001]
	loss= ['Gatys','spectrumTFabs']
	#scalesStrat = ['Init','']
	MSS = ''
	MS_Strat= MSS
	padding = 'SAME'
	for i in range(number_of_restarts):
		for j,eps in enumerate(eps_list):
			
			name_img_wt_ext,_ = name_img.split('.')
			path_output_tmp = path_output+name_img_wt_ext
			#do_mkdir(path_output_tmp)
			tf.reset_default_graph() # Necessity to use a new graph !! 
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_'+padding
			for loss_item in loss:
				output_img_name += '_' + loss_item
			if eps==10**(-16) or j==0:
				output_img_name += '_eps10m16'
			elif eps==0.001 or j ==1:
				output_img_name += '_eps10m3'
			output_img_name +='_'+str(i)
			if not(MSS==''):
				output_img_name += '_MSS' +MSS
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,clipping_type=clipping_type,
				vgg_name=vgg_name,maxcor=maxcor,config_layers=config_layers,padding=padding,MS_Strat=MS_Strat,
				eps=eps)
			args = parser.parse_args()
			output_img_name_full = path_output + output_img_name + '.png'
			if DrawAgain or not(os.path.isfile(output_img_name_full)):
				st.style_transfer(args)
				#src=output_img_name_full
				#dst = path_output_tmp+'/'+ output_img_name + '.png'
				#copyfile(src, dst)	
	
if __name__ == '__main__':
	generation_Texture_LegoTest()	
