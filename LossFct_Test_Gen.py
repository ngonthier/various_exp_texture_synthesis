"""
Created on  Wed 28 June

This script have the goal to generate texture with different loss function

@author: nicolas
"""

import Style_Transfer as st
from Arg_Parser import get_parser_args 
import os
import tensorflow as tf

def get_list_of_images(path_origin):
	dirs = os.listdir(path_origin)
	dirs = sorted(dirs, key=str.lower)
	return(dirs)

def do_mkdir(path):
	if not(os.path.isdir(path)):
		os.mkdir(path)
	return(0)

def generation_Texture_LossFct():
	path_origin = '/home/nicolas/Style-Transfer/InitializationTest/data/'
	path_output = '/home/nicolas/Style-Transfer/LossFct/results/'
	do_mkdir(path_output)
		
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 100
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	n = 4
	p = 4
	losses_to_test = [['autocorr'],['Lp'],['fft3D'],['bizarre']]
	losses_to_test = [['variance']]
	losses_to_test = [['texture']]
	list_img = get_list_of_images(path_origin)
	
	for loss in losses_to_test:
		for name_img in list_img:
			tf.reset_default_graph() # Necessity to use a new graph !! 
			name_img_wt_ext,_ = name_img.split('.')
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_' + loss[0]
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range,p=p,n=n)
			args = parser.parse_args()
			st.style_transfer(args)

if __name__ == '__main__':
	generation_Texture_LossFct()
	# python LossFct_Test_Gen.py >> /home/nicolas/Style-Transfer/LossFct/results/output.txt
