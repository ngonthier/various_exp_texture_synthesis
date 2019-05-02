"""
Created on  Wed 28 July

This script have the goal to generate differente texture with different
use of the layer 

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

def generation_Texture_():
	path_origin = '/home/nicolas/Style-Transfer/dataImages/'
	path_output = '/home/nicolas/Style-Transfer/LossFct/Layers/'
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
	loss = ['texture']
	config_layers_tab = [['conv1_1'],['conv1_1','pool1'],['conv1_1','pool1','pool2'],['conv1_1','pool1','pool2','pool3'],['conv1_1','pool1','pool2','pool3','pool4'],['conv1_1','conv2_1','conv3_1']]
	list_img = ['pebbles.png','glass.png','AbstractVarious0013_S.png','FoodVarious0033_1_S.png']
	config_layers = 'Custom'

	for style_layers in config_layers_tab:
		style_layer_weights = [1]*len(style_layers)
		for name_img in list_img:
			tf.reset_default_graph() # Necessity to use a new graph !! 
			name_img_wt_ext,_ = name_img.split('.')
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext
			for layer in style_layers:
				output_img_name += '_' + layer
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,config_layers=config_layers,
				optimizer=optimizer,loss=loss,clipping_type=clipping_type,style_layers=style_layers,style_layer_weights=style_layer_weights,
				vgg_name=vgg_name)
			args = parser.parse_args()
			st.style_transfer(args)

if __name__ == '__main__':
	generation_Texture_()
	# python LossFct_Test_Gen.py >> /home/nicolas/Style-Transfer/LossFct/results/output.txt
