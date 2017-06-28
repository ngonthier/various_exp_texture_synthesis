#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Wed 28 June

This script have the goal to generate texture with different initialization 

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

def generation_Texture_init():
	path_origin = '/home/nicolas/Style-Transfer/InitializationTest/data/'
	path_output = '/home/nicolas/Style-Transfer/InitializationTest/results/'
	do_mkdir(path_output)
		
	parser = get_parser_args()
	max_iter = 2000
	print_iter = 100
	start_from_noise = 1
	init_noise_ratio = 1.0
	optimizer = 'lbfgs'
	init_list = [('Gaussian',0.),('Cst',0.),('Uniform',127.5),('Uniform',20.),('Uniform',1.),('smooth_grad',0.)]
	loss = 'texture'
	
	list_img = get_list_of_images(path_origin)
	print(list_img)
	
	for init,init_range in init_list:
		for name_img in list_img:
			tf.reset_default_graph() # Necessity to use a new graph !! 
			name_img_wt_ext,_ = name_img.split('.')
			print("New Generation",init,name_img_wt_ext)
			img_folder = path_origin
			img_output_folder = path_origin
			output_img_name = name_img_wt_ext + '_' + init + '_' + str(init_range)
			parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
				img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
				init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
				optimizer=optimizer,loss=loss,init=init,init_range=init_range)
			args = parser.parse_args()
			st.style_transfer(args)

if __name__ == '__main__':
	generation_Texture_init()
	# python Init_Test_Gen.py >> /home/nicolas/Style-Transfer/InitializationTest/results/output.txt
