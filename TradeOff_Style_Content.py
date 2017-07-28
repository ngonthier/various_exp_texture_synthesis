"""
Created on July 28

This script have the goal to generate Style image with different value 
for the content_strengh

@author: nicolas
"""

import Style_Transfer as st
from Arg_Parser import get_parser_args 
import os
import tensorflow as tf

def generation():
	path_origin = '/home/nicolas/Style-Transfer/'
	path_output = '/home/nicolas/Style-Transfer/'

	parser = get_parser_args()
	max_iter = 2000
	print_iter = 500
	start_from_noise = 0
	init_noise_ratio = 0.05
	optimizer = 'lbfgs'
	init = 'Gaussian'
	init_range = 0.0
	clipping_type = 'ImageStyleBGR'
	vgg_name = 'normalizedvgg.mat'
	loss = ['texture','content']
	
	content_strengh_tab = [10.,1.,0.1,0.01,0.001,0.0001,0.00001,0.000001]
	
	list_img = get_list_of_images(path_origin)
	
	for content_strengh in content_strengh_tab:	
		tf.reset_default_graph() # Necessity to use a new graph !! 
		name_img_wt_ext,_ = name_img.split('.')
		img_folder = path_origin
		img_output_folder = path_origin
		output_img_name = name_img_wt_ext + '_' + str(content_strengh)
		for loss_item in loss:
			output_img_name += '_' + loss_item
		parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
			img_output_folder=path_output,style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
			init_noise_ratio=init_noise_ratio,start_from_noise=start_from_noise,output_img_name=output_img_name,
			optimizer=optimizer,loss=loss,init=init,init_range=init_range,p=p,n=n,clipping_type=clipping_type,content_strengh=content_strengh,
			vgg_name=vgg_name)
		args = parser.parse_args()
		st.style_transfer(args)

if __name__ == '__main__':
	generation()
	# python LossFct_Test_Gen.py >> /home/nicolas/Style-Transfer/LossFct/results/output.txt
