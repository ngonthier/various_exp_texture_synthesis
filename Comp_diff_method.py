"""
Created on  Wed 28 July

This script have the goal to compare different 

@author: nicolas
"""

import Style_Transfer as st
from Arg_Parser import get_parser_args 
import os
import tensorflow as tf
import numpy as np

# TODO regroup all those utility function in one file ! 
def get_list_of_images(path_origin):
	dirs = os.listdir(path_origin)
	dirs = sorted(dirs, key=str.lower)
	return(dirs)

def do_mkdir(path):
	if not(os.path.isdir(path)):
		os.mkdir(path)
	return(0)

def CompResult():
	path_origin_gatys_output = '/home/nicolas/Style-Transfer/LossFct/random_phase_noise_v1.3/'
	path_origin = '/home/nicolas/random_phase_noise_v1.3/im/'
	path_origin_rand = '/home/nicolas/random_phase_noise_v1.3/src/output/'
		
	list_img = get_list_of_images(path_origin)
		
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
	style_layers = ['conv1_1','pool1','pool2','pool3','pool4']
	style_layer_weights = [1]*len(style_layers)
	
	
	f = open('ComparisonRandomGatys.txt', 'w')
	
	for name_img in list_img:
		print(name_img)
		tf.reset_default_graph()
		name_img_wt_ext,_  = name_img.split('.') 
		parser.set_defaults(verbose=False,max_iter=max_iter,print_iter=print_iter,img_folder=path_origin,
		style_img_name=name_img_wt_ext,content_img_name=name_img_wt_ext,
		optimizer=optimizer,loss=loss,style_layers=style_layers,style_layer_weights=style_layer_weights,
		vgg_name=vgg_name)
		args = parser.parse_args()
		
		f.write(name_img_wt_ext+'\n')
		image_content = st.load_img(args,name_img_wt_ext)
		image_style = st.load_img(args,name_img_wt_ext)
		_,image_h, image_w, number_of_channels = image_content.shape 
		M_dict = st.get_M_dict(image_h,image_w)
		
		pooling_type = args.pooling_type
		padding = args.padding
		vgg_layers = st.get_vgg_layers(args.vgg_name)
		
		# Precomputation Phase :
		
		dict_gram = st.get_Gram_matrix_wrap(args,vgg_layers,image_style,pooling_type,padding)
		dict_features_repr = st.get_features_repr_wrap(args,vgg_layers,image_content,pooling_type,padding)
		net = st.net_preloaded(vgg_layers, image_content,pooling_type,padding) # The output image as the same size as the content one
		placeholder = tf.placeholder(tf.float32, shape=image_style.shape)
		assign_op = net['input'].assign(placeholder)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		loss_total,list_loss,list_loss_name = st.get_losses(args,sess, net, dict_features_repr,M_dict,image_style,dict_gram,pooling_type,padding)
		
		image_gatys =  name_img_wt_ext + '_texture'
		args.img_folder = path_origin_gatys_output
		image_gaty_get = st.load_img(args,image_gatys)
		sess.run(assign_op, {placeholder: image_gaty_get})
		loss_ref_gatys = sess.run(loss_total)
		string = 'Gatys method = {:.2e} \n'.format(loss_ref_gatys)
		print(string)
		f.write(string)
		list_img_rand = get_list_of_images(path_origin_rand+name_img_wt_ext)
		loss_tab = []
		for img_name in list_img_rand:
			img_name_wt,_ = img_name.split('.')
			img_name_path = name_img_wt_ext +'/'+img_name_wt
			args.img_folder = path_origin_rand
			image_loaded = st.load_img(args,img_name_path)
			sess.run(assign_op, {placeholder: image_loaded})
			loss_ref_rand = sess.run(loss_total)
			loss_tab += [loss_ref_rand]
		mean = np.mean(loss_tab)
		std = np.std(loss_tab)
		string2 = 'Random method : {:.2e} et std : {:.2e} \n'.format(mean,std)
		print(string2)
		f.write(string2)
		f.flush()
		
	f.close()
	

if __name__ == '__main__':
	CompResult()
	# python LossFct_Test_Gen.py >> /home/nicolas/Style-Transfer/LossFct/results/output.txt
