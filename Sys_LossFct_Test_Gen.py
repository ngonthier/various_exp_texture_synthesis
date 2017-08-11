#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 1

Try to launch the script for the fft and rfft cases

@author: nicolas
"""

from Arg_Parser import get_parser_args 
import sys
from utils import get_list_of_images,do_mkdir
import os


def generation_Texture_LossFct():
	path_origin = '/home/nicolas/Style-Transfer/dataImages/'
	path_output = '/home/nicolas/Style-Transfer/LossFct/resultsDiff_loss_function/'
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
	losses_to_test = [['autocorr_rfft'],['autocorr'],['Lp'],['texture'],['phaseAlea'],['phaseAleaSimple'],['autocorr_rfft','texture'],['texture','spectrum'],['autocorr_rfft','spectrum'],['autocorrLog'],['variance']]
	losses_to_test = [['variance'],['phaseAlea']]
	config_layers_tab = ['FirstConvs','PoolConfig']
	config_layers_tab = ['PoolConfig']
	# Il manque les phaseAlea pour PoolConfig mais aussi variance
	list_img = get_list_of_images(path_origin)
	
	Stop = False
	
	for config_layers in config_layers_tab:
		for loss in losses_to_test:
			
			if loss[0] == 'nmoments':
				for n in n_list: 
					if(config_layers=='FirstConvs') and (n > 2):
						loss = ['nmoments_reduce']
					for name_img in list_img:
						name_img_wt_ext,_ = name_img.split('.')
						img_folder = path_origin
						output_img_name = name_img_wt_ext + '_' + config_layers
						loss_str = ''
						for loss_item in loss:
							output_img_name += '_' + loss_item + '_' + str(n)
							loss_str += loss_item + ' '
						main_command = 'python Main_Style_Transfer.py --verbose --max_iter ' + str(max_iter) +' --print_iter ' + str(print_iter) + ' --start_from_noise ' + str(start_from_noise) + ' --init_noise_ratio ' + str(init_noise_ratio) +  ' --img_folder ' + path_origin + ' --output_img_name ' + output_img_name+ ' --img_output_folder ' + path_output + ' --style_img_name ' + name_img_wt_ext + ' --content_img_name ' + name_img_wt_ext+ ' --loss ' + loss_str + ' --init ' + str(init) + ' --init_range ' + str(init_range)  +' --n ' + str(n) + ' --p ' + str(p) + ' --clipping_type ' + clipping_type +  ' --vgg_name ' + vgg_name + ' --config_layers ' + config_layers
						print(main_command)
						try:
							if not(Stop): os.system(main_command)
						except:
							raise
							Stop = True
							sys.exit(0)

			else:
				n = 4
				for name_img in list_img:
					name_img_wt_ext,_ = name_img.split('.')
					img_folder = path_origin
					output_img_name = name_img_wt_ext + '_' + config_layers
					loss_str = ''
					for loss_item in loss:
						output_img_name += '_' + loss_item
						loss_str += loss_item + ' '
					main_command = 'python Main_Style_Transfer.py --verbose --max_iter ' + str(max_iter) +' --print_iter ' + str(print_iter) + ' --start_from_noise ' + str(start_from_noise) + ' --init_noise_ratio ' + str(init_noise_ratio) +  ' --img_folder ' + path_origin + ' --output_img_name ' + output_img_name+ ' --img_output_folder ' + path_output + ' --style_img_name ' + name_img_wt_ext + ' --content_img_name ' + name_img_wt_ext+ ' --loss ' + loss_str + ' --init ' + str(init) + ' --init_range ' + str(init_range) + ' --n ' + str(n) + ' --p ' + str(p) +  ' --clipping_type ' + clipping_type + ' --vgg_name ' + vgg_name + ' --config_layers ' + config_layers
					print(main_command)
					try:
						if not(Stop): os.system(main_command)
					except:
						raise
						Stop = True
						sys.exit(0)

if __name__ == '__main__':
	generation_Texture_LossFct()
