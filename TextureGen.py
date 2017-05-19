#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 2017

This script have the goal to generate texture with different seetings

@author: nicolas
"""

from Arg_Parser import get_parser_args 
import utils

def main():
	parser = get_parser_args()
	style_img_name = "StarryNight"
	content_img_name = "Louvre"
	max_iter = 1000
	print_iter = 100
	start_from_noise = 1 # True
	init_noise_ratio = 1.0
	content_strengh = 0.001
	optimizer = 'lbfgs'
	learning_rate = 10 # 10 for adam and 10**(-10) for GD
	maxcor = 10
	sampling = 'up'
	# In order to set the parameter before run the script
	parser.set_defaults(style_img_name=style_img_name,max_iter=max_iter,
		print_iter=print_iter,start_from_noise=start_from_noise,
		content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,
		content_strengh=content_strengh,optimizer=optimizer,maxcor=maxcor,
		learning_rate=learning_rate,sampling=sampling)
	args = parser.parse_args()
	print(args)
	#pooling_type='avg'
	#padding='SAME'
	#style_transfer(args,pooling_type,padding)

if __name__ == '__main__':
	main()
