#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:32:49 2017

Le but de ce code est de comprendre pourquoi on se retrouve avec des nan 
pour certaines valeurs de moments ou des std = 0

@author: nicolas
"""

import scipy
import numpy as np
import tensorflow as tf
import Style_Transfer as st
from Arg_Parser import get_parser_args 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
from tensorflow.python.framework import dtypes
import matplotlib.gridspec as gridspec
import math
from skimage import exposure
from PIL import Image

# Name of the 19 first layers of the VGG19
VGG19_LAYERS = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	'relu5_3', 'conv5_4', 'relu5_4'
)

VGG19_LAYERS_INDICES = {'conv1_1' : 0,'conv1_2' : 2,'conv2_1' : 5,'conv2_2' : 7,
	'conv3_1' : 10,'conv3_2' : 12,'conv3_3' : 14,'conv3_4' : 16,'conv4_1' : 19,
	'conv4_2' : 21,'conv4_3' : 23,'conv4_4' : 25,'conv5_1' : 28,'conv5_2' : 30,
	'conv5_3' : 32,'conv5_4' : 34}

VGG19_LAYERS_INTEREST = (
    'conv1_1','conv2_1', 'conv3_1'
)

def estimate_std(args):
	""" Compute mean and std of tehe activation maps """
	#sns.set_style("white")
	#path = 'Results/STD/' 
	#pltname = path+args.style_img_name +'_std.pdf'
	#pp = PdfPages(pltname)
	image_style_path = args.img_folder + args.style_img_name + args.img_ext
	image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
	
	vgg_layers = st.get_vgg_layers()
	net = st.net_preloaded(vgg_layers, image_style) # net for the style image
	sess = tf.Session()
	sess.run(net['input'].assign(image_style))
	for layer in VGG19_LAYERS:
		list_of_stdNull = []
		#print(layer)
		a = net[layer].eval(session=sess)
		a = a[0]
		h,w,number_of_features = a.shape
		a_reshaped = np.reshape(a,(h*w,number_of_features))
		#print(h*w)
		for i in range(number_of_features):
			#print(i)
			samples = a_reshaped[:,i]
			mean = np.mean(samples)
			std = np.std(samples)
			if(std==0):
				list_of_stdNull += [i]
		if(not(len(list_of_stdNull)==0)):
			print("STD == 0 for ",layer," for ",len(list_of_stdNull)," features.")
			print(list_of_stdNull)
				#print(a[:,:,i].shape)
				#f = plt.figure()
				#plt.matshow(a[:,:,i])
				#titre =  layer + ' STD == 0 for features number '+str(i)
				#plt.suptitle(titre)
				#plt.savefig(pp, format='pdf')
				#plt.close()
	#pp.close()
	#plt.clf()
	
	### For pebbles : 
	#STD == 0 for  relu5_1  for  6  features.
#[70, 72, 483, 491, 498, 511]
#STD == 0 for  relu5_2  for  25  features.
#[38, 90, 92, 104, 119, 154, 196, 203, 224, 280, 293, 294, 323, 326, 349, 362, 368, 398, 399, 400, 408, 418, 437, 449, 454]
#STD == 0 for  relu5_3  for  29  features.
#[82, 101, 119, 143, 154, 191, 205, 238, 247, 256, 265, 273, 281, 292, 298, 301, 320, 333, 339, 367, 370, 381, 405, 427, 456, 460, 480, 492, 501]
#STD == 0 for  relu5_4  for  151  features.
#[4, 5, 6, 8, 9, 11, 13, 15, 16, 27, 28, 31, 33, 37, 38, 39, 46, 47, 49, 51, 53, 58, 61, 65, 66, 68, 69, 77, 80, 83, 88, 90, 93, 97, 99, 101, 107, 108, 111, 114, 115, 118, 121, 126, 132, 135, 139, 140, 141, 142, 143, 144, 145, 148, 159, 161, 162, 163, 165, 167, 169, 172, 173, 180, 186, 188, 192, 196, 197, 198, 202, 203, 205, 207, 209, 222, 228, 232, 235, 236, 242, 246, 258, 262, 268, 269, 274, 277, 278, 280, 285, 286, 296, 297, 300, 304, 307, 308, 312, 313, 318, 321, 324, 326, 327, 329, 334, 337, 345, 353, 360, 361, 364, 366, 368, 369, 370, 372, 374, 385, 389, 390, 411, 412, 416, 417, 421, 424, 426, 432, 435, 436, 437, 443, 446, 448, 450, 456, 458, 462, 475, 477, 484, 485, 490, 493, 496, 497, 498, 499, 501]
	
	### For BrickSmallBrown0293_1_S
	#STD == 0 for  relu1_1  for  4  features.
#[36, 39, 50, 61]
#STD == 0 for  relu2_1  for  2  features.
#[49, 123]
#STD == 0 for  relu3_1  for  1  features.
#[77]
#STD == 0 for  relu3_2  for  2  features.
#[100, 249]
#STD == 0 for  relu3_4  for  1  features.
#[202]
#STD == 0 for  pool3  for  1  features.
#[202]
#STD == 0 for  relu4_1  for  48  features.
#[13, 16, 42, 47, 66, 68, 75, 83, 100, 113, 122, 135, 165, 179, 194, 199, 209, 213, 225, 231, 235, 243, 246, 259, 271, 274, 285, 298, 312, 314, 345, 354, 362, 379, 420, 450, 459, 460, 466, 478, 479, 487, 491, 492, 496, 498, 506, 511]
#STD == 0 for  relu4_2  for  34  features.
#[29, 41, 50, 70, 84, 85, 89, 92, 95, 114, 124, 125, 150, 162, 180, 233, 259, 261, 281, 291, 298, 314, 323, 333, 334, 345, 348, 366, 411, 423, 434, 486, 497, 503]
#STD == 0 for  relu4_3  for  22  features.
#[0, 40, 73, 116, 136, 148, 167, 173, 191, 230, 257, 288, 322, 350, 358, 393, 418, 430, 435, 466, 501, 511]
#STD == 0 for  relu4_4  for  53  features.
#[5, 18, 30, 31, 38, 43, 54, 60, 62, 70, 79, 101, 125, 126, 135, 146, 167, 174, 176, 178, 188, 195, 234, 242, 253, 254, 276, 278, 286, 304, 306, 308, 327, 328, 334, 336, 347, 348, 353, 366, 367, 379, 382, 388, 401, 417, 420, 424, 425, 446, 470, 490, 504]
#STD == 0 for  pool4  for  53  features.
#[5, 18, 30, 31, 38, 43, 54, 60, 62, 70, 79, 101, 125, 126, 135, 146, 167, 174, 176, 178, 188, 195, 234, 242, 253, 254, 276, 278, 286, 304, 306, 308, 327, 328, 334, 336, 347, 348, 353, 366, 367, 379, 382, 388, 401, 417, 420, 424, 425, 446, 470, 490, 504]
#2017-08-10 17:14:00.079514: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.14GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
#2017-08-10 17:14:00.079586: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.08GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
#STD == 0 for  relu5_1  for  77  features.
#[2, 6, 10, 17, 25, 29, 34, 41, 46, 48, 49, 53, 68, 70, 77, 81, 83, 93, 95, 100, 123, 124, 130, 133, 134, 144, 146, 148, 153, 171, 174, 175, 184, 195, 212, 226, 230, 249, 250, 251, 253, 255, 269, 280, 281, 291, 301, 302, 305, 307, 309, 311, 316, 320, 327, 331, 347, 351, 367, 379, 386, 404, 414, 422, 438, 455, 456, 457, 460, 491, 492, 493, 496, 503, 506, 507, 511]
#STD == 0 for  relu5_2  for  104  features.
#[6, 14, 15, 22, 27, 28, 34, 35, 45, 46, 48, 59, 66, 67, 68, 73, 74, 82, 92, 104, 111, 118, 135, 140, 146, 151, 152, 153, 154, 156, 160, 165, 169, 170, 173, 174, 184, 186, 189, 192, 193, 195, 196, 201, 214, 217, 219, 227, 232, 235, 238, 241, 249, 251, 252, 264, 270, 273, 280, 291, 293, 294, 297, 300, 301, 302, 303, 304, 308, 309, 310, 312, 320, 322, 323, 339, 365, 366, 368, 373, 379, 380, 382, 386, 388, 398, 403, 412, 413, 415, 425, 427, 434, 437, 442, 447, 449, 479, 485, 491, 497, 501, 502, 507]
#STD == 0 for  relu5_3  for  142  features.
#[5, 6, 7, 9, 11, 14, 17, 18, 19, 23, 24, 26, 29, 39, 40, 43, 51, 52, 56, 59, 63, 68, 73, 75, 79, 81, 100, 108, 109, 112, 121, 130, 131, 132, 134, 135, 141, 142, 144, 146, 147, 152, 154, 156, 158, 159, 164, 167, 168, 171, 174, 178, 180, 183, 188, 189, 190, 191, 192, 193, 195, 196, 197, 201, 202, 205, 211, 219, 221, 225, 228, 229, 232, 249, 252, 253, 256, 260, 266, 282, 296, 299, 300, 301, 313, 314, 317, 321, 322, 324, 325, 329, 331, 335, 340, 350, 352, 360, 365, 367, 368, 369, 374, 375, 379, 380, 384, 385, 389, 391, 397, 399, 400, 401, 403, 404, 405, 407, 411, 415, 422, 429, 438, 444, 450, 456, 458, 464, 467, 475, 477, 478, 481, 487, 490, 494, 497, 498, 500, 505, 506, 507]
#STD == 0 for  relu5_4  for  287  features.
#[0, 1, 2, 3, 5, 8, 9, 11, 13, 14, 16, 18, 20, 21, 23, 24, 29, 30, 31, 33, 34, 36, 38, 39, 40, 45, 47, 48, 50, 53, 60, 63, 64, 65, 66, 67, 68, 72, 74, 75, 78, 80, 81, 84, 85, 87, 90, 91, 92, 94, 97, 98, 100, 101, 102, 103, 104, 105, 106, 109, 110, 111, 114, 118, 120, 121, 122, 125, 126, 127, 129, 132, 134, 135, 137, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149, 151, 152, 153, 154, 156, 157, 158, 160, 161, 164, 165, 167, 168, 169, 170, 171, 173, 177, 179, 181, 182, 185, 189, 190, 193, 195, 196, 197, 198, 199, 200, 202, 203, 206, 208, 209, 213, 214, 218, 219, 220, 222, 224, 225, 226, 227, 228, 231, 233, 234, 235, 236, 237, 238, 241, 243, 245, 247, 251, 254, 260, 261, 263, 265, 267, 268, 269, 270, 274, 275, 277, 278, 279, 280, 281, 284, 285, 286, 287, 288, 289, 290, 293, 294, 295, 297, 301, 302, 303, 304, 305, 306, 310, 312, 313, 318, 320, 323, 324, 327, 329, 330, 333, 334, 336, 337, 338, 341, 348, 349, 351, 353, 354, 356, 357, 359, 360, 361, 362, 363, 364, 368, 370, 371, 372, 374, 375, 376, 377, 378, 380, 383, 385, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 399, 400, 405, 408, 410, 411, 412, 413, 414, 415, 416, 419, 420, 421, 422, 423, 424, 426, 427, 428, 429, 430, 431, 432, 434, 436, 437, 439, 443, 446, 447, 448, 449, 452, 455, 457, 462, 464, 467, 472, 473, 475, 476, 477, 478, 479, 485, 487, 488, 489, 490, 496, 497, 498, 499, 500, 501, 509, 511]
	
	return(0)

def main(name=None):
	"""
	Estimate the distribution of the distribution 
	"""
	parser = get_parser_args()
	if(name==None):
		style_img_name = "BrickSmallBrown0293_1_S"
	else:
		style_img_name = name
	parser.set_defaults(style_img_name=style_img_name)
	args = parser.parse_args()
	estimate_std(args)
	
if __name__ == '__main__':
	main()
	
