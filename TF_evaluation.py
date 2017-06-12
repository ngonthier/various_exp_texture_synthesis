#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  TF_evaluation.py
#  
#  Copyright 2017 Nicolas <nicolas@Clio>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # 1 to remove info, 2 to remove warning and 3 for all
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn
import scipy 
from PIL import Image

def test_fft2d_of_tensorflow():
	#a = np.mgrid[:5, :5][0]
	#print(a)
	#np_fft_a = np.fft.fft2(a)
	#print(np_fft_a)
	#array([[ 50.0 +0.j        ,   0.0 +0.j        ,   0.0 +0.j        ,
          #0.0 +0.j        ,   0.0 +0.j        ],
       #[-12.5+17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,
          #0.0 +0.j        ,   0.0 +0.j        ],
       #[-12.5 +4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,
          #0.0 +0.j        ,   0.0 +0.j        ],
       #[-12.5 -4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,
            #0.0 +0.j        ,   0.0 +0.j        ],
       #[-12.5-17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,
          #0.0 +0.j        ,   0.0 +0.j        ]])
    # check if np.fft2d of TF.fft2d and NP have the same result

	size = 2
	testimage = np.random.rand(size, size)
	testimage = testimage+0j
	print(type(testimage))

	ft_testimage = np.fft.fft2(testimage)
	print("Test avec 2D element")
	print("Numpy fft")
	print(ft_testimage)
	np_result = np.sum(ft_testimage)
	print(np_result)
		
	sess = tf.Session()
	with sess.as_default():
		tf_ft_testimage = tf.fft2d(testimage)
		tf_result = np.sum(tf_ft_testimage.eval())
		print("Tensorflow fft")
		print(tf_ft_testimage.eval())
		print(tf_result)
		
		tensor3D = tf.constant(np.expand_dims(testimage, axis=2),dtype=tf.complex64)
		print("Tensorflow fft with expand dim")
		print('Dims tensor',tensor3D.shape)
		#tensor3D = tf.constant(y)
		tf_ft_testimage = tf.fft2d(tensor3D)
		tf_result = np.sum(tf_ft_testimage.eval())
		print(tf_ft_testimage.eval())
		print(tf_result)
		
		tensor3D = tf.transpose(tensor3D, [2, 0, 1])
		print("Tensorflow fft with expand dim transpose")
		print('Dims tensor',tensor3D.shape)
		#tensor3D = tf.constant(y)
		tf_ft_testimage = tf.fft2d(tensor3D)
		tf_result = np.sum(tf_ft_testimage.eval())
		print(tf_ft_testimage.eval())
		print(tf_result)
   
def load_img(img_folder,img_name,img_ext='.png'):
	"""
	This function load the image and convert it to a numpy array and do 
	the preprocessing
	"""
	image_path = img_folder + img_name +img_ext
	new_img_ext = img_ext
	try:
		img = scipy.misc.imread(image_path,mode='RGB')  # Float between 0 and 255
	except IOError:
		print("Exception when we try to open the image, try with a different extension format",str(img_ext))
		if(img_ext==".jpg"):
			new_img_ext = ".png"
		elif(img_ext==".png"):
			new_img_ext = ".jpg"
		try:
			image_path = args.img_folder + img_name +new_img_ext # Try the new path
			img = scipy.misc.imread(image_path,mode='RGB')
			print("The image have been sucessfully loaded with a different extension")
		except IOError:
			print("Exception when we try to open the image, we already test the 2 differents extension.")
			raise
	if(len(img.shape)==2):
		img = gray2rgb(img) # Convertion greyscale to RGB
	img = img.astype('float32')
	return(img)
    
def TestShift():
	plt.ion()
	nameImg = 'TilesOrnate0158_1_S'
	path = 'images/'
	path_to_image = path + nameImg + '.png'
	img =  Image.open(path_to_image)
	img = np.asarray(img)
	shift = 50
	#img2 = np.roll(img , shift, axis=0)
	img2 = img.copy()
	for i in range(3):
		img2[:,:,i] = np.roll(img[:,:,i] , shift, axis=0)
	#sess = tf.Session()
	f, ax = plt.subplots(4,2)
	ax[0,0].imshow(img)
	ax[0,0].set_title("Image")
	ax[0,1].imshow(img2)
	ax[0,1].set_title("Image shifte")
	ft_testimage = np.fft.fftshift(np.fft.fft2(img))
	ft_testimage2 = np.fft.fftshift(np.fft.fft2(img2))
	module1 = np.abs(ft_testimage)
	module2 = np.abs(ft_testimage2)
	diff_module = module1-module2
	phase1 = np.angle(ft_testimage)
	phase2 = np.angle(ft_testimage2)
	diff_phase = phase1 - phase2
	#diff_phase = (diff_phase + np.pi) % 2*np.pi
	for i in range(3):
		im = ax[1+i,0].matshow(diff_module[:,:,i], cmap='jet')
		ax[1+i,0].set_title("Module Diff")
		plt.colorbar(im, cax=ax[1+i,0])
		im =ax[1+i,1].matshow(diff_phase[:,:,i], cmap='jet')
		ax[1+i,1].set_title("Phase Diff")
		plt.colorbar(im, cax=ax[1+i,1])
	plt.suptitle("Difference")
	f.tight_layout()
	
	f, ax = plt.subplots(4,2)
	ax[0,0].imshow(img)
	ax[0,0].set_title("Image")
	ax[0,1].imshow(img2)
	ax[0,1].set_title("Image shifte")
	ft_testimage = np.fft.fftshift(np.fft.fft2(img))
	ft_testimage2 = np.fft.fftshift(np.fft.fft2(img2))
	module1 = np.abs(ft_testimage)
	module2 = np.abs(ft_testimage2)
	diff_module = np.roll(module1, shift, axis=0)-module2
	phase1 = np.angle(ft_testimage)
	phase2 = np.angle(ft_testimage2)
	diff_phase = np.roll(phase1, shift, axis=0) - phase2
	diff_phase = (diff_phase + np.pi) % 2*np.pi
	for i in range(3):
		im = ax[1+i,0].matshow(diff_module[:,:,i], cmap='jet')
		ax[1+i,0].set_title("Module Diff")
		plt.colorbar(im, cax=ax[1+i,0])
		im =ax[1+i,1].matshow(diff_phase[:,:,i], cmap='jet')
		ax[1+i,1].set_title("Phase Diff")
		plt.colorbar(im, cax=ax[1+i,1])
	plt.suptitle("Difference shiftee")
	f.tight_layout()
		
	f, ax = plt.subplots(4,2)
	ax[0,0].imshow(img)
	ax[0,0].set_title("Image")
	ax[0,1].imshow(img2)
	ax[0,1].set_title("Image shifte")
	ft_testimage = np.fft.fft2(img)
	ft_testimage2 = np.fft.fft2(img2)
	module1 = np.abs(ft_testimage)
	module2 = np.abs(ft_testimage2)
	diff_module = module1-module2
	phase1 = np.angle(ft_testimage)
	phase2 = np.angle(ft_testimage2)
	for i in range(3):
		im = ax[1+i,0].matshow(np.roll(phase1[:,:,i], shift, axis=0), cmap='jet')
		plt.colorbar(im, cax=ax[1+i,0])
		im =ax[1+i,1].matshow(phase2[:,:,i], cmap='jet')
		plt.colorbar(im, cax=ax[1+i,1])
	plt.suptitle("Phase")
	f.tight_layout()	

	
	f, ax = plt.subplots(4,2)
	ax[0,0].imshow(img)
	ax[0,0].set_title("Image")
	ax[0,1].imshow(img2)
	ax[0,1].set_title("Image shifte")
	ft_testimage = np.fft.fft2(img)
	ft_testimage2 = np.fft.fft2(img2)
	module1 = np.abs(ft_testimage)
	module2 = np.abs(ft_testimage2)
	diff_module = module1-module2
	phase1 = np.angle(ft_testimage)
	phase2 = np.angle(ft_testimage2)
	for i in range(3):
		im = ax[1+i,0].matshow(np.roll(module1[:,:,i], shift, axis=0), cmap='jet')
		plt.colorbar(im, cax=ax[1+i,0])
		im =ax[1+i,1].matshow(module2[:,:,i], cmap='jet')
		plt.colorbar(im, cax=ax[1+i,1])
	plt.suptitle("Module")
	f.tight_layout()
	
	
	
	plt.show()
	input("Wait input")
	

if __name__ == '__main__':
    #test_fft2d_of_tensorflow()
    TestShift()
