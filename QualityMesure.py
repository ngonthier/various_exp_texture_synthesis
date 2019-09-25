#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  The goal of this script is to evaluate a quality metrics between the 
# synthetic images and the reference one
# The idea is to compute the distance of the distribution of the images 
# to do so we can use a KL distance at 3 different scale or 
# but maybe need to use a small number of bins for the histogram
# or compute alpha and etha then we have an explicit formula up to KL 
#  
# it is a quality measure after the synthesis, this quality measure is not
# used during synthesis process
#
#  Copyright 2019 gonthier <gonthier@Morisot>

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
import os
import os.path
from scipy import fftpack
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.color import rgb2hsv
import pathlib

directory = "./im/References/"
ResultsDir = "./im/"
path_base  = os.path.join('C:\\','Users','gonthier')
if not(os.path.exists(path_base)):
	path_base  = os.path.join(os.sep,'media','gonthier','HDD')
if os.path.exists(path_base):
	ResultsDir = os.path.join(path_base,'ownCloud','These Gonthier Nicolas Partage','Images Textures Résultats')
	directory = os.path.join(path_base,'owncloud','These Gonthier Nicolas Partage','Images Textures References Subset')
else:
	print(path_base,'not found')
	raise(NotImplementedError)


extension = ".png"
files = [file for file in os.listdir(directory) if file.lower().endswith(extension)]
listofmethod =['','_SAME_Gatys','_EfrosLeung','_EfrosFreeman',
'_SAME_Gatys_spectrumTFabs_eps10m16','MultiScale_o5_l3_8_psame',
'_DCor','_SAME_Gatys_MSSInit','_SAME_Gatys_spectrumTFabs_eps10m16_MSSInit'
,'_SAME_autocorr','_SAME_autocorr_MSSInit','1','2','3','4']
# Si vous voulez afficher plus de choses
#['','_SAME_Gatys','_EfrosLeung','_EfrosFreeman','_SAME_Gatys_spectrum','_SAME_Gatys_spectrumTFabs_eps10m16','MultiScale_o5_l3_8_psame','_DCor','_Gatys_Gang','_SAME_Gatys_MSSInit','_SAME_Gatys_spectrum_MSSInit','_SAME_Gatys_spectrumTFabs_eps10m16_MSSInit','_Gatys_Gang_MSInit','_SAME_autocorr','_SAME_autocorr_MSSInit','_SAME_phaseAlea_MSSInit']
# ,'_SAME_texture_spectrum_MSSInit','_SAME_phaseAlea'
listNameMethod = ['Reference','Gatys','EfrosLeung','EfrosFreeman','Gatys + Spectrum TF','Snelgorove','Deep Corr','Gatys + MSInit','Gatys + Spectrum TF + MSInit','Autocorr','Autocorr + MSInit','1 : OT Galerne Leclair','2 :Guisong method','3 : Tartavel','GAN Zalando Jetchev']
# listNameMethod = ['Reference','Gatys','EfrosLeung','EfrosFreeman','Gatys + Spectrum TF','Gatys + Spectrum TF eps10m16','Snelgorove','Deep Corr','Gang Spectrum Code','Gatys + MSInit','Gatys + Spectrum TF + MSInit','Gatys + Spectrum TF eps10m16 + MSInit','Gang code for MSInit','Autocorr','Autocorr + MSInit','PhaseAlea + MSInit']
#'Gatys + Spectrum + multi-scale Init','PhaseAlea'

listofmethod = ['','_SAME_Gatys','_SAME_Gatys_spectrumTFabs_eps10m16','_SAME_autocorr']
listofmethod = ['','_SAME_Gatys']
listNameMethod = ['Reference','Gatys','Gatys + Spectrum TF','Autocorr']

cmap='viridis' 
#cmap='plasma' 
files_short = files
#files_short = [files[-1]]
files_short = ['TexturesCom_FloorsCheckerboard0046_4_seamless_S_1024.png']

def kl(p, q):
	p = np.asarray(p, dtype=np.float)
	q = np.asarray(q, dtype=np.float)

	return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=10, sigma=1):
	ahist, bhist = (np.histogram(a, bins=nbins)[0],
					np.histogram(b, bins=nbins)[0])

	asmooth, bsmooth = (gaussian_filter(ahist, sigma),
						gaussian_filter(bhist, sigma))
	return kl(asmooth, bsmooth)
	
def hist_kl_distance(a, b, nbins=10):
	ahist, bhist = (np.histogram(a, bins=nbins)[0],
					np.histogram(b, bins=nbins)[0])
	return kl(ahist, bhist)

def main(args):
	"""
	compute the quality measure
	"""
	# path_output = os.path.join('Spectrum','Crops',str(dim))
	# pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
	number_of_scale = 3
	
	for file in files_short:
		filewithoutext = '.'.join(file.split('.')[:-1])
		print('Image :',filewithoutext)
		filewithoutextreplaced = filewithoutext.replace('_','$\_$')
		dict_imgs = {}
		
		# Load the images
		for i,zipped in enumerate(zip(listofmethod,listNameMethod)):
			method,nameMethod = zipped
			if method =='':
				stringname = os.path.join(directory,filewithoutext + method)
			else:
				stringname = os.path.join(ResultsDir,filewithoutext,filewithoutext + method)
			stringnamepng = stringname + '.png'
			stringnamejpg = stringname + '.jpg'
			if os.path.isfile(stringnamepng):
				filename = stringnamepng
			elif os.path.isfile(stringnamejpg):
				filename = stringnamejpg
			else:
				print('The image does not exist')
			
			imagergb = io.imread(filename)
			dict_imgs[nameMethod] = imagergb
		
		# Compute the metrics between the reference and the images
		for method,nameMethod in zip(listofmethod,listNameMethod):
			if nameMethod=='Reference': # Reference one
				pass
			ref_img = dict_imgs['Reference']
			syn_img = dict_imgs[nameMethod]
			for s in range(number_of_scale): # scale
				if s==0:
					ref_img_s = ref_img
					syn_img_s = syn_img
				else:
					ref_img_s = resize(ref_img, (ref_img_s.shape[0] // (2*i), ref_img.shape[1] // (2*i)),
					   anti_aliasing=True)
					syn_img_s = resize(syn_img, (syn_img.shape[0] // (2*i), syn_img.shape[1] //(2*i)),
					   anti_aliasing=True)
				
				for c in range(3): # color loop
					ref_img_s_c = ref_img_s[:,:,c].ravel()
					syn_img_s_c = syn_img_s[:,:,c].ravel()
					kl_s_c = hist_kl_distance(ref_img_s_c, syn_img_s_c, nbins=10)
					print(nameMethod,s,c,'kl = ',kl_s_c)
			
			

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
