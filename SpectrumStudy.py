# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:43:37 2019

The goal of this script is to study the outputs textures of the differents
methodes considered for the texture synthesis tasks 

@author: gonthier
"""

import os
import os.path
from scipy import fftpack
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

directory = "./im/References/"
ResultsDir = "./im/"
path_base  = os.path.join('C:\\','Users','gonthier')
if os.path.exists(path_base):
    ResultsDir = os.path.join(path_base,'ownCloud','These Gonthier Nicolas Partage','Images Textures Résultats')
    directory = os.path.join(path_base,'ownCloud','These Gonthier Nicolas Partage','Images Textures References Subset')


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
files = [files[0]]
listofmethod = ['','_SAME_Gatys','_SAME_Gatys_spectrumTFabs_eps10m16','_SAME_autocorr']
for file in files:
    filewithoutext = '.'.join(file.split('.')[:-1])
    print('Image :',filewithoutext)
    filewithoutextreplaced = filewithoutext.replace('_','$\_$')
    for method,nameMethod in zip(listofmethod,listNameMethod):
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
        
        image = io.imread(filename)
        M, N, k = image.shape
        #image = np.mean(image,axis=2) # Mean of the RGB channels
        
#        f, ax = plt.subplots(figsize=(4.8, 4.8))
#        ax.imshow(image)
#        ax.set_title(filewithoutext + method)

        print((M, N), image.dtype) 
        # Take the fourier transform of the image.
        F1 = fftpack.fft2(image)
        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F2 = fftpack.fftshift( F1 )
 
        # Calculate a 2D power spectrum
        F_magnitude = np.abs(F1)
        F_magnitude = fftpack.fftshift(F_magnitude)
        
#        f, ax = plt.subplots(figsize=(4.8, 4.8))
#
#        ax.imshow(F_magnitude, cmap='viridis',
#          extent=(-N // 2, N // 2, -M // 2, M // 2))
#        ax.set_title('Spectrum magnitude '+filewithoutext + method)
        
        f, ax = plt.subplots(figsize=(4.8, 4.8))
        log_F_magnitude = np.log(1 + F_magnitude)
        ax.imshow((log_F_magnitude-np.min(log_F_magnitude))/(np.max(log_F_magnitude)-np.min(log_F_magnitude)),\
#                  cmap='viridis',\
                  extent=(-N // 2, N // 2, -M // 2, M // 2))
        ax.set_title('Log Spectrum magnitude '+filewithoutext + method)
        plt.show()
plt.close('all')
            

