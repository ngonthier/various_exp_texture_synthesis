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
import matplotlib.gridspec as gridspec
from skimage.color import rgb2hsv
import pathlib

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

listofmethod = ['','_SAME_Gatys','_SAME_Gatys_spectrumTFabs_eps10m16','_SAME_autocorr']
listofmethod = ['','_SAME_Gatys']
listNameMethod = ['Reference','Gatys','Gatys + Spectrum TF','Autocorr']

cmap='viridis' 
#cmap='plasma' 
files_short = files
#files_short = [files[-1]]
files_short = ['TexturesCom_FloorsCheckerboard0046_4_seamless_S_1024.png']

def crop(im, height, width):
    #im = Image.open(input)
    list_crops = [] 
    shapes = im.shape
    if len(shapes)==2:
        imgwidth, imgheight = im.shape
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                a = im[i:i+height,j:j+width]
                list_crops += [a]
    else:
        imgwidth, imgheight,channel = im.shape
    
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
    #            box = (j, i, j+width, i+height)
                a = im[i:i+height,j:j+width,:]
                list_crops += [a]
    np_crops = np.stack(list_crops)
    return(np_crops)

def plot_Spectrum_crops(dim=256):
    """
    Plot the mean of the crop of size of the images
    """
    path_output = os.path.join('Spectrum','Crops',str(dim))
    color_tab  = ['RGB','HSV']
    pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
    for file in files_short:
        
        for color_type in color_tab:
    
            f = plt.figure(constrained_layout=True)  
            number_img_large = 4
            num_methods = len(listofmethod)
            numberFFT =4
            width_ratios = [100]*numberFFT
            gs00 = gridspec.GridSpec(num_methods, numberFFT, width_ratios=width_ratios,
                                      wspace=0.05,hspace=0.05)
            axes = []
            for j in range(num_methods*numberFFT):
                ax = plt.subplot(gs00[j])
                axes += [ax]
    
            filewithoutext = '.'.join(file.split('.')[:-1])
            print('Image :',filewithoutext,color_type)
            filewithoutextreplaced = filewithoutext.replace('_','$\_$')
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
                if color_type=='HSV':
                    imagehsv= rgb2hsv(imagergb)
                

                if color_type=='RGB':
                    crops = crop(imagergb,dim,dim)
                if color_type=='HSV':
                    crops = crop(imagehsv,dim,dim)
        
                for r in range(numberFFT):
                    
                    if r ==3 :
                        Amplitude =  255
                        im_mean = np.mean(imagergb,axis=-1)
                        image = crop(im_mean,dim,dim)
                    else:
                        if color_type=='RGB': 
                            image = crops[:,:,:,r]
                            Amplitude =255
                        if color_type=='HSV':
                            image = crops[:,:,:,r]
                            if r ==0:
                                Amplitude = 360
                            elif r==1 or r==2:
                                Amplitude = 1.
                    ax = axes[i*numberFFT+r]
                    #image = np.mean(image,axis=-1)
                    shapes = image.shape
                    if len(shapes)==2:
                        M, N = shapes
                    else:
                        M, N,k =shapes
                    num_pixels = dim**2
            #        print(shapes, image.dtype) 
                    #image = np.mean(image,axis=2) # Mean of the RGB channels
                    
            #        f, ax = plt.subplots(figsize=(20, 20))
            #        ax.imshow(image,cmap='Greys')
            #        ax.set_title(filewithoutext + method)
            
                    
                    # Take the fourier transform of the image.
                    F1 = fftpack.fft2(image)
                    
                    #print('F1.shape',F1.shape)
                    # Now shift the quadrants around so that low spatial frequencies are in
                    # the center of the 2D fourier transformed image.
#                    F2 = fftpack.fftshift( F1 )
             
                    # Calculate a 2D power spectrum
                    F_magnitude = np.abs(F1)
                    F_magnitude = np.mean(F_magnitude,axis=-1)
                    assert(num_pixels*Amplitude >= np.max(F_magnitude))
                    #F_magnitude[0,0] = np.median(F_magnitude)
                    F_magnitude = fftpack.fftshift(F_magnitude)
    
            #        f, ax = plt.subplots(figsize=(4.8, 4.8))
            #
            #        ax.imshow(F_magnitude, cmap=cmap,
            #          extent=(-N // 2, N // 2, -M // 2, M // 2))
            #        ax.set_title('Spectrum magnitude '+filewithoutext + method)
                    
                    #f, ax = plt.subplots(figsize=(20, 20))
                    
                    log_F_magnitude = np.log(1 + F_magnitude)
                    F_for_print = (255 *(log_F_magnitude/np.log(1+num_pixels*Amplitude))).astype(np.uint8)
                    #F_for_print = (255 *(log_F_magnitude-np.min(log_F_magnitude))/(np.max(log_F_magnitude)-np.min(log_F_magnitude))).astype(np.uint8)
                    
            #        f_bounded = 20 * np.log(F_magnitude)
            #        f_img = 255 * f_bounded / np.max(f_bounded)
            #        f_img = f_img.astype(np.uint8)
            #        F_for_print = f_img
                    
                    # Besoin de la meme echelle pour faire des comparaisons !
            
                    ax.imshow(F_for_print,\
                              cmap=cmap, extent=(-dim // 2, dim// 2, -dim // 2, dim // 2))
                    #ax.axis('off')
                    if i==0:
                        if r==0:
                            if color_type=='RGB':
                                ax.set_title('Red')
                            if color_type=='HSV':
                                ax.set_title('H')
                        elif r==1:
                            if color_type=='RGB':
                                ax.set_title('Green')
                            if color_type=='HSV':
                                ax.set_title('S')
                        elif r==2:
                            if color_type=='RGB':
                                ax.set_title('Blue')
                            if color_type=='HSV':
                                ax.set_title('V')
                        elif r==3:
                            ax.set_title('Mean')
                    if r==0:
                        if nameMethod=='':
                            ax.set_ylabel('Ref')
                        else:
                            ax.set_ylabel(nameMethod.replace('_',' '))
                            
                    ff, axf = plt.subplots(figsize=(20, 20))
                    axf.imshow(F_for_print,cmap=cmap, extent=(-dim // 2, dim // 2, -dim// 2, dim // 2))
                    if r==3:
                        ext_str = 'Mean'
                    else:
                        ext_str = color_type[r]
                        
                    axf.set_title('Log Spectrum magnitude '+filewithoutext + method + ' '+ext_str)
                    fname = os.path.join(path_output,filewithoutext +'_'+color_type+str(r)+'_' + method + '_'+ext_str+'_'+str(dim)+'.png')
    #                plt.show()
                    if not(color_type=='HSV' and r==3):
                        plt.savefig(fname, dpi=300)
                    plt.close(ff)
                                  
                        
                    #ax.set_title('Log Spectrum magnitude '+filewithoutext + method)
            plt.suptitle('Log Spectrum magnitude '+filewithoutext+' '+color_type)
            fname = os.path.join(path_output,filewithoutext + '_'+color_type+'_'+str(dim)+'.png')
            plt.savefig(fname, dpi=300)
            plt.show()
            plt.close(f)
        
            
    plt.close('all')



def plot_Spectrum_full_image(not_close=False):
    color_tab  = ['RGB','HSV']
    color_tab  = ['RGB']
    for file in files_short:
        
        for color_type in  color_tab:
    
            f = plt.figure(constrained_layout=True)  
            number_img_large = 4
            num_methods = len(listofmethod)
            numberFFT =4
            width_ratios = [100]*numberFFT
            gs00 = gridspec.GridSpec(num_methods, numberFFT, width_ratios=width_ratios,
                                      wspace=0.05,hspace=0.05)
            axes = []
            for j in range(num_methods*numberFFT):
                ax = plt.subplot(gs00[j])
                axes += [ax]
    
            filewithoutext = '.'.join(file.split('.')[:-1])
            print('Image :',filewithoutext,color_type)
            filewithoutextreplaced = filewithoutext.replace('_','$\_$')
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
                if color_type=='HSV':
                    imagehsv= rgb2hsv(imagergb)
        
                for r in range(numberFFT):
                    
                    if r ==3 :
                        Amplitude =  255
                        image = np.mean(imagergb,axis=-1)
                    else:
                        if color_type=='RGB': 
                            image = imagergb[:,:,r]
                            Amplitude =255
                        if color_type=='HSV':
                            image = imagehsv[:,:,r]
                            if r ==0:
                                Amplitude = 360
                            elif r==1 or r==2:
                                Amplitude = 1.
                    ax = axes[i*numberFFT+r]
                    #image = np.mean(image,axis=-1)
                    shapes = image.shape
                    if len(shapes)==2:
                        M, N = shapes
                    else:
                        M, N,k =shapes
                    num_pixels = M*N
            #        print(shapes, image.dtype) 
                    #image = np.mean(image,axis=2) # Mean of the RGB channels
                    
            #        f, ax = plt.subplots(figsize=(20, 20))
            #        ax.imshow(image,cmap='Greys')
            #        ax.set_title(filewithoutext + method)
            
                    
                    # Take the fourier transform of the image.
                    F1 = fftpack.fft2(image)
                    #print('F1.shape',F1.shape)
                    # Now shift the quadrants around so that low spatial frequencies are in
                    # the center of the 2D fourier transformed image.
#                    F2 = fftpack.fftshift( F1 )
             
                    # Calculate a 2D power spectrum
                    F_magnitude = np.abs(F1)
                    assert(num_pixels*Amplitude >= np.max(F_magnitude))
                    F_magnitude[0,0] = np.median(F_magnitude)
                    F_magnitude = fftpack.fftshift(F_magnitude)
    
            #        f, ax = plt.subplots(figsize=(4.8, 4.8))
            #
            #        ax.imshow(F_magnitude, cmap=cmap,
            #          extent=(-N // 2, N // 2, -M // 2, M // 2))
            #        ax.set_title('Spectrum magnitude '+filewithoutext + method)
                    
                    #f, ax = plt.subplots(figsize=(20, 20))
                    
                    log_F_magnitude = np.log(1 + F_magnitude)
                    F_for_print = (255 *(log_F_magnitude/np.log(1+num_pixels*Amplitude))).astype(np.uint8)
                    #F_for_print = (255 *(log_F_magnitude-np.min(log_F_magnitude))/(np.max(log_F_magnitude)-np.min(log_F_magnitude))).astype(np.uint8)
                    
            #        f_bounded = 20 * np.log(F_magnitude)
            #        f_img = 255 * f_bounded / np.max(f_bounded)
            #        f_img = f_img.astype(np.uint8)
            #        F_for_print = f_img
                    
                    # Besoin de la meme echelle pour faire des comparaisons !
            
                    ax.imshow(F_for_print,\
                              cmap=cmap, extent=(-N // 2, N // 2, -M // 2, M // 2))
                    #ax.axis('off')
                    if i==0:
                        if r==0:
                            if color_type=='RGB':
                                ax.set_title('Red')
                            if color_type=='HSV':
                                ax.set_title('H')
                        elif r==1:
                            if color_type=='RGB':
                                ax.set_title('Green')
                            if color_type=='HSV':
                                ax.set_title('S')
                        elif r==2:
                            if color_type=='RGB':
                                ax.set_title('Blue')
                            if color_type=='HSV':
                                ax.set_title('V')
                        elif r==3:
                            ax.set_title('Mean')
                    if r==0:
                        if nameMethod=='':
                            ax.set_ylabel('Ref')
                        else:
                            ax.set_ylabel(nameMethod.replace('_',' '))
                            
                    ff, axf = plt.subplots(figsize=(20, 20))
                    axf.imshow(F_for_print,cmap=cmap, extent=(-N // 2, N // 2, -M // 2, M // 2))
                    if r==3:
                        ext_str = 'Mean'
                    else:
                        ext_str = color_type[r]
                        
                    axf.set_title('Log Spectrum magnitude '+filewithoutext + method + ' '+ext_str)
                    fname = os.path.join('Spectrum',filewithoutext +'_'+color_type+str(r)+'_' + method + '_'+ext_str+'.png')
    #                plt.show()
                    if not(color_type=='HSV' and r==3):
                        if not(not_close):
                            plt.savefig(fname, dpi=300)
                    if not_close:
                        plt.show(block=False)
                        plt.pause(0.01)
                        input('Enter to close')
                        plt.close(ff)
                                  
                        
                    #ax.set_title('Log Spectrum magnitude '+filewithoutext + method)
            plt.suptitle('Log Spectrum magnitude '+filewithoutext+' '+color_type)
            fname = os.path.join('Spectrum',filewithoutext + '_'+color_type+'.png')
            if not(not_close):
                plt.savefig(fname, dpi=300)
            
            if not(not_close):
                plt.close(f)
            else:
                plt.show()
                input('Enter to close')
                plt.close('all')
        
    if not(not_close):
        plt.close('all')
    else:
        input('Enter to close')
        plt.close('all')
            

