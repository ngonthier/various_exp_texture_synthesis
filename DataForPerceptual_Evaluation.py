# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:55:38 2020

Resize and crop output image for the perceptual test

@author: gonthier
"""

import numpy as np
import os
import os.path
import pathlib
from shutil import copyfile
import cv2

directory = "./im/References/"
ResultsDir = "./im/"
#if os.environ.get('OS','') == 'Windows_NT':
path_base  = os.path.join('C:\\','Users','gonthier')
ownCloudname = 'ownCloud'

if not(os.path.exists(path_base)):
    path_base  = os.path.join(os.sep,'media','gonthier','HDD')
    ownCloudname ='owncloud'
if os.path.exists(path_base):
    ResultsDir = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','ForTexturePaper','Output','1024')
    directory = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','Images Textures References Subset')
    directory_betaTexture = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','ForTexturePaper','Output','1024_Beta')
    output_folder = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','ForTexturePaper','Output','ForPerceptualTest')
    output_merge_folder = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','ForTexturePaper','Output','ForPerceptualTestMerge')
    output_Ref_folder = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','ForTexturePaper','Output','ForPerceptualRef')
else:
    print(path_base,'not found')
    raise(NotImplementedError)

# Uniquement les images que l'on garde pour la partie d'estimation visuelle de l'utilisateur
listofmethod = ['','_SAME_Gatys','_SAME_Gatys_MSSInit','_SAME_Gatys_spectrumTFabs_eps10m16_MSSInit',\
    '_Snelgorove_MultiScale_o5_l3_8_psame','_DCor']
listNameMethod = ['Reference','Gatys','Gatys + MSInit','Gatys + Spectrum TF + MSInit',\
    'Snelgorove','Deep Corr']

extension = ".png"
files = [file for file in os.listdir(directory) if file.lower().endswith(extension)]
files_short = files
#files_short = [files[0],files[-1]]


def Resize_and_crop_center():
    """
    This function create a downsampled version of each image and a crop of it center
    """
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_merge_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_Ref_folder).mkdir(parents=True, exist_ok=True)
    
    downsampled_size = (256,256)
    target_width = 256
    target_height = 256
    
    white_column = np.uint8(255*np.ones((target_height,25,3)))

    for file in files_short:
        filewithoutext = '.'.join(file.split('.')[:-1])
        print('Image :',filewithoutext)
        
        # Load the images
        for i,zipped in enumerate(zip(listofmethod,listNameMethod)):
            method,nameMethod = zipped
            
            filewithoutext_with_method = filewithoutext + method
            
            if method =='':
                stringname = os.path.join(directory,filewithoutext_with_method)
            else:
                stringname = os.path.join(ResultsDir,filewithoutext,filewithoutext_with_method)
            #print(method,nameMethod)
            stringnamepng = stringname + '.png'
            stringnamejpg = stringname + '.jpg'
            if os.path.isfile(stringnamepng):
                filename = stringnamepng
            elif os.path.isfile(stringnamejpg):
                filename = stringnamejpg
            else:
                print('Neither',stringnamepng,' or ',stringnamejpg,'exist. Wrong path ?')
            
            # Copy file
            dst = os.path.join(output_Ref_folder,filewithoutext_with_method+'.png')
            copyfile(filename, dst)  # src,dst
            
            # Read image
            image = cv2.imread(filename)
            # Resize image
            # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. 
            small = cv2.resize(image, downsampled_size, interpolation = cv2.INTER_AREA)
            # save the image
            small_img_name = filewithoutext_with_method +'_downsampled.png'
            path_small_img = os.path.join(output_folder,small_img_name)
            cv2.imwrite(path_small_img,small)
            
            # Crop center 
            height , width, channel = image.shape
            left_corner = int(round(width/2)) - int(round(target_width/2))
            top_corner = int(round(height/2)) - int(round(target_height/2))
            
            crop_img = image[left_corner:left_corner+target_width, top_corner:top_corner+target_height,:]
            
            # save the image
            crop_img_name = filewithoutext_with_method +'_cropcenter.png'
            path_crop_img = os.path.join(output_folder,crop_img_name)
            cv2.imwrite(path_crop_img,crop_img)
            
            # Concatenate the two images
            vis= np.concatenate((small,white_column,crop_img), axis=1)
            merge_img_name = filewithoutext_with_method +'_merge.png'
            path_merge_img = os.path.join(output_merge_folder,merge_img_name)
            cv2.imwrite(path_merge_img,vis)
#            cv2.imshow("test", vis)
#            cv2.waitKey()

if __name__ == '__main__':
    Resize_and_crop_center()
