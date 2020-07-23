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
from itertools import permutations,combinations
import random
import math
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.table import Table
matplotlib.rcParams['text.usetex'] = True
import pandas as pd

import choix
import pickle
import scipy
import tikzplotlib
import platform

if platform.system()=='Windows':
    os.environ["path"] += os.path.join('C:\\','Program Files','MiKTeX','miktex','bin','x64')
else:
   os.environ["PATH"] += os.path.join( 'usr','bin')

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
    ForPerceptualTestAllMerge = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','ForTexturePaper','Output','ForPerceptualTestAllMerge')
    output_Ref_folder = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','ForTexturePaper','Output','ForPerceptualRef')
    ForPerceptualTestPsyToolkitSurvey = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','ForTexturePaper','Output','PsyToolkitSurvey')
else:
    print(path_base,'not found')
    raise(NotImplementedError)

# Uniquement les images que l'on garde pour la partie d'estimation visuelle de l'utilisateur
listofmethod = ['','_SAME_Gatys','_SAME_Gatys_MSSInit','_SAME_Gatys_spectrumTFabs_eps10m16_MSSInit',\
    '_Snelgorove_MultiScale_o5_l3_8_psame','_DCor']
listofmethod_onlySynth = ['_SAME_Gatys','_SAME_Gatys_MSSInit','_SAME_Gatys_spectrumTFabs_eps10m16_MSSInit',\
    '_Snelgorove_MultiScale_o5_l3_8_psame','_DCor']

listNameMethod = ['Reference','Gatys','Gatys + MSInit','Gatys + Spectrum TF + MSInit',\
    'Snelgrove','Deep Corr']
listNameMethod_onlySynth = ['Gatys','Gatys + MSInit','Gatys + Spectrum TF + MSInit',\
    'Snelgrove','Deep Corr']
listNameMethod_onlySynth_withoutTF = ['Gatys','Gram + MSInit','Gram + Spectrum + MSInit',\
    'Snelgrove','Deep Corr']
listNameMethod_onlySynth_withoutTF_withCite = [r'Gatys \cite{gatys_texture_2015}',r'Gram + MSInit',r'Gram + Spectrum + MSInit',\
    r'Snelgrove \cite{snelgrove_highresolution_2017}',r'Deep Corr \cite{sendik_deep_2017}']
listNameMethod_onlySynth_withoutTF_withCiteForpgf = [r'Gatys \cite{gatys_texture_2015}',r'Gram + MSInit',r'Gram + Spectrum + MSInit',\
    r'Snelgrove \cite{snelgrove_highresolution_2017}',r'Deep Corr \cite{sendik_deep_2017}']

extension = ".png"
files = [file for file in os.listdir(directory) if file.lower().endswith(extension)]
files_short = files
#files_short = [files[0],files[-1]]

# List of regular images decided with Yann on 12/06/20 : 11 elements
listRegularImages = ['BrickRound0122_1_seamless_S',
                     'CRW_5751_1024',
                     'Pierzga_2006_1024',
                     'fabric_white_blue_1024',
                     'lego_1024',
                     'TexturesCom_BrickSmallBrown0473_1_M_1024',
                     'TexturesCom_FloorsCheckerboard0046_4_seamless_S_1024',
                     'TexturesCom_TilesOrnate0085_1_seamless_S',
                     'TexturesCom_TilesOrnate0158_1_seamless_S',
                     'metal_ground_1024']
    


def Resize_and_crop_center():
    """
    This function create a downsampled version of each image and a crop of it center
    """
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_merge_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_Ref_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ForPerceptualTestAllMerge).mkdir(parents=True, exist_ok=True)
    
    downsampled_size = (256,256)
    target_width = 256
    target_height = 256
    
    white_column = np.uint8(255*np.ones((target_width,26,3)))

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
            
    size_one_im = 256
    with_space_size = 26
    ImWidth = size_one_im*2+with_space_size # Size of the couple image
    ImHeight = 256
    WIDTH = size_one_im*4+with_space_size*6 # Also two band on the border
    HEIGHT=size_one_im*2+with_space_size*4 # Also border around
    
    ## Creation avec les deux images de la meme methode l une a cote de l'autre
#    for file in files_short:
#        filewithoutext = '.'.join(file.split('.')[:-1])
#        print('Image :',filewithoutext)
#            
#        # Load the images
#        all_pairs = permutations(listofmethod_onlySynth, 2) # 
#        for pair in all_pairs:
#            print(pair)
#            
#            methodA, methodB = pair
#            
#            output_name = filewithoutext + methodA + methodB +'.png'
#            
#            filewithoutext_with_method = filewithoutext + ''
#            stringname_Ref = os.path.join(output_merge_folder,filewithoutext_with_method+'_merge.png')
#            imgRef = cv2.imread(stringname_Ref) 
#            
#            filewithoutext_with_methodA = filewithoutext + methodA
#            stringname_A= os.path.join(output_merge_folder,filewithoutext_with_methodA+'_merge.png')
#            imgA = cv2.imread(stringname_A) 
#            
#            filewithoutext_with_methodB = filewithoutext + methodB
#            stringname_B= os.path.join(output_merge_folder,filewithoutext_with_methodB+'_merge.png')
#            imgB = cv2.imread(stringname_B) 
#            
#            big_white_Image = np.uint8(255*np.ones((HEIGHT,WIDTH,3)))
#            
#            y0_ref = int((WIDTH-ImWidth)/2) 
#            y1_ref = y0_ref + ImWidth 
#            x0_ref = with_space_size
#            x1_ref = ImHeight + x0_ref
#            big_white_Image[x0_ref:x1_ref,y0_ref:y1_ref,:] = imgRef
#            
#            y0_A = with_space_size
#            y1_A = ImWidth + y0_A
#            x0_A= ImHeight + with_space_size*3
#            x1_A = x0_A + ImHeight
#            big_white_Image[x0_A:x1_A,y0_A:y1_A,:] = imgA
#            
#            y0_B = ImWidth + with_space_size*3
#            y1_B = ImWidth + y0_B
#            x0_B= ImHeight + with_space_size*3
#            x1_B = ImHeight + x0_B
#            big_white_Image[x0_B:x1_B,y0_B:y1_B,:] = imgB
#            
#            
#            path_merge_img = os.path.join(ForPerceptualTestAllMerge,output_name)
#            cv2.imwrite(path_merge_img,big_white_Image)
            
    
    font = ImageFont.truetype("arial.ttf", 32)
    size_text = font.getsize("1")
    size_line = size_text[1]
    Height_bandeau = size_line*2+with_space_size*2
    bandeau_Image = Image.new('RGB', (WIDTH,Height_bandeau), color = (255, 255, 255))
    d = ImageDraw.Draw(bandeau_Image)
    # Sous la premiere image
    height_first_line = 0
    d.text((int(with_space_size+size_one_im//2-size_text[0]//2),height_first_line), "1", fill=(0,0,0),font=font)
    # Sous la seconde
    d.text((int(with_space_size*2+size_one_im+size_one_im//2-size_text[0]//2),height_first_line), "2", fill=(0,0,0),font=font)
    # Sous la troisieme et la quatrieme
    d.text((int(with_space_size*4+2*size_one_im+size_one_im//2-size_text[0]//2),height_first_line), "1", fill=(0,0,0),font=font)
    d.text((int(with_space_size*5+3*size_one_im+size_one_im//2-size_text[0]//2),height_first_line), "2", fill=(0,0,0),font=font)
    # Texte Global
    espace_line = with_space_size
    height_second_line = height_first_line+size_line+espace_line
    size_text = font.getsize("Global")
    d.text((int(with_space_size+size_one_im+with_space_size//2-size_text[0]//2),height_second_line), "Global", fill=(0,0,0),font=font)
    size_text = font.getsize("Local")
    d.text((int(with_space_size*4+3*size_one_im+with_space_size//2-size_text[0]//2),height_second_line), "Local", fill=(0,0,0),font=font)
    bandeau_Image.save('bandeau.png')
    array_bandeau_Image = np.array(bandeau_Image)
    h_bandeau, w_bandeau, _ = array_bandeau_Image.shape
    
    ## Creation avec les deux images global l une a cote de l autre sinon les deux zooms
    for file in files_short:
        filewithoutext = '.'.join(file.split('.')[:-1])
        print('Image :',filewithoutext)
            
        # Load the images
        all_pairs = permutations(listofmethod_onlySynth, 2) # 
        for pair in all_pairs:
            print(pair)
            
            methodA, methodB = pair
            
            output_name = filewithoutext + methodA + methodB +'.png'
            
            filewithoutext_with_method = filewithoutext + ''
            small_img_name_ref = os.path.join(output_folder,filewithoutext_with_method +'_downsampled.png')
            img_global_ref = cv2.imread(small_img_name_ref)
            crop_img_name_ref = os.path.join(output_folder,filewithoutext_with_method +'_cropcenter.png')
            img_crop_ref = cv2.imread(crop_img_name_ref)
            
            stringname_Ref = os.path.join(output_merge_folder,filewithoutext_with_method+'_merge.png')
            imgRef = cv2.imread(stringname_Ref) 
            
            filewithoutext_with_methodA = filewithoutext + methodA
            small_img_name_A = os.path.join(output_folder,filewithoutext_with_methodA +'_downsampled.png')
            img_global_A = cv2.imread(small_img_name_A)
            
            filewithoutext_with_methodB = filewithoutext + methodB
            small_img_name_B = os.path.join(output_folder,filewithoutext_with_methodB +'_downsampled.png')
            img_global_B = cv2.imread(small_img_name_B)
            
            crop_img_nameA = os.path.join(output_folder,filewithoutext_with_methodA +'_cropcenter.png')
            img_crop_A = cv2.imread(crop_img_nameA)
            
            crop_img_nameB = os.path.join(output_folder,filewithoutext_with_methodB +'_cropcenter.png')
            img_crop_B = cv2.imread(crop_img_nameB)
            
            big_white_Image = np.uint8(255*np.ones((HEIGHT+h_bandeau,WIDTH,3)))
            big_white_Image[HEIGHT:,:,:] = array_bandeau_Image
            
            y0_ref = size_one_im//2 + with_space_size +with_space_size//2
            y1_ref = y0_ref + size_one_im 
            x0_ref = with_space_size
            x1_ref = size_one_im + x0_ref
            big_white_Image[x0_ref:x1_ref,y0_ref:y1_ref,:] = img_global_ref
            
            y0_ref = ImWidth + with_space_size*3 + size_one_im//2 + with_space_size//2
            y1_ref = y0_ref + size_one_im 
            x0_ref = with_space_size
            x1_ref = size_one_im + x0_ref
            big_white_Image[x0_ref:x1_ref,y0_ref:y1_ref,:] = img_crop_ref

            vis_global= np.concatenate((img_global_A,white_column,img_global_B), axis=1)
            vis_crop= np.concatenate((img_crop_A,white_column,img_crop_B), axis=1)
            
            y0_G = with_space_size
            y1_G = ImWidth + y0_G
            x0_G= ImHeight + with_space_size*3
            x1_G = x0_G + ImHeight
            big_white_Image[x0_G:x1_G,y0_G:y1_G,:] = vis_global
            
            y0_C = ImWidth + with_space_size*3
            y1_C = ImWidth + y0_C
            x0_C= ImHeight + with_space_size*3
            x1_C = ImHeight + x0_C
            big_white_Image[x0_C:x1_C,y0_C:y1_C,:] = vis_crop
            
            
            path_merge_img = os.path.join(ForPerceptualTestAllMerge,output_name)
            cv2.imwrite(path_merge_img,big_white_Image)
            
            
def create_survey_for_PsyToolkit():
    """
    This function create the script of the 
    """
    pathlib.Path(ForPerceptualTestPsyToolkitSurvey).mkdir(parents=True, exist_ok=True)
    
    # First we will create the 400 possibles questions 
    
#    l: examplequestion1
#    t: radio
#    #o: random # You can use the option line o: random to randomize the order of the items.
#    i: {center} https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualTestAllMerge/BrickRound0122_1_seamless_S_DCor_SAME_Gatys_MSSInit.png
#    /BrickRound0122_1_seamless_S_SAME_Gatys_MSSInit.png
#    q: Which one is closer to the <a href="https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/BrickRound0122_1_seamless_S.png">Reference Image</a>?
#    - <a href="https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/BrickRound0122_1_seamless_S_DCor.png">Left Image</a> 
#    - <a href="https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/BrickRound0122_1_seamless_S_SAME_Gatys_MSSInit.png">Right Image</a>  
#        
    template_of_question = 'l: {0} \n t: radio \n i: {{ center }} https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualTestAllMerge/{1} \n q: Which one is closer to the <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{2}>Reference Image</a>? \n - <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{3}>Left Image</a>  \n - <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{4}>Right Image</a> \n' 
        
    list_all_questions = []
    
    for file in files_short:
        filewithoutext = '.'.join(file.split('.')[:-1])
        #print('Image :',filewithoutext)
            
        # All the possible pairs
        all_pairs = permutations(listofmethod_onlySynth, 2) # [['_DCor','_SAME_Gatys_MSSInit']]
        for pair in all_pairs:
            #print(pair)
            
            methodA, methodB = pair
            name_question = filewithoutext + methodA + methodB
            main_image_name = filewithoutext + methodA + methodB +'.png'
            ref_image_name = filewithoutext +'.png'
            methodA_img_nam = filewithoutext + methodA  +'.png'
            methodB_img_nam = filewithoutext + methodB +'.png'
            
            question_for_this_pair = template_of_question.format(name_question,\
                                        main_image_name,ref_image_name,methodA_img_nam,methodB_img_nam)
            
            list_all_questions += [question_for_this_pair]
            
    number_of_questions_per_survey = 40 
    total_number_of_ques = len(list_all_questions)
    nb_surveys = math.ceil(total_number_of_ques/number_of_questions_per_survey)
    beginning_survey = 'random: begin \n \n'
    ending = 'random: end \n'
    
    for survey_id in range(nb_surveys):
        txt_survey = '# Survey number {0} \n \n'.format(survey_id)
        txt_survey += beginning_survey
        random.seed(survey_id)
        random.shuffle(list_all_questions)
        first_questions = list_all_questions[0:number_of_questions_per_survey]
        list_all_questions = list_all_questions[number_of_questions_per_survey:]
        for ques in first_questions:
            txt_survey += ques
            txt_survey += '\n'
        txt_survey += ending
        
        # Save the txt file
        survey_txt_file_name = os.path.join(ForPerceptualTestPsyToolkitSurvey,'Survey_'+str(survey_id)+'.txt')
        text_file = open(survey_txt_file_name, "w")
        n = text_file.write(txt_survey)
        text_file.close()
        
def create_survey_for_PsyToolkit_4ques():
    """
    This function create the script of the 
    """
    pathlib.Path(ForPerceptualTestPsyToolkitSurvey).mkdir(parents=True, exist_ok=True)
    
    # First we will create the 400 possibles questions 
    
#    l: examplequestion1
#    t: radio
#    #o: random # You can use the option line o: random to randomize the order of the items.
#    i: {center} https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualTestAllMerge/BrickRound0122_1_seamless_S_DCor_SAME_Gatys_MSSInit.png
#    /BrickRound0122_1_seamless_S_SAME_Gatys_MSSInit.png
#    q: Which one is closer to the <a href="https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/BrickRound0122_1_seamless_S.png">Reference Image</a>?
#    - <a href="https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/BrickRound0122_1_seamless_S_DCor.png">Left Image</a> 
#    - <a href="https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/BrickRound0122_1_seamless_S_SAME_Gatys_MSSInit.png">Right Image</a>  
#        
    begin_template_of_question = 'l: {0} \n t: radio \n i: {{ center }} https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualTestAllMerge/{1} \n q: Which one is the best compared to the <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{2}>Reference Image</a>? \n '
    
    question1 = '- Global <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{3}>1</a> - Local <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{3}>1</a> \n '
    question2 = '- Global <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{4}>2</a> - Local <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{4}>2</a> \n '
    question3 = '- Global <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{3}>1</a> - Local <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{4}>2</a> \n '
    question4 = '- Global <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{4}>2</a> - Local <a href=https://perso.telecom-paristech.fr/gonthier/data/ForPerceptualRef/{3}>1</a> \n '
            
    template_of_question = begin_template_of_question + question1 + question2 + question3 + question4
    list_all_questions = []
    
    for file in files_short:
        filewithoutext = '.'.join(file.split('.')[:-1])
        #print('Image :',filewithoutext)
            
        # All the possible pairs
        all_pairs = permutations(listofmethod_onlySynth, 2) # [['_DCor','_SAME_Gatys_MSSInit']]
        for pair in all_pairs:
            #print(pair)
            
            methodA, methodB = pair
            name_question = filewithoutext + methodA + methodB
            main_image_name = filewithoutext + methodA + methodB +'.png'
            ref_image_name = filewithoutext +'.png'
            methodA_img_nam = filewithoutext + methodA  +'.png'
            methodB_img_nam = filewithoutext + methodB +'.png'
            
            question_for_this_pair = template_of_question.format(name_question,\
                                        main_image_name,ref_image_name,methodA_img_nam,methodB_img_nam)
            
            list_all_questions += [question_for_this_pair]
            
    number_of_questions_per_survey = 40 
    total_number_of_ques = len(list_all_questions)
    nb_surveys = math.ceil(total_number_of_ques/number_of_questions_per_survey)
    beginning_survey = 'random: begin \n \n'
    ending = 'random: end \n'
    
    for survey_id in range(nb_surveys):
        txt_survey = '# Survey number {0} \n \n'.format(survey_id)
        txt_survey += beginning_survey
        random.seed(survey_id)
        random.shuffle(list_all_questions)
        first_questions = list_all_questions[0:number_of_questions_per_survey]
        list_all_questions = list_all_questions[number_of_questions_per_survey:]
        for ques in first_questions:
            txt_survey += ques
            txt_survey += '\n'
        txt_survey += ending
        
        # Save the txt file
        survey_txt_file_name = os.path.join(ForPerceptualTestPsyToolkitSurvey,'Survey_'+str(survey_id)+'_4ques.txt')
        text_file = open(survey_txt_file_name, "w")
        n = text_file.write(txt_survey)
        text_file.close()

def regrouper_resultats_psytoolkit():
    """
    This fct helps to regroup all the votes of the pystoolkit questionnary 
    Remove old answers (testing time) + empty answer
    """
    
    #pathlib.Path(ForPerceptualTestPsyToolkitSurvey).mkdir(parents=True, exist_ok=True)

#    Value = =
#    1 : Global 1 - Local 1 
#    2 : Global 2 - Local 2
#    3 : Global 1 - Local 2
#    4 : Global 2 - Local 1

    dict_data =  {}

    all_df = None
    
    number_answer_per_participant = None
    
    for i in range(0,10):
        folder_name = 'data_'+str(i)
        print(folder_name)
        path_folder = os.path.join(ForPerceptualTestPsyToolkitSurvey,folder_name)
        name_csv_file = 'data.csv'
        path_csv_file = os.path.join(path_folder,name_csv_file)
        df_i = pd.read_csv(path_csv_file,sep=',',index_col=False)
        # Il va falloir filtrer les dates ici car certaines réponses peuvent dater des tests !
        # Toutes les réponses avant le 4 mai sont a retirer
        date_start = np.datetime64('2020-05-04').astype('datetime64[ns]')
        df_i['TIME_start'] =  pd.to_datetime(df_i['TIME_start'], format='%Y-%m-%d-%H-%M').astype('datetime64[ns]')
        #df_i['TIME_start'].astype('datetime64[ns]')
        #print(len(df_i))
        #print(df_i['TIME_start'])
        #print(df_i['TIME_end'])
        df_i = df_i[df_i['TIME_start']>date_start]
        #print(len(df_i))
        
        df_i = df_i.drop(columns=['participant','TIME_start', 'TIME_end', 'TIME_total'])
        
        local_number_answer_per_participant = 40 - df_i.isna().sum(axis=1) 
        df_i = df_i[local_number_answer_per_participant!=0]
        dict_data[i] = df_i
        
        if number_answer_per_participant is None:
            number_answer_per_participant = local_number_answer_per_participant
        else:
            number_answer_per_participant = number_answer_per_participant.append(local_number_answer_per_participant,ignore_index=True)
    
    print('=======================')
    print("On all the participants we have :")
    print("A mean of number of answer of :",number_answer_per_participant.mean())
    print("A std of number of answer of :",number_answer_per_participant.std())
    print("A median of number of answer of :",number_answer_per_participant.median())
    Not_all_anwser = number_answer_per_participant[number_answer_per_participant!=40]
    print('On ',len(number_answer_per_participant),'participants',len(Not_all_anwser),'don t answer all the 40 questions')
    print("A mean of number of answer of :",number_answer_per_participant.mean())
    print("On this subset of incomplete answers we have")
    print("A mean of number of answer of :",Not_all_anwser.mean())
    print("A std of number of answer of :",Not_all_anwser.std())
    print("A median of number of answer of :",Not_all_anwser.median())
    print("A maximum of number of answer of :",Not_all_anwser.max())
    print("A minimum of number of answer of :",Not_all_anwser.min())
    print('=======================')
    
    diff_case=['global','local','both']
    for case in diff_case:
    
        number_win_all_images = pd.DataFrame(columns=['image','methodA','methodB','winA','winB'])
        dict_number_win = {}  
        
        dict_correspondance = {}
        dict_correspondance_order = {}
        
        for j,file in enumerate(files_short):
            filewithoutext = '.'.join(file.split('.')[:-1])
            all_pairs = combinations(listofmethod_onlySynth, 2) # Without repetition 
            for i,pair in enumerate(all_pairs):
                methodA, methodB = pair
                index = j + i*len(files_short)
                number_win_all_images.loc[index] = [filewithoutext,methodA,methodB,0,0]
                dict_correspondance[filewithoutext+methodA + methodB] = index
                dict_correspondance_order[filewithoutext+methodA + methodB] = True
                dict_correspondance[filewithoutext+methodB + methodA] = index
                dict_correspondance_order[filewithoutext+methodB + methodA] = False

        for i in range(0,10): 
            df = dict_data[i]
            #df = df.drop(columns=['participant','TIME_start', 'TIME_end', 'TIME_total'])
            for row in df.iterrows():
                row_values = row[1]
                for c,value in zip(df.columns,row_values):
                    if np.isnan(value):
                        continue
                    assert(value > 0)
                    assert(value <= 4)
                    
                    c = c.split(':')[0]
                    index = dict_correspondance[c]
                    right_order = dict_correspondance_order[c]
                    if case=='global': # Global winning
                        if value==1.0 or value==3.0:
                            if right_order:
                                number_win_all_images.loc[index,'winA'] += 1
                            else:
                                number_win_all_images.loc[index,'winB'] += 1
                        else:
                            if right_order:
                                number_win_all_images.loc[index,'winB'] += 1
                            else:
                                number_win_all_images.loc[index,'winA'] += 1
                    elif case=='local': # Local winning
                        if value==1.0 or value==4.0:
                            if right_order:
                                number_win_all_images.loc[index,'winA'] += 1
                            else:
                                number_win_all_images.loc[index,'winB'] += 1
                        else:
                            if right_order:
                                number_win_all_images.loc[index,'winB'] += 1
                            else:
                                number_win_all_images.loc[index,'winA'] += 1
                    elif case=='both': # Local winning
                        if value==1.0:
                            if right_order:
                                number_win_all_images.loc[index,'winA'] += 2
                            else:
                                number_win_all_images.loc[index,'winB'] += 2
                        elif value==2.0:
                            if right_order:
                                number_win_all_images.loc[index,'winB'] += 2
                            else:
                                number_win_all_images.loc[index,'winA'] += 2
                        else: # Tie
                            number_win_all_images.loc[index,'winB'] += 1
                            number_win_all_images.loc[index,'winA'] += 1
        
        # Number of votes for each given question 
        number_win_all_images['NumberVote'] = number_win_all_images['winA'] + number_win_all_images['winB']
        print('===',case,'===')
        print("Per question (we gather the two possible question : image A on right and image B on left and the opposite we have :")
        print("A maximum of number of answer of :",number_win_all_images['NumberVote'].max())
        print("A minimum of number of answer of :",number_win_all_images['NumberVote'].min())
        print("A mean of number of answer of :",number_win_all_images['NumberVote'].mean())
        print("A median of number of answer of :",number_win_all_images['NumberVote'].median())
        print("A std of number of answer of :",number_win_all_images['NumberVote'].std())
                
        path_df = os.path.join(ForPerceptualTestPsyToolkitSurvey,'Number_of_wins_'+case+'.csv')
        number_win_all_images.to_csv(path_df,sep=',',index=False)
        
        
#    =======================
#    On all the participants we have :
#    A mean of number of answer of : 34.086021505376344
#    A std of number of answer of : 13.064768350147315
#    A median of number of answer of : 40.0
#    On  93 participants 17 don t answer all the 40 questions
#    A mean of number of answer of : 34.086021505376344
#    On this subset of incomplete answers we have
#    A mean of number of answer of : 7.647058823529412
#    A std of number of answer of : 8.52159885577956
#    A median of number of answer of : 6.0
#    A maximum of number of answer of : 27
#    A minimum of number of answer of : 0
#    =======================
#    === global ===
#    Per question (we gather the two possible question : image A on right and image B on left and the opposite we have :
#    A maximum of number of answer of : 23
#    A minimum of number of answer of : 9
#    A mean of number of answer of : 15.85
#    A median of number of answer of : 16.0
#    A std of number of answer of : 3.303340778894533
       
def convert_pd_df_to_list_wins_lost(df):
    data = []
            
    for row in df.iterrows():
        methodA = row[1]['methodA']
        winA = row[1]['winA']
        methodB = row[1]['methodB']
        winB = row[1]['winB']
        methodA_index = listofmethod_onlySynth.index(methodA)
        methodB_index = listofmethod_onlySynth.index(methodB)
        data += [(methodA_index,methodB_index)]*int(winA)
        data += [(methodB_index,methodA_index)]*int(winB)

    return(data)
    
def get_Wi_Ei(sub_part):
    """
    Return a list of winning probabilty per method and the standard
    deviation of the winning probability 
    """
    n_method = len(listofmethod_onlySynth)
    W_list = []
    E_list = []
    for i,method in enumerate(listofmethod_onlySynth):
        pij_A = sub_part[sub_part['methodA']==method]['pA'].values
        pij_B = sub_part[sub_part['methodB']==method]['pB'].values
        pij = np.concatenate([pij_A,pij_B])
        assert(len(pij)==n_method-1)
        sum_pij = np.sum(pij)
        #print(sum_pij)
        Wi = sum_pij / (n_method-1.)
        assert(Wi<=1.0)
        #print(Wi)
        W_list += [Wi]  
        Ei = np.sqrt(np.mean((pij-Wi)**2))
        E_list += [Ei]
        
    # Wi represents the probability that a candidate i was preferred over all other candidates.
    return(W_list,E_list)
    
def get_Wi_Ei_fromParams(params):
    """
    Return a list of winning probabilty per method and the standard
    deviation of the winning probability 
    """
    n_method = len(listofmethod_onlySynth)
    W_list = []
    E_list = []
    for i,method in enumerate(listofmethod_onlySynth):
        pij_tab = []
        for j,_ in enumerate(listofmethod_onlySynth):
            if not(i==j):
                pij, pji = choix.probabilities([i,j],params)
                #print(i,j,pij, pji)
                pij_tab += [pij]
        
#        pij_A = sub_part[sub_part['methodA']==method]['pA'].values
#        pij_B = sub_part[sub_part['methodB']==method]['pB'].values
#        pij_tab = np.concatenate([pij_A,pij_B])
        assert(len(pij_tab)==n_method-1)
        sum_pij = np.sum(pij_tab)
        #print(sum_pij)
        Wi = sum_pij / (n_method-1.)
        assert(Wi<=1.0)
        #print(Wi)
        W_list += [Wi]
        
        Ei = np.sqrt(np.mean((pij_tab-Wi)**2)) # Cela n a pas vraiment de realite quelconque en fait
        E_list += [Ei]
        
    # Wi represents the probability that a candidate i was preferred over all other candidates.
    return(W_list,E_list)
    
def test_if_sparsity(sub_part):
    """
    Return True if one method always wins or always loss : sparsity case
    """
    
    for i,method in enumerate(listofmethod_onlySynth):
        pij_A = sub_part[sub_part['methodA']==method]['pA'].values
        pij_B = sub_part[sub_part['methodB']==method]['pB'].values
        pij_tab = np.concatenate([pij_A,pij_B]) 
        pi = np.mean(pij_tab)
        #print(pi)
        if pi==0.0 or pi==1.0:
            return(True)
            
    return(False)

def draw_duels_results(sub_part):
    """
    This function will draw duel results virtually and return the new dataframe
    of this virtual study with the same number of duels as previously
    """
    
    output_df = sub_part.copy()
    
    n = 1
    data = []
    for row in sub_part.iterrows():
        methodA = row[1]['methodA']
        pA = row[1]['pA']
        methodB = row[1]['methodB']
        pB = row[1]['pB']
        NumberVote = row[1]['NumberVote']
        bernouilli_output = np.random.binomial(n,pA, NumberVote)
        
        
        methodA_index = listofmethod_onlySynth.index(methodA)
        methodB_index = listofmethod_onlySynth.index(methodB)
        data += [(methodA_index,methodB_index)]*int(winA)
        data += [(methodB_index,methodA_index)]*int(winB)
    
    
    for i,method in enumerate(listofmethod_onlySynth):
        for j,method in enumerate(listofmethod_onlySynth):
            if i < j:
                pij_A = sub_part[sub_part['methodA']==method]['pA'].values
                pij_B = sub_part[sub_part['methodB']==method]['pB'].values
        pij_tab = np.concatenate([pij_A,pij_B]) 
        pi = np.mean(pij_tab)
        #print(pi)
        if pi==0.0 or pi==1.0:
            return(True)

def get_Wi_Betaij_stdij_fromDF_Empirical(sub_part,estimation_method='mm',
                                           max_iter=100000,tol=10**(-8),
                                           std_estimation='boostraping'):
    """
    Estimation of the std_estimation by boostraping
    """
    
    N_virtual_study = 3000
    N_virtual_study = 100
    n_method = len(listofmethod_onlySynth)
    
    betas=np.zeros((N_virtual_study,n_method))
    #probs_tirages=np.zeros((N_virtual_study,n_method,n_method))
    Wi_tirages=np.zeros((N_virtual_study,n_method))

    sub_part.loc[:,'pA'] = sub_part['winA'] /  sub_part['NumberVote']
    sub_part.loc[:,'pB'] = sub_part['winB'] /  sub_part['NumberVote']
    
    if std_estimation=='boostraping': 
        data_based = convert_pd_df_to_list_wins_lost(sub_part)
        total_num_votes = sub_part['NumberVote'].sum()
        sparsity = test_if_sparsity(sub_part)

    for n in range(N_virtual_study):
        output_df = sub_part.copy()
        # Do a new virtual study 
        
        if std_estimation=='binomial':
            data = []
            for row in sub_part.iterrows():
                index_row = row[0]
                methodA = row[1]['methodA']
                #pA = row[1]['pA']
                methodB = row[1]['methodB']
                pB = row[1]['pB']
                NumberVote = row[1]['NumberVote']
                methodA_index = listofmethod_onlySynth.index(methodA)
                methodB_index = listofmethod_onlySynth.index(methodB)

                bernouilli_output = np.random.binomial(1,pB, NumberVote)
                winA = list(bernouilli_output).count(0)
                winB = list(bernouilli_output).count(1)
  
                output_df.loc[index_row,'pA'] = winA / NumberVote
                output_df.loc[index_row,'pB'] = winB / NumberVote
                
                # If pB == 1 it will output only 1.0 values i.e . only votes of methods B
                data += [(methodA_index,methodB_index)]*int(winA)
                data += [(methodB_index,methodA_index)]*int(winB)
                
            sparsity = test_if_sparsity(output_df)
                
        elif std_estimation=='boostraping': 
            data_based = convert_pd_df_to_list_wins_lost(sub_part)
            data_indices = np.random.choice(np.arange(len(data_based)), size=total_num_votes, replace=True, p=None) 
            data = [data_based[i] for i in data_indices] #data_based[]
            
        if sparsity:
            alpha = 10**(-4)
        else:
            alpha = 0.0

        if estimation_method=='mm':
            params = choix.mm_pairwise(n_method,data,max_iter=max_iter,tol=tol,alpha=alpha) 
        elif estimation_method=='ilsr':
            params = choix.ilsr_pairwise(n_method,data,max_iter=max_iter,tol=tol,alpha=alpha) 
        elif estimation_method=='opt_pairwise':
            params = choix.opt_pairwise(n_method,data,max_iter=max_iter,tol=tol,alpha=alpha) 
        
        betas[n,:] = params
        W_list = []
        for i,method in enumerate(listofmethod_onlySynth):
            pij_tab = []
            for j,_ in enumerate(listofmethod_onlySynth):
                if not(i==j):
                    pij, pji = choix.probabilities([i,j],params)
                    pij_tab += [pij]
            assert(len(pij_tab)==n_method-1)
            sum_pij = np.sum(pij_tab)
            Wi = sum_pij / (n_method-1.)
            assert(Wi<=1.0)
            W_list += [Wi]
        Wi_tirages[n,:] = W_list
        
    betas = betas - np.mean(betas,axis=1,keepdims=True) # As the important stuff is the difference
    # between Beta_i and Beta_j => we can substract to all of them the mean of Beta_i 
    # For one given virtual study
    params_mean = np.mean(betas,axis=0) # Cela est faux  !!!! il faut considerer les 
    # ecarts a Beta_i - Beta_j et non les beta_i seul car cela depend de la constante dans 
    # l optimisation ! 
    # Idem std_matrix est faux !!! 
#    var_beta = np.var(betas,axis=0)
    cov = np.cov(betas,rowvar=0)
    std_matrix = np.zeros((n_method,n_method))
    for i,_ in enumerate(listofmethod_onlySynth):
        for j,_ in enumerate(listofmethod_onlySynth):
            if not(i==j):
                stdij = np.sqrt(cov[i,i]+cov[j,j]-2*cov[i,j])
                assert(stdij >= 0.)
                std_matrix[i,j] = stdij
    print('Wi_tirages',Wi_tirages.shape)
    print('Wi_tirages',Wi_tirages)
    W_list = np.mean(Wi_tirages,axis=0)
    std_Wi_tirages = np.std(Wi_tirages,axis=0)
    print('std_Wi_tirages',std_Wi_tirages.shape)
    print('std_Wi_tirages',std_Wi_tirages)
    
    return(W_list,params_mean,std_matrix,std_Wi_tirages)    
        

def get_Wi_Betaij_stdij_fromDF(sub_part,estimation_method='mm',
                               std_estimation='hessian',
                               max_iter=100000,tol=10**(-8)):
    
    if std_estimation=='hessian':
        W_list,params_mean,std_matrix = get_Wi_Betaij_stdij_fromDF_compute_beta_std(sub_part,
                                                            estimation_method=estimation_method,
                                                       std_estimation=std_estimation,
                                                       max_iter=max_iter,tol=tol)
        stdW_list = get_std_Wi(params_mean,std_Bi_minus_Bj=std_matrix)

    elif std_estimation=='boostraping' or std_estimation=='binomial':
        W_list,params_mean,std_matrix,stdW_list = get_Wi_Betaij_stdij_fromDF_Empirical(
                                            sub_part,estimation_method=estimation_method,
                                            max_iter=max_iter,tol=tol,
                                            std_estimation=std_estimation)
        
    return(W_list,stdW_list,params_mean,std_matrix)
    
    
def get_Wi_Betaij_stdij_fromDF_compute_beta_std(sub_part,estimation_method='mm',
                               std_estimation='hessian',
                               max_iter=100000,tol=10**(-8)):
    n_method = len(listofmethod_onlySynth)
    
    sub_part.loc[:,'pA'] = sub_part['winA'] /  sub_part['NumberVote']
    sub_part.loc[:,'pB'] = sub_part['winB'] /  sub_part['NumberVote'] 
#            hand_p_list,hand_e_list = get_Wi_Ei(sub_part)

#    print(sub_part['pA'])
#    print(sub_part['pB'])
    
    data = convert_pd_df_to_list_wins_lost(sub_part)
    
    sparsity = test_if_sparsity(sub_part)
    if sparsity:
        print('In a sparsity case !')
        alpha = 10**(-4)
    else:
        alpha = 0.0
    
#            if filewithoutext=='marbre_1024':
#                tol=10**(-5)
#            else:
#                tol =1e-8
    if estimation_method=='mm':
        params = choix.mm_pairwise(n_method,data,max_iter=max_iter,tol=tol,alpha=alpha) 
    elif estimation_method=='ilsr':
        params = choix.ilsr_pairwise(n_method,data,max_iter=max_iter,tol=tol,alpha=alpha) 
    elif estimation_method=='opt_pairwise':
        params = choix.opt_pairwise(n_method,data,max_iter=max_iter,tol=tol,alpha=alpha) 
    # Provide de si : the score per method
    #  maximum-likelihood estimate of params with minorization-maximization (MM) algorithm [Hun04]_
   
    W_list = []
    for i,method in enumerate(listofmethod_onlySynth):
        pij_tab = []
        for j,_ in enumerate(listofmethod_onlySynth):
            if not(i==j):
                pij, pji = choix.probabilities([i,j],params)
                #pij, pji = choix.probabilities([i,j],params)
                #print(i,j,pij, pji)
                pij_tab += [pij]
        assert(len(pij_tab)==n_method-1)
        sum_pij = np.sum(pij_tab)
        Wi = sum_pij / (n_method-1.)
        assert(Wi<=1.0)
        W_list += [Wi]
    
    if std_estimation=='hessian':
        f = choix.opt.PairwiseFcts(data, alpha)
        # provides methods to compute the negative log-likelihood 
        hessian = f.hessian(params)
#        gradient = f.gradient(params)
#        objective = f.objective(params)
#        print('hessian',hessian)
#        print('gradient',gradient)
#        print('objective',objective)
#        print(np.linalg.cond(hessian))
        
        Iy = hessian # as we compute the negative log-likelihood
        # It is the Fisher Information
        if np.linalg.cond(Iy) > np.finfo(Iy.dtype).eps: 
            # Pseudo inverse
            #print('pinv')
            #Iy_inv = np.linalg.pinv(Iy) # uses the linalg.lstsq
            Iy_inv = scipy.linalg.pinv2(Iy) # Use SVD 
        else:
            Iy_inv = np.linalg.inv(Iy)

        #print('Iy_inv',Iy_inv)
        std_matrix = np.zeros_like(hessian)
        for i,_ in enumerate(listofmethod_onlySynth):
            for j,_ in enumerate(listofmethod_onlySynth):
                if not(i==j):
                    stdij = np.sqrt(Iy_inv[i,i]+Iy_inv[j,j]-2*Iy_inv[i,j])
                    assert(stdij >= 0.)
                    std_matrix[i,j] = stdij
        #print('std_matrix',std_matrix)
                    
                    
    return(W_list,params,std_matrix)
    
def get_Wi_Ei_fromDF(sub_part,estimation_method='mm',max_iter=100000,tol=10**(-8)):
    n_method = len(listofmethod_onlySynth)
    
    sub_part.loc[:,'pA'] = sub_part['winA'] /  sub_part['NumberVote']
    sub_part.loc[:,'pB'] = sub_part['winB'] /  sub_part['NumberVote'] 
#            hand_p_list,hand_e_list = get_Wi_Ei(sub_part)

#    print(sub_part['pA'])
#    print(sub_part['pB'])
    
    data = convert_pd_df_to_list_wins_lost(sub_part)
    
    sparsity = test_if_sparsity(sub_part)
    if sparsity:
        print('In a sparsity case !')
        alpha = 10**(-4)
    else:
        alpha = 0.0
    
#            if filewithoutext=='marbre_1024':
#                tol=10**(-5)
#            else:
#                tol =1e-8
    if estimation_method=='mm':
       params = choix.mm_pairwise(n_method,data,max_iter=max_iter,tol=tol,alpha=alpha) 
    elif estimation_method=='ilsr':
        params = choix.ilsr_pairwise(n_method,data,max_iter=max_iter,tol=tol,alpha=alpha) 
    # Provide de si : the score per method
    #  maximum-likelihood estimate of params with minorization-maximization (MM) algorithm [Hun04]_
   
    W_list,E_list = get_Wi_Ei_fromParams(params)
    
    return(W_list,E_list)
    
def run_statistical_study(estimation_method='mm',
                          std_estimation='hessian',
                          protocol='all_together'):
    """
    In this function we will compute the mean score per method per image + all image 
    together but also the near convergence consistency metric (kind of std)
    @param estimation_method : mm or ilsr or opt_pairwise
    @param protocol : all_together : We compute all the pij (prob i>j) with 
                                        all the images together 
           Individual_image : we consider each reference images as an independant study
    """
    
    # A propos du cas Tie : 
#    For our next experiment we choose the equal-split method: if an observer chooses “no-
#    preference”, we split the vote in two and add a half-vote to each condition. This may result in a
#    non-integer number of votes,
    
    # En fait il semble y avoir plusieurs cas a considerer :
    # Le cas ou l on regroupe toutes les images ensemble differemment
    # Le cas ou l'on considère chacune des images de référence comme une étude 
    # et que l'on calcule ensuite  he winning probabilities for one candidate across different studies,
    # And thus the near convergence consistency metric 
    
    
    n_method = len(listofmethod_onlySynth) # Number of methods    
    
    diff_case=['global','local','both']
#    diff_case=['global','local']
#    diff_case=['both']
    
    # TODO solve problem :  BubbleMarbel seem to be a absorbing class : need to check what it is 
    # TODO : need to find a way to have statistical test between the winning probability 
    
    dict_couple_W_E = {}
    
    max_iter = 500000
    tol = 10**(-8)
    
    for case in diff_case:
        print("===",case,"===")
#        if case=='global':
#            tol=10**(-5)
#        elif case=='local':
#            tol=10**(-2)
#        else:
#            tol=10**(-5)
        path_df = os.path.join(ForPerceptualTestPsyToolkitSurvey,'Number_of_wins_'+case+'.csv')
        number_win_all_images =  pd.read_csv(path_df,sep=',')
        
        
        if protocol=='Individual_image':
            
            W_per_image_all = np.zeros([len(files_short),n_method])
            W_per_image_Reg = np.zeros([len(listRegularImages),n_method])
            W_per_image_Irreg = np.zeros([len(files_short)-len(listRegularImages),n_method])
            stdW_per_image_all = np.zeros([len(files_short),n_method])
            stdW_per_image_Reg = np.zeros([len(listRegularImages),n_method])
            stdW_per_image_Irreg = np.zeros([len(files_short)-len(listRegularImages),n_method])

            # Per images
            j_reg = 0
            j_irreg = 0 
            for j,file in enumerate(files_short):
                filewithoutext = '.'.join(file.split('.')[:-1])
                print(j,filewithoutext)
                sub_part = number_win_all_images[number_win_all_images['image']==filewithoutext]
                 
                W_list,stdW_list,params,std_matrix = get_Wi_Betaij_stdij_fromDF(sub_part=sub_part,
                                                           estimation_method=estimation_method,
                                                           std_estimation=std_estimation,
                                                           max_iter=max_iter,tol=tol)
                #stdW_list = get_std_Wi(params,std_Bi_minus_Bj=std_matrix)
                #print(j,stdW_list)
                dict_couple_W_E[filewithoutext] = [W_list,stdW_list,params,std_matrix]
                
                W_per_image_all[j,:] = W_list
                stdW_per_image_all[j,:] = stdW_list
                #print(listRegularImages)
                if filewithoutext in listRegularImages:
                     W_per_image_Reg[j_reg,:] = W_list
                     stdW_per_image_Reg[j_reg,:] = stdW_list
                     j_reg +=1
                else:
                     W_per_image_Irreg[j_irreg,:] = W_list
                     stdW_per_image_Irreg[j_irreg,:] = stdW_list
                     j_irreg += 1
            #print('stdW_per_image_all',stdW_per_image_all)
            #print('np.mean(W_per_image_all,axis=0),np.std(W_per_image_all,axis=0),np.mean(stdW_per_image_all,axis=0)')
            print(np.mean(W_per_image_all,axis=0),np.std(W_per_image_all,axis=0),np.mean(stdW_per_image_all,axis=0))
            # Now we will compute the mean of the winning probabilities and the near-convergene metric
            # And then the std of the Wi
            dict_couple_W_E['All'] = [np.mean(W_per_image_all,axis=0),np.std(W_per_image_all,axis=0),np.mean(stdW_per_image_all,axis=0)]
            dict_couple_W_E['Reg'] = [np.mean(W_per_image_Reg,axis=0),np.std(W_per_image_Reg,axis=0),np.mean(stdW_per_image_Reg,axis=0)]
            dict_couple_W_E['Irreg'] = [np.mean(W_per_image_Irreg,axis=0),np.std(W_per_image_Irreg,axis=0),np.mean(stdW_per_image_Irreg,axis=0)]
            
        if protocol=='all_together':
            # We will regroup all the images (20) together 
            print('All together')
            df_all = number_win_all_images.groupby(['methodA','methodB'])["winA", "winB",'NumberVote'].apply(lambda x : x.sum()).reset_index()
            W_list,stdW_list,params,std_matrix = get_Wi_Betaij_stdij_fromDF(sub_part=df_all,
                                                           estimation_method=estimation_method,
                                                           std_estimation=std_estimation,
                                                           max_iter=max_iter,tol=tol)
            
            dict_couple_W_E['All'] = [W_list,stdW_list,params,std_matrix]
            
            # We will work on the two subsets : regular and non-regular image 
            print("Regular")
            sub_part_reg = number_win_all_images[number_win_all_images['image'].isin(listRegularImages)]
            df_reg = sub_part_reg.groupby(['methodA','methodB'])["winA", "winB",'NumberVote'].apply(lambda x : x.sum()).reset_index()
            W_list,stdW_list,params,std_matrix = get_Wi_Betaij_stdij_fromDF(sub_part=df_reg,
                                                           estimation_method=estimation_method,
                                                           std_estimation=std_estimation,
                                                           max_iter=max_iter,tol=tol)

            dict_couple_W_E['Reg'] = [W_list,stdW_list,params,std_matrix]
            
            print("Irregular")
            sub_part_irreg = number_win_all_images[~number_win_all_images['image'].isin(listRegularImages)]
            df_irreg = sub_part_irreg.groupby(['methodA','methodB'])["winA", "winB",'NumberVote'].apply(lambda x : x.sum()).reset_index()
            W_list,stdW_list,params,std_matrix = get_Wi_Betaij_stdij_fromDF(sub_part=df_irreg,
                                                           estimation_method=estimation_method,
                                                           std_estimation=std_estimation,
                                                           max_iter=max_iter,tol=tol)

            dict_couple_W_E['Irreg'] = [W_list,stdW_list,params,std_matrix]

        # Save the data :
        data_path_save = os.path.join(ForPerceptualTestPsyToolkitSurvey,'WBS_'+case+'_'+protocol+'_'+estimation_method+'_'+std_estimation+'.pkl')
        with open(data_path_save, 'wb') as pkl:
            pickle.dump(dict_couple_W_E,pkl)
            
            
def run_statistical_study_old(estimation_method='mm',protocol='all_together'):
    """
    In this function we will compute the mean score per method per image + all image 
    together but also the near convergence consistency metric (kind of std)
    @param estimation_method : mm or ilsr or opt_pairwise
    @param protocol : all_together : We compute all the pij (prob i>j) with 
                                        all the images together 
           Individual_image : we consider each reference images as an independant study
    """
    
    # A propos du cas Tie : 
#    For our next experiment we choose the equal-split method: if an observer chooses “no-
#    preference”, we split the vote in two and add a half-vote to each condition. This may result in a
#    non-integer number of votes,
    
    # En fait il semble y avoir plusieurs cas a considerer :
    # Le cas ou l on regroupe toutes les images ensemble differemment
    # Le cas ou l'on considère chacune des images de référence comme une étude 
    # et que l'on calcule ensuite  he winning probabilities for one candidate across different studies,
    # And thus the near convergence consistency metric 
    
    
    n_method = len(listofmethod_onlySynth) # Number of methods    
    
    diff_case=['global','local','both']
#    diff_case=['global','local']
#    diff_case=['both']
    
    # TODO solve problem :  BubbleMarbel seem to be a absorbing class : need to check what it is 
    # TODO : need to find a way to have statistical test between the winning probability 
    
    dict_couple_W_E = {}
    
    max_iter = 500000
    tol = 10**(-5)
    
    for case in diff_case:
        print("===",case,"===")
#        if case=='global':
#            tol=10**(-5)
#        elif case=='local':
#            tol=10**(-2)
#        else:
#            tol=10**(-5)
        path_df = os.path.join(ForPerceptualTestPsyToolkitSurvey,'Number_of_wins_'+case+'.csv')
        number_win_all_images =  pd.read_csv(path_df,sep=',')
        
        
        if protocol=='Individual_image':
            
            W_per_image_all = np.zeros([len(files_short),n_method])
            W_per_image_Reg = np.zeros([len(listRegularImages),n_method])
            W_per_image_Irreg = np.zeros([len(files_short)-len(listRegularImages),n_method])

            # Per images
            j_reg = 0
            j_irreg = 0 
            for j,file in enumerate(files_short):
                filewithoutext = '.'.join(file.split('.')[:-1])
                print(j,filewithoutext)
                sub_part = number_win_all_images[number_win_all_images['image']==filewithoutext]
                 
                W_list,E_list = get_Wi_Ei_fromDF(sub_part=sub_part,estimation_method=estimation_method,max_iter=max_iter,tol=tol)
                
                dict_couple_W_E[filewithoutext] = [W_list,E_list]
                
                W_per_image_all[j,:] = W_list
                #print(listRegularImages)
                if filewithoutext in listRegularImages:
                     W_per_image_Reg[j_reg,:] = W_list
                     j_reg +=1
                else:
                     print('irrge',filewithoutext)
                     W_per_image_Irreg[j_irreg,:] = W_list
                     j_irreg += 1
                
            # Now we will compute the mean of the winning probabilities and the near-convergene metric
            dict_couple_W_E['All'] = [np.mean(W_per_image_all,axis=0),np.std(W_per_image_all,axis=0)]
            dict_couple_W_E['Reg'] = [np.mean(W_per_image_Reg,axis=0),np.std(W_per_image_Reg,axis=0)]
            dict_couple_W_E['Irreg'] = [np.mean(W_per_image_Irreg,axis=0),np.std(W_per_image_Irreg,axis=0)]
            
        if protocol=='all_together':
            # We will regroup all the images (20) together 
            print('All together')
            df_all = number_win_all_images.groupby(['methodA','methodB'])["winA", "winB",'NumberVote'].apply(lambda x : x.sum()).reset_index()
            W_list,E_list = get_Wi_Ei_fromDF(sub_part=df_all,estimation_method=estimation_method,max_iter=max_iter,tol=tol)
            dict_couple_W_E['All'] = [W_list,E_list]
            
            # We will work on the two subsets : regular and non-regular image 
            print("Regular")
            sub_part_reg = number_win_all_images[number_win_all_images['image'].isin(listRegularImages)]
            df_reg = sub_part_reg.groupby(['methodA','methodB'])["winA", "winB",'NumberVote'].apply(lambda x : x.sum()).reset_index()
            W_list,E_list = get_Wi_Ei_fromDF(sub_part=df_reg,estimation_method=estimation_method,max_iter=max_iter,tol=tol)
            dict_couple_W_E['Reg'] = [W_list,E_list]
            
            print("Irregular")
            sub_part_irreg = number_win_all_images[~number_win_all_images['image'].isin(listRegularImages)]
            df_irreg = sub_part_irreg.groupby(['methodA','methodB'])["winA", "winB",'NumberVote'].apply(lambda x : x.sum()).reset_index()
            W_list,E_list = get_Wi_Ei_fromDF(sub_part=df_irreg,estimation_method=estimation_method,max_iter=max_iter,tol=tol)
            dict_couple_W_E['Irreg'] = [W_list,E_list]

        # Save the data :
        data_path_save = os.path.join(ForPerceptualTestPsyToolkitSurvey,'WinningProb_'+case+'_'+protocol+'.pkl')
        with open(data_path_save, 'wb') as pkl:
            pickle.dump(dict_couple_W_E,pkl)
 
def create_save_bar_plot(heights,error,path='',ext_name='',subset='',title='',
                         output_img='png'):
    # Build the plot
    if output_img=='pgf':
        matplotlib.use('pgf')
        plt.rcParams.update({"pgf.texsystem" : "pdflatex"})
        matplotlib.rcParams['text.usetex'] = True
        #matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{hyperref}'] #if needed
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        matplotlib.rcParams['pgf.preamble'] = [r'\usepackage{hyperref}', ]
    elif output_img=='tikz':
        plt.rc('text', usetex=True)
        
    x_pos = np.arange(len(listofmethod_onlySynth))
    # Color blind color cycle
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00','#A2C8EC', '#FFBC79']
    fig, ax = plt.subplots()
    ax.bar(x_pos, heights, yerr=error, align='center', alpha=0.5,color=CB_color_cycle, ecolor='black', capsize=10)
    ax.set_ylabel('Winning Prob')
    ax.set_xticks(x_pos)
    if output_img=='png': 
        ax.set_xticklabels(listNameMethod_onlySynth_withoutTF, rotation=45,fontsize=8)
    elif output_img=='pgf' or output_img=='tikz':
        ax.set_xticklabels(listNameMethod_onlySynth_withoutTF_withCite, rotation=45,fontsize=8,fontdict={'horizontalalignment': 'center'})
    plt.ylim((0,1))
    ax.set_title(title)
    ax.yaxis.grid(True)
    
    # Save the figure and show
    plt.tight_layout()
    if output_img=='png':
        path_fig = os.path.join(path,'Bar_plot_'+ext_name+'_'+subset+'.png')
        plt.savefig(path_fig,bbox_inches='tight')
    if output_img=='tikz':
        path_fig = os.path.join(path,'Bar_plot_'+ext_name+'_'+subset+'.tex')
        tikzplotlib.save(path_fig)
        # Two workaround sorry 
        # https://sourceforge.net/p/pgfplots/mailman/message/25027720/
        modify_underscore(path_fig)
        modify_labels(path_fig)
    elif output_img=='pgf':
        path_fig = os.path.join(path,'Bar_plot_'+ext_name+'_'+subset+'.pgf')
        plt.savefig(path_fig)
    #plt.show()
    plt.close()
           
def modify_underscore(path_fig):
    with open (path_fig, "r") as myfile:
        data=myfile.read()
    data = data.replace('\_','_')
    WriteTxtFile = open(path_fig, "w")
    WriteTxtFile.write(data)
    WriteTxtFile.close()
    
def modify_labels(path_fig):
    with open (path_fig, "r") as myfile:
        data=myfile.read()
    data = data.replace('rotate=45.0','rotate=45.0,align=center')
    data = data.replace('Gram + MSInit',r'{Gram +\\ MSInit}')
    data = data.replace('Gram + Spectrum + MSInit',r'{Gram +\\ Spectrum +\\ MSInit}')
    WriteTxtFile = open(path_fig, "w")
    WriteTxtFile.write(data)
    WriteTxtFile.close()
    
def create_significant_comp(params,std_matrix,path='',ext_name='',subset='',title='',zalpha=1.,
                            output_img='png'):
    
    if output_img=='png': 
        extension = 'png'
        local_list_label = listNameMethod_onlySynth_withoutTF
    elif output_img=='tikz':
        extension = 'tex'
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        local_list_label = listNameMethod_onlySynth_withoutTF_withCite 
    elif output_img=='pgf':
        extension = 'pgf'
        local_list_label = listNameMethod_onlySynth_withoutTF_withCiteForpgf 
        matplotlib.use('pgf')
        #plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        matplotlib.rcParams['pgf.preamble'] = [r'\usepackage{hyperref}', ]
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax) # , bbox=[0,0,1,1]
    tb.auto_set_font_size(False)
    tb.set_fontsize(18)

    color_win = 'lightgreen'
    color_loss = 'lightcoral'
    color_neutral = 'white'

    nrows, ncols = len(params),len(params)
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for i in range(len(params)):
        for j in range(len(params)):
            if not(i==j):
                #print(i,j)
                b_ij = params[i]-params[j]
                std_ij = std_matrix[i,j]
                text_ij = '{0:.2e}\n({1:.2e})'.format(b_ij,zalpha*std_ij)
                if b_ij > 0:
                    if b_ij - zalpha*std_ij > 0:
                        color = color_win
                    else:
                        color= color_neutral
                else:
                    if b_ij + zalpha*std_ij < 0:
                        color = color_loss
                    else:
                        color = color_neutral
                tb.add_cell(i, j, width, height, text=text_ij, 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label_raw in enumerate(local_list_label):
        if not(output_img=='pgf'):
            label_raw = label_raw.replace('_',' ')
        label = label_raw.replace('+','+\n')
        tb.add_cell(i, -1, width, height, text=label, loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label_raw in enumerate(local_list_label):
        if not(output_img=='pgf'):
            label_raw = label_raw.replace('_',' ')
        label = label_raw.replace('+','+\n')
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    
#    
#    fig, ax = plt.subplots()
#    
#    dict_ij = {}
#    
#    image = np.zeros(nrows*ncols)
#
#
#    
#    for i in range(len(params)):
#        for j in range(len(params)):
#            if not(i==j):
#                print(i,j)
#                b_ij = params[i]-params[j]
#                std_ij = std_matrix[i,j]
#                text_ij = '{0:.2e} ({1:.2e})'.format(b_ij,std_ij)
#                dict_ij[[i,j]] =  text_ij
#                
#                ax.text(i+1/2, j+1/2, text_ij, va='center', ha='center')
#    
#    plt.xticks(range(len(params)), listNameMethod_onlySynth,rotation=45)
#    plt.yticks(range(len(params)), listNameMethod_onlySynth)
#    
#    for tick in ax.xaxis.get_minor_ticks():
#        tick.tick1line.set_markersize(0)
#        tick.tick2line.set_markersize(0)
#        tick.label1.set_horizontalalignment('center')
#    for tick in ax.yaxis.get_minor_ticks():
#        tick.tick1line.set_markersize(0)
#        tick.tick2line.set_markersize(0)
#        tick.label1.set_horizontalalignment('center')
#    
#    ax.grid()
    
    ax.set_title(title)

    # Save the figure and show
    plt.tight_layout()
    if not(zalpha==1.0):
        name_fig = r'BetaValue_plot_'+ext_name+'_'+subset+'_zalpha'+str(zalpha).replace('.','')+'.'+extension
        path_fig = os.path.join(path,name_fig)
        if not(output_img=='tikz'):
            plt.savefig(path_fig,bbox_inches='tight',dpi=300)
        else:
            from tabulate import tabulate
            print(tabulate(tb, tablefmt="latex"))
            #tikzplotlib.save(path_fig) # Ne marche pas avec Table !!!
    else:
        name_fig = r'BetaValue_plot_'+ext_name+'_'+subset+'.'+extension
        path_fig = os.path.join(path,name_fig)
        if not(output_img=='tikz'):
            plt.show()
            plt.savefig(path_fig,bbox_inches='tight',dpi=300)
        else:
            from tabulate import tabulate
            print(tabulate(tb, tablefmt="latex"))
    plt.close()
    
    
def create_significant_compOutputToTex(params,std_matrix,path='',ext_name='',subset='',title='',zalpha=1.,
                            output_img='png'):
    """
    In this case we will output a tex file 
    """
    
    if output_img=='png': 
        extension = 'png'
        local_list_label = listNameMethod_onlySynth_withoutTF
    elif output_img=='tikz':
        extension = 'tex'
        local_list_label = listNameMethod_onlySynth_withoutTF_withCite 
    elif output_img=='pgf':
        extension = 'pgf'
        local_list_label = listNameMethod_onlySynth_withoutTF_withCite 
        matplotlib.use('pgf')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        matplotlib.rcParams['pgf.preamble'] = [r'\usepackage{hyperref}', ]
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax) # , bbox=[0,0,1,1]
    tb.auto_set_font_size(False)
    tb.set_fontsize(18)

    color_win = 'lightgreen'
    color_loss = 'lightcoral'
    color_neutral = 'white'

    nrows, ncols = len(params),len(params)
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for i in range(len(params)):
        for j in range(len(params)):
            if not(i==j):
                #print(i,j)
                b_ij = params[i]-params[j]
                std_ij = std_matrix[i,j]
                text_ij = '{0:.2e}\n({1:.2e})'.format(b_ij,zalpha*std_ij)
                if b_ij > 0:
                    if b_ij - zalpha*std_ij > 0:
                        color = color_win
                    else:
                        color= color_neutral
                else:
                    if b_ij + zalpha*std_ij < 0:
                        color = color_loss
                    else:
                        color = color_neutral
                tb.add_cell(i, j, width, height, text=text_ij, 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label_raw in enumerate(local_list_label):
        label = label_raw.replace('_',' ')
        label = label.replace('+','+\n')
        tb.add_cell(i, -1, width, height, text=label, loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label_raw in enumerate(local_list_label):
        label = label_raw.replace('_',' ')
        label = label.replace('+','+\n')
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    
#    
#    fig, ax = plt.subplots()
#    
#    dict_ij = {}
#    
#    image = np.zeros(nrows*ncols)
#
#
#    
#    for i in range(len(params)):
#        for j in range(len(params)):
#            if not(i==j):
#                print(i,j)
#                b_ij = params[i]-params[j]
#                std_ij = std_matrix[i,j]
#                text_ij = '{0:.2e} ({1:.2e})'.format(b_ij,std_ij)
#                dict_ij[[i,j]] =  text_ij
#                
#                ax.text(i+1/2, j+1/2, text_ij, va='center', ha='center')
#    
#    plt.xticks(range(len(params)), listNameMethod_onlySynth,rotation=45)
#    plt.yticks(range(len(params)), listNameMethod_onlySynth)
#    
#    for tick in ax.xaxis.get_minor_ticks():
#        tick.tick1line.set_markersize(0)
#        tick.tick2line.set_markersize(0)
#        tick.label1.set_horizontalalignment('center')
#    for tick in ax.yaxis.get_minor_ticks():
#        tick.tick1line.set_markersize(0)
#        tick.tick2line.set_markersize(0)
#        tick.label1.set_horizontalalignment('center')
#    
#    ax.grid()
    
    ax.set_title(title)

    # Save the figure and show
    plt.tight_layout()
    if not(zalpha==1.0):
        name_fig = 'BetaValue_plot_'+ext_name+'_'+subset+'_zalpha'+str(zalpha).replace('.','')+'.'+extension
        path_fig = os.path.join(path,name_fig)
        if not(output_img=='tikz'):
            plt.savefig(path_fig,bbox_inches='tight',dpi=300)
        else:
            tikzplotlib.save(path_fig)
    else:
        name_fig = 'BetaValue_plot_'+ext_name+'_'+subset+'.'+extension
        path_fig = os.path.join(path,name_fig)
        if not(output_img=='tikz'):
            plt.savefig(path_fig,dpi=300)
        else:
            tikzplotlib.save(path_fig)
    plt.close()
   
def get_std_Wi(params,std_Bi_minus_Bj):
    std_pij_matrix = get_std_pij(params,std_Bi_minus_Bj)
    n_method = len(listNameMethod_onlySynth)
    list_std_Wi = []
    for i,_ in enumerate(listofmethod_onlySynth):
        std_pij_tab = []
        for j,_ in enumerate(listofmethod_onlySynth):
            if not(i==j):
                std_pij = std_pij_matrix[i,j]
                #print(i,j,pij, pji)
                std_pij_tab += [std_pij]
#        std_pij_tab = np.concatenate(std_pij_tab)
        assert(len(std_pij_tab)==n_method-1)
        sum_std_pij = np.sqrt(np.sum(np.array(std_pij_tab)**2))
        #print(sum_pij)
        std_Wi = sum_std_pij / (n_method-1.)
        list_std_Wi += [std_Wi]
    return(list_std_Wi)
    
def get_std_Wi_old(std_Bi_minus_Bj):
    std_pij_matrix = get_std_pij_old(std_Bi_minus_Bj)
    n_method = len(listNameMethod_onlySynth)
    list_std_Wi = []
    for i,_ in enumerate(listofmethod_onlySynth):
        std_pij_tab = []
        for j,_ in enumerate(listofmethod_onlySynth):
            if not(i==j):
                std_pij = std_pij_matrix[i,j]
                #print(i,j,pij, pji)
                std_pij_tab += [std_pij]
#        std_pij_tab = np.concatenate(std_pij_tab)
        assert(len(std_pij_tab)==n_method-1)
        sum_std_pij = np.sqrt(np.sum(std_pij_tab**2))
        #print(sum_pij)
        std_Wi = sum_std_pij / (n_method-1.)
        list_std_Wi += [std_Wi]
    return(list_std_Wi)
 
def _safe_exp(x):
    x = np.clip(x,-np.inf,500)
    return np.exp(x)
    
def get_std_pij(params,std_Bi_minus_Bj):
    bij_matrix = np.zeros((len(params),len(params)))
    for i in range(len(params)):
        for j in range(len(params)):
            b_ij = params[i]-params[j]
            bij_matrix[i,j] = b_ij
    
    std_pij_matrix = (_safe_exp(bij_matrix)/((1+_safe_exp(bij_matrix))**2))*std_Bi_minus_Bj
    return(std_pij_matrix)
    
def get_std_pij_old(std_Bi_minus_Bj):
    std_pij_matrix = _safe_exp(std_Bi_minus_Bj)/(1+_safe_exp(std_Bi_minus_Bj))
    return(std_pij_matrix)
            
    
def plot_evaluation(estimation_method='mm',std_estimation='hessian',
                    protocol_tab = ['all_together','Individual_image'],
                    output_img='png'):
    """
    This function will plot the differents images 
    @param : output_img : kind of output used to save the image png is the default
        pgf is to gave the \cite in the figure itself : ne marche pas car il y a besoin de la taille des \cite qui ne sont pas la
    """        
    print("== Start Plotting the bar plot ==")
    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    
    diff_case=['global','local','both']
    #diff_case=['local','both']
     
    #protocol_tab = ['Individual_image']  

    output_im_path = os.path.join(ForPerceptualTestPsyToolkitSurvey,'BarPlot_score',std_estimation)
    pathlib.Path(output_im_path).mkdir(parents=True, exist_ok=True)
    for protocol in protocol_tab:
        print('=',protocol,'=')
        if protocol=='all_together':
            protocol_str = 'All images together'
        if protocol=='Individual_image':
            protocol_str = 'Each image as a study'
            
        for case in diff_case:
            print("=",case,"=")
            if case=='global':
                case_str = 'Global'
            elif case=='local':
                case_str = 'Local'
            else:
                case_str = 'Both'
            
            ext_name = protocol + '_'+case+'_'+estimation_method+'_'+std_estimation
            
            data_path_save =os.path.join(ForPerceptualTestPsyToolkitSurvey,'WBS_'+case+'_'+protocol+'_'+estimation_method+'_'+std_estimation+'.pkl')
            if not(os.path.exists(data_path_save)):
                print(data_path_save,'does not exist')
                break
            with open(data_path_save, 'rb') as pkl:
                dict_couple_W_E = pickle.load(pkl)
            
            if protocol=='Individual_image':
                # Per images
#                for j,file in enumerate(files_short):
#                    filewithoutext = '.'.join(file.split('.')[:-1])    
#                    print(j,filewithoutext)
#                    [W_list,stdW_list,params,std_matrix] = dict_couple_W_E[filewithoutext]
#                    create_save_bar_plot(W_list,stdW_list,path=output_im_path,ext_name=ext_name,subset=filewithoutext,title='')
#                    create_significant_comp(params,std_matrix,path=output_im_path,ext_name=ext_name,subset=filewithoutext,title='')
#                    create_significant_comp(params,std_matrix,path=output_im_path,ext_name=ext_name,subset=filewithoutext,zalpha=1.96,title='')

                print('All, Reg, Irreg')
                [W_list,E_list,stdW_list] = dict_couple_W_E['All'] # All images together
                create_save_bar_plot(W_list,E_list,path=output_im_path,ext_name=ext_name,subset='All',title='',output_img=output_img)
#                print('stdW_list',stdW_list)
#                print('E_list',E_list)
#                print('W_list',W_list)
                create_save_bar_plot(W_list,stdW_list,path=output_im_path,ext_name=ext_name+'_stdW',subset='All',title='',output_img=output_img)
                [W_list,E_list,stdW_list] = dict_couple_W_E['Reg'] # All images together
                create_save_bar_plot(W_list,E_list,path=output_im_path,ext_name=ext_name,subset='Reg',title='')
                create_save_bar_plot(W_list,stdW_list,path=output_im_path,ext_name=ext_name+'_stdW',subset='Reg',title='',output_img=output_img)
                [W_list,E_list,stdW_list] = dict_couple_W_E['All'] # All images together
                create_save_bar_plot(W_list,E_list,path=output_im_path,ext_name=ext_name,subset='Irreg',title='')
                create_save_bar_plot(W_list,stdW_list,path=output_im_path,ext_name=ext_name+'_stdW',subset='Irreg',title='',output_img=output_img)


            if protocol=='all_together':
                print('All, Reg, Irreg')
                [W_list,stdW_list,params,std_matrix] = dict_couple_W_E['All'] # All images together
                create_save_bar_plot(W_list,stdW_list,path=output_im_path,ext_name=ext_name,subset='All',title='',output_img=output_img)
                #create_significant_comp(params,std_matrix,path=output_im_path,ext_name=ext_name,subset='All',title='',output_img=output_img)
                #create_significant_comp(params,std_matrix,path=output_im_path,ext_name=ext_name,subset='All',zalpha=1.96,title='',output_img=output_img)
                [W_list,stdW_list,params,std_matrix] = dict_couple_W_E['Reg'] # All images together
                create_save_bar_plot(W_list,stdW_list,path=output_im_path,ext_name=ext_name,subset='Reg',title='',output_img=output_img)
                #create_significant_comp(params,std_matrix,path=output_im_path,ext_name=ext_name,subset='Reg',title='',output_img=output_img)
                #create_significant_comp(params,std_matrix,path=output_im_path,ext_name=ext_name,subset='Reg',zalpha=1.96,title='',output_img=output_img)
                [W_list,stdW_list,params,std_matrix] = dict_couple_W_E['All'] # All images together
                create_save_bar_plot(W_list,stdW_list,path=output_im_path,ext_name=ext_name,subset='Irreg',title='',output_img=output_img)
                #create_significant_comp(params,std_matrix,path=output_im_path,ext_name=ext_name,subset='Irreg',title='',output_img=output_img)
                #create_significant_comp(params,std_matrix,path=output_im_path,ext_name=ext_name,subset='Irreg',zalpha=1.96,title='',output_img=output_img)
            
    
def plot_evaluation_old():
    """
    This function will plot the differents images 
    """        
    print("== Start Plotting the bar plot ==")
    matplotlib.use('Agg') # To avoid to have the figure that's pop up during execution
    
    diff_case=['global','local','both']
    protocol_tab = ['all_together','Individual_image']
    

    output_im_path = os.path.join(ForPerceptualTestPsyToolkitSurvey,'BarPlot_score')
    pathlib.Path(output_im_path).mkdir(parents=True, exist_ok=True)
    for protocol in protocol_tab:
        print('=',protocol,'=')
        if protocol=='all_together':
            protocol_str = 'All images together'
        if protocol=='Individual_image':
            protocol_str = 'Each image as a study'
            
        for case in diff_case:
            print("=",case,"=")
            if case=='global':
                case_str = 'Global'
            elif case=='local':
                case_str = 'Local'
            else:
                case_str = 'Both'
            
            ext_name = protocol + '_'+case
            
            data_path_save = os.path.join(ForPerceptualTestPsyToolkitSurvey,'WinningProb_'+case+'_'+protocol+'.pkl')
            with open(data_path_save, 'rb') as pkl:
                dict_couple_W_E = pickle.load(pkl)
            
            if protocol=='Individual_image':
                # Per images
                for j,file in enumerate(files_short):
                    filewithoutext = '.'.join(file.split('.')[:-1])    
                    print(j,filewithoutext)
                    [W_list,E_list] = dict_couple_W_E[filewithoutext]
                    create_save_bar_plot(W_list,None,path=output_im_path,ext_name=ext_name,subset=filewithoutext,\
                                         title=filewithoutext+' '+case_str+' '+protocol_str)
                print('All, Reg, Irreg')
                [W_list,E_list] = dict_couple_W_E['All'] # All images together
                create_save_bar_plot(W_list,E_list,path=output_im_path,ext_name=ext_name,subset='All',title='All images'+' '+case_str+' '+protocol_str)
                [W_list,E_list] = dict_couple_W_E['Reg'] # All images together
                create_save_bar_plot(W_list,E_list,path=output_im_path,ext_name=ext_name,subset='Reg',title='Regular images'+' '+case_str+' '+protocol_str)
                [W_list,E_list] = dict_couple_W_E['All'] # All images together
                create_save_bar_plot(W_list,E_list,path=output_im_path,ext_name=ext_name,subset='Irreg',title='Irregular images'+' '+case_str+' '+protocol_str)

            if protocol=='all_together':
                print('All, Reg, Irreg')
                [W_list,E_lis] = dict_couple_W_E['All'] # All images together
                create_save_bar_plot(W_list,None,path=output_im_path,ext_name=ext_name,subset='All',title='All images'+' '+case_str+' '+protocol_str)
                [W_list,E_lis] = dict_couple_W_E['Reg'] # All images together
                create_save_bar_plot(W_list,None,path=output_im_path,ext_name=ext_name,subset='Reg',title='Regular images'+' '+case_str+' '+protocol_str)
                [W_list,E_lis] = dict_couple_W_E['All'] # All images together
                create_save_bar_plot(W_list,None,path=output_im_path,ext_name=ext_name,subset='Irreg',title='Irregular images'+' '+case_str+' '+protocol_str)
            
    


if __name__ == '__main__':
    #Resize_and_crop_center()
    #create_survey_for_PsyToolkit_4ques()
    #regrouper_resultats_psytoolkit()
    #run_statistical_study(estimation_method='mm',protocol='Individual_image')
#    run_statistical_study(estimation_method='opt_pairwise',
#                          protocol='Individual_image',
#                          std_estimation='hessian')
#    run_statistical_study(estimation_method='opt_pairwise',
#                          protocol='all_together',
#                          std_estimation='hessian')
    
    #plot_evaluation(estimation_method='opt_pairwise',std_estimation='hessian')
    
#    run_statistical_study(estimation_method='mm',protocol='all_together')
#    plot_evaluation()

    ## Cas avec le boostraping !
#    run_statistical_study(estimation_method='opt_pairwise',
#                          protocol='all_together',
#                          std_estimation='binomial')
#    plot_evaluation(estimation_method='opt_pairwise',std_estimation='binomial',
#                    protocol_tab = ['all_together'])
#    run_statistical_study(estimation_method='opt_pairwise',
#                          protocol='all_together',
#                          std_estimation='hessian')
#    plot_evaluation(estimation_method='opt_pairwise',std_estimation='hessian',
#                    protocol_tab = ['all_together'],output_img='tikz')
    plot_evaluation(estimation_method='opt_pairwise',std_estimation='hessian',
                    protocol_tab = ['all_together'],output_img='tikz')
