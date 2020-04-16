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
from itertools import permutations
import random
import math

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
#            small_img_name = filewithoutext_with_method +'_downsampled.png'
#            path_small_img = os.path.join(output_folder,small_img_name)
#            cv2.imwrite(path_small_img,small)
            
            # Crop center 
            height , width, channel = image.shape
            left_corner = int(round(width/2)) - int(round(target_width/2))
            top_corner = int(round(height/2)) - int(round(target_height/2))
            
            crop_img = image[left_corner:left_corner+target_width, top_corner:top_corner+target_height,:]
            
            # save the image
#            crop_img_name = filewithoutext_with_method +'_cropcenter.png'
#            path_crop_img = os.path.join(output_folder,crop_img_name)
#            cv2.imwrite(path_crop_img,crop_img)
            
            # Concatenate the two images
            vis= np.concatenate((small,white_column,crop_img), axis=1)
#            merge_img_name = filewithoutext_with_method +'_merge.png'
#            path_merge_img = os.path.join(output_merge_folder,merge_img_name)
#            cv2.imwrite(path_merge_img,vis)
            
    ImWidth = 256*2+26
    ImHeight = 256
    for file in files_short:
        filewithoutext = '.'.join(file.split('.')[:-1])
        print('Image :',filewithoutext)
            
        # Load the images
        all_pairs = permutations(listofmethod_onlySynth, 2) # [['_DCor','_SAME_Gatys_MSSInit']]
        for pair in all_pairs:
            print(pair)
            
            methodA, methodB = pair
            
            output_name = filewithoutext + methodA + methodB +'.png'
            
            filewithoutext_with_method = filewithoutext + ''
            stringname_Ref = os.path.join(output_merge_folder,filewithoutext_with_method+'_merge.png')
            imgRef = cv2.imread(stringname_Ref) 
            
            filewithoutext_with_methodA = filewithoutext + methodA
            stringname_A= os.path.join(output_merge_folder,filewithoutext_with_methodA+'_merge.png')
            imgA = cv2.imread(stringname_A) 
            
            filewithoutext_with_methodB = filewithoutext + methodB
            stringname_B= os.path.join(output_merge_folder,filewithoutext_with_methodB+'_merge.png')
            imgB = cv2.imread(stringname_B) 
            
            WIDTH = 256*4+26*4
            HEIGHT=256*2+26*2
            big_white_Image = np.uint8(255*np.ones((HEIGHT,WIDTH,3)))
            
            y0_ref = int((WIDTH-ImWidth)/2)
            y1_ref = y0_ref + ImWidth
            x0_ref = 0
            x1_ref = ImHeight
            big_white_Image[x0_ref:x1_ref,y0_ref:y1_ref,:] = imgRef
            
            y0_A = 0
            y1_A = ImWidth
            x0_A= ImHeight + 26*2
            x1_A = ImHeight*2 + 26*2
            big_white_Image[x0_A:x1_A,y0_A:y1_A,:] = imgA
            
            y0_B = ImWidth + 26*2
            y1_B = ImWidth*2 +26*2
            x0_B= ImHeight + 26*2
            x1_B = ImHeight*2 + 26*2
            big_white_Image[x0_B:x1_B,y0_B:y1_B,:] = imgB
            
            
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
        
        
    
    

if __name__ == '__main__':
#    Resize_and_crop_center()
    create_survey_for_PsyToolkit()
