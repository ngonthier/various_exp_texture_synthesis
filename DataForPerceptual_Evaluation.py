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

import pandas as pd

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
                                number_win_all_images.loc[index,'winA'] += 1
                            else:
                                number_win_all_images.loc[index,'winB'] += 1
                        elif value==2.0:
                            if right_order:
                                number_win_all_images.loc[index,'winB'] += 1
                            else:
                                number_win_all_images.loc[index,'winA'] += 1
                        else: # Tie
                            number_win_all_images.loc[index,'winB'] += 0.5
                            number_win_all_images.loc[index,'winA'] += 0.5
        
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
       
#def convert_pd_df_to_list_wins
        
def run_statistical_study():
        
    diff_case=['global','local','both']
    for case in diff_case:
        path_df = os.path.join(ForPerceptualTestPsyToolkitSurvey,'Number_of_wins_'+case+'.csv')
        number_win_all_images =  pd.read_csv(path_df,sep=',')
        
        # Per images
        for j,file in enumerate(files_short):
            filewithoutext = '.'.join(file.split('.')[:-1])
            sub_part = number_win_all_images[number_win_all_images['image']==filewithoutext]
            
            
            

            
    
    


if __name__ == '__main__':
    #Resize_and_crop_center()
    create_survey_for_PsyToolkit_4ques()
