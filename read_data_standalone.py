# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:06:36 2020

@author: gonthier
"""

import os
import os.path

import pickle

directory = "./im/References/"
ResultsDir = "./im/"
#if os.environ.get('OS','') == 'Windows_NT':
path_base  = os.path.join('C:\\','Users','gonthier')
ownCloudname = 'ownCloud'

if not(os.path.exists(path_base)):
    path_base  = os.path.join(os.sep,'media','gonthier','HDD')
    ownCloudname ='owncloud'
if os.path.exists(path_base):
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
listNameMethod_onlySynth = ['Gatys','Gatys + MSInit','Gatys + Spectrum TF + MSInit',\
    'Snelgorove','Deep Corr']

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

def read_data_standalone(protocol='all_together',
                         case='global',
                         set_of_images = 'All',
                         estimation_method='opt_pairwise',
                         std_estimation='hessian'
                         ):
    """
    This function return the values computed by Nicolas in the different cases :
        One have to mentionned 
    @param protocol : the protocol used : 'all_together' or 'Individual_image'
    @param case : the scale (global local or both)        
    @param : set_of_images of images considered All, Reg or Irreg
    
    In the case of the all_together protocol, the function will return a list of 4 elements :
        - the winning probability per method W_i 
        - the std of this winning probability GrandSigma_i
        - the list of the strenght Beta_i
        - the matrix of the standard error se_ij of the difference Beta_i - Beta_j
        the parameters Beta_i computed but also the 
    
    
    """        
    estimation_method='opt_pairwise'
    std_estimation='hessian'
    diff_case=['global','local','both']
    protocol_tab = ['all_together','Individual_image']  
    set_of_images_tab = ['All','Reg','Irreg']
    # In the case of all 
    assert(protocol in protocol_tab)
    assert(case in diff_case)
    assert(set_of_images in set_of_images_tab)

            
    data_path_save =os.path.join(ForPerceptualTestPsyToolkitSurvey,'WBS_'+case+'_'+protocol+'_'+estimation_method+'_'+std_estimation+'.pkl')
    if not(os.path.exists(data_path_save)):
        print(data_path_save,'does not exist')
        raise(ValueError(data_path_save))
    with open(data_path_save, 'rb') as pkl:
        dict_couple_W_E = pickle.load(pkl)
    
    if protocol=='Individual_image':
        # Per images
        list_outputs = [] 
        for j,file in enumerate(files_short):
            filewithoutext = '.'.join(file.split('.')[:-1])    
            #print(j,filewithoutext)
            [W_list,stdW_list,list_Beta_i,seij_matrix] = dict_couple_W_E[filewithoutext]
            if set_of_images=='All':
                list_outputs += [[W_list,stdW_list,list_Beta_i,seij_matrix]]
            elif set_of_images=='Reg':
                if filewithoutext in listRegularImages:
                    list_outputs += [[W_list,stdW_list,list_Beta_i,seij_matrix]]
            elif set_of_images=='Irreg':
                if not(filewithoutext in listRegularImages):
                    list_outputs += [[W_list,stdW_list,list_Beta_i,seij_matrix]]
            
        [W_list,E_list,stdW_list] = dict_couple_W_E[set_of_images]
        list_winning_prob_over_images = [W_list,E_list,stdW_list]
        return(list_winning_prob_over_images,list_outputs)

    if protocol=='all_together':
        
        [W_list,stdW_list,list_Beta_i,seij_matrix] = dict_couple_W_E[set_of_images]
        return([W_list,stdW_list,list_Beta_i,seij_matrix])
        
if __name__ == '__main__':
    [W_list,stdW_list,list_Beta_i,seij_matrix] = read_data_standalone(protocol='all_together',
                         case='global',
                         set_of_images = 'All')
    print('Liste parameters beta i for all_together global All images',list_Beta_i)
    print('seij for all_together global All images',seij_matrix)
    list_winning_prob_over_images,list_outputs = read_data_standalone(protocol='Individual_image',
                         case='local',
                         set_of_images = 'Reg')
    [W_list,E_list,stdW_list] = list_winning_prob_over_images
    print('Winning probability over images',W_list)
    
    print('len(list_outputs)',len(list_outputs))
    [W_list,stdW_list,list_Beta_i,seij_matrix] = list_outputs[0]
    print('Liste parameters beta i for the first regular images in the case local',list_Beta_i)
    print('seij for the first regular images in the case local',seij_matrix)
    