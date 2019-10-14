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
import pywt # Wavelet
import pickle

from scipy.stats import gennorm
from scipy.special import gamma

directory = "./im/References/"
ResultsDir = "./im/"
path_base  = os.path.join('C:\\','Users','gonthier')
ownCloudname = 'ownCloud'
if not(os.path.exists(path_base)):
    path_base  = os.path.join(os.sep,'media','gonthier','HDD')
    ownCloudname ='owncloud'
if os.path.exists(path_base):
    ResultsDir = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','Images Textures Résultats')
    directory = os.path.join(path_base,ownCloudname,'These Gonthier Nicolas Partage','Images Textures References Subset')
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

listofmethod = ['','_SAME_Gatys','_SAME_Gatys_MSSInit','_SAME_Gatys_spectrumTFabs_eps10m16','_SAME_Gatys_spectrumTFabs_eps10m16_MSSInit',\
    '_SAME_autocorr','_SAME_autocorr_MSSInit','MultiScale_o5_l3_8_psame','_DCor','_EfrosLeung','_EfrosFreeman']
listNameMethod = ['Reference','Gatys','Gatys + MSInit','Gatys + Spectrum TF','Gatys + Spectrum TF + MSInit',\
    'Autocorr','Autocorr + MSInit','Snelgorove','Deep Corr','EfrosLeung','EfrosFreeman']

trucEnPlus = ['_SAME_OnInput_autocorr','_SAME_OnInput_autocorr_MSSInit','_SAME_OnInput_SpectrumOnFeatures']

listofmethod += trucEnPlus
listNameMethod += trucEnPlus

cmap='viridis' 
#cmap='plasma' 
files_short = files
#files_short = [files[-1]]
files_short = ['BrickRound0122_1_seamless_S.png','TexturesCom_FloorsCheckerboard0046_4_seamless_S_1024.png']

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
    ahist, bhist = (np.histogram(a, bins=nbins,density=True)[0],
                    np.histogram(b, bins=nbins,density=True)[0])
    return kl(ahist, bhist)
    


def gennorm_kl_distance(a, b,verbose=False):
    beta_a,loc_a,scale_a = gennorm.fit(a) # beta =  shape parameter and scale = alpha
    beta_b,loc_b,scale_b = gennorm.fit(b)
    if verbose:
        print('beta_a,loc_a,scale_a')
        print(beta_a,loc_a,scale_a)
        print('beta_b,loc_b,scale_b')
        print(beta_b,loc_b,scale_b )
    D = np.log((beta_a*scale_b*gamma(1./beta_b)) / (beta_b*scale_a*gamma(1./beta_a))) +  (((scale_a/scale_b)**(beta_b)) * (gamma((beta_b+1)/beta_a)/gamma(1/beta_a))) - 1./beta_a

    return D

def plot_hist_of_coeffs(coeffs,namestr=''):
    
    n_bins= 50
    axes = []
    number_img_h = len(coeffs)-1
    number_img_w = 3
    #cAn = coeffs[0]
    plt.figure()
    gs00 = gridspec.GridSpec(number_img_h, number_img_w)
    list_str = ['cH','cV','cD']

    for j in range(number_img_w*number_img_h):
        ax = plt.subplot(gs00[j])
        axes += [ax]

    for k,ax in enumerate(axes):
        scale_ = (k // 3  ) + 1
        o_ = k % 3 # Orientation
#        print('scale_,o_',scale_,o_)
        coeffs_s_o = coeffs[scale_][o_]
        coeffs_s_o = coeffs_s_o.reshape(-1,3)
#        print('len(coeffs_s_o)',len(coeffs_s_o))
#        print('shape(coeffs_s_o)',coeffs_s_o.shape)
        title_local_str = list_str[o_] + str(len(coeffs)-scale_)
        _ = ax.hist(coeffs_s_o,n_bins, density=False,label=['R','G','B'],color=['r','g','b'])
        ax.tick_params(axis='both', which='major', labelsize=3)
        ax.tick_params(axis='both', which='minor', labelsize=3)
        ax.legend(loc='upper right', prop={'size': 2})
        ax.set_title(title_local_str)
    titre = 'DB4 coeffs ' +namestr
    plt.suptitle(titre)
    plt.show()

def main(args):
    """
    compute the quality measure
    """
    # path_output = os.path.join('Spectrum','Crops',str(dim))
    # pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
    verbose = False
    plot_hist = False
    With_formula = True # If False we will use the histogram
    number_of_scale = 3
    
    name = 'Wavelets_KL_'+str(number_of_scale)+'Scale'
    if With_formula:
        name += '_ExplicitFormula'
    else:
        name +=  '_Hist'
    name += '.pkl'
    data_path_save = os.path.join('data',name)
    
    wavelet_db4 = pywt.Wavelet('db4') # Daubechies D4 : lenght filter = 8
    # In this experiment, we employed the conventional pyramid
    # wavelet decomposition with three levels using the Daubechies’
    # maximally flat orthogonal filters of length 8 ( filters)
    dictTotal = {}
    dictTotal_all = {}
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
            #print(method,nameMethod)
            stringnamepng = stringname + '.png'
            stringnamejpg = stringname + '.jpg'
            if os.path.isfile(stringnamepng):
                filename = stringnamepng
            elif os.path.isfile(stringnamejpg):
                filename = stringnamejpg
            else:
                print('Neither',stringnamepng,' or ',stringnamejpg,'exist. Wrong path ?')
            #print(filename)
            imagergb = io.imread(filename)
            
            # Multilevel 2D Discrete Wavelet Transform. if level = None take the maximum
            coeffs = pywt.wavedec2(imagergb, wavelet_db4, mode='symmetric', level=number_of_scale,\
                          axes=(0,1))
            
            if plot_hist:
                plot_hist_of_coeffs(coeffs,namestr=nameMethod)
            # len(coeffs)  = level + 1 
            
            dict_imgs[nameMethod] = coeffs
        
#        # Compute the metrics between the reference and the images

        if With_formula:
            dict_scores = {}
            dict_all_scores = {}
        else:
            dict_all_scores_hist = {}
            dict_scores_hist = {}
        for method,nameMethod in zip(listofmethod,listNameMethod):
            if nameMethod=='Reference': # Reference one
                continue
            ref_coeff = dict_imgs['Reference']
            syn_coeff = dict_imgs[nameMethod]
            if With_formula:
                dict_all_scores[nameMethod] = []
                dict_scores[nameMethod] = 0.
            else:
                dict_scores_hist[nameMethod] = 0.
                dict_all_scores_hist[nameMethod] = []
#            ref_img = dict_imgs['Reference']
#            syn_img = dict_imgs[nameMethod]
            for s in range(number_of_scale+1): # scale
                if s==0:
                    # Average case : cAn : ie approximation at rank n 
                    # we will pass it 
                    cAn = ref_coeff[s]
                    continue
                else:
                    # (cHn, cVn, cDn) : the coefficient are provide in the decreasing
                    # order
                    (ref_cHn, ref_cVn, ref_cDn) = ref_coeff[s]
                    (syn_cHn, syn_cVn, syn_cDn) = syn_coeff[s]
                    # We will extract 2 * 9  parameters (2 per generalised gaussian, for 3 scale times 3 orientations)
                    o = 0
                    for ref_coeffs_s_o,syn_coeffs_s_o in zip([ref_cHn, ref_cVn, ref_cDn],[syn_cHn, syn_cVn, syn_cDn]):
                        for c in range(3): # Color channet
                            ref_coeffs_s_o_c = ref_coeffs_s_o[:,:,c].ravel()
                            syn_coeffs_s_o_c = syn_coeffs_s_o[:,:,c].ravel()
                            
                            if With_formula:
                                # KL divergence with explicit formula
                                gennorm_kl_s_o_c = gennorm_kl_distance(ref_coeffs_s_o_c, syn_coeffs_s_o_c,verbose=verbose)
                                if verbose : print('kl with gennorm at s : ',s,'o : ',o,'c : ',c,' = ',gennorm_kl_s_o_c)
                                dict_all_scores[nameMethod] += [gennorm_kl_s_o_c]
                                dict_scores[nameMethod] += gennorm_kl_s_o_c
                            else:
                                # Use Histogram to estimate KL divergence
                                hist_kl_s_c = hist_kl_distance(ref_coeffs_s_o_c, syn_coeffs_s_o_c, nbins=10)
                                if verbose : print('kl with hist at s : ',s,'o : ',o,'c : ',c,' = ',hist_kl_s_c)
                                dict_all_scores_hist[nameMethod] += [hist_kl_s_c]
                                dict_scores_hist[nameMethod] += hist_kl_s_c
                            
                        o += 1
            # Print one global score = sum at different scale of the different orientation and for the 3 color     
            if With_formula:       
                print('Reference against ',nameMethod,dict_scores[nameMethod])
            else:
                print('Reference against ',nameMethod,dict_scores_hist[nameMethod],' with hist')
		if With_formula:
			dictTotal[filewithoutext] = dict_scores
			dictTotal_all[filewithoutext] = dict_all_scores
		else:
			dictTotal[filewithoutext] = dict_scores_hist
			dictTotal_all[filewithoutext] = dict_all_scores_hist
    
    with open(data_path_save, 'wb') as pkl:
        pickle.dump(data,pkl)

def readData():
    With_formula = True # If False we will use the histogram
    number_of_scale = 3
    
    name = 'Wavelets_KL_'+str(number_of_scale)+'Scale'
    if With_formula:
        name += '_ExplicitFormula'
    else:
        name +=  '_Hist'
    name += '.pkl'
    data_path_save = os.path.join('data',name)
    with open(data_path_save, 'rb') as pkl:
         data = pickle.load(pkl)
    dict_all_scores,dict_scores = data
    print(dict_scores)
                        
#                if s==0:
#                    ref_img_s = ref_img
#                    syn_img_s = syn_img
#                else:
#                    ref_img_s = resize(ref_img, (ref_img.shape[0] // (2*i), ref_img.shape[1] // (2*i)),
#                       anti_aliasing=True)
#                    syn_img_s = resize(syn_img, (syn_img.shape[0] // (2*i), syn_img.shape[1] //(2*i)),
#                       anti_aliasing=True)
                
#                for c in range(3): # color loop
#                    ref_img_s_c = ref_coeff[:,:,c].ravel()
#                    syn_img_s_c = syn_coeff[:,:,c].ravel()
#                    hist_kl_s_c = hist_kl_distance(ref_img_s_c, syn_img_s_c, nbins=10)
#                    gennorm_kl_s_c = gennorm_kl_distance(ref_img_s_c, syn_img_s_c)
#                    print(nameMethod,'scale :',s,'color :',c,'kl with hist = ',hist_kl_s_c)
#                    print('kl with gennorm = ',gennorm_kl_s_c)
#            
            

if __name__ == '__main__':
    # main()
    readData()
    # import sys
    # sys.exit(main(sys.argv))
