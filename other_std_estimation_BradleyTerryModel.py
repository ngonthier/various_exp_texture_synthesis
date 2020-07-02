# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:48:17 2020

@author: gonthier
"""

import os
import os.path
import pandas as pd
import pickle
import numpy as np
from sympy.combinatorics.prufer   import Prufer

from read_data_standalone import read_duels_data,read_data_standalone

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
#ForPerceptualTestPsyToolkitSurvey=\
#    '/Users/said/Nextcloud_maison/Boulot/test_stats_texture'
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
files_short = ['BrickRound0122_1_seamless_S.png',
                 'BubbleMarbel.png',
                 'bubble_1024.png',
                 'CRW_3241_1024.png',
                 'CRW_3444_1024.png',
                 'CRW_5751_1024.png',
                 'fabric_white_blue_1024.png',
                 'glass_1024.png',
                 'lego_1024.png',
                 'marbre_1024.png',
                 'metal_ground_1024.png',
                 'Pierzga_2006_1024.png',
                 'rouille_1024.png',
                 'Scrapyard0093_1_seamless_S.png',
                 'TexturesCom_BrickSmallBrown0473_1_M_1024.png',
                 'TexturesCom_FloorsCheckerboard0046_4_seamless_S_1024.png',
                 'TexturesCom_TilesOrnate0085_1_seamless_S.png',
                 'TexturesCom_TilesOrnate0158_1_seamless_S.png',
                 'tricot_1024.png',
                 'vegetable_1024.png']

listImages = ['BrickRound0122_1_seamless_S',
                 'BubbleMarbel',
                 'bubble_1024',
                 'CRW_3241_1024',
                 'CRW_3444_1024',
                 'CRW_5751_1024',
                 'fabric_white_blue_1024',
                 'glass_1024',
                 'lego_1024',
                 'marbre_1024',
                 'metal_ground_1024',
                 'Pierzga_2006_1024',
                 'rouille_1024',
                 'Scrapyard0093_1_seamless_S',
                 'TexturesCom_BrickSmallBrown0473_1_M_1024',
                 'TexturesCom_FloorsCheckerboard0046_4_seamless_S_1024',
                 'TexturesCom_TilesOrnate0085_1_seamless_S',
                 'TexturesCom_TilesOrnate0158_1_seamless_S',
                 'tricot_1024',
                 'vegetable_1024']

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
listIrregularImages=list(set(listImages).difference(set(listRegularImages)))


#%% PARTIE SAID

def method_1_2(m1,m2,donnees,imgs='both'): #renvoie les votes pour m1 contre m2
    s=donnees.shape[0]
    v1=0
    v2=0
    if imgs=='both':
        listevoulue=listImages
    elif imgs=='regular':
        listevoulue=listRegularImages
    elif imgs=='irregular':
        listevoulue=listIrregularImages
    else:
        raise Exception('Ni regular ni irregular ni both')
    #print(listevoulue)
        
    for k in range(s):
        if donnees[k,0] in listevoulue:
            if donnees[k,1]==listofmethod[m1] and donnees[k,2]==listofmethod[m2]:
                
                #print('DIR')
                v1+=donnees[k,3]
                v2+=donnees[k,4]
            
            elif donnees[k,1]==listofmethod[m2] and donnees[k,2]==listofmethod[m1]:
                #print('INV')
                v2+=donnees[k,3]
                v1+=donnees[k,4]
    return (v1,v2)
    
def tab_duels(donnees,imgs='both'): #tableau des duels
    tabout=np.zeros((5,5))
    for k in range(1,5):
        for m in range(k+1,6):
            tmp=method_1_2(k,m,donnees,imgs=imgs)
            tabout[k-1,m-1]=tmp[0]
            tabout[m-1,k-1]=tmp[1]
    return tabout


def tab_p(tab_d): #calcul de la table des probas a partir des duels
    out=np.zeros((5,5))
    for k in range(5):
        for m in range(5):
            if m!=k:
                out[k,m]=tab_d[k,m]/(tab_d[k,m]+tab_d[m,k])
            else:
                out[k,m]=0.5
    return out

def genere_puissances(tab_p,graphe): #prend un graph qui est un arbre couvrant 
# et renvoie les puissances. on pose puissance de beta_0=0
    # un graph est une matrice 5x5 symétrique. 
    # le parcours du graphe est fait betement. 
    # tant qu'une puissance n'est pas connue on cherche à étendre 
    # depuis l'ensemble des puissances connues.
    idx=np.arange(0,5).astype(np.int)
    def puiss(p):
        return np.log(p/(1-p))
    def voisins(l): # a partir d'une liste d'indices renvoie les voisins
        out=set([])
        for k in l:
            out=out.union(set(idx[graphe[k]==1]))
        return out
    idxconnus=[0]
    valconnus=[0]
    while len(idxconnus)<5:
        vs=voisins(idxconnus)
        vs=vs.difference(set(idxconnus))
        vs=list(vs)
        for k in vs:
            idxconnus.append(k)
            valconnus.append(0)
            for m in idxconnus:
                if graphe[k,m]==1:
                    pm=valconnus[idxconnus.index(m)]
                    valconnus[-1]=(pm+puiss(tab_p[k,m]))
    out=np.zeros(5)
    for k in range(5):
        pk=idxconnus.index(k)
        out[k]=valconnus[pk]
    return out

def all_betas_trees(tab_probas):
    #from sympy.combinatorics.prufer   import Prufer
    betas=np.zeros((125,5)) # 125=5^3 le nombre d'arbres a 5 sommets
    c=Prufer([[0,1],[0,2],[0,3],[0,4]])
    for k in range(125):
        t=c.tree_repr
        M=np.zeros((5,5),dtype=np.int)
        for l in t:
            M[l[0],l[1]]=1
            M[l[1],l[0]]=1
        betas[k,:]=genere_puissances(tab_probas, M)
        c=c.next()
    return betas

def probas_from_betas(betas):
    lb=betas.shape[0]
    out=np.zeros((lb,5,5))
    for k in range(lb):
        for i in range(5):
            for j in range(5):
                di=betas[k,i]-betas[k,j]
                out[k,i,j]=1-1/(1+np.exp(di))
    return out

#%% Avec une estimation des beta_i avec l optimisation : 
    
def tirer(tabN,tabd):
    """
    la fonction tirer(tabN,tabd) 
    revoie un tableau de probas après tirage de tabN[i,j] +1 
    suivant la probailité tabd[i,j]/tabN[i,j]
    """
    
    
    
    

def std_estimation_by_combination(case='both',imgs='regular'):
    """
    Dans ce cas, on va regarder les differentes possibilitees d 'estimer les 
    valeurs de Beta_i avec les differentes configurations
    
    Cela revient a faire par boostraping !
    
    """
    
    donnees=read_duels_data(case=case).values
    tab_d=tab_duels(donnees,imgs=imgs)
    tab_probas=tab_p(tab_d)
    
    N_virutal_study = 3000
    # 3000 tirages (comme si tu simulais une autre étude en te basant sur les 
    # duels effectivement mesurés pour en générer des plausibles)
    
    betas=np.zeros((N_virutal_study,5))
    probs_tirages=np.zeros((N_virutal_study,5,5))

    for k in range(N_virutal_study):
         probas=tirer(tabN,tabd) 
         # la fonction tirer(tabN,tabd) 
         # revoie un tableau de probas après tirage de tabN[i,j] +1 
         # suivant la probailité tabd[i,j]/tabN[i,j]
         betas[k]=calcul_betas(probas) #(ta machinerie d'optimisation)
         probs_tirages[k]= probas_depuis_betas(betas[k])    
         # probas_depuis_betas calcule les probas depuis les betas suivant
         # la formule e(beta-beta)/(1+e(beta-beta))

    #Puis tu fais les statistiques sur probs_tirage


      
#%% utilisation 

# case peut both ou local ou global
#  imgs peut etre both ou regular ou irregular
donnees=read_duels_data(case='both').values
tab_d=tab_duels(donnees,imgs='regular')
tab_probas=tab_p(tab_d)

graphsimple=np.zeros((5,5),dtype=np.int)
for k in range(5):
    graphsimple[k,(k-1)%5]=1
    graphsimple[(k-1)%5,k]=1

betasimple=genere_puissances(tab_probas,graphsimple)

graphsimple2=np.zeros((5,5),dtype=np.int)
graphsimple2[0,1:]=1
graphsimple2[1:,0]=1
betasimple2=genere_puissances(tab_probas,graphsimple2)           

#%% On genere tous les arbres possibles sur 5 sommets
# chaque arbre permet d'inférer les puissances 

# from sympy.combinatorics.prufer   import Prufer
# betas=np.zeros((125,5)) # 125=5^3 le nombre d'arbres a 5 sommets
# c=Prufer([[0,1],[0,2],[0,3],[0,4]])
# for k in range(125):
#     t=c.tree_repr
#     M=np.zeros((5,5),dtype=np.int)
#     for l in t:
#         M[l[0],l[1]]=1
#         M[l[1],l[0]]=1
#     betas[k,:]=genere_puissances(tab_probas, M)
#     c=c.next()
# print(c.rank)

#%% GENERATTION DES FIGURES 

import matplotlib.pyplot as plt


idxs=np.arange(0,5)
for typeimage in ['both','regular','irregular']:
    for local in ['local']: #['both','global','local']:
        donnees=read_duels_data(case=local).values
        tab_d=tab_duels(donnees,imgs=typeimage)
        tab_probas=tab_p(tab_d)
        betas=all_betas_trees(tab_probas)
        probs=probas_from_betas(betas)
        print(probs)
        winnings=probs.sum(axis=0).sum(axis=0)/125/5
        wins=probs.sum(axis=1)/5
        stds=wins.std(axis=0)
        print('stds',stds.shape,stds)
        fig=plt.figure()
        fig.suptitle('type image:'+typeimage+' echelle:'+local)
        plt.stem(idxs,(1-winnings)-stds)
        plt.stem(idxs,(1-winnings)+stds)
plt.show()

idxs=np.arange(0,5)
for typeimage in ['both','regular','irregular']:
    for local in ['local']: #['both','global','local']:
        donnees=read_duels_data(case=local).values
        tab_d=tab_duels(donnees,imgs=typeimage)
        tab_probas=tab_p(tab_d)
        betas=all_betas_trees(tab_probas)
        probs=probas_from_betas(betas)
        for i in range(len(probs)):    
            prob_i = probs[i,:,:]
            prob_i[np.diag_indices_from(probs[i,:,:])] = 0.
            probs[i,:,:] = prob_i
        print(probs)
        winnings=probs.sum(axis=0).sum(axis=0)/125/4
        wins=probs.sum(axis=1)/4
        print('wins',wins.shape,wins)
        stds=wins.std(axis=0)
        print('stds',stds.shape,stds)
        fig=plt.figure()
        fig.suptitle('type image:'+typeimage+' echelle:'+local+' with i != j')
        plt.stem(idxs,(1-winnings)-stds)
        plt.stem(idxs,(1-winnings)+stds)
plt.show()

        

#%%