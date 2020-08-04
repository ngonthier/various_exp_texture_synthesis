# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:02:24 2019

@author: gonthier
"""

import Style_Transfer as st
from Arg_Parser import get_parser_args 

def testTexture():
    max_iter = 1000
    print_iter = 200
    parser = get_parser_args()
    name_texture = 'Camouflage_1'
    output_img_name = name_texture +'_firstlayer_DCT_net'
    parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,\
                        style_img_name=name_texture,loss=['texture'],\
                        output_img_name=output_img_name,\
                        vgg_name='firstlayer_DCT_net.mat',
                        recomputePrepocess=True)
    args = parser.parse_args()
    st.style_transfer(args)
    
def testTexture2():
    max_iter = 1000
    print_iter = 200
    parser = get_parser_args()
    name_texture = 'Camouflage_1'
    output_img_name = name_texture +'FirstLayer_dict_Camouflage_1'
    parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,\
                        style_img_name=name_texture,loss=['texture'],\
                        output_img_name=output_img_name,\
                        vgg_name='FirstLayer_dict_Camouflage_1.mat',
                        recomputePrepocess=True)
    args = parser.parse_args()
    st.style_transfer(args)
    
def testTexture3():
    max_iter = 1000
    print_iter = 200
    parser = get_parser_args()
    name_texture = 'TilesOrnate0158_1_S'
    output_img_name = name_texture +'FirstLayer_dict_TilesOrnate0158_1_S'
    parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,\
                        style_img_name=name_texture,loss=['texture'],\
                        output_img_name=output_img_name,\
                        vgg_name='firstlayer_atomSparseDict_net.mat',
                        recomputePrepocess=True)
    args = parser.parse_args()
    st.style_transfer(args)
    parser = get_parser_args()
    name_texture = 'TilesOrnate0158_1_S'
    output_img_name = name_texture +'Gatys_original'
    parser.set_defaults(verbose=True,max_iter=max_iter,print_iter=print_iter,\
                        style_img_name=name_texture,loss=['texture'],\
                        output_img_name=output_img_name,\
                        recomputePrepocess=True)
    args = parser.parse_args()
    st.style_transfer(args)
    
if __name__ == '__main__':
    testTexture2()
    testTexture3()