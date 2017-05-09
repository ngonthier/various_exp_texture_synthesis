#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 02 2017

The goal of this script is to vizualised the reponse of the filter of the
different convolution of the network

@author: nicolas
"""

import scipy
import numpy as np
import tensorflow as tf
import Style_Transfer as st
from Arg_Parser import get_parser_args 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

# Name of the 19 first layers of the VGG19
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


VGG19_LAYERS_INTEREST = (
    'conv1_1','conv2_1', 'conv3_1'
)

#VGG19_LAYERS_INTEREST = {'conv1_1'}

def hist(values,value_range,nbins=100,dtype=dtypes.float32):
    nbins_float = float(nbins)
    # Map tensor values that fall within value_range to [0, 1].
#    scaled_values = math_ops.truediv(values - value_range[0],
#                                     value_range[1] - value_range[0],
#                                     name='scaled_values') # values - value_range[0] / value_range[1] - value_range[0]
    scaled_values = tf.truediv(values - value_range[0],value_range[1] - value_range[0])
    scaled_values =tf.multiply(nbins_float,scaled_values)
    # map tensor values within the open interval value_range to {0,.., nbins-1},
    # values outside the open interval will be zero or less, or nbins or more.
   # indices = math_ops.floor(nbins_float * scaled_values, name='indices')
    indices = tf.floor(scaled_values)
    print(indices)
    print(type(indices))
    histo = indices
    # Clip edge cases (e.g. value = value_range[1]) or "outliers."
    #indices = math_ops.cast(
    #    clip_ops.clip_by_value(indices, 0, nbins_float- 1), dtypes.int32)

    # TODO(langmore) This creates an array of ones to add up and place in the
    # bins.  This is inefficient, so replace when a better Op is available.
    #histo= math_ops.unsorted_segment_sum(array_ops.ones_like(indices, dtype=dtype),indices,nbins)
    return(histo)

def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True

def plot_and_save(Matrix,path,name=''):
     Matrix = Matrix[0] # Remove first dim
     h,w,channels = Matrix.shape
     df_Matrix = pd.DataFrame(np.reshape(Matrix,(h*w,channels)))
     len_columns = len(df_Matrix.columns)
     if(len_columns<6):
         fig, axes = plt.subplots(1,len_columns)
     else:
         if(len_columns%4==0):
             fig, axes = plt.subplots(len_columns//4, 4)
         elif(len_columns%3==0):
             fig, axes = plt.subplots(len_columns//3, 3)
         elif(len_columns%5==0):
             fig, axes = plt.subplots(len_columns//5, 5)
         elif(len_columns%2==0):
             fig, axes = plt.subplots(len_columns//2, 2)
         else:
             j=6
             while(not(len_columns%j==0)):
                 j += 1
             fig, axes = plt.subplots(len_columns//j, j)
     
     i = 0
     axes = axes.flatten()
     for axis in zip(axes):
         df_Matrix.hist(column = i, bins = 64, ax=axis)
         i += 1
     pltname = path+name+'.png'
     # TODO avoid to Plot some ligne on the screen
     fig.savefig(pltname, dpi = 1000)

def plot_and_save_pdf(Matrix,path,name=''):
    pltname = path+name+'_hist.pdf'
    pltname_rep = path+name+'_img.pdf'
    pp = PdfPages(pltname)

    Matrix = Matrix[0] # Remove first dim
    h,w,channels = Matrix.shape
    df_Matrix = pd.DataFrame(np.reshape(Matrix,(h*w,channels)))
    len_columns = len(df_Matrix.columns)
    for i in range(len_columns):
        df_Matrix.hist(column = i, bins = 128)
        plt.savefig(pp, format='pdf')
        plt.close()
    pp.close()

    plt.clf()
    # Result of the convolution 
    pp_img = PdfPages(pltname_rep)
    for i in range(len_columns):
        plt.imshow(Matrix[:,:,i], cmap='gray')
        plt.savefig(pp_img, format='pdf')
        plt.close()
    pp_img.close()


def plot_Rep(args):
    
    directory_path = 'Results/Filter_Rep/'+args.style_img_name+'/' 
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    sns.set()
    image_style_path = args.img_folder + args.style_img_name + args.img_ext
    image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
    _,image_h_art, image_w_art, _ = image_style.shape 
    plot_and_save_pdf(image_style,directory_path,'ProcessIm')
    print("Plot initial image")
    
    vgg_layers = st.get_vgg_layers()
    net = st.net_preloaded(vgg_layers, image_style) # net for the style image
    sess = tf.Session()
    sess.run(net['input'].assign(image_style))
    for layer in VGG19_LAYERS:
        a = net[layer].eval(session=sess)
        print(layer,a.shape)
        plot_and_save_pdf(a,directory_path,layer)

def estimate_gennorm(args):
    
    	sns.set_style("white")
    image_style_path = args.img_folder + args.style_img_name + args.img_ext
    image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
    
    vgg_layers = st.get_vgg_layers()
    net = st.net_preloaded(vgg_layers, image_style) # net for the style image
    sess = tf.Session()
    sess.run(net['input'].assign(image_style))
    Distrib_Estimation = {}
    dict_pvalue = {}
    alpha = 0.1
    for layer in VGG19_LAYERS_INTEREST:
        print(layer)
        a = net[layer].eval(session=sess)
        a = a[0]
        h,w,number_of_features = a.shape
        a_reshaped = np.reshape(a,(h*w,number_of_features))
        print(h*w)
        Distrib_Estimation[layer] = np.array([])
        dict_pvalue[layer] = []
        for i in range(number_of_features):
            print(i)
            samples = a_reshaped[:,i]
            # This fit is computed by maximizing a log-likelihood function, with
            # penalty applied for samples outside of range of the distribution. The
            # returned answer is not guaranteed to be the globally optimal MLE, it
            # may only be locally optimal, or the optimization may fail altogether.
            beta, loc, scale = stats.gennorm.fit(samples)
            if(len(Distrib_Estimation[layer])==0):
                print("Number of points",len(samples))
                Distrib_Estimation[layer] = np.array([beta,loc,scale])
            else:
                Distrib_Estimation[layer] =  np.vstack((Distrib_Estimation[layer],np.array([beta,loc,scale])))
            # The KS test is only valid for continuous distributions. and with a theoritical distribution
            D,pvalue = stats.kstest(samples, 'gennorm',(beta, loc, scale ))
            dict_pvalue[layer]  += [pvalue]
            if(pvalue > alpha ): #p-value> α
                print(layer,i,pvalue)
                pass
        #print(Distrib_Estimation[layer])
        #print(dict_pvalue[layer])
    return(Distrib_Estimation)

def genTexture(args):
    
    image_style_path = args.img_folder + args.style_img_name + args.img_ext
    image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
    
    vgg_layers = st.get_vgg_layers()
    net = st.net_preloaded(vgg_layers, image_style) # net for the style image
    sess = tf.Session()
    sess.run(net['input'].assign(image_style))
    Distrib_Estimation = {}
    dict_pvalue = {}
    alpha = 0.1
    for layer in VGG19_LAYERS_INTEREST:
        print(layer)
        a = net[layer].eval(session=sess)
        a = a[0]
        h,w,number_of_features = a.shape
        a_reshaped = np.reshape(a,(h*w,number_of_features))
        print(h*w)
        Distrib_Estimation[layer] = np.array([])
        dict_pvalue[layer] = []
        for i in range(number_of_features):
            print(i)
            samples = a_reshaped[:,i]
            # This fit is computed by maximizing a log-likelihood function, with
            # penalty applied for samples outside of range of the distribution. The
            # returned answer is not guaranteed to be the globally optimal MLE, it
            # may only be locally optimal, or the optimization may fail altogether.
            beta, loc, scale = stats.gennorm.fit(samples)
            if(len(Distrib_Estimation[layer])==0):
                print("Number of points",len(samples))
                Distrib_Estimation[layer] = np.array([beta,loc,scale])
            else:
                Distrib_Estimation[layer] =  np.vstack((Distrib_Estimation[layer],np.array([beta,loc,scale])))
            # The KS test is only valid for continuous distributions. and with a theoritical distribution
            D,pvalue = stats.kstest(samples, 'gennorm',(beta, loc, scale ))
            dict_pvalue[layer]  += [pvalue]
    
#    VGG_Invert = ()
#    for layer in VGG_Invert:
#        if(layer in VGG19_LAYERS_INTEREST):
#            tf.Variable(np.zeros((1, height, width, numberChannels), dtype=np.float32))
        
    vgg_layers = st.get_vgg_layers()
    net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
    net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
  net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
  net['pool1']   = build_net('pool',net['conv1_2'])
  net['conv2_1'] = build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5))
  net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
  net['pool2']   = build_net('pool',net['conv2_2'])
net[] = build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10))

    generative_net = {}
    generative_net['conv3_1_input'] = tf.Variable(np.zeros(net['conv3_1'].shape, dtype=np.float32))
    weights = tf.constant(np.transpose(vgg_layers[10][0][0][2][0][0], (1, 0, 2, 3)))
    bias = -tf.constant(vgg_layers[10][0][0][2][0][1].reshape(-1))
    generative_net['conv3_1'] = tf.conv2d_transpose(tf.nn.bias_add(generative_net['conv3_1_input'],bias), weights, strides=(1, 1, 1, 1),	padding='SAME')
    generative_net['pool2'] = 
    
             
def generateArt(args):
    if args.verbose:
        print("verbosity turned on")
    
    output_image_path = args.img_folder + args.output_img_name +args.img_ext
    image_style_path = args.img_folder + args.style_img_name + args.img_ext
    image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
    _,image_h_art, image_w_art, _ = image_style.shape 

    t1 = time.time()
    vgg_layers = st.get_vgg_layers()
    net = st.net_preloaded(vgg_layers, image_style) # The output image as the same size as the content one
    t2 = time.time()
    if(args.verbose): print("net loaded and gram computation after ",t2-t1," s")

    try:
        sess = tf.Session()
        init_img = st.get_init_noise_img(image_style,1)
        loss_total =  hist_style_loss(sess,net,image_style)
        
        if(args.verbose): print("init loss total")
        print(tf.trainable_variables())
        #optimizer = tf.train.AdamOptimizer(args.learning_rate) # Gradient Descent
        #train = optimizer.minimize(loss_total)
        bnds = st.get_lbfgs_bnds(init_img)
        optimizer_kwargs = {'maxiter': args.max_iter,'iprint': args.print_iter}
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_total,bounds=bnds, method='L-BFGS-B',options=optimizer_kwargs)
        sess.run(tf.global_variables_initializer())
        sess.run(net['input'].assign(init_img)) # This line must be after variables initialization ! 
        optimizer.minimize(sess)
        t3 = time.time()
        if(args.verbose): print("sess Adam initialized after ",t3-t2," s")
        if(args.verbose): print("loss before optimization")
        if(args.verbose): print(sess.run(loss_total))
#        for i in range(args.max_iter):
#            if(i%args.print_iter==0):
#                t3 =  time.time()
#                sess.run(train)
#                t4 = time.time()
#                if(args.verbose): print("Iteration ",i, "after ",t4-t3," s")
#                if(args.verbose): print(sess.run(loss_total))
#                result_img = sess.run(net['input'])
#                result_img_postproc = st.postprocess(result_img)
#                scipy.misc.toimage(result_img_postproc).save(output_image_path)
#            else:
#                sess.run(train)
    except:
        print("Error")
        result_img = sess.run(net['input'])
        result_img_postproc = st.postprocess(result_img)
        output_image_path_error = args.img_folder + args.output_img_name+'_error' +args.img_ext
        scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
        raise 
    finally:
        if(args.verbose): print("Close Sess")
        sess.close()
    
def hist_style_loss(sess,net,style_img):
    #value_range = [-2000.0,2000.0] # TODO change according to the layer
    value_range = [-2000.0,2000.0] 
    style_value_range = {'conv1_1' : [-200.0,200.0],'conv2_1': [-500.0,500.0],'conv3_1' :  [-2000.0,2000.0] }
    nbins = 2048
    style_layers = [('conv1_1',1.),('conv2_1',1.),('conv3_1',1.)]
    #style_layers = [('conv1_1',1.)]
    #style_layers_size =  {'conv1' : 64,'conv2' : 128,'conv3' : 256,'conv4': 512,'conv5' : 512}
    length_style_layers = float(len(style_layers))
    sess.run(net['input'].assign(style_img))
    style_loss = 0.0
    weight_help_convergence = 10**9
    for layer, weight in style_layers:
        value_range = style_value_range[layer]
        style_loss_layer = 0.0
        a = sess.run(net[layer])
        _,h,w,N = a.shape
        M =h*w
        tf_M = tf.to_int32(M)
        tf_N = tf.to_int32(N)
        a_reshaped = tf.reshape(a,[tf_M,tf_N])
        a_split = tf.unstack(a_reshaped,axis=1)
        x = net[layer]
        #print("x.op",x.op)
        x_reshaped =  tf.reshape(x,[tf_M,tf_N])
        x_split = tf.unstack(x_reshaped,axis=1)
        
        for a_slide,x_slide in zip(a_split,x_split): # N iteration 
            # Descripteur des representations des histogrammes moment d'ordre 1 a N
            #hist_a = hist(a_slide,value_range, nbins=nbins,dtype=tf.float32)
            #hist_x = hist(x_slide,value_range, nbins=nbins,dtype=tf.float32)
            hist_a = tf.histogram_fixed_width(a_slide, value_range, nbins=nbins,dtype=tf.float32)
            hist_x = tf.histogram_fixed_width(x_slide, value_range, nbins=nbins,dtype=tf.float32)
            #hist_a = tf.floor(a_slide)
            #hist_x = tf.floor(x_slide)
            # TODO normalized les histogrammes avant le calcul plutot qu'apres
            #style_loss_layer += tf.to_float(tf.reduce_mean(tf.abs(hist_a- hist_x))) # norm L1
            #style_loss_layer += tf.reduce_mean(tf.pow(hist_a- hist_x,2)) # Norm L2
            style_loss_layer += tf.sqrt(1-tf.reduce_sum(tf.multiply(tf.sqrt(hist_a),tf.sqrt(hist_x))))
            # TODO use bhattacharyya distance
            
        style_loss_layer *= weight * weight_help_convergence  / (2.*tf.to_float(N**2)*tf.to_float(M**2)*length_style_layers)
        style_loss += style_loss_layer
    return(style_loss)

def 

def main_plot():
    parser = get_parser_args()
    style_img_name = "StarryNight"
    #style_img_name = "Louvre_Big"
    parser.set_defaults(style_img_name=style_img_name)
    args = parser.parse_args()
    plot_Rep(args)
    
def main_distrib():
    parser = get_parser_args()
    style_img_name = "StarryNight"
    parser.set_defaults(style_img_name=style_img_name)
    args = parser.parse_args()
    estimate_gennorm(args)

if __name__ == '__main__':
    parser = get_parser_args()
    style_img_name = "StarryNight"
    output_img_name = "Texture"
    max_iter = 10
    print_iter = 1
    parser.set_defaults(style_img_name=style_img_name,output_img_name=output_img_name,
                        max_iter=max_iter,    print_iter=print_iter)
    args = parser.parse_args()
    generateArt(args)
    