#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:33:29 2017

The goal of this script is to study the gradient values

@author: nicolas
"""
import scipy
import time
import numpy as np
import tensorflow as tf
import Style_Transfer as st
import pickle
from Arg_Parser import get_parser_args 

def grad_computation(args):
    
    output_image_path = args.img_folder + args.output_img_name +args.img_ext
    image_content_path = args.img_folder + args.content_img_name +args.img_ext
    image_style_path = args.img_folder + args.style_img_name + args.img_ext
    image_content = st.preprocess(scipy.misc.imread(image_content_path).astype('float32')) # Float between 0 and 255
    image_style = st.preprocess(scipy.misc.imread(image_style_path).astype('float32')) 
    _,image_h, image_w, number_of_channels = image_content.shape 
    _,image_h_art, image_w_art, _ = image_style.shape 
    M_dict = st.get_M_dict(image_h,image_w)
    #print("Content")
    #plt.figure()
    #plt.imshow(postprocess(image_content))
    #plt.show()
    #print("Style")
    #plt.figure()
    #plt.imshow(postprocess(image_style))
    #plt.show()
    # TODO add something that reshape the image 
    # TODO : be able to have two different size for the image
    t1 = time.time()

    
    vgg_layers = st.get_vgg_layers()
    
    data_style_path = args.data_folder + "gram_"+args.style_img_name+"_"+str(image_h_art)+"_"+str(image_w_art)+".pkl"
    try:
        dict_gram = pickle.load(open(data_style_path, 'rb'))
    except(FileNotFoundError):
        if(args.verbose): print("The Gram Matrices doesn't exist, we will generate them.")
        dict_gram = st.get_Gram_matrix(vgg_layers,image_style)
        with open(data_style_path, 'wb') as output_gram_pkl:
            pickle.dump(dict_gram,output_gram_pkl)
        if(args.verbose): print("Pickle dumped")

    data_content_path = args.data_folder +args.content_img_name+"_"+str(image_h)+"_"+str(image_w)+".pkl"
    try:
        dict_features_repr = pickle.load(open(data_content_path, 'rb'))
    except(FileNotFoundError):
        if(args.verbose): print("The dictionnary of features representation of content image doesn't exist, we will generate it.")
        dict_features_repr = st.get_features_repr(vgg_layers,image_content)
        with open(data_content_path, 'wb') as output_content_pkl:
            pickle.dump(dict_features_repr,output_content_pkl)
        if(args.verbose): print("Pickle dumped")


    net = st.net_preloaded(vgg_layers, image_content) # The output image as the same size as the content one
    t2 = time.time()
    print("net loaded and gram computation after ",t2-t1," s")

    try:
        sess = tf.Session()
        
        if(not(args.start_from_noise)):
            try:
                init_img = st.preprocess(scipy.misc.imread(output_image_path).astype('float32'))
            except(FileNotFoundError):
                if(args.verbose): print("Former image not found, use of white noise mixed with the content image as initialization image")
                # White noise that we use at the beginning of the optimization
                init_img = st.get_init_noise_img(image_content,args.init_noise_ratio)
        else:
            init_img = st.get_init_noise_img(image_content,args.init_noise_ratio)

        #
        # TODO add a plot mode ! 
        #noise_imgS = postprocess(noise_img)
        #plt.figure()
        #plt.imshow(noise_imgS)
        #plt.show()
        
        # Propose different way to compute the lossses 
        style_loss = st.sum_style_losses(sess,net,dict_gram,M_dict)
        content_loss = args.content_strengh * st.sum_content_losses(sess, net, dict_features_repr) # alpha/Beta ratio 
        loss_total =  content_loss + style_loss
        
        if(args.verbose): print("init loss total")

        optimizer = tf.train.AdamOptimizer(args.learning_rate) # Gradient Descent
        # TODO function in order to use different optimization function
        train = optimizer.minimize(loss_total)

        sess.run(tf.global_variables_initializer())
        sess.run(net['input'].assign(init_img)) # This line must be after variables initialization ! 
        t3 = time.time()
        print("sess Adam initialized after ",t3-t2," s")
        # turn on interactive mode
        print("loss before optimization")
        st.print_loss(sess,loss_total,content_loss,style_loss)
        for i in range(args.max_iter):
            t3 =  time.time()
            sess.run(train)
            t4 = time.time()
            print("Iteration ",i, "after ",t4-t3," s")
            st.print_loss(sess,loss_total,content_loss,style_loss)
            result_img = sess.run(net['input'])
            result_img_postproc = st.postprocess(result_img)
            scipy.misc.toimage(result_img_postproc).save(output_image_path)
        
    except:
        print("Error")
        result_img = sess.run(net['input'])
        result_img_postproc = st.postprocess(result_img)
        #plt.imshow(result_img)
        #plt.show()
        output_image_path_error = args.img_folder + args.output_img_name+'_error' +args.img_ext
        scipy.misc.toimage(result_img_postproc).save(output_image_path_error)
        # In the case of the lbfgs optimizer we only get the init_img if we did not do a check point before
        raise 
    finally:
        print("Close Sess")
        sess.close()
        
def main():
    parser = get_parser_args()
    #style_img_name = "StarryNightBig"
    style_img_name = "wave_crop"
    content_img_name = "Louvre"
    max_iter = 100
    print_iter = 10
    start_from_noise = 1 # True
    init_noise_ratio = 0.7
    content_strengh = 0.001
    # In order to set the parameter before run the script
    parser.set_defaults(style_img_name=style_img_name,max_iter=max_iter,
        print_iter=print_iter,start_from_noise=start_from_noise,
        content_img_name=content_img_name,init_noise_ratio=init_noise_ratio,
        content_strengh=content_strengh)
    args = parser.parse_args()
    grad_computation(args)

if __name__ == '__main__':
    main()    
    
