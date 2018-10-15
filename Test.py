#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 2017

The goal of this script is to test some function from the project

@author: nicolas
"""

import Style_Transfer as st
import pickle
from Arg_Parser import get_parser_args 
import seaborn as sns
import scipy
import scipy.stats
import tensorflow as tf

def test_moments_computation():
    shape_tensor = (1,300,400,64)
    dist_names = ['expon', 'norm']
    sess = tf.Session()
    epsilon = 10**(-1)
    epsilonmean = 10**(-1)
    for dist_name in dist_names:
        print(dist_name)
        dist = getattr(scipy.stats, dist_name)
        tensor4D = dist.rvs(size=shape_tensor)
        mean_x,variance_x,skewness_x,kurtosis_x = st.compute_4_moments(tensor4D)
        assert sess.run(tf.shape(mean_x))[0] == shape_tensor[3]
        assert sess.run(tf.shape(variance_x))[0] == shape_tensor[3]
        assert sess.run(tf.shape(skewness_x))[0] == shape_tensor[3]
        assert sess.run(tf.shape(kurtosis_x))[0] == shape_tensor[3]
        mean, var, skew, kurt = dist.stats(moments='mvsk')  
        kurt += 3 # because fisher definition of the formula 
        mean_x = sess.run(mean_x)
        variance_x = sess.run(variance_x)
        skewness_x= sess.run(skewness_x)
        kurtosis_x = sess.run(kurtosis_x)
        testOk = True
        for i in range(shape_tensor[3]):
            if(abs(mean-mean_x[i])>epsilonmean):
                print("Mean Error",i,"mean theoritical = ",mean," mean computed = ",mean_x[i])
                testOk = False
            if(abs(var-variance_x[i])/var>epsilon):
                print("Var Error",i,"Var theoritical = ",var," var computed = ",variance_x[i])
                testOk = False
            if(abs(skew-skewness_x[i])>epsilonmean):
                print("skewness Error",i,"skewness theoritical = ",skew," skewness computed = ",skewness_x[i])
                testOk = False
            if(abs(kurt-kurtosis_x[i])/kurt>epsilon):
                print("kurtosis Error",i,"kurtosis theoritical = ",kurt," kurtosis computed = ",kurtosis_x[i])
                testOk = False
    if(testOk) : 
        print("Test OK")
    else:
        print("Test not OK")        
    sess.close()
        
def test_n_moments_computation():
    
    shape_tensor = (1,300,400,64)
    dist_names = ['expon', 'norm']
    sess = tf.Session()
    epsilon = 10**(-1)
    epsilonmean = 10**(-1)
    for dist_name in dist_names:
        print(dist_name)
        dist = getattr(scipy.stats, dist_name)
        tensor4D = dist.rvs(size=shape_tensor)
        mean_x,variance_x,skewness_x,kurtosis_x = st.compute_n_moments(tensor4D,4)
        assert sess.run(tf.shape(mean_x))[0] == shape_tensor[3]
        assert sess.run(tf.shape(variance_x))[0] == shape_tensor[3]
        assert sess.run(tf.shape(skewness_x))[0] == shape_tensor[3]
        assert sess.run(tf.shape(kurtosis_x))[0] == shape_tensor[3]
        mean, var, skew, kurt = dist.stats(moments='mvsk')  
        kurt += 3 # because fisher definition of the formula 
        mean_x = sess.run(mean_x)
        variance_x = sess.run(variance_x)
        skewness_x= sess.run(skewness_x)
        kurtosis_x = sess.run(kurtosis_x)
        testOk = True
        for i in range(shape_tensor[3]):
            if(abs(mean-mean_x[i])>epsilonmean):
                print("Mean Error",i,"mean theoritical = ",mean," mean computed = ",mean_x[i])
                testOk = False
            if(abs(var-variance_x[i])/var>epsilon):
                print("Var Error",i,"Var theoritical = ",var," var computed = ",variance_x[i])
                testOk = False
            if(abs(skew-skewness_x[i])>epsilonmean):
                print("skewness Error",i,"skewness theoritical = ",skew," skewness computed = ",skewness_x[i])
                testOk = False
            if(abs(kurt-kurtosis_x[i])/kurt>epsilon):
                print("kurtosis Error",i,"kurtosis theoritical = ",kurt," kurtosis computed = ",kurtosis_x[i])
                testOk = False
    if(testOk) : 
        print("Test OK pour n moments")
    else:
        print("Test not OK pour n moments")     
    sess.close()

def resizer():
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Oct  1 18:26:33 2018

    @author: gonthier
    """

    import matplotlib.pyplot as plt
    import tensorflow as tf
    from skimage import data, color
    from skimage.transform import rescale, resize, downscale_local_mean
    import numpy as np
    import cv2

    image = data.chelsea()
    image = image[0:64,0:64,:]
    tf_resized = tf.image.resize_area(np.expand_dims(image,axis=0),[image.shape[0]//2, image.shape[1]//2],
                                          align_corners=True)
    sess = tf.Session()
    tf_resized_value = sess.run(tf_resized)[0]
    tf_resized_value = tf_resized_value.astype('uint8')
    #image_rescaled = rescale(image, 1.0 / 2.0, anti_aliasing=False)
    image_resized = cv2.resize(image, (image.shape[0]//2, image.shape[1]//2),
                           interpolation=cv2.INTER_AREA
                           ).astype('uint8')

    diff = np.abs(tf_resized_value - image_resized)
    print('Max of diff between TF et skimage',np.max(diff))
    diff2 = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    image_upscaled = resize(image, (image.shape[0]*2, image.shape[1] *2),anti_aliasing=True)

    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(tf_resized_value, cmap='gray')
    ax[1].set_title("Resized by TF image (aliasing ?)")

    ax[2].imshow(image_resized, cmap='gray')
    ax[2].set_title("Resized by Skimage (no aliasing)")
    #
    #ax[3].imshow(image_upscaled, cmap='gray')
    #ax[3].set_title("Upscale image (no aliasing)")
    ax[3].imshow(diff2, cmap='gray')
    ax[3].set_title("Difference")

    #ax[0].set_xlim(0, 512)
    #ax[0].set_ylim(512, 0)
    plt.tight_layout()
    plt.show()

def autreTest():
    import tensorflow as tf
    sess = tf.Session()
    innerProd = tf.complex(tf.constant(5.),tf.constant(6.))
    longformula  = tf.pow(tf.multiply(innerProd,tf.conj(innerProd)),0.5)
    byabs= tf.complex(tf.abs(innerProd),0.)
    print('Abs with tf.abs : {:.20e} '.format(sess.run(byabs)))
    print('Abs with long formula : {:.20e} '.format(sess.run(longformula)))
    #Abs with tf.abs :       7.81024980545043945312e+00+0.00000000000000000000e+00j 
	#Abs with long formula : 7.81024885177612304688e+00+0.00000000000000000000e+00j 
	# math.sqrt(6**2+5**2) = 7.810249675906654
if __name__ == '__main__':
    #test_moments_computation()
    #test_n_moments_computation()
    autreTest()
