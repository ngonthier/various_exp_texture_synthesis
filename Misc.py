#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function for image processing
"""


import numpy as np
import scipy


def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    
    @author: Gatys
    '''

    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r>cum_values.max()] = cum_values.max()    
        matched_image = inv_cdf(r).reshape(org_image.shape) 
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
            r[r>cum_values.max()] = cum_values.max()    
            matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)
        
    return matched_image
    
def histogram_matching_gradient(org_image, match_image, grey=False, n_bins=100):
	'''
	This function realize an histogram matching on the gradient of the image 
	TODO
	'''

	[b, h, w, d] = x.get_shape()
	b, h, w, d = tf.to_int32(b),tf.to_int32(h),tf.to_int32(w),tf.to_int32(d)
	tv_y_size = tf.to_float(b * (h-1) * w * d) # Nombre de pixels
	tv_x_size = tf.to_float(b * h * (w-1) * d)
	loss_y = tf.nn.l2_loss(x[:,1:,:,:] - x[:,:-1,:,:]) 
	loss_y /= tv_y_size
	loss_x = tf.nn.l2_loss(x[:,:,1:,:] - x[:,:,:-1,:]) 
	loss_x /= tv_x_size

def histogram_matching_for_tf(org_image, match_image, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image

    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''
    matched_image = np.zeros_like(org_image)
    _,_,_,numchannels = org_image.shape
    for i in range(numchannels):
        hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
        r[r>cum_values.max()] = cum_values.max()    
        matched_image[0,:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)
        
    return(matched_image)

def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram
    
    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))
   
   
# Those functions come from https://gist.github.com/bistaumanga/6309599    
def imhist(im):
  # calculates normalized histogram of an image
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]]+=1
    return np.array(h)/(m*n)

def cumsum(h):
    # finds cumulative sum of a numpy array, list
    return [sum(h[:i+1]) for i in range(len(h))]

def histeq(img):
    #calculate Histogram
    img2 = np.zeros(img.shape)
    for k in range(3):
        im = img[:,:,k]
        h = imhist(im)
        cdf = np.array(cumsum(h)) #cumulative distribution function
        sk = np.uint8(255 * cdf) #finding transfer function values
        s1, s2 = im.shape
        Y = np.zeros_like(im)
        # applying transfered values for each pixels
        for i in range(0, s1):
            for j in range(0, s2):
                Y[i, j] = sk[im[i, j]]
        img2[:,:,k] = Y
    #return transformed image, original and new istogram, 
    # and transform function
    return(img2)
