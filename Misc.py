#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 2017

Utility function for image processing

@author: Nicolas
"""


import numpy as np
import scipy
from skimage.util import dtype, dtype_limits

#def _prepare_colorarray(arr):
    #"""Check the shape of the array and convert it to
    #floating point representation.
    #"""
    #arr = np.asanyarray(arr)

    #if arr.ndim not in [3, 4] or arr.shape[-1] != 3:
        #msg = ("the input array must be have a shape == (.., ..,[ ..,] 3)), " +
               #"got (" + (", ".join(map(str, arr.shape))) + ")")
        #raise ValueError(msg)

    #return dtype.img_as_float(arr)

#def rgb2hsi(rgb):
    #"""RGB to HSI color space conversion.
    #Parameters
    #----------
    #rgb : array_like
        #The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    #Returns
    #-------
    #out : ndarray
        #The image in HSI format, in a 3-D array of shape ``(.., .., 3)``.
    #Raises
    #------
    #ValueError
        #If `rgb` is not a 3-D array of shape ``(.., .., 3)``.
    #Notes
    #-----
    
    #References
    #----------
    #.. [1] http://en.wikipedia.org/wiki/HSL_and_HSV
    #Examples
    #--------
    #>>> from skimage import color
    #>>> from skimage import data
    #>>> img = data.astronaut()
    #>>> img_hsi = rgb2hsi(img)
    #Fonction non testee !!!! TODO
    #"""
    #arr = _prepare_colorarray(rgb)
    #out = np.empty_like(arr)

    ## -- V channel
    #out_I = np.mean(arr,axis=2)
    
    ## -- H channel
    #alpha = 0.5*(2*arr[:, :, 0] - arr[:, :, 1] - arr[:, :, 2])
    #beta = (np.sqrt(3)/2)*(arr[:, :, 2] - arr[:, :, 1])
    #H2 = np.arctan2(alpha,beta)
    #out_h = H2
    
    ## -- S channel
    #m = arr.min(-1)
    ## Ignore warning for zero divided by zero
    #old_settings = np.seterr(invalid='ignore')
    #out_s = 1. -( m / out_I )
    #out_s[out_I == 0.] = 0.
    #out_s[out_s <= 0.] = 0.
    #np.seterr(**old_settings)

    ## -- output
    #out[:, :, 0] = out_h
    #out[:, :, 1] = out_s
    #out[:, :, 2] = out_I

    ## remove NaN
    #out[np.isnan(out)] = 0.
        
    #assert((out_I >= 0.0).all() and (out_I <= 1.0).all())
    #assert((out_s >= 0.0).all() and (out_s <= 1.0).all())
    
    
    #print("out_h",np.max(out_h),np.min(out_h))
    
    #return out


#def hsi2rgb(hsi):
    #"""HSI to RGB color space conversion.
    #Parameters
    #----------
    #hsv : array_like
        #The image in HSI format, in a 3-D array of shape ``(.., .., 3)``.
    #Returns
    #-------
    #out : ndarray
        #The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    #Raises
    #------
    #ValueError
        #If `hsi` is not a 3-D array of shape ``(.., .., 3)``.
    #Notes
    #-----
    #Conversion between RGB and HS Icolor spaces 
    #Given an HSI color with hue H ∈ [0°, 360°], saturation SHSI ∈ [0, 1],
     #and intensity I ∈ [0, 1],
    #References
    #----------
    #.. [1] http://en.wikipedia.org/wiki/HSL_and_HSV
    #Examples
    #--------
    #>>> from skimage import data
    #>>> img = data.astronaut()
    #>>> img_hsi = rgb2hsi(img)
    #>>> img_rgb = hsi2rgb(img_hsv)
    #Fonction non testee !!!! TODO
    #"""
    #arr = _prepare_colorarray(hsi)

    #hi = np.floor(arr[:, :, 0] * 6)
    
    #z = 1. - np.abs((hi % 2) -1) 
    
    #c = 3*arr[:, :, 2]*arr[:, :, 1]/(1+z)

    #m = arr[:, :, 2]*(1-arr[:, :, 1])

    #x  = c*z

    #zeros = np.zeros_like(x)

    #hi = np.dstack([hi, hi, hi]).astype(np.uint8) % 6
    #out1 = np.choose(hi, [np.dstack((c, x, zeros)),
                         #np.dstack((x, c, zeros)),
                         #np.dstack((zeros, c, x)),
                         #np.dstack((zeros, x, c)),
                         #np.dstack((x, zeros, c)),
                         #np.dstack((c, zeros, x))])
                         
    #out = out1 + np.dstack((m,m,m))
    #print(out[0,0,0])
    #return out

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

if __name__ == '__main__':
    from skimage import data
    import matplotlib.pyplot as plt
    img = data.astronaut()
    print(np.max(img),np.min(img))
    plt.ion()
    plt.figure()
    plt.imshow(img)
    img_hsi = rgb2hsi(img)
    img_rgb = hsi2rgb(img_hsi)
    print(img_rgb[0,0,0])
    print(np.max(img_rgb),np.min(img_rgb))
    plt.figure()
    img_rgb_uint8 = (255*img_rgb).astype(np.uint8)
    plt.imshow(img_rgb_uint8)
    input("wait to end")
    

    
