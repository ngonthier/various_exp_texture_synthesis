"""
Created on Wed 10 october 2018

This script have the goal to investigate the spectrum constraint case

@author: nicolas
"""

import tensorflow as tf
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import Style_Transfer as st
from Arg_Parser import get_parser_args 
from skimage.color import gray2rgb

eps = 0.001

cmap = 'gray'
cmap = 'magma'

def getImageXwithPhaseOfA_maison(x,a):
    """ Get from the loss_spectrum """
    print("Fonction Maison")
    a = tf.transpose(a, [0,3,1,2]) # On passe l image de batch,h,w,canaux à  batch,canaux,h,w
    F_a = tf.fft2d(tf.complex(a,0.)) # TF de l image de reference

    x_t = tf.transpose(x, [0,3,1,2])
    F_x = tf.fft2d(tf.complex(x_t,0.)) # Image en cours de synthese 

    # Element wise multiplication of FFT and conj of FFT
    if tf.__version__ < '1.8':
        innerProd = tf.reduce_sum(tf.multiply(F_x,tf.conj(F_a)), 1, keep_dims=True)  # sum(ftIm .* conj(ftRef), 3);
    else:
        innerProd = tf.reduce_sum(tf.multiply(F_x,tf.conj(F_a)), 1, keepdims=True)  # sum(ftIm .* conj(ftRef), 3);
    # Shape = [  1   1 512 512] pour une image 512*512
    #module_InnerProd = tf.pow(tf.multiply(innerProd,tf.conj(innerProd)),0.5) # replace by tf.abs
    print('innerProd',innerProd)
    #if tf.__version__ > '1.4':
        #module_InnerProd = tf.complex(tf.abs(innerProd),0.) # Possible with tensorflow 1.4
    #else:
    module_InnerProd = tf.pow(tf.multiply(innerProd,tf.conj(innerProd)),0.5)
    print('module_InnerProd',module_InnerProd)
    dephase = tf.divide(innerProd,tf.add(module_InnerProd,eps))
    print('dephase',dephase)
    ftNew =  tf.multiply(dephase,F_a) #compute the new version of the FT of the reference image
    # Element wise multiplication
    imF = tf.ifft2d(ftNew)
    imF =  tf.real(tf.transpose(imF, [0,2,3,1]))
    return(imF,ftNew,dephase,module_InnerProd,innerProd)
    
    
def getImageXwithPhaseOfA_tfabs(x,a):
    """ loss_spectrumGang """
    print("With tf.asb")
    a = tf.transpose(a, [0,3,1,2]) # On passe l image de batch,h,w,canaux à  batch,canaux,h,w
    F_a = tf.fft2d(tf.complex(a,0.)) # TF de l image de reference

    x_t = tf.transpose(x, [0,3,1,2])
    F_x = tf.fft2d(tf.complex(x_t,0.)) # Image en cours de synthese 

    # Element wise multiplication of FFT and conj of FFT
    if tf.__version__ < '1.8':
        innerProd = tf.reduce_sum(tf.multiply(F_x,tf.conj(F_a)), 1, keep_dims=True)  # sum(ftIm .* conj(ftRef), 3);
    else:
        innerProd = tf.reduce_sum(tf.multiply(F_x,tf.conj(F_a)), 1, keepdims=True)  # sum(ftIm .* conj(ftRef), 3);
    # Shape = [  1   1 512 512] pour une image 512*512
    #module_InnerProd = tf.pow(tf.multiply(innerProd,tf.conj(innerProd)),0.5) # replace by tf.abs
    print('innerProd',innerProd)
    if tf.__version__ > '1.4':
        #module_InnerProd= tf.complex(tf.sqrt(tf.real(tf.add(tf.pow(tf.real(innerProd),2),tf.pow(tf.imag(innerProd),2)))),0.)
        module_InnerProd = tf.complex(tf.abs(innerProd),0.) # Possible with tensorflow 1.4
    else:
        raise(NotImplemented)
    print('module_InnerProd',module_InnerProd)
    dephase = tf.divide(innerProd,tf.add(module_InnerProd,eps))
    print('dephase',dephase)
    ftNew =  tf.multiply(dephase,F_a) #compute the new version of the FT of the reference image
    # Element wise multiplication
    imF = tf.ifft2d(ftNew)
    imF =  tf.real(tf.transpose(imF, [0,2,3,1]))
    return(imF,ftNew,dephase,module_InnerProd,innerProd)
    
def plot_image(args,image,name="",fig=None,withoutZeroTake=False):
    """
    Plot the image using matplotlib
    """
    if(fig is None):
        fig = plt.figure()
    if not(withoutZeroTake):
        imagepost = image[0] 
    else:
        imagepost = image
    #imagepost = (imagepost - np.min(imagepost))*255 / ( np.max(imagepost)- np.min(imagepost))
    #imagepost = np.clip(imagepost,0,255).astype('uint8') 
    #print(imagepost.shape)
    plt.imshow(imagepost, cmap=cmap)
    plt.title(name)
    plt.colorbar()
    if(args.verbose): print("Plot",name)
    fig.canvas.flush_events()
    #time.sleep(10**(-6))
    return(fig)
    
def plot_imagegreyLog(args,image,name="",fig=None):
    """
    Plot the image using matplotlib
    """
    if(fig is None):
        fig = plt.figure()
    imagepost = np.log(1 + image[0,0] - np.min(image))
    #imagepost = (imagepost)*255 / ( np.max(imagepost)- np.min(imagepost))
    #imagepost = np.clip(imagepost,0,255).astype('uint8') 
    plt.imshow(imagepost, cmap=cmap)
    plt.title(name +' in log')
    plt.colorbar()
    if(args.verbose): print("Plot",name)
    fig.canvas.flush_events()
    #time.sleep(10**(-6))
    return(fig)
    
def investigationSpectrumModulus():

    seuil = 0.001

    parser = get_parser_args()
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    #print(args.init,args.start_from_noise,args.init_noise_ratio)
    style_img_name = 'lego_1024Ref_256'
    #style_img_name = 'purple_32'
    
    image_style = st.load_img(args,style_img_name)
    st.plot_image_with_postprocess(args,image_style.copy(),name="a",fig=None)
    
    x = st.get_init_img_wrap(args,'',image_style).astype('float32')
    st.plot_image_with_postprocess(args,x.copy(),name="x",fig=None)
    sess = tf.Session()
    
    imF_maison,ftNew_maison,dephase_maison,module_InnerProd_maison,innerProd_maison = sess.run(getImageXwithPhaseOfA_maison(image_style,image_style))
    imF_tfabs,ftNew_tfabs,dephase_tfabs,module_InnerProd_tfabs,innerProd_tfabs = sess.run(getImageXwithPhaseOfA_tfabs(image_style,image_style))
    print('case with a and itself')
    module_InnerProd_tfabs_real = module_InnerProd_tfabs.real
    print('Max and min  of module_InnerProd_tfabs_real',np.max(module_InnerProd_tfabs_real),np.min(module_InnerProd_tfabs_real))
    module_InnerProd_tfabs_imag = module_InnerProd_tfabs.imag
    module_InnerProd_maison_real = module_InnerProd_maison.real
    print('Max and min of module_InnerProd_maison_real',np.max(module_InnerProd_maison_real),np.min(module_InnerProd_maison_real))
    module_InnerProd_maison_imag = module_InnerProd_maison.imag
    diff_realpart_module_innerProd  = module_InnerProd_tfabs_real- module_InnerProd_maison_real
    ratio_realpart_module_innerProd  = module_InnerProd_maison_real  / module_InnerProd_tfabs_real
    diff_tfabs_main = imF_tfabs - imF_maison
    plot_imagegreyLog(args,module_InnerProd_tfabs_real,name="module_InnerProd_tfabs_real",fig=None)
    plot_imagegreyLog(args,module_InnerProd_maison_real,name="module_InnerProd_maison_real",fig=None)
    #plot_imagegreyLog(args,module_InnerProd_tfabs_imag,name="module_InnerProd_tfabs_imag",fig=None)
    #plot_imagegreyLog(args,module_InnerProd_maison_imag,name="module_InnerProd_maison_imag",fig=None)
    plot_imagegreyLog(args,diff_realpart_module_innerProd,name="diff_realpart_module_innerProd",fig=None)
    plot_image(args,diff_realpart_module_innerProd[0],name="diff_realpart_module_innerProd",fig=None)
    plot_imagegreyLog(args,ratio_realpart_module_innerProd,name="ratio_realpart_module_innerProd",fig=None)
    plot_image(args,ratio_realpart_module_innerProd[0],name="ratio_realpart_module_innerProd",fig=None)
    print('Maximum of the absolute value of the ratio minus 1 : ',np.max(np.abs(1-ratio_realpart_module_innerProd)))
    print('Mean and std of the ratio',np.mean(ratio_realpart_module_innerProd),np.std(ratio_realpart_module_innerProd))
    print('Maximum and minimum of the imag part of module_InnerProd_maison',np.max(module_InnerProd_maison_imag),np.min(module_InnerProd_maison_imag))
    print('Maximum and minimum of the imag part of module_InnerProd_tfabs',np.max(module_InnerProd_tfabs_imag),np.min(module_InnerProd_tfabs_imag))
    print('Maximum of the difference between module_InnerProd_tfabs_real and module_InnerProd_maison_real',np.max(np.abs(diff_realpart_module_innerProd)))
    print('MSE between module_InnerProd_tfabs_real and module_InnerProd_maison_imag',np.sum(diff_realpart_module_innerProd**2))
    print('Maximum of the difference between imF_tfabs and imF_maison : the image reconstruction ',np.max(np.abs(diff_tfabs_main)))
    print('Maximum in percentage of the difference between imF_tfabs and imF_maison',100*np.max(np.abs(diff_tfabs_main))/np.max([np.max(2*np.abs(module_InnerProd_maison_real)),np.max(2*np.abs(module_InnerProd_tfabs_real))]))
    print('MSE between imF_tfabs and imF_maison',np.sum(diff_tfabs_main**2))
    diff_realpart_innerProd_mask = np.zeros_like(diff_realpart_module_innerProd)
    diff_realpart_innerProd_mask[np.where(np.abs(diff_realpart_module_innerProd) > seuil)]= 1.
    print('max et min du mask',np.max(diff_realpart_innerProd_mask),np.min(diff_realpart_innerProd_mask))
    #print("Index where np.where(np.abs(diff_realpart_module_innerProd) > seuil)",np.where(np.abs(diff_realpart_innerProd_mask) > seuil))
    #print(diff_realpart_innerProd_mask.shape)
    plot_imagegreyLog(args,diff_realpart_innerProd_mask,name="diff_realpart_module_innerProd_mask",fig=None)
    
    
    print('Case with x and a')
    
    imF_maison,ftNew_maison,dephase_maison,module_InnerProd_maison,innerProd_maison = sess.run(getImageXwithPhaseOfA_maison(x,image_style))
    imF_tfabs,ftNew_tfabs,dephase_tfabs,module_InnerProd_tfabs,innerProd_tfabs = sess.run(getImageXwithPhaseOfA_tfabs(x,image_style))
    #print(module_InnerProd_maison)
    st.plot_image_with_postprocess(args,imF_maison,name="imF_maison with x et a",fig=None)
    st.plot_image_with_postprocess(args,imF_tfabs,name="imF_tfabs with x et a",fig=None)
    module_InnerProd_tfabs_real = module_InnerProd_tfabs.real
    print('Max and min  of module_InnerProd_tfabs_real',np.max(module_InnerProd_tfabs_real),np.min(module_InnerProd_tfabs_real))
    module_InnerProd_tfabs_imag = module_InnerProd_tfabs.imag
    module_InnerProd_maison_real = module_InnerProd_maison.real
    print('Max and min of module_InnerProd_maison_real',np.max(module_InnerProd_maison_real),np.min(module_InnerProd_maison_real))
    module_InnerProd_maison_imag = module_InnerProd_maison.imag
    diff_realpart_module_innerProd  = module_InnerProd_tfabs_real- module_InnerProd_maison_real
    ratio_realpart_module_innerProd  = module_InnerProd_maison_real  / module_InnerProd_tfabs_real
    diff_tfabs_main = imF_tfabs - imF_maison
    plot_imagegreyLog(args,module_InnerProd_tfabs_real,name="module_InnerProd_tfabs_real",fig=None)
    plot_imagegreyLog(args,module_InnerProd_maison_real,name="module_InnerProd_maison_real",fig=None)
    #plot_imagegreyLog(args,module_InnerProd_tfabs_imag,name="module_InnerProd_tfabs_imag",fig=None)
    #plot_imagegreyLog(args,module_InnerProd_maison_imag,name="module_InnerProd_maison_imag",fig=None)
    plot_imagegreyLog(args,diff_realpart_module_innerProd,name="diff_realpart_module_innerProd",fig=None)
    plot_image(args,diff_realpart_module_innerProd[0],name="diff_realpart_module_innerProd",fig=None)
    plot_imagegreyLog(args,ratio_realpart_module_innerProd,name="ratio_realpart_module_innerProd",fig=None)
    plot_image(args,ratio_realpart_module_innerProd[0],name="ratio_realpart_module_innerProd",fig=None)
    print('Maximum of the absolute value of the ratio minus 1 : ',np.max(np.abs(1-ratio_realpart_module_innerProd)))
    print('Mean and std of the ratio',np.mean(ratio_realpart_module_innerProd),np.std(ratio_realpart_module_innerProd))
    print('Maximum and minimum of the imag part of module_InnerProd_maison',np.max(module_InnerProd_maison_imag),np.min(module_InnerProd_maison_imag))
    print('Maximum and minimum of the imag part of module_InnerProd_tfabs',np.max(module_InnerProd_tfabs_imag),np.min(module_InnerProd_tfabs_imag))
    print('Maximum of the difference between module_InnerProd_tfabs_real and module_InnerProd_maison_real',np.max(np.abs(diff_realpart_module_innerProd)))
    print('MSE between module_InnerProd_tfabs_real and module_InnerProd_maison_imag',np.sum(diff_realpart_module_innerProd**2))
    print('Maximum of the difference between imF_tfabs and imF_maison : the image reconstruction ',np.max(np.abs(diff_tfabs_main)))
    print('Maximum in percentage of the difference between imF_tfabs and imF_maison',100*np.max(np.abs(diff_tfabs_main))/np.max([np.max(2*np.abs(module_InnerProd_maison_real)),np.max(2*np.abs(module_InnerProd_tfabs_real))]))
    print('MSE between imF_tfabs and imF_maison',np.sum(diff_tfabs_main**2))
    diff_realpart_innerProd_mask = np.zeros_like(diff_realpart_module_innerProd)
    diff_realpart_innerProd_mask[np.where(np.abs(diff_realpart_module_innerProd) > seuil)]= 1.
    print('max et min du mask',np.max(diff_realpart_innerProd_mask),np.min(diff_realpart_innerProd_mask))
    #print("Index where np.where(np.abs(diff_realpart_module_innerProd) > seuil)",np.where(np.abs(diff_realpart_innerProd_mask) > seuil))
    #print(diff_realpart_innerProd_mask.shape)
    plot_imagegreyLog(args,diff_realpart_innerProd_mask,name="diff_realpart_module_innerProd_mask",fig=None)
    
    wheresup99centile = np.where(np.abs(diff_realpart_module_innerProd) > np.percentile(np.abs(diff_realpart_module_innerProd),99))
    whereunder99centile = np.where(np.abs(diff_realpart_module_innerProd) < np.percentile(np.abs(diff_realpart_module_innerProd),99))
    
    plt.figure()
    plt.hist(np.ravel(np.abs(diff_realpart_module_innerProd)), color='r', alpha=0.95, bins=100)
    plt.legend(loc='best')
    plt.title('Histogram of abs diff_realpart_module_innerProd')
       
    
    print('Mean and min of module_InnerProd_maison_real sup 99 centile',np.mean(module_InnerProd_maison_real[wheresup99centile]),np.min(module_InnerProd_maison_real[wheresup99centile]))
    print('Mean and max of module_InnerProd_maison_real under 99 centile',np.mean(module_InnerProd_maison_real[whereunder99centile]),np.max(module_InnerProd_maison_real[whereunder99centile]))
    
    plt.style.use('seaborn-deep')
    bins = np.linspace(np.min([np.min(module_InnerProd_tfabs_real),np.min(module_InnerProd_maison_real)]), np.max([np.max(module_InnerProd_tfabs_real),np.max(module_InnerProd_maison_real)]), 20)

    print('Mean and std of module_InnerProd_maison_real',np.mean(np.ravel(module_InnerProd_maison_real)),np.std(np.ravel(module_InnerProd_maison_real)))
    print('Mean and std of module_InnerProd_tfabs_real', np.mean(np.ravel(module_InnerProd_tfabs_real)),np.std(np.ravel(module_InnerProd_tfabs_real)))

    plt.figure()
    plt.hist([np.ravel(module_InnerProd_maison_real), np.ravel(module_InnerProd_tfabs_real)], label=['maison', 'tfabs'], color=['g','r'], alpha=0.5, bins=100)
    plt.legend(loc='best')
    
    plt.show()
    input('wait input to close')
    
#def testFFT():
	### DEFINE FFTs
	#fft1D_np = np.fft.fft
	#fft1D_tf = tf.fft
	#ifft1D_np = np.fft.ifft
	#ifft1D_tf = tf.ifft
	#fft2D_np = np.fft.fft2
	#fft2D_tf = tf.fft2d
	#ifft2D_np = np.fft.ifft2
	#ifft2D_tf = tf.ifft2d
	### TEST 2D FFT
	#np.random.seed(13)
	#M = np.random.random([1,2,2]).astype(np.complex64)
	#M_tf = tf.placeholder(dtype=M.dtype, shape=[None,*M.shape[1:]], name='M_tf')
	#eval_dict = {M_tf: M}
	#sess, gsaver = nn_utils.init_network()
	#Mhat = fft2D_np(M) 
	#Mhat_tf = fft2D_tf(M_tf)
	#Mhat_tfeval = Mhat_tf.eval(feed_dict=eval_dict)
	#e = Mhat - Mhat_tfeval
	#err = np.linalg.norm(e.reshape([-1,1]))
	#assert( np.isclose(err, 0.0) )

    
if __name__ == '__main__':
    investigationSpectrumModulus()
    
