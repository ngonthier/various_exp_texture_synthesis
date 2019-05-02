"""
Created on  Wed 21 Februar 2018

This script have the goal to realized an gradient histogram matching of two 
images (the reference image and the synthetised one)

@author: nicolas
"""
from skimage import filters
import numpy as np 
import scipy
import cv2
from scipy.sparse import lil_matrix,csr_matrix
from scipy.sparse.linalg import spsolve
from skimage import color
import colour # conda install -y -c conda-forge colour-science
from sklearn.decomposition import PCA,FastICA
npt=np.float64
import ot

r = np.random.RandomState(42)


def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(I):
    return np.clip(I, 0, 1)

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

def poisson_solver_function(gx,gy,boundary_image):
    """function [img_direct] = poisson_solver_function(gx,gy,boundary_image) 
    Inputs; Gx and Gy -> Gradients 
    Boundary Image -> Boundary image intensities 
    Gx Gy and boundary image should be of same size
    """
    H,W,C = boundary_image.shape
    gxx = np.zeros_like(boundary_image)
    gyy = np.zeros_like(boundary_image)
    f = np.zeros_like(boundary_image)    
    #j = 1:H-1;   k = 1:W-1; 
 
    #Laplacian
    gyy = gy[1:,:,:] - gy[:-1,:,:]
    gxx = gx[:,1:,:] - gx[:,:-1,:] 
    f = gxx + gyy                        
    # boundary image contains image intensities at boundaries 
    #boundary_image(1:end-1,1:end-1,0) = 0
    # Solving Poisson Equation Using DST 
    #j = 2:H-1;          k = 2:W-1;   f_bp = zeros(H,W); 
    #f_bp(j,k) = -4*boundary_image(j,k) + boundary_image(j,k+1) + 
    #boundary_image(j,k-1) + boundary_image(j-1,k) + boundary_image(j+1,k); 
    #clear j k  
    #f1 = f - reshape(f_bp,H,W);% 
    #subtract boundary points contribution
    #clear f_bp f 
    #% 
    #DST Sine Transform
     #algo starts here 
    #f2       =       f1(2:end-1,2:end-1);                            clear       f1       
    #%compute sine transform 
    #tt = dst(f2);       f2sin = dst(tt’)’;         clear f2 
    #%compute 
    #Eigen Values
    #[x,y] = meshgrid(1:W-2,1:H-2);   denom = (2*cos(pi*x/(W-1))-2) + (2*cos(pi*y/(H-1)) - 2) ; 
    #%divide 
    #f3 = f2sin./denom;                              clear f2sin x y 
    #%compute 
    #Inverse Sine Transform 
    #tt = idst(f3);      clear f3;    img_tt = idst(tt’)’;       clear tt 
    #time_used = toc;    disp(sprintf(’Time for Poisson Reconstruction = %f secs’,time_used)); 
    #% put solution in inner points; outer points obtained from boundary image 
    #img_direct = boundary_image; 
    #img_direct(2:end-1,2:end-1) = 0; 
    #img_direct(2:end-1,2:end-1) = img_tt; 
    return(0)


def get_gradientSobel(ref_image):
    '''
    This function return the gradient of the ref_image input a RGB image
    '''
    
    hsobel_ref_image = np.zeros_like(ref_image)
    vsobel_ref_image = np.zeros_like(ref_image)
    
    for i in range(3):
        hsobel_ref_image[:,:,i] = filters.sobel_h(ref_image[:,:,i]) # Find the horizontal edges of an image using the Sobel transform. use this kernel  
        # 1   2   1 
        # 0   0   0
        # -1  -2  -1
        vsobel_ref_image[:,:,i] = filters.sobel_v(ref_image[:,:,i]) # TODO compute a simpler gradient x and y 
    
    return(hsobel_ref_image,vsobel_ref_image)
    
    
def gradients_matching(ref_image, syn_image, n_bins=100):
    '''
    This function realize an histogram matching on the gradient of the images
    '''
    
    hsobel_ref_image = np.zeros_like(ref_image)
    vsobel_ref_image = np.zeros_like(ref_image)
    hsobel_syn_image = np.zeros_like(syn_image)
    vsobel_syn_image = np.zeros_like(syn_image)
    
    for i in range(3):
        hsobel_ref_image[:,:,i] = filters.sobel_h(ref_image[:,:,i]) # Find the horizontal edges of an image using the Sobel transform. use this kernel  
        # 1   2   1 
        # 0   0   0
        # -1  -2  -1
        vsobel_ref_image[:,:,i] = filters.sobel_v(ref_image[:,:,i]) # TODO compute a simpler gradient x and y 
        hsobel_syn_image[:,:,i] = filters.sobel_h(syn_image[:,:,i])
        vsobel_syn_image[:,:,i] = filters.sobel_v(syn_image[:,:,i])

    # Horizontal Matching
    hsobel_syn_image = histogram_matching(hsobel_syn_image,hsobel_ref_image, grey=False, n_bins=100)
    # Vertical Matching
    vsobel_syn_image = histogram_matching(vsobel_syn_image,vsobel_ref_image, grey=False, n_bins=100)
    
    # 
    return(hsobel_syn_image,vsobel_syn_image)

def getLaplacianFromDerivativesSobel(gx,gy):
    """
    Compute the Laplacian from the Derivatives
    """
    m,n,c = gx.shape
    Laplacian = np.zeros((m, n, c),np.float)

    for i in range(c):
        Laplacian[:,:,i] = filters.sobel_h(gx[:,:,i])  + filters.sobel_v(gy[:,:,i])
    return(Laplacian)
    
def grad_im(im):
    """ Fct from Said : compute the gradient of an image defined in the TP 3  MVA"""
    (dy,dx)=im.shape
    champ_grad=np.zeros((2,dy,dx),dtype=npt)
    champ_grad[0,:,-1]=0.0
    champ_grad[0,:,:-1]=im[:,1:]-im[:,:-1]
    
    champ_grad[1,-1,:]=0.0
    champ_grad[1,:-1,:]=im[1:,:]-im[:-1,:]
    return champ_grad
    
def div_champ(ch):
    """ Fct from Said : compute the div of a field defined in the TP 3  MVA"""
    (_,dy,dx)=ch.shape
    div=np.zeros((dy,dx),dtype=npt)
    
    div[1:-1,:]+=ch[1,1:-1,:]-ch[1,0:-2,:]
    div[:,1:-1]+=ch[0,:,1:-1]-ch[0,:,0:-2]
    
    div[:,0]+=ch[0,:,0]
    div[:,-1]-=ch[0,:,-2]

    div[0,:]+=ch[1,0,:]
    div[-1,:]-=ch[1,-2,:]
    return div

    
def get_gradient(ref_image):
    '''
    This function return the gradient of the ref_image input a RGB image
    '''
    #np.gradient(ref_image) # Second order first derivative 
    m,n,c = ref_image.shape
    gx = np.zeros_like(ref_image)
    gy = np.zeros_like(ref_image)
    kernelx =  np.array([[0,0,0],[0,1,0],[0,-1,0]])
    kernely =  np.array([[0,0,0],[0,1,-1],[0,0,0]])
    for i in range(c):
        #gradients = np.gradient(ref_image[:,:,i])
        #gradh_ref_image[:,:,i] = gradients[0]
        #gradv_ref_image[:,:,i] = gradients[1]
        #gradh_ref_image[:,:,i] = np.diff(np.pad(ref_image[:,:,i],((0,1),(0,0)), 'constant', constant_values=0),axis=0)
        #gradv_ref_image[:,:,i] = np.diff(np.pad(ref_image[:,:,i],((0,0),(0,1)), 'constant', constant_values=0),axis=1)
        gx[:,:,i] = scipy.signal.convolve2d(ref_image[:,:,i], kernelx,mode='same',boundary='wrap')
        gy[:,:,i] = scipy.signal.convolve2d(ref_image[:,:,i], kernely,mode='same',boundary='wrap')
        #champ_grad = grad_im(ref_image[:,:,i])
        #gradh_ref_image[:,:,i] = champ_grad[0,:,:]
        #gradv_ref_image[:,:,i] =  champ_grad[1,:,:]
        #if i == 0:
            #print(gx[254:257,254:257])
            #print(gy[254:257,254:257])
    return(gx,gy)
    
    
def getLaplacianFromDerivatives(gx,gy):
    """
    Compute the Divergence of the Derivatives to get the Laplacian 
    Laplacian(Phi) = Div(g) if grad(Phi) = (gx,gy)
    """
    m,n,c = gx.shape
    Laplacian = np.zeros((m, n, c),np.float)
    kernelx =  np.array([[0,-1,0],[0,1,0],[0,0,0]])
    kernely =  np.array([[0,0,0],[-1,1,0],[0,0,0]])

    for i in range(c):
        #gxx= np.gradient(gx[:,:,i])[0]
        #gyy= np.gradient(gy[:,:,i])[1]
        #gxx= -np.flip(np.diff(np.pad(np.flip(gx[:,:,i],axis=0),((0,1),(0,0)), 'constant', constant_values=0),axis=0),axis=0)
        #gyy= -np.flip(np.diff(np.pad(np.flip(gy[:,:,i],axis=1),((0,0),(0,1)), 'constant', constant_values=0),axis=1),axis=1)
        gxx= scipy.signal.convolve2d(gx[:,:,i], kernelx,mode='same',boundary='wrap')
        gyy= scipy.signal.convolve2d(gy[:,:,i], kernely,mode='same',boundary='wrap')
        #g =  np.stack([gx[:,:,i],gy[:,:,i]], axis=0)
        #Laplacian[:,:,i] = div_champ(g)
        Laplacian[:,:,i] = gxx + gyy
        #if i == 0:
            #print(gxx[254:257,254:257])
            #print(gyy[254:257,254:257])
    return(Laplacian)

def getRotationelOfDerivative(gx,gy):
    m,n,c = gx.shape
    Rotationel = np.zeros((m, n, c),np.float)
    #kernelx =  np.array([[0,-1,0],[0,1,0],[0,0,0]])
    #kernely =  np.array([[0,0,0],[-1,1,0],[0,0,0]])

    kernelx =  np.array([[0,0,0],[0,1,0],[0,-1,0]])
    kernely =  np.array([[0,0,0],[0,1,-1],[0,0,0]])

    for i in range(c):
        gxy= scipy.signal.convolve2d(gx[:,:,i], kernely,mode='same',boundary='wrap')
        gyx= scipy.signal.convolve2d(gy[:,:,i], kernelx,mode='same',boundary='wrap')
        Rotationel[:,:,i] = gyx - gxy
       
    return(Rotationel)
    
def ReconstructionFromLaplacian(ref_image,Laplacian):
    border_img = ref_image.astype(np.float)
    m,n,c = ref_image.shape
    num_pixels = m*n 
    
    Coeff_matr = lil_matrix((num_pixels,num_pixels)) # Sparse Matrix
    B = np.zeros((num_pixels, c),np.float)
    
    # Creation of an index to give a numerotation for the pixels 
    indices = np.zeros((m,n),np.uint32)
    count = 0
    # On prendra x indices sur les colonnes et y sur les lignes
    for x, line in enumerate(ref_image):
        for y, col in enumerate(line):
            indices[x, y] = count
            count += 1
    assert(count==m*n)
    
    # iterate over every pixel in the image to define the values of the matrix
    for x in range(0,m):
        for y in range(0,n):
            # boundaries
            if (y == 0) or (y== m-1) or (x==0) or (x==n-1):
            #if  (y== m-1) and (x==n-1):
                Coeff_matr[indices[x, y], indices[x, y]] = 1
                for chnl in range(c):
                    B[indices[x, y], chnl] = B[indices[x, y], chnl] + border_img[x, y, chnl] # Dirichlet boundary
                    # Les differentes contributions s'additionnent
            else:
                for chnl in range(c):
                    assert(chnl <= c)
                    B[indices[x, y], chnl] = B[indices[x, y], chnl] + Laplacian[x, y, chnl]
                Coeff_matr[indices[x, y], indices[x, y]] = -4
                # take care of neighbours
                Coeff_matr[indices[x, y], indices[x - 1, y]] = 1
                Coeff_matr[indices[x, y], indices[x, y - 1]] = 1
                Coeff_matr[indices[x, y], indices[x + 1, y]] = 1
                Coeff_matr[indices[x, y], indices[x, y + 1]] = 1

    Coeff_matr = Coeff_matr.tocsr() # Convert the matrix in CSR
    solns = spsolve(Coeff_matr,B) # solving Ax = B : Solve the sparse linear system Ax=b, where b may be a vector or a matrix.
    
    # From the vector solution to the image 2D
    final_img = (ref_image.copy()).astype(np.float)
    solns = solns.reshape(num_pixels,c)
    k = 0
    for x, line in enumerate(ref_image):
        for y, col in enumerate(line):
            for ch in range(c):
                final_img[x,y,ch] = solns[k,ch]
            k += 1

    return(final_img)
    
def ReconstructionFromDerivativeTest():
    """
    This is a test fonction that test the reconstruction from the gradients via Poisson equation
    """
    img_ext = '.png'
    img_folder = 'images/GradHist/'
    name_im_test = 'TilesOrnate0158_512'
    #name_im_test = 'OneDot'
    #name_im_test = 'Line'
    image_path = img_folder + name_im_test + img_ext
    ref_image =  scipy.misc.imread(image_path).astype(np.float)
    m,n,c = ref_image.shape
    num_pixels = m*n
    border_img = ref_image.astype(np.float)
    
    # Compute the Derivatives
    gx,gy = get_gradient(ref_image)
    output_image_path = img_folder + name_im_test +'_gx' +img_ext
    gx_img = gx.astype(np.uint8)
    scipy.misc.imsave(output_image_path,gx_img)
    
    output_image_path = img_folder + name_im_test +'_gy' +img_ext
    gy_img = gy.astype(np.uint8)
    scipy.misc.imsave(output_image_path,gy_img)
    
    # Compute the Laplacian
    Laplacian = -getLaplacianFromDerivatives(gx,gy)
    #print(Laplacian[254:257,254:257,0])
    output_image_path = img_folder + name_im_test +'_Laplacian' +img_ext
    Laplacian_img =  (Laplacian-np.min(Laplacian))*255/(np.max(Laplacian)-np.min(Laplacian))
    scipy.misc.imsave(output_image_path,Laplacian_img)
    
    LaplacianVerif = Laplacian.copy()
    kernel =  np.array([[0,1,0],[1,-4,1],[0,1,0]])
    for i in range(3):
        LaplacianVerif[:,:,i] = scipy.signal.convolve2d(ref_image[:,:,i], kernel,mode='same')
    print(LaplacianVerif[254:257,254:257,0])
    
    print('Laplacian',np.max(Laplacian),np.min(Laplacian),np.std(Laplacian))
    print('LaplacianVerif',np.max(LaplacianVerif),np.min(LaplacianVerif),np.std(LaplacianVerif))
    LaplacianDiff = LaplacianVerif - Laplacian
    LaplacianDiff =  np.delete(LaplacianDiff,0,axis=0)
    LaplacianDiff =  np.delete(LaplacianDiff,0,axis=1)
    print("np.max(np.abs(LaplacianDiff))",np.max(np.abs(LaplacianDiff)))
    output_image_path = img_folder + name_im_test +'_LaplaciansDiff' +img_ext
    LaplacianDiff = (LaplacianDiff-np.min(LaplacianDiff))*255/(np.max(LaplacianDiff)-np.min(LaplacianDiff))
    print("After rescale",np.max(np.abs(LaplacianDiff)))
    scipy.misc.imsave(output_image_path,LaplacianDiff)
    
    
    output_image_path = img_folder + name_im_test +'_LaplacianByConvolution' +img_ext
    LaplacianVerif_img = (LaplacianVerif-np.min(LaplacianVerif))*255/(np.max(LaplacianVerif)-np.min(LaplacianVerif))
    scipy.misc.imsave(output_image_path,LaplacianVerif_img)
    
    Rot = getRotationelOfDerivative(gx,gy)
    print("np.max(np.abs(Rot))",np.max(np.abs(Rot)))
    output_image_path = img_folder + name_im_test +'_RotGrad' +img_ext
    if(np.max(np.abs(Rot)) > 0):
        Rot_img = ((Rot-np.min(Rot))*255/(np.max(Rot)-np.min(Rot))).astype(np.uint8)
    else:
        Rot_img = Rot
    scipy.misc.imsave(output_image_path,Rot_img)
    
    final_img = ReconstructionFromLaplacian(ref_image,Laplacian)

    print(np.max(final_img),np.min(final_img))
    final_img_uint8 = final_img.astype(np.uint8)

    # Save the reconstructed image
    output_image_path = img_folder + name_im_test +'_ReconstructTest' +img_ext
    scipy.misc.imsave(output_image_path,final_img_uint8)
    
    final_img_2 = (final_img - np.min(final_img))/(np.max(final_img)- np.min(final_img))
    final_img_2 = final_img_2.astype(np.uint8)

    # Save the reconstructed image rescaled
    output_image_path = img_folder + name_im_test +'_ReconstructTestRescale' +img_ext
    scipy.misc.imsave(output_image_path,final_img_2)
    
    final_img_3 = np.clip(final_img,0,255) 
    final_img_3 = final_img_3.astype(np.uint8)

    # Save the reconstructed image clipped
    output_image_path = img_folder + name_im_test +'_ReconstructTestClip' +img_ext
    scipy.misc.imsave(output_image_path,final_img_3)
    
    # Test with a passage throw the HSV domain color 
    print("Start working on HSV")
    img_hsv = color.rgb2hsv(ref_image.copy())
    # We will only consider the V value and reconstruct on it 
    img_V = np.expand_dims(img_hsv[:,:,2],axis=-1)
    gx,gy = get_gradient(img_V)
    Laplacian = -getLaplacianFromDerivatives(gx,gy)
    final_img = ReconstructionFromLaplacian(img_V,Laplacian)
    img_hsv_output = img_hsv.copy()
    img_hsv_output[:,:,2] = final_img[:,:,0]
    img_hsv_rgb = color.hsv2rgb(img_hsv_output)
    output_image_path = img_folder + name_im_test +'_viaHSV' +img_ext
    scipy.misc.imsave(output_image_path,img_hsv_rgb)
    
    # Test with a pasa in the HSL domain color 
    print("Start working on HSL")
    img_hsl = colour.RGB_to_HSL(ref_image.copy())
    # We will only consider the L luminance and reconstruct on it 
    img_L = np.expand_dims(img_hsv[:,:,2],axis=-1)
    gx,gy = get_gradient(img_L)
    Laplacian = -getLaplacianFromDerivatives(gx,gy)
    final_img = ReconstructionFromLaplacian(img_L,Laplacian)
    img_hsl_output = img_hsl.copy()
    img_hsl_output[:,:,2] = final_img[:,:,0]
    img_hsl_rgb = colour.HSL_to_RGB(img_hsl_output)
    output_image_path = img_folder + name_im_test +'_viaHSL' +img_ext
    scipy.misc.imsave(output_image_path,img_hsv_rgb)
    
def HistogramOfGradientMatchingMultipleTimes():
    """
    This is a test fonction that test the reconstruction from the gradients via Poisson equation
    Cela ne donne rien du tout !!! 
    """
    img_ext = '.png'
    img_folder = 'images/GradHist/'
    name_im_ref = 'TilesOrnate0158_512'
    image_path = img_folder + name_im_ref + img_ext
    ref_image =  scipy.misc.imread(image_path).astype(np.float)
    m,n,c = ref_image.shape
    num_pixels = m*n
    
    print("Range Im ref",np.max(ref_image),np.min(ref_image),np.std(ref_image))
    
    name_img_whose_needHGM = 'TilesOrnate0158_512_SAME_texture_spectrum'
    name_img_whose_needHGM = 'TilesOrnate0158_512_SAME_autocorr'
    name_img_whose_needHGM = 'TilesOrnate0158_512_SAME_texture'
    image_path = img_folder + name_img_whose_needHGM +img_ext
    img_needHGM =  scipy.misc.imread(image_path).astype(np.float)
    print("Range Im remap",np.max(img_needHGM),np.min(img_needHGM),np.std(img_needHGM))
    maxiter = 10
    
    img_rep = np.zeros_like(img_needHGM)
    
    for i in range(maxiter):
        print(i)
        # Histogram of color Channel matching
        img_HCM = histogram_matching(img_needHGM, ref_image, grey=False, n_bins=100)
        img_HCM_uint8 = img_HCM.astype(np.uint8)
        #output_image_path = img_folder + name_img_whose_needHGM +'_HCM' +img_ext
        #scipy.misc.imsave(output_image_path,img_HCM_uint8)
        
        # Compute the Derivatives
        ref_gx,ref_gy = get_gradient(ref_image)
        needHGM_gx,needHGM_gy = get_gradient(img_needHGM)
        
        # Matching of the Gradient images
        # First argument of histogram_matching is the image whose distribution should be remapped
        matched_gx = histogram_matching(needHGM_gx,ref_gx, grey=False, n_bins=100) 
        matched_gy = histogram_matching(needHGM_gy,ref_gy, grey=False, n_bins=100)
       
        border_img = img_needHGM.astype(np.float)

        # Compute the Laplacian
        Laplacian = -getLaplacianFromDerivatives(matched_gx,matched_gy)
        
        Coeff_matr = lil_matrix((num_pixels,num_pixels)) # Sparse Matrix
        B = np.zeros((num_pixels, 3),np.float)
        
        # Creation of an index to give a numerotation for the pixels 
        indices = np.zeros((m,n),np.uint32)
        count = 0
        # On prendra x indices sur les colonnes et y sur les lignes
        for x, line in enumerate(ref_image):
            for y, col in enumerate(line):
                indices[x, y] = count
                count += 1
        assert(count==m*n)
        
        # iterate over every pixel in the image to define the values of the matrix
        for x in range(0,m):
            for y in range(0,n):
                # boundaries
                if (y == 0) or (y== m-1) or (x==0) or (x==n-1):
                #if  (y== m-1) and (x==n-1):
                    Coeff_matr[indices[x, y], indices[x, y]] = 1
                    for chnl in range(0,3):
                        B[indices[x, y], chnl] = B[indices[x, y], chnl] + border_img[x, y, chnl] # Dirichlet boundary
                        # Les differentes contributions s'additionnent
                else:
                    for chnl in range(0,3):
                        B[indices[x, y], chnl] = B[indices[x, y], chnl] + Laplacian[x, y, chnl]
                    Coeff_matr[indices[x, y], indices[x, y]] = -4
                    # take care of neighbours
                    Coeff_matr[indices[x, y], indices[x - 1, y]] = 1
                    Coeff_matr[indices[x, y], indices[x, y - 1]] = 1
                    Coeff_matr[indices[x, y], indices[x + 1, y]] = 1
                    Coeff_matr[indices[x, y], indices[x, y + 1]] = 1

        Coeff_matr = Coeff_matr.tocsr() # Convert the matrix in CSR
        solns = spsolve(Coeff_matr,B) # solving Ax = B : Solve the sparse linear system Ax=b, where b may be a vector or a matrix.
        
        # From the vector solution to the image 2D
        final_img = (ref_image.copy()).astype(np.float)
        k = 0
        for x, line in enumerate(ref_image):
            for y, col in enumerate(line):
                for ch in range(0,3):
                    final_img[x,y,ch] = solns[k,ch]
                k += 1

        #print("Range Im final",np.max(final_img),np.min(final_img),np.std(final_img))
        final_img_uint8 = final_img.astype(np.uint8)

        ## Save the reconstructed image
        #output_image_path = img_folder + name_img_whose_needHGM +'_HGM' +img_ext
        #scipy.misc.imsave(output_image_path,final_img_uint8)



        final_img_HCM = histogram_matching(final_img, ref_image, grey=False, n_bins=100)
        #print("Range Im final with HCM",np.max(final_img_HCM),np.min(final_img_HCM),np.std(final_img_HCM))
        final_img_HCM_uint8 = final_img_HCM.astype(np.uint8)
        output_image_path = img_folder + name_img_whose_needHGM +'_HGM_HCM' + str(i) +img_ext
        scipy.misc.imsave(output_image_path,final_img_HCM_uint8)
        
        if np.max(np.abs(img_rep-final_img_HCM)) < 0.1:
            print("Images are identical")
            break
        else:
            img_rep = final_img_HCM.copy()  
            img_needHGM = final_img_HCM.copy()  
                    
        #final_img_scaled = final_img.copy()
        #for i in range(3):
            #final_img_scaled[:,:,i] = (final_img_scaled[:,:,i]-np.min(final_img_scaled[:,:,i]))*255/(np.max(final_img_scaled[:,:,i])-np.min(final_img_scaled[:,:,i]))
        #output_image_path = img_folder + name_img_whose_needHGM +'_HGM_scaled' +img_ext
        #final_img_uint8_scaled = final_img_scaled.astype(np.uint8)
        #scipy.misc.imsave(output_image_path,final_img_uint8_scaled)
        #final_img_HCM_scaled_HCM = histogram_matching(final_img_scaled, ref_image, grey=False, n_bins=100)
        #final_img_HCM_scaled_HCM_uint8 = final_img_HCM_scaled_HCM.astype(np.uint8)
        #output_image_path = img_folder + name_img_whose_needHGM +'_HGM_scaled_HCM' +img_ext
        #scipy.misc.imsave(output_image_path,final_img_HCM_scaled_HCM_uint8)
    
def HistogramOfGradientMatching():
    """
    This is a test fonction that test the reconstruction from the gradients via Poisson equation
    """
    img_ext = '.png'
    img_folder = 'images/GradHist/'
    name_im_ref = 'TilesOrnate0158_512'
    name_im_ref = 'Food_0008'
    image_path = img_folder + name_im_ref + img_ext
    ref_image =  scipy.misc.imread(image_path).astype(np.float)
    m,n,c = ref_image.shape
    num_pixels = m*n
    
    print("Range Im ref",np.max(ref_image),np.min(ref_image),np.std(ref_image))
    
    name_img_whose_needHGM = 'TilesOrnate0158_512_SAME_texture_spectrum'
    name_img_whose_needHGM = 'TilesOrnate0158_512_SAME_autocorr'
    #name_img_whose_needHGM = 'TilesOrnate0158_512_SAME_texture'
    name_img_whose_needHGM = 'Food_0008_SAME_texture'
    name_img_whose_needHGM = 'Food_0008_SAME_autocorr'
    image_path = img_folder + name_img_whose_needHGM +img_ext
    img_needHGM =  scipy.misc.imread(image_path).astype(np.float)
    print("Range Im remap",np.max(img_needHGM),np.min(img_needHGM),np.std(img_needHGM))
    
    # Histogram of color Channel matching
    img_HCM = histogram_matching(img_needHGM, ref_image, grey=False, n_bins=100)
    img_HCM_uint8 = img_HCM.astype(np.uint8)
    output_image_path = img_folder + name_img_whose_needHGM +'_HCM' +img_ext
    scipy.misc.imsave(output_image_path,img_HCM_uint8)
    
    # Compute the Derivatives
    ref_gx,ref_gy = get_gradient(ref_image)
    needHGM_gx,needHGM_gy = get_gradient(img_needHGM)
    
    # Matching of the Gradient images
    # First argument of histogram_matching is the image whose distribution should be remapped
    matched_gx = histogram_matching(needHGM_gx,ref_gx, grey=False, n_bins=100) 
    matched_gy = histogram_matching(needHGM_gy,ref_gy, grey=False, n_bins=100)
   
    border_img = img_needHGM.astype(np.float)

    # Compute the Laplacian
    Laplacian = -getLaplacianFromDerivatives(matched_gx,matched_gy)
    
    final_img = ReconstructionFromLaplacian(ref_image,Laplacian)

    print("Range Im final",np.max(final_img),np.min(final_img),np.std(final_img))
    final_img_uint8 = final_img.astype(np.uint8)

    # Save the reconstructed image
    output_image_path = img_folder + name_img_whose_needHGM +'_HGM' +img_ext
    scipy.misc.imsave(output_image_path,final_img_uint8)



    final_img_HCM = histogram_matching(final_img, ref_image, grey=False, n_bins=100)
    print("Range Im final with HCM",np.max(final_img_HCM),np.min(final_img_HCM),np.std(final_img_HCM))
    final_img_HCM_uint8 = final_img_HCM.astype(np.uint8)
    output_image_path = img_folder + name_img_whose_needHGM +'_HGM_HCM' +img_ext
    scipy.misc.imsave(output_image_path,final_img_HCM_uint8)
    
    final_img_scaled = final_img.copy()
    for i in range(3):
        final_img_scaled[:,:,i] = (final_img_scaled[:,:,i]-np.min(final_img_scaled[:,:,i]))*255/(np.max(final_img_scaled[:,:,i])-np.min(final_img_scaled[:,:,i]))
    output_image_path = img_folder + name_img_whose_needHGM +'_HGM_scaled' +img_ext
    final_img_uint8_scaled = final_img_scaled.astype(np.uint8)
    scipy.misc.imsave(output_image_path,final_img_uint8_scaled)
    final_img_HCM_scaled_HCM = histogram_matching(final_img_scaled, ref_image, grey=False, n_bins=100)
    final_img_HCM_scaled_HCM_uint8 = final_img_HCM_scaled_HCM.astype(np.uint8)
    output_image_path = img_folder + name_img_whose_needHGM +'_HGM_scaled_HCM' +img_ext
    scipy.misc.imsave(output_image_path,final_img_HCM_scaled_HCM_uint8)
    
    ## An other solution is to only work on the V or L channel of an HSV/HSL decomposition of the image
    # Test with a passage throw the HSV domain color 
    print("Start working on HSV")
    img_hsv = color.rgb2hsv(ref_image.copy())
    img_needHGM_hsv = color.rgb2hsv(img_needHGM.copy())
    # We will only consider the V value and reconstruct on it 
    img_V = np.expand_dims(img_hsv[:,:,2],axis=-1)
    ref_gx,ref_gy = get_gradient(img_V)
    img_needHGM_hsv_V = np.expand_dims(img_needHGM_hsv[:,:,2],axis=-1)
    needHGM_gx,needHGM_gy = get_gradient(img_needHGM_hsv_V)
    
    matched_gx = histogram_matching(needHGM_gx,ref_gx, grey=True, n_bins=100) 
    matched_gy = histogram_matching(needHGM_gy,ref_gy, grey=True, n_bins=100)
    
    Laplacian = -getLaplacianFromDerivatives(matched_gx,matched_gy)
    final_img = ReconstructionFromLaplacian(img_V,Laplacian)
    print("V ref",np.min(img_V[:,:,0]),np.max(img_V[:,:,0])) 
    print("V final",np.min(final_img[:,:,0]),np.max( final_img[:,:,0])) 
    img_hsv_output = img_needHGM_hsv.copy()
    img_hsv_output[:,:,2] = final_img[:,:,0]
    img_hsv_rgb = color.hsv2rgb(img_hsv_output)
    print(np.mean(ref_image),np.mean(img_needHGM),np.mean(img_hsv_rgb))
    output_image_path = img_folder + name_img_whose_needHGM +'_HGM_viaHSV' +img_ext
    scipy.misc.imsave(output_image_path,img_hsv_rgb.astype(np.uint8))
    
    img_hsv_output[:,:,2] = (final_img[:,:,0]-np.min(final_img[:,:,0]))*225./(np.max(final_img[:,:,0])-np.min(final_img[:,:,0]))
    img_hsv_rgb_scaled = color.hsv2rgb(img_hsv_output)
    output_image_path = img_folder + name_img_whose_needHGM +'_HGM_viaHSV_sclaed' +img_ext
    scipy.misc.imsave(output_image_path,img_hsv_rgb_scaled.astype(np.uint8))
    
    # Test with a pasa in the HSL domain color 
    print("Start working on HSL")
    img_hsl = colour.RGB_to_HSL(ref_image.copy())
    img_needHGM_hsl = colour.RGB_to_HSL(img_needHGM.copy())
    # We will only consider the L luminance and reconstruct on it 
    img_L = np.expand_dims(img_hsl[:,:,2],axis=-1)
    ref_gx,ref_gy = get_gradient(img_L)
    img_needHGM_hsl_L = np.expand_dims(img_needHGM_hsl[:,:,2],axis=-1)
    needHGM_gx,needHGM_gy = get_gradient(img_needHGM_hsl_L)
    
    matched_gx = histogram_matching(needHGM_gx,ref_gx, grey=True, n_bins=100) 
    matched_gy = histogram_matching(needHGM_gy,ref_gy, grey=True, n_bins=100)
    
    Laplacian = -getLaplacianFromDerivatives(matched_gx,matched_gy)
    final_img = ReconstructionFromLaplacian(img_L,Laplacian)
    print("L ref",np.min(img_L[:,:,0]),np.max(img_L[:,:,0])) 
    print("L final",np.min(final_img[:,:,0]),np.max( final_img[:,:,0])) # -93.81092802999521 369.4265694531914 alors que l on derait avoir quelque chose entre 0 et 1
    
    img_hsl_output = img_needHGM_hsl.copy()
    img_hsl_output[:,:,2] = final_img[:,:,0]
    img_hsl_rgb = colour.HSL_to_RGB(img_hsl_output)
    print(np.mean(ref_image),np.mean(img_needHGM),np.mean(img_hsl_rgb))
    output_image_path = img_folder + name_img_whose_needHGM +'_HGM_viaHSL' +img_ext
    scipy.misc.imsave(output_image_path,img_hsl_rgb.astype(np.uint8))
    
    img_hsl_output[:,:,2] = (final_img[:,:,0]-np.min(final_img[:,:,0]))*225./(np.max(final_img[:,:,0])-np.min(final_img[:,:,0]))
    img_hsl_rgb_scaled = colour.HSL_to_RGB(img_hsl_output)
    output_image_path = img_folder + name_img_whose_needHGM +'_HGM_viaHSL_sclaed' +img_ext
    scipy.misc.imsave(output_image_path,img_hsl_rgb_scaled.astype(np.uint8))
    
    ## Tentative with a PCA decomposition
    #print('Via PCA')
    #X = ref_image.copy()
    #X = X.reshape(-1, 3)
    #pca = PCA(n_components=3)
    #X_new =pca.fit_transform(X)
    #h,w,c = ref_image.shape
    #ref_image_pca = X_new.reshape(h,w,c)
    #img_needHGM_pca = pca.transform(img_needHGM.reshape(-1,3))
    #img_needHGM_pca =  img_needHGM_pca.reshape(h,w,c)
    #ref_gx,ref_gy = get_gradient(ref_image_pca)
    #needHGM_gx,needHGM_gy = get_gradient(img_needHGM_pca)
    ## Matching of the Gradient images
    ## First argument of histogram_matching is the image whose distribution should be remapped
    #matched_gx = histogram_matching(needHGM_gx,ref_gx, grey=False, n_bins=100) 
    #matched_gy = histogram_matching(needHGM_gy,ref_gy, grey=False, n_bins=100)
    #Laplacian = -getLaplacianFromDerivatives(matched_gx,matched_gy)
    #final_img = ReconstructionFromLaplacian(ref_image,Laplacian)
    #print(final_img.shape)
    #X_final_img = final_img.copy()
    #X_final_img = X_final_img.reshape(-1,3)
    #X_final_img_inv = pca.inverse_transform(X_final_img)
    #final_img = X_final_img_inv.reshape(h,w,c)
    #print(np.mean(ref_image),np.mean(img_needHGM),np.mean(final_img))
    #output_image_path = img_folder + name_img_whose_needHGM +'_HGM_viaPCA' +img_ext
    #scipy.misc.imsave(output_image_path,final_img.astype(np.uint8))
    #final_img_scaled = final_img.copy()
    #for i in range(3):
        #final_img_scaled[:,:,i] = (final_img_scaled[:,:,i]-np.min(final_img_scaled[:,:,i]))*255./(np.max(final_img_scaled[:,:,i])-np.min(final_img_scaled[:,:,i]))
    #output_image_path = img_folder + name_img_whose_needHGM +'_HGM_viaPCA_scaled' +img_ext
    #scipy.misc.imsave(output_image_path,final_img_scaled.astype(np.uint8))
    
    ## Tentative with a PCA decomposition
    #print('Via ICA')
    #ica = FastICA(n_components=3)
    #X = ref_image.copy()
    #X = X.reshape(-1, 3)
    #X_new =ica.fit_transform(X)
    #h,w,c = ref_image.shape
    #ref_image_ica = X_new.reshape(h,w,c)
    #img_needHGM_ica = ica.transform(img_needHGM.reshape(-1,3))
    #img_needHGM_ica =  img_needHGM_ica.reshape(h,w,c)
    #ref_gx,ref_gy = get_gradient(ref_image_ica)
    #needHGM_gx,needHGM_gy = get_gradient(img_needHGM_ica)
    ## Matching of the Gradient images
    ## First argument of histogram_matching is the image whose distribution should be remapped
    #matched_gx = histogram_matching(needHGM_gx,ref_gx, grey=False, n_bins=100) 
    #matched_gy = histogram_matching(needHGM_gy,ref_gy, grey=False, n_bins=100)
    #Laplacian = -getLaplacianFromDerivatives(matched_gx,matched_gy)
    #final_img = ReconstructionFromLaplacian(ref_image,Laplacian)
    #X_final_img = final_img.reshape(-1,3)
    #X_final_img_inv = ica.inverse_transform(X_final_img)
    #final_img = X_final_img_inv.reshape(h,w,c)
    #print(np.mean(ref_image),np.mean(img_needHGM),np.mean(final_img))
    #output_image_path = img_folder + name_img_whose_needHGM +'_HGM_viaICA' +img_ext
    #scipy.misc.imsave(output_image_path,final_img.astype(np.uint8))
    #final_img_scaled = final_img.copy()
    #for i in range(3):
        #final_img_scaled[:,:,i] = (final_img_scaled[:,:,i]-np.min(final_img_scaled[:,:,i]))*255./(np.max(final_img_scaled[:,:,i])-np.min(final_img_scaled[:,:,i]))
    #output_image_path = img_folder + name_img_whose_needHGM +'_HGM_viaICA_scaled' +img_ext
    #scipy.misc.imsave(output_image_path,final_img_scaled.astype(np.uint8))
    
    ### Color Transfert by Optimal Transport
    print("OT")
    I1 = ref_image.copy().astype(np.float64) / 256
    I2 = img_needHGM.copy().astype(np.float64) / 256
    X1 = im2mat(I1)
    X2 = im2mat(I2)

    # training samples
    nb = 2000
    idx1 = r.randint(X1.shape[0], size=(nb,))
    idx2 = r.randint(X2.shape[0], size=(nb,))

    Xs = X1[idx1, :]
    Xt = X2[idx2, :]
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)

    # SinkhornTransport
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)

    # prediction between images (using out of sample prediction as in [6])
    transp_Xs_emd = ot_emd.transform(Xs=X1)
    transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)
    transp_Xs_sinkhorn = ot_emd.transform(Xs=X1)
    transp_Xt_sinkhorn = ot_emd.inverse_transform(Xt=X2)
    I2t = minmax(mat2im(transp_Xt_emd, I2.shape))
    I2te = minmax(mat2im(transp_Xt_sinkhorn, I2.shape))
    
    output_image_path = img_folder + name_img_whose_needHGM +'_OT' +img_ext
    scipy.misc.imsave(output_image_path,I2t)
    output_image_path = img_folder + name_img_whose_needHGM +'_OTE' +img_ext
    scipy.misc.imsave(output_image_path,I2te)
    
    ## Gradient Transfert by Optimal Transport
    print("Grad OT")
    ref_gx,ref_gy = get_gradient(ref_image)
    needHGM_gx,needHGM_gy = get_gradient(img_needHGM)
    
    ref_grad = np.concatenate((ref_gx,ref_gy),axis=2)
    needHGM_grad = np.concatenate((needHGM_gx,needHGM_gy),axis=2)
    max_grad = np.max((np.max(ref_grad),np.max(needHGM_grad)))
    min_grad = np.min((np.min(ref_grad),np.min(needHGM_grad)))
    
    I1 = (ref_grad.copy().astype(np.float64)-min_grad) / (max_grad-min_grad)
    I2 = (needHGM_grad.copy().astype(np.float64)-min_grad) / (max_grad-min_grad)
    X1 = im2mat(I1)
    X2 = im2mat(I2)

    # training samples
    nb = 2000
    idx1 = r.randint(X1.shape[0], size=(nb,))
    idx2 = r.randint(X2.shape[0], size=(nb,))

    Xs = X1[idx1, :]
    Xt = X2[idx2, :]
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)

    # SinkhornTransport
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)

    # prediction between images (using out of sample prediction as in [6])
    #transp_Xs_emd = ot_emd.transform(Xs=X1)
    transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)
    #transp_Xs_sinkhorn = ot_emd.transform(Xs=X1)
    transp_Xt_sinkhorn = ot_emd.inverse_transform(Xt=X2)
    I2t = minmax(mat2im(transp_Xt_emd, I2.shape))
    I2te = minmax(mat2im(transp_Xt_sinkhorn, I2.shape))
    print("End OT")
    
    matched_gx = (I2t[:,:,0:3] *(max_grad-min_grad)) +min_grad
    matched_gy = (I2t[:,:,3:6] *(max_grad-min_grad)) +min_grad
    border_img = img_needHGM.astype(np.float)
    Laplacian = -getLaplacianFromDerivatives(matched_gx,matched_gy)
    final_img = ReconstructionFromLaplacian(ref_image,Laplacian)
    output_image_path = img_folder + name_img_whose_needHGM +'_OTGrad' +img_ext
    scipy.misc.imsave(output_image_path,final_img.astype(np.uint8))
    final_img_scaled = final_img.copy()
    for i in range(3):
        final_img_scaled[:,:,i] = (final_img_scaled[:,:,i]-np.min(final_img_scaled[:,:,i]))*255/(np.max(final_img_scaled[:,:,i])-np.min(final_img_scaled[:,:,i]))
    output_image_path = img_folder + name_img_whose_needHGM +'_OTGrad_scaled' +img_ext
    final_img_uint8_scaled = final_img_scaled.astype(np.uint8)
    scipy.misc.imsave(output_image_path,final_img_uint8_scaled)
    
    final_img_HCM_scaled_HCM = histogram_matching(final_img_scaled, ref_image, grey=False, n_bins=100)
    final_img_HCM_scaled_HCM_uint8 = final_img_HCM_scaled_HCM.astype(np.uint8)
    output_image_path = img_folder + name_img_whose_needHGM +'_OTGrad_scaled_HCM' +img_ext
    scipy.misc.imsave(output_image_path,final_img_HCM_scaled_HCM_uint8)
    
    matched_gx = (I2te[:,:,0:3] *(max_grad-min_grad)) +min_grad
    matched_gy = (I2te[:,:,3:6] *(max_grad-min_grad)) +min_grad
    border_img = img_needHGM.astype(np.float)
    Laplacian = -getLaplacianFromDerivatives(matched_gx,matched_gy)
    final_img = ReconstructionFromLaplacian(ref_image,Laplacian)
    output_image_path = img_folder + name_img_whose_needHGM +'_OTeGrad' +img_ext
    scipy.misc.imsave(output_image_path,final_img.astype(np.uint8))
    final_img_scaled = final_img.copy()
    for i in range(3):
        final_img_scaled[:,:,i] = (final_img_scaled[:,:,i]-np.min(final_img_scaled[:,:,i]))*255/(np.max(final_img_scaled[:,:,i])-np.min(final_img_scaled[:,:,i]))
    output_image_path = img_folder + name_img_whose_needHGM +'_OTeGrad_scaled' +img_ext
    final_img_uint8_scaled = final_img_scaled.astype(np.uint8)
    scipy.misc.imsave(output_image_path,final_img_uint8_scaled)
    
    final_img_HCM_scaled_HCM = histogram_matching(final_img_scaled, ref_image, grey=False, n_bins=100)
    final_img_HCM_scaled_HCM_uint8 = final_img_HCM_scaled_HCM.astype(np.uint8)
    output_image_path = img_folder + name_img_whose_needHGM +'_OTeGrad_scaled_HCM' +img_ext
    scipy.misc.imsave(output_image_path,final_img_HCM_scaled_HCM_uint8)
    
    
    
def seamless():
    
    
    # TODO  use those gradient to compute the laplacien by l = gxx + gyy
    
    
    border_img = syn_img.astype(np.float)
    m,n = len(ref_img),len(ref_img[0])
    points = []
    #cv2.imshow('get_input',ref_img)
    #cv2.setMouseCallback('get_input',construct_polygon,points)
    #cv2.waitKey(0)
    #cv2.destroyWindow('get_input')


    '''Create mask from the input image'''
    mask = 255*np.ones((m,n),np.uint8)
    mask[0,:] = 0
    mask[:,0] = 0
    mask[m-1,:] = 0
    mask[:,n-1] = 0
    #mask[1,:] = 1
    #mask[:,1] = 1
    #mask[m-2,:] = 1
    #mask[:,n-2] = 1
    #cv2.fillPoly(mask, np.array([points]), (255,255,255))
    #ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    #cv2.destroyWindow('mask')
    #''' Find length of the coeff matrix'''
    nz_pixels = np.nonzero(mask)
    num_pixels = len(nz_pixels[0])
    #print("num of white pixels - ", num_pixels)
    indices = np.zeros((m,n),np.uint32)
    count = 0
    for i in range(0,num_pixels):
        y = nz_pixels[0][i]
        x = nz_pixels[1][i]
        indices[y, x] = count
        count += 1
    ''' Calculate the laplacian at each point'''

    grad_img = np.zeros((m, n, 3),np.float)
    ref_img = ref_img.astype(np.float)
    H = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],np.float)
    #grad_ref_img = cv2.filter2D(ref_img, -1, H)
    #grad_syn_img = cv2.filter2D(syn_img, -1, H)
    #print(grad_img.shape)
    for i in range(3):
        #grad_img[:,:,i] = filters.sobel_v(vsobel_syn_image[:,:,i]) +  filters.sobel_h(hsobel_syn_image[:,:,i])
        grad_img[:,:,i] = filters.sobel_v(filters.sobel_v(syn_img[:,:,i])) +  filters.sobel_h(filters.sobel_h(syn_img[:,:,i]))
    
    # TODO use laplacian_img instead of grad_img
    print("Laplacians calculated")
    '''max of 5 pixels'''
    Coeff_matr = lil_matrix((num_pixels,num_pixels))        # this matrix has 4*num_pixels vales at maximum
    #Coeff_matr = csr_matrix((num_pixels,num_pixels))        # this matrix has 4*num_pixels vales at maximum
    #Coeff_matr = np.zeros((num_pixels,num_pixels))        # this matrix has 4*num_pixels vales at maximum
    B = np.zeros((num_pixels, 3),np.float)

    ''' iterate over every pixel in the image'''
    print("size of image is ",m, "  ",n)
    for y in range(1,m-2+1):
        for x in range(1,n-2+1):
            ''' only add points that are in the mask'''
            if mask[y,x] == 255:
                neighbours = 1
                '''take care of neighbours'''
                '''top boundary'''
                if mask[y - 1, x] == 1:
                    print("top boundary")
                    Coeff_matr[indices[y, x], indices[y - 1, x]] = -1
                    neighbours += 1
                else:
                    for chnl in range(0,3):
                        B[indices[y, x], chnl] = B[indices[y, x], chnl] +  (border_img[y - 1, x, chnl])
                        #print indices[y,x],"  ",border_img[y - 1, x, chnl]
                '''left boundary'''
                if mask[y, x - 1] == 1:
                    Coeff_matr[indices[y, x], indices[y, x - 1]] = -1
                    neighbours += 1
                else:
                    for chnl in range(0,3):
                        B[indices[y, x], chnl] = B[indices[y, x], chnl] +  (border_img[y, x - 1, chnl])
                        #print indices[y, x], "  ", border_img[y , x-1, chnl]
                ''' bottom boundary '''
                if mask[y + 1, x] == 1:
                    Coeff_matr[indices[y, x], indices[y + 1, x]] = -1
                    neighbours += 1
                else:
                    for chnl in range(0,3):
                        B[indices[y, x], chnl] = B[indices[y, x], chnl] +  (border_img[y + 1, x, chnl])
                        #print indices[y, x], "  ", border_img[y + 1, x, chnl]
                ''' right boundary '''
                if mask[y, x + 1] == 1:
                    Coeff_matr[indices[y, x], indices[y, x + 1]] = -1
                    neighbours += 1
                else:
                    for chnl in range(0,3):
                        B[indices[y, x], chnl] = B[indices[y, x], chnl] +  (border_img[y, x + 1, chnl])
                        #print indices[y, x], "  ", border_img[y, x + 1, chnl]
                for chnl in range(0,3):
                    B[indices[y, x], chnl] = B[indices[y, x], chnl] + grad_img[y, x, chnl]
                Coeff_matr[indices[y, x], indices[y, x]] = 4

    print("End Border condition")
    final_img = border_img.astype(np.float)
    ''' solving Ax = B'''

    Coeff_matr = Coeff_matr.tocsr()
    #print(len(Coeff_matr.nonzero()[0]))
    solns = spsolve(Coeff_matr,B)
    #solns = solve(Coeff_matr,B)
    #print solns
    for k  in range(0,num_pixels):
        y = nz_pixels[0][k]
        x = nz_pixels[1][k]
        for ch in range(0,3):
            final_img[y,x,ch] = solns[k,ch]
        #final_img[y, x, :] = solns[k,:]

    print("End optimisation")
    final_img = final_img.astype(np.uint8)

    output_image_path = img_folder + name_match +'_gradMatching' +img_ext
    scipy.misc.imsave(output_image_path,final_img)

    #cv2.imshow('final',final_img)
    #cv2.waitKey(0)
    #cv2.destroyWindow('final')
    
    
    return(0)
    
    
def main():
    #/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/images/GradHist/TilesOrnate0158_512_SAME_autocorr.png
    #/home/gonthier/Travail_Local/Texture_Style/Style_Transfer/images/GradHist/TilesOrnate0158_512_SAME_texture_spectrum.png
    img_ext = '.png'
    img_folder = 'images/GradHist/'
    image_path = img_folder + 'TilesOrnate0158_512' +img_ext
    ref_image = scipy.misc.imread(image_path)
    image_path = img_folder + 'TilesOrnate0158_512_SAME_texture_spectrum' +img_ext
    syn_image = scipy.misc.imread(image_path)
    
    gradients_matching(ref_image, syn_image, n_bins=100)
    
if __name__ == '__main__':
    #main()
    #seamless()
    #ReconstructionFromDerivativeTest()
    HistogramOfGradientMatching()
