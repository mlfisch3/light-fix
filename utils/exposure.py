import numpy as np
import streamlit as st

from utils.array_entropy import entropy
from utils.array_tools import geometric_mean, imresize, autoscale_array

#### Exposure Functions
@st.experimental_memo(show_spinner=False)
def applyK(G, k, a=-0.3293, b=1.1258, verbose=False):
    #log_memory('applyK||B')
    if k==1.0:
        return G.astype(np.float32)

    if k<=0:
        return np.ones_like(G, dtype=np.float32)

    gamma = k**a
    beta = np.exp((1-gamma)*b)

    if verbose:
        print(f'a: {a:.4f}, b: {b:.4f}, k: {k:.4f}, gamma: {gamma:.4f}, beta: {beta}.  ----->  output = {beta:.4} * image^{gamma:.4f}')
    #log_memory('applyK||E')
    return (np.power(G,gamma)*beta).astype(np.float32)


@st.experimental_memo(show_spinner=False)
def get_dim_pixels(image,dim_pixels,dim_size=(50,50)):
    #log_memory('get_dim_pixels||B')
    dim_pixels = imresize(dim_pixels,size=dim_size)

    image = imresize(image,size=dim_size)
    image = np.where(image>0,image,0)
    Y = geometric_mean(image)
    #log_memory('get_dim_pixels||E')
    return Y[dim_pixels]

@st.experimental_memo(show_spinner=False)
def optimize_exposure_ratio(array, a, b, lo=1, hi=7, npoints=20):
    #log_memory('optimize_exposure_ratio||B')  
    if sum(array.shape)==0:
        #log_memory('optimize_exposure_ratio||B')  
        return 1.0

    sample_ratios = np.r_[lo:hi:np.complex(0,npoints)].tolist()
    entropies = np.array(list(map(lambda k: entropy(applyK(array, k, a, b)), sample_ratios)))
    optimal_index = np.argmax(entropies)
    #log_memory('optimize_exposure_ratio||B')  
    return sample_ratios[optimal_index]

    return fusion_weights

@st.experimental_memo(show_spinner=False)
def adjust_exposure(image, illumination_map, a, b, exposure_ratio=-1, dim_threshold=0.5, dim_size=(50,50), lo=1, hi=7, npoints=20, color_gamma=0.3981, verbose=False):
    #log_memory('bimef|autoscale_array|B')
    image_01 = autoscale_array(image)
    #log_memory('bimef|autoscale_array|E')
    #log_memory('bimef|np.zeros_like|B')
    dim_pixels = np.zeros_like(illumination_map)
    #log_memory('bimef|np.zeros_like|E')
    
    if exposure_ratio==-1:
        #log_memory('bimef|dim_threshold|B')
        dim_pixels = illumination_map<dim_threshold
        #log_memory('bimef|dim_threshold|E')
        #log_memory('bimef|get_dim_pixels|B')
        Y = get_dim_pixels(image_01, dim_pixels, dim_size=dim_size) 
        #log_memory('bimef|get_dim_pixels|E')
        #log_memory('bimef|optimize_exposure_ratio|B')
        exposure_ratio = optimize_exposure_ratio(Y, a, b, lo=lo, hi=hi, npoints=npoints)
        #log_memory('bimef|optimize_exposure_ratio|E')


    #log_memory('bimef|applyK|B')
    image_01_K = applyK(image_01, exposure_ratio, a, b) 
    if (image_01.ndim == 2) or ((exposure_ratio < 1) & (exposure_ratio > 0)):
        image_exposure_adjusted = image_01_K
    else:
        image_01_ave = image_01.mean(axis=2)[:,:,None]
        image_01_dRGB = image_01 - image_01_ave
        image_01_ave_K = applyK(image_01_ave, exposure_ratio, a, b)
        image_exposure_adjusted = color_gamma * (image_01_ave_K + image_01_dRGB) + (1 - color_gamma)* image_01_K

    #log_memory('bimef|applyK|E')
    #log_memory('bimef|np.where|B')
    image_exposure_adjusted = np.where(image_exposure_adjusted>1,1,image_exposure_adjusted)    
    #log_memory('bimef|np.where|E')

    return image_exposure_adjusted, exposure_ratio