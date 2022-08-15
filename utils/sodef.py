import numpy as np
import datetime
import streamlit as st

MAX_ENTRIES = 20

from utils.array_tools import float32_to_uint8, imresize, autoscale_array
from utils.logging import timestamp
from utils.edge_preserving_smoother import smooth
from utils.exposure import adjust_exposure
from utils.fusion import calculate_fusion_weights, fuse_image


#@st.experimental_memo(show_spinner=False)
def bimef(image, 
          exposure_ratio=-1, 
          enhance=0.5, 
          a=-0.3293, 
          b=1.1258, 
          lamda=0.5, 
          texture_style='I',
          kernel_shape=(5,1), 
          scale=0.3, 
          sharpness=0.001, 
          dim_threshold=0.5, 
          dim_size=(50,50), 
          solver='cg', 
          CG_prec='ILU', 
          CG_TOL=0.1, 
          LU_TOL=0.015, 
          MAX_ITER=50, 
          FILL=50, 
          lo=1, 
          hi=7, 
          npoints=20, 
          color_gamma=0.3981,
          verbose=False, 
          print_info=True, 
          return_texture_weights=True):

    #log_memory('bimef||B')

    tic = datetime.datetime.now()

    #log_memory('bimef|image_01|B')
    image_01 = autoscale_array(image)
    #log_memory('bimef|image_01|E')

    #log_memory('bimef|image_01_maxRGB|B')
    if image_01.ndim == 3: 
       # image_01 = np.copy(image_01[:,:,:3])
        image_01_maxRGB = image_01.max(axis=2)
    else: 
        image_01_maxRGB = np.copy(image_01)
    #log_memory('bimef|image_01_maxRGB|E')
    
    #log_memory('bimef|scale|B')
    if (scale <= 0) | (scale >= 1) : 
        image_01_maxRGB_reduced = image_01_maxRGB
    else: 
        image_01_maxRGB_reduced = imresize(image_01_maxRGB, scale)  
    #log_memory('bimef|scale|E')

    #log_memory('bimef|smooth|B')
    smooth_out = smooth(image_01_maxRGB_reduced, 
                        image_01_maxRGB.shape, 
                        texture_style=texture_style, 
                        kernel_shape=kernel_shape, 
                        sharpness=sharpness, 
                        lamda=lamda, 
                        solver=solver, 
                        CG_prec=CG_prec, 
                        CG_TOL=CG_TOL, 
                        LU_TOL=LU_TOL, 
                        MAX_ITER=MAX_ITER, 
                        FILL=FILL, 
                        return_texture_weights=return_texture_weights)

    #log_memory('bimef|smooth|E')
    if return_texture_weights:
        gradient_v, gradient_h, texture_weights_v, texture_weights_h, illumination_map = smooth_out
    else:
        illumination_map = smooth_out
    
    illumination_map = np.where(illumination_map<0,0,illumination_map)
#    illumination_map = np.where(illumination_map>1,1,illumination_map)
    fusion_weights = calculate_fusion_weights(illumination_map, enhance, image_01.ndim) 

    image_exposure_adjusted, exposure_ratio = adjust_exposure(image_01, 
                                                              illumination_map, 
                                                              a, 
                                                              b, 
                                                              exposure_ratio=exposure_ratio, 
                                                              dim_threshold=dim_threshold, 
                                                              dim_size=dim_size, 
                                                              lo=lo, 
                                                              hi=hi, 
                                                              color_gamma=color_gamma, 
                                                              npoints=npoints)

    enhancement_map, enhanced_image = fuse_image(image_01, image_exposure_adjusted, fusion_weights)

    toc = datetime.datetime.now()

    if print_info:
        print(f'[{timestamp()}] exposure_ratio: {exposure_ratio:.4f}, enhance: {enhance:.4f}, lamda: {lamda:.4f}, scale: {scale:.4f}, runtime: {(toc-tic).total_seconds():.4f}s')
    
    #log_memory('bimef||E')
    
    return image_01_maxRGB, gradient_v, gradient_h, texture_weights_v, texture_weights_h, illumination_map, fusion_weights, image_exposure_adjusted, enhancement_map, enhanced_image, exposure_ratio