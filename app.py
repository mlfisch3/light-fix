import streamlit as st

VERSION = 2
title = f'SODEF{VERSION}'
st.set_page_config(page_title=title, layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

#st.markdown("Main page")
DEFAULT_DIR_PATH = f'C:\GIT_REPOS\BIMEF_MF{VERSION}\DOWNLOADS'

#st.write(st.session_state)
import cv2
import numpy as np
from matplotlib import image as img
import datetime
from psutil import Process
from utils.io_tools import change_extension, load_binary, mkpath
from utils.sodef import bimef
from utils.array_tools import float32_to_uint8, autoscale_array, array_info
from psutil import virtual_memory, swap_memory, Process
from utils.logging import timestamp
import os
from os import getpid
import sys
from PIL import Image, ImageOps
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
#python_dir = r'C:\CODE\PYTHON'
#sys.path.insert(1,python_dir)

#from sodef import view_memory, timestamp, float32_to_uint8
#from sodef import #log_memory, entropy, xentropy, kl_divergence, autoscale_array, array_info, print_array_info, geometric_mean
#from sodef.sodef import sodef

#from sodef import calculate_texture_weights_I #, calculate_texture_weights_II, calculate_texture_weights_III, calculate_texture_weights_IV, calculate_texture_weights_V
#from sodef import joint_entropy, mutual_information, variation_of_information
#from sodef import normalized_variation_of_information, conditional_entropy

                

# def reset_calculation(condition, x):
#     if st.session_state.live_updates == 'Automatic':
#         del st.session_state['ai_out']

# def reset_calculation(condition, x):
#     if st.session_state.live_updates == 'Automatic':
#         del st.session_state[x]

# def update_state(name, value):
#     if name not in st.session_state:
#         st.session_state.name = value




#@st.cache(max_entries=1, show_spinner=False, suppress_st_warning=True)
#@st.cache(show_spinner=False, suppress_st_warning=True)
#@st.experimental_memo
def adjust_intensity(array, 
                     exposure_ratio=-1, enhance=0.8, 
                     a=-0.3293, b=1.1258, lamda=0.3, 
                     texture_style='I',
                     kernel_shape=(5,1), scale=0.1, sharpness=0.001, 
                     dim_threshold=0.5, dim_size=(50,50), 
                     solver='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, 
                     lo=1, hi=7, color_gamma=0.5, npoints=20, return_texture_weights=True
                     ):
    
    '''
    Wrapper for function bimef(). Exists in order to avoid hard-coding a particular set of numerical default values in the definition of bimef()
    '''

    return bimef(array[:,:,[2,1,0]], 
                 exposure_ratio=exposure_ratio, 
                 enhance=enhance, 
                 a=a, 
                 b=b, 
                 lo=lo, 
                 hi=hi, 
                 lamda=lamda,
                 texture_style=texture_style,
                 kernel_shape=kernel_shape, 
                 scale=scale, 
                 sharpness=sharpness, 
                 dim_threshold=dim_threshold, 
                 dim_size=dim_size, 
                 solver=solver, 
                 CG_prec='ILU', 
                 CG_TOL=CG_TOL, 
                 LU_TOL=LU_TOL, 
                 MAX_ITER=MAX_ITER, 
                 FILL=FILL, 
                 color_gamma=color_gamma, 
                 npoints=npoints, 
                 return_texture_weights=return_texture_weights) 


SCRAPYARD_FILE_PATH = '.\\examples\\scrapyard.jpg'
SELFIE_FILE_PATH = '.\\examples\\selfie.jpg'
CYLINDER_FILE_PATH = '.\\examples\\cylinder.jpg'
PARK_FILE_PATH = '.\\examples\\park.jpg'
SCHOOL_FILE_PATH = '.\\examples\\school.jpg'
SPIRAL_FILE_PATH = '.\\examples\\spiral.jpg'

def limit_cache():
    if 'limit_cache_use' not in st.session_state:
        st.session_state.limit_cache_use = False
    elif st.session_state.limit_cache_use == False:
        st.session_state.limit_cache_use = True
    elif st.session_state.limit_cache_use == True:
        st.session_state.limit_cache_use = False

def clear_cache():
    st.experimental_memo.clear()

def reset():
    if 'image_np' in st.session_state:
        del st.session_state.image_np
    if 'input_file_name' in st.session_state:
        del st.session_state.input_file_name
    
    if st.session_state.limit_cache_use:
        clear_cache()


if 'full_clear' not in st.session_state:
    st.session_state.full_clear = False


def full_reset():
    if 'image_np' in st.session_state:
        del st.session_state.image_np
    if 'input_file_name' in st.session_state:
        del st.session_state.input_file_name
    
    if st.session_state.limit_cache_use:
        clear_cache()

    st.session_state.full_clear = True



def run_app(default_power=0.5, 
            default_smoothness=0.3, 
            default_texture_style='I',
            default_kernel_parallel=5, 
            default_kernel_orthogonal=1,
            default_sharpness=0.001,
            CG_TOL=0.1, 
            LU_TOL=0.015, 
            MAX_ITER=50, 
            FILL=50,
            default_dim_size=(50), 
            default_dim_threshold=0.5, 
            default_a=-0.3293, 
            default_b=1.1258, 
            default_lo=1, 
            default_hi=7,
            default_exposure_ratio=-1, 
            default_color_gamma=0.3981):

    
    if 'limit_cache_use' not in st.session_state:
        st.session_state.limit_cache_use = False

    if 'input_file_name' not in st.session_state:
        st.session_state.input_file_name = ''

    #log_memory('run_app||B')

    with st.sidebar:
    
        with st.expander("Reset App"):
            with st.form("Apply"):
                st.form_submit_button("Reset Now", on_click=full_reset)
                if st.session_state.full_clear:
                    st.session_state.full_clear = False
                    st.experimental_rerun()

        # logging = st.radio("Process Log:", ('OFF', 'ON'), on_click=set_logging, args=)
        input_selection = st.radio("Select Example:", ('scrapyard', 'selfie', 'cylinder', 'park', 'school', 'spiral'), horizontal=True, on_change=reset)
        example_paths = {'scrapyard': SCRAPYARD_FILE_PATH, 'selfie': SELFIE_FILE_PATH, 'cylinder': CYLINDER_FILE_PATH, 'park':PARK_FILE_PATH, 'school':SCHOOL_FILE_PATH, 'spiral':SPIRAL_FILE_PATH}
        IMAGE_EXAMPLE_PATH = example_paths[input_selection]

        ################################################################################################
        ################### INPUT IMAGE ################################################################
        #log_memory('run_app|file_uploader|B')
        #st.text("\nOr Upload Your Own Image:")
        fImage = st.file_uploader("Or Upload Your Own Image:", on_change=reset) #("Process new image:")
        #log_memory('run_app|file_uploader|E')
        ################################################################################################

        with st.expander("Parameter Settings"):
            with st.form('Parameter Settings'):
                submitted = st.form_submit_button('Apply')
                ################################################################################################
                ################### GUI PARAMETERS #########################################################
                #speed = float(st.text_input(f'Speed   (default = {default_speed})', str(default_speed), on_change=reset_calculation, args=(st.session_state.live_updates == 'Automatic', 'ai_out')))
                granularity_selection = st.radio("Illumination detail", ('standard', 'boost', 'max'), horizontal=True)
                granularity_dict = {'standard': 0.1, 'boost': 0.3, 'max': 0.5}
                granularity = granularity_dict[granularity_selection]

                power = float(st.text_input(f'Power     (default = {default_power})', str(default_power)))
            
                smoothness = float(st.text_input(f'Smoothness   (default = {default_smoothness})', str(default_smoothness)))
                sharpness = float(st.text_input(f'Sharpness   (default = {default_sharpness})', str(default_sharpness)))
                kernel_parallel = int(st.text_input(f'Kernel Parallel   (default = {default_kernel_parallel})', str(default_kernel_parallel)))
                kernel_orthogonal = int(st.text_input(f'Kernel Orthogonal   (default = {default_kernel_orthogonal})', str(default_kernel_orthogonal))) 
                a = float(st.text_input(f'Camera A   (default = {default_a})', str(default_a)))
                b = float(st.text_input(f'Camera B   (default = {default_b})', str(default_b)))
                lo = int(st.text_input(f'Min Gain   (default = {default_lo})', str(default_lo), help="Sets lower bound of search range for Exposure Ratio.  Only relevant if Exposure Ratio is in 'auto' mode"))
                hi = int(st.text_input(f'Max Gain   (default = {default_hi})', str(default_hi), help="Sets upper bound of search range for Exposure Ratio.  Only relevant if Exposure Ratio is in 'auto' mode"))
                exposure_ratio = float(st.text_input(f'Exposure Ratio   (default = -1 (auto))', str(default_exposure_ratio)))
                color_gamma = float(st.text_input(f'Color Gamma   (default = {default_color_gamma})', str(default_color_gamma)))
                texture_weight_calculator = st.radio("Select texture weight calculator", ('I', 'II', 'III', 'IV', 'V'), horizontal=True) 
                texture_weight_calculator_dict = {
                            'I':  ('I', CG_TOL, LU_TOL, MAX_ITER, FILL),
                            'II': ('II', CG_TOL, LU_TOL, MAX_ITER, FILL),
                            'III':('III', 0.1*CG_TOL, LU_TOL, 10*MAX_ITER, FILL),
                            'IV': ('IV', 0.5*CG_TOL, LU_TOL, MAX_ITER, FILL/2),
                            'V':  ('V', CG_TOL, LU_TOL, MAX_ITER, FILL)
                            }

                texture_style, cg_tol, lu_tol, max_iter, fill = texture_weight_calculator_dict[texture_weight_calculator]
              #  if speed <= 2:
              #      fill *= 2

        checkbox = st.checkbox('Show Mid-Process Images')

    col1, col2, col3 = st.columns(3)

    if fImage is not None:

        st.session_state.input_file_name = str(fImage.__dict__['name'])
        #st.session_state.image_currently_loaded = True
       
        #log_memory('run_app|np.frombuffer|B')
        np_array = np.frombuffer(fImage.getvalue(), np.uint8)
        #log_memory('run_app|np.frombuffer|E')
        #log_memory('run_app|cv2.imdecode|B')
        st.session_state.image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        #log_memory('run_app|cv2.imdecode|E')

        # #st.session_state.image_np = image_np

        # st.session_state.input_file_name = input_file_name
        # if 'ai_out' in st.session_state:         
        #     del st.session_state['ai_out']
    
    else:

        st.session_state.input_file_name = IMAGE_EXAMPLE_PATH.split('\\')[-1]
        st.session_state.image_np = np.array(ImageOps.exif_transpose(Image.open(IMAGE_EXAMPLE_PATH)))[:,:,[2,1,0]]

    if 'image_np' in st.session_state:

        image_np = st.session_state.image_np
        input_file_name = st.session_state.input_file_name
        #########################################################################################################
        #########################################################################################################
        start = datetime.datetime.now()
        #log_memory('run_app|adjust_intensity|B')

        ai_out = adjust_intensity(image_np, 
                                   exposure_ratio=exposure_ratio, 
                                   scale=granularity, 
                                   enhance=power, 
                                   lamda=smoothness, 
                                   a=a, 
                                   b=b, 
                                   lo=lo, 
                                   hi=hi,
                                   texture_style=texture_style, 
                                   kernel_shape=(kernel_parallel, kernel_orthogonal),
                                   sharpness=sharpness, 
                                   color_gamma=color_gamma,
                                   CG_TOL=cg_tol, 
                                   LU_TOL=lu_tol, 
                                   MAX_ITER=max_iter, 
                                   FILL=fill,
                                   return_texture_weights=True)

        image_np_maxRGB, image_np_gradient_v, image_np_gradient_h, image_np_texture_weights_v, image_np_texture_weights_h, illumination_map, fusion_weights, image_exposure_maxent, image_np_simulation, image_np_fused, exposure_ratio = ai_out
        titles = ['image_np_maxRGB', 'image_np_gradient_v', 'image_np_gradient_h', 'image_np_texture_weights_v', 'image_np_texture_weights_h', 'illumination_map', 'fusion_weights', 'image_exposure_maxent', 'image_np_simulation', 'image_np_fused']

        #log_memory('run_app|adjust_intensity|E')
        end = datetime.datetime.now()
        process_time = (end - start).total_seconds()
        print(f'[{datetime.datetime.now().isoformat()}]  Processing time: {process_time:.5f} s')
        #########################################################################################################

        image_np_L1_map = np.abs(image_np_gradient_v * image_np_texture_weights_v + image_np_gradient_h * image_np_texture_weights_h)
        image_np_L1_map = image_np_L1_map.clip(min=0, max=image_np_L1_map.flatten().mean()+image_np_L1_map.flatten().std())
        image_np_L1_map = float32_to_uint8(autoscale_array(image_np_L1_map))
        image_np_L1_map = np.tile(image_np_L1_map.T, (3,1,1)).T

        image_np_wls_map = (image_np_texture_weights_v + image_np_texture_weights_h)/2
        image_np_wls_map_med = image_np_wls_map.flatten().mean()
        image_np_wls_map_std = image_np_wls_map.flatten().std()
        image_np_wls_map = image_np_wls_map.clip(min=image_np_wls_map_med-image_np_wls_map_std, max=image_np_wls_map_med+image_np_wls_map_std)
        image_np_wls_map = float32_to_uint8(autoscale_array(image_np_wls_map))
        image_np_wls_map = np.tile(image_np_wls_map.T, (3,1,1)).T

        image_np_fine_texture_map = float32_to_uint8(image_np_maxRGB - illumination_map)
        #image_np_fine_texture_map_info_str = array_info(image_np_fine_texture_map, print_info=True, name='image_np_fine_texture_map_info_str')#, return_info=True, return_info_str=True)
        image_np_fine_texture_map = np.tile(image_np_fine_texture_map.T, (3,1,1)).T

        illumination_map = float32_to_uint8(illumination_map)
        illumination_map = np.tile(illumination_map.T, (3,1,1)).T

        fusion_weights = float32_to_uint8(fusion_weights)
        fusion_weights = np.tile(fusion_weights.T, (3,1,1)).T

        # array_info(image_np_maxRGB, print_info=True, name='image_np_maxRGB')
        # array_info(image_np_gradient_v, print_info=True, name='image_np_gradient_v')
        # array_info(image_np_gradient_h, print_info=True, name='image_np_gradient_h')
        # array_info(image_np_texture_weights_v, print_info=True, name='image_np_texture_weights_v')
        # array_info(image_np_texture_weights_h, print_info=True, name='image_np_texture_weights_h')
        # array_info(image_np_wls_map, print_info=True, name='image_np_wls_map')
        # array_info(illumination_map, print_info=True, name='illumination_map')

        # array_info(fusion_weights, print_info=True, name='fusion_weights')
        # array_info(image_exposure_maxent, print_info=True, name='image_exposure_maxent')
        # array_info(image_np_simulation, print_info=True, name='image_np_simulation')
        # array_info(image_np_L1_map, print_info=True, name='image_L1_map')
        # array_info(image_np_fine_texture_map, print_info=True, name='image_np_fine_texture_map')
        # array_info(image_np_fused, print_info=True, name='image_np_fused')


        # gradient
        granularity_param_str = f'_{granularity*100:.0f}'

        # convolution
        convolution_param_str = granularity_param_str + f'_{kernel_parallel:d}_{kernel_orthogonal:d}'

        # texture_weights
        texture_param_str = convolution_param_str + f'_{sharpness*1000:.0f}_{texture_weight_calculator:s}'

        # illumination map, texture map
        smooth_param_str = texture_param_str + f'_{smoothness*100:.0f}'

        # fused image        
        fusion_param_str = smooth_param_str + f'_{color_gamma*100:.0f}_{power*100:.0f}_{-a*1000:.0f}_{b*1000:.0f}_{exposure_ratio*100:.0f}'

        #input_file_name = str(fImage.__dict__['name'])
        input_file_ext = '.' + str(input_file_name.split('.')[-1])
        input_file_basename = input_file_name.replace(input_file_ext, '')
        output_wls_map_file_name = input_file_basename + '_WLS' + texture_param_str + input_file_ext
        output_L1_map_file_name = input_file_basename + '_L1' + texture_param_str + input_file_ext
        output_fine_texture_map_file_name = input_file_basename + '_FTM' + smooth_param_str + input_file_ext
        output_illumination_map_file_name = input_file_basename + '_ILL' + smooth_param_str + input_file_ext
        output_simulation_file_name = input_file_basename + '_SIM' + fusion_param_str + input_file_ext
        output_exposure_maxent_file_name = input_file_basename + '_EME' + smooth_param_str + input_file_ext
        output_fusion_weights_file_name = input_file_basename + '_FW' + fusion_param_str + input_file_ext
        output_fused_file_name = input_file_basename + '_FUSION' + fusion_param_str + input_file_ext
        

        with col1:        

            ###########################
            # st.header(f'Original Image')
            # #log_memory('run_app|st.image|B')
            # st.image(image_np[:,:,[2,1,0]])
            # #log_memory('run_app|st.image|E')

            # #checkbox = st.checkbox('download')
            # #if checkbox:
            # input_file_name = st.text_input('Download Original Image As', input_file_name)
            # ext = '.' + input_file_name.split('.')[-1]
            # #log_memory('run_app|cv2.imencode|B')
            # image_np_binary = cv2.imencode(ext, st.session_state['image_np'])[1].tobytes()
            # #log_memory('run_app|cv2.imencode|E')
            # #button = BigButton(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")
            # #button = st.download_button(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")
            # data = b'0'
            # button = st.download_button(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png", on_click=load_binary, args=(ext, st.session_state['image_np'], data))
            
            ###########################
            st.markdown("<h3 style='text-align: center; color: white;'>Original</h3>", unsafe_allow_html=True)
            #st.header(f'Original')
            #log_memory('run_app|st.image|B')
            st.image(image_np[:,:,[2,1,0]])
            #log_memory('run_app|st.image|E')

            #checkbox = st.checkbox('download')
            #if checkbox:
            input_file_name = st.text_input('Download Original Image As', input_file_name)
            ext = '.' + input_file_name.split('.')[-1]
            #log_memory('run_app|cv2.imencode|B')
            image_np_binary = cv2.imencode(ext, image_np)[1].tobytes()
            #log_memory('run_app|cv2.imencode|E')
            #button = BigButton(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")
            #button = st.download_button(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")
            #data = b'0'
            button = st.download_button(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")

            if checkbox:
                ###########################
                st.header(f'WLS Map')
                #log_memory('run_app|st.image|B')    
                st.image(image_np_wls_map, clamp=True)
                #log_memory('run_app|st.image|E')

                output_wls_map_file_name = st.text_input('Download WLS Map As', output_wls_map_file_name)
                ext = '.' + output_wls_map_file_name.split('.')[-1]
                #log_memory('run_app|cv2.imencode|B')           
                #image_np_wls_map_binary = cv2.imencode(ext, image_np_wls_map[:,:,[2,1,0]])[1].tobytes()
                image_np_wls_map_binary = load_binary(ext, image_np_wls_map, color_channel='bgr')
                #log_memory('run_app|cv2.imencode|E')

                button = st.download_button(label = "Download WLS Map", data = image_np_wls_map_binary, file_name = output_wls_map_file_name, mime = "image/png")
                
                ###########################
                st.header(f'L1 Map')
                #log_memory('run_app|st.image|B')
                st.image(image_np_L1_map, clamp=True)
                #log_memory('run_app|st.image|E')

                output_L1_map_file_name = st.text_input('Download L1 Map As', output_L1_map_file_name)
                ext = '.' + output_L1_map_file_name.split('.')[-1]
                #log_memory('run_app|cv2.imencode|B')
                image_np_L1_map_binary = cv2.imencode(ext, image_np_L1_map[:,:,[2,1,0]])[1].tobytes()
                #log_memory('run_app|cv2.imencode|E')

                button = st.download_button(label = "Download L1 Map", data = image_np_L1_map_binary, file_name = output_L1_map_file_name, mime = "image/png")

        with col2:
        ###########################
            st.markdown("<h3 style='text-align: center; color: white;'>Simulation</h3>", unsafe_allow_html=True)
            #st.header(f'Simulation')
            #log_memory('run_app|st.image|B')            
            st.image(image_np_simulation, clamp=True)
            #log_memory('run_app|st.image|E')

            output_simulation_file_name = st.text_input('Download Simulation As', output_simulation_file_name)
            ext = '.' + output_simulation_file_name.split('.')[-1]
            #log_memory('run_app|cv2.imencode|B')
            image_np_simulation_binary = cv2.imencode(ext, image_np_simulation[:,:,[2,1,0]])[1].tobytes()
            #log_memory('run_app|cv2.imencode|E')

            button = st.download_button(label = "Download Simulation", data = image_np_simulation_binary, file_name = output_simulation_file_name, mime = "image/png")        

            if checkbox:

                ###########################
                st.header(f'Illumination Map')
                #log_memory('run_app|st.image|B')            
                st.image(illumination_map, clamp=True)
                #log_memory('run_app|st.image|E')

                output_illumination_map_file_name = st.text_input('Download Illumination Map As', output_illumination_map_file_name)
                ext = '.' + output_illumination_map_file_name.split('.')[-1]
                #log_memory('run_app|cv2.imencode|B')
                illumination_map_binary = cv2.imencode(ext, illumination_map[:,:,[2,1,0]])[1].tobytes()
                #log_memory('run_app|cv2.imencode|E')

                button = st.download_button(label = "Download Illumination Map", data = illumination_map_binary, file_name = output_illumination_map_file_name, mime = "image/png")

                ###########################
                st.header(f'Fusion Weights')
                #log_memory('run_app|st.image|B')            
                st.image(fusion_weights, clamp=True)
                #log_memory('run_app|st.image|E')

                output_fusion_weights_file_name = st.text_input('Download Fusion Weights As', output_fusion_weights_file_name)
                ext = '.' + output_fusion_weights_file_name.split('.')[-1]
                #log_memory('run_app|cv2.imencode|B')
                fusion_weights_binary = cv2.imencode(ext, fusion_weights)[1].tobytes()
                #log_memory('run_app|cv2.imencode|E')

                button = st.download_button(label = "Download Fusion Weights", data = fusion_weights_binary, file_name = output_fusion_weights_file_name, mime = "image/png")

        with col3:

            ###########################
            st.markdown("<h3 style='text-align: center; color: white;'>Fusion</h3>", unsafe_allow_html=True)
            #st.header(f'Fusion')
            #log_memory('run_app|st.image|B')            
            st.image(image_np_fused, clamp=True)
            #log_memory('run_app|st.image|E')

            output_fused_file_name = st.text_input('Download Fused Image As', output_fused_file_name)
            ext = '.' + output_fused_file_name.split('.')[-1]
            #log_memory('run_app|cv2.imencode|B')
            image_np_fused_binary = cv2.imencode(ext, image_np_fused[:,:,[2,1,0]])[1].tobytes()
            #log_memory('run_app|cv2.imencode|E')

            button = st.download_button(label = "Download Fused Image", data = image_np_fused_binary, file_name = output_fused_file_name, mime = "image/png")
        
            if checkbox:

                ###########################
                st.header(f'MaxEnt Exposure')
                #log_memory('run_app|st.image|B')            
                st.image(image_exposure_maxent, clamp=True)
                #log_memory('run_app|st.image|E')

                output_exposure_maxent_file_name = st.text_input('Download Maxent Exposure As', output_exposure_maxent_file_name)
                ext = '.' + output_exposure_maxent_file_name.split('.')[-1]
                #log_memory('run_app|cv2.imencode|B')
                #image_exposure_maxent_binary = cv2.imencode(ext, image_exposure_maxent[:,:,[2,1,0]])[1].tobytes()
                image_exposure_maxent_binary = load_binary(ext, image_exposure_maxent, color_channel='bgr')
                #log_memory('run_app|cv2.imencode|E')

                button = st.download_button(label = "Download Maxent Exposure", data = image_exposure_maxent_binary, file_name = output_exposure_maxent_file_name, mime = "image/png")            
                ###########################
                st.header(f'Fine Texture Map')
                #log_memory('run_app|st.image|B')   
                st.image(image_np_fine_texture_map, clamp=True)
                #log_memory('run_app|st.image|E')

                output_fine_texture_map_file_name = st.text_input('Download Fine Texture Map As', output_fine_texture_map_file_name)
                ext = '.' + output_fine_texture_map_file_name.split('.')[-1]
                #log_memory('run_app|cv2.imencode|B')
                image_np_fine_texture_map_binary = cv2.imencode(ext, image_np_fine_texture_map[:,:,[2,1,0]])[1].tobytes()
                #log_memory('run_app|cv2.imencode|E')

                button = st.download_button(label = "Download Fine Texture Map", data = image_np_fine_texture_map_binary, file_name = output_fine_texture_map_file_name, mime = "image/png")

            # else:                
            #     with st.form("Show MaxEnt Exposure & Fine Texture Map"):
            #         if st.form_submit_button('Show MaxEnt Exposure & Fine Texture Map'):
            #             st.header(f'MaxEnt Exposure')       
            #             st.image(image_exposure_maxent, clamp=True)
            #             st.header(f'Fine Texture Map')
            #             st.image(image_np_fine_texture_map, clamp=True)

        st.text('\n\n\n\n\n\n\n\n')
        st.text('*Supported file extensions: jpg, jpeg, png, gif, bmp, pdf, svg, eps')
        st.text('\n\n\n\n\n\n\n\n')

        with st.sidebar:
            
            if st.checkbox("View Image Info"):
                col_left, col_mid, col_right = st.columns(3)


                #log_memory('run_app|array_info|B')
                image_np_info, image_np_info_str = array_info(image_np, print_info=False, return_info=True, return_info_str=True, name='Original Image')
                #log_memory('run_app|array_info|E')
                
                #log_memory('run_app|array_info|B')
                image_np_fused_info, image_np_fused_info_str = array_info(uint8_to_float32(image_np_fused), print_info=False, return_info=True, return_info_str=True, name='Fused Image')
                #log_memory('run_app|array_info|E')

                st.text(f'exposure ratio: {exposure_ratio:.4f}')
                
                entropy_change_abs = image_np_fused_info['entropy'] - image_np_info['entropy']
                entropy_change_rel = (image_np_fused_info['entropy'] / image_np_info['entropy']) - 1.0
                st.text(f'entropy change: {entropy_change_abs:.4f} ({entropy_change_rel * 100.0:.4f} %)\n')   

                st.text(image_np_info_str)
                
                st.text("\n\n\n\n\n")
               
                st.text(image_np_fused_info_str)
                st.text(image_np_fused_info['bytes'])

                
                image_maxent_info, image_maxent_info_str = array_info(image_exposure_maxent, print_info=False, return_info=True, return_info_str=True, name='MaxEnt Image')
                st.text("\n\n\n\n\n")
                st.text(image_maxent_info_str)

                
                # if st.checkbox("Advanced"):
                #     #log_memory('run_app|show_statistics|E')



                #     kl = kl_divergence(image_np, image_np_fused)
                #     js = js_divergence(image_np, image_np_fused)
                #     s_joint = joint_entropy(image_np, image_np_fused)
                #     i_mutual = mutual_information(image_np, image_np_fused)
                #     voi = variation_of_information(image_np, image_np_fused)
                #     nvoi = normalized_variation_of_information(image_np, image_np_fused)

                #     st.text("\n\n\n\n\n")

                #     st.text(f'KL divergence: {kl:.4f}')
                #     st.text(f'JS divergence: {js:.4f}')
                #     st.text(f'joint entropy: {s_joint:.4f}')
                #     st.text(f'mutual information: {i_mutual:.4f}')
                #     st.text(f'variation of information (VOI): {voi:.4f}')
                #     st.text(f'normalized VOI: {nvoi:.4f}\n')
            
            with st.expander("Resource Settings"):
                checkbox_limit_cache = st.checkbox('Limit Caching', on_change=limit_cache, 
                                                   help="Purge existing subprocess results from cache whenever a new image is loaded.  \
                                                   Helps protect the app from exceeding available memory resources. \
                                                   Might cause longer processing times."
                                                   )                                                 
                pid = getpid()
                mem = Process(pid).memory_info()[0]/float(2**20)
                virt = virtual_memory()[3]/float(2**20)
                swap = swap_memory()[1]/float(2**20)

                st.text(f'[{timestamp()}]\nPID: {pid}')

                st.text(f'rss: {mem:.2f} MB\nvirt: {virt:.2f} MB\nswap: {swap:.2f} MB')

        with st.form("Download Batch"):
            st.text('Download All Output Files to Local Folder:')     
 
            colI, colII = st.columns(2)
            with colI:
                default_dir_path = DEFAULT_DIR_PATH
                dir_path = st.text_input('Folder Name:', default_dir_path)

                last_download_time = '-'
            # with colII:
            #    info = array_info(image_np, print_info=False, return_info=True, return_info_str=False)
            #    st.text(f"\n\n  height: {info['shape'][0]}\n  width:  {info['shape'][1]}\n        ")
 
            colA, colB, colC, colD, colE = st.columns(5)
            with colA:
                ext_batch = st.text_input('File extension:', 'jpg')

                illumination_map_fullpath = os.path.join(dir_path,output_illumination_map_file_name)
                wls_map_fullpath = os.path.join(dir_path, output_wls_map_file_name)
                L1_map_fullpath = os.path.join(dir_path, output_L1_map_file_name)
                fine_texture_map_fullpath = os.path.join(dir_path, output_fine_texture_map_file_name)
                simulation_fullpath = os.path.join(dir_path,output_simulation_file_name)
                exposure_maxent_fullpath = os.path.join(dir_path,output_exposure_maxent_file_name)
                fusion_weights_fullpath = os.path.join(dir_path,output_fusion_weights_file_name)
                fused_fullpath = os.path.join(dir_path,output_fused_file_name)

            with colB:
                st.text('\n')
                st.text('\n')
                if st.form_submit_button('DOWNLOAD ALL', on_click=mkpath, args=[dir_path]):
                #if st.button('DOWNLOAD ALL', on_click=mkpath, args=[dir_path]):
                    #log_memory('run_app|download all|B')
                    mkpath(dir_path)
                    img.imsave(change_extension(wls_map_fullpath, ext_batch), image_np_wls_map)
                    img.imsave(change_extension(L1_map_fullpath, ext_batch), image_np_L1_map)
                    img.imsave(change_extension(illumination_map_fullpath, ext_batch), illumination_map)
                    img.imsave(change_extension(fine_texture_map_fullpath, ext_batch), image_np_fine_texture_map)
                    img.imsave(change_extension(simulation_fullpath, ext_batch), image_np_simulation)
                    img.imsave(change_extension(fusion_weights_fullpath, ext_batch), fusion_weights)
                    img.imsave(change_extension(exposure_maxent_fullpath, ext_batch), image_exposure_maxent)
                    img.imsave(change_extension(fused_fullpath, ext_batch), image_np_fused)
                    last_download_time = datetime.datetime.now()
                    #log_memory('run_app|download all|E')

            st.text(f'last batch download completed at {last_download_time}')
        
        #log_memory('run_app||E')

if __name__ == '__main__':
    # tracemalloc.start()
    # snapshot1 = tracemalloc.take_snapshot()
    # print(tracemalloc.get_traced_memory())
    total_start = datetime.datetime.now()
    log_memory('main|run_app|B')

    run_app()

    log_memory('main|run_app|E')
    total_end = datetime.datetime.now()
    total_process_time = (total_end - total_start).total_seconds()
    print(f'[{timestamp()}]  Total processing time: {total_process_time:.5f} s')
    # gc.collect()
    # snapshot2 = tracemalloc.take_snapshot()
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    # print("[ Top 10 differences ]")
    # for stat in top_stats[:10]:
    #     print(stat)

    # #print(tracemalloc.get_traced_memory())
    # tracemalloc.stop()