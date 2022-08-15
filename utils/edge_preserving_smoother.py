import streamlit as st
import numpy as np
from scipy import signal
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import cg, spsolve, spilu, LinearOperator, use_solver
from utils.array_tools import diff, cyclic_diff, imresize

def delta(x):   

    gradient_v = np.vstack([diff(x, axis=0),cyclic_diff(x,axis=0)])
    gradient_h = np.hstack([diff(x,axis=1),cyclic_diff(x,axis=1)])

    return gradient_v, gradient_h

def convolve(gradient_v, gradient_h, kernel_shape, return_level=1):
    ''' 
        gradient_v:         forward-difference in vertical direction
        gradient_h:         forward-difference in horizontal direction 
        kernel_shape:       2-tuple of odd positive integers. (Auto-incremented if even)
        return_level 1:     return only standard convolution
        return_level 2:     return standard convolution and convolution of absolute values
        return_level -2:    return only convolution of absolute values
    '''

    sigma_v, sigma_h = kernel_shape
    sigma_v += 1-sigma_v%2
    sigma_h += 1-sigma_h%2

    n_pad = [int(sigma_v/2),int(sigma_h/2)]
    kernel = np.ones(kernel_shape)

    out = []
    if (return_level > 0) or (return_level==-1):
        convolution_v = signal.convolve(gradient_v, kernel, method='fft')[n_pad[0]:n_pad[0]+gradient_v.shape[0],n_pad[1]:n_pad[1]+gradient_v.shape[1]]
        convolution_h = signal.convolve(gradient_h, kernel.T, method='fft')[n_pad[1]:n_pad[1]+gradient_h.shape[0],n_pad[0]:n_pad[0]+gradient_h.shape[1]]
        out.append(convolution_v)
        out.append(convolution_h)

    if (return_level > 1) or (return_level==-2):
        convolution_v_abs = signal.convolve(np.abs(gradient_v), kernel, method='fft')[n_pad[0]:n_pad[0]+gradient_v.shape[0],n_pad[1]:n_pad[1]+gradient_v.shape[1]]
        convolution_h_abs = signal.convolve(np.abs(gradient_h), kernel.T, method='fft')[n_pad[1]:n_pad[1]+gradient_h.shape[0],n_pad[0]:n_pad[0]+gradient_h.shape[1]]
        out.append(convolution_v_abs)
        out.append(convolution_h_abs)

    return tuple(out)

def calculate_texture_weights(image_01_maxRGB_reduced, kernel_shape=(5,1), sharpness=0.001, lamda=0.5, texture_style='I'):
    
    gradient_v, gradient_h = delta(image_01_maxRGB_reduced)
    convolution_v, convolution_h = convolve(gradient_v, gradient_h, kernel_shape)

    if texture_style == 'I':
        texture_weights_v = 1/(np.abs(convolution_v) * np.abs(gradient_v) + sharpness)
        texture_weights_h = 1/(np.abs(convolution_h) * np.abs(gradient_h) + sharpness)
        return gradient_v, gradient_h, texture_weights_v, texture_weights_h

    if texture_style == 'II':
        texture_weights_v = 1/(np.abs(gradient_v) + sharpness)
        texture_weights_h = 1/(np.abs(gradient_h) + sharpness)
        return gradient_v, gradient_h, texture_weights_v, texture_weights_h

    if texture_style == 'III':
        texture_weights_v = 1/(np.abs(convolution_v) + sharpness)
        texture_weights_h = 1/(np.abs(convolution_h) + sharpness)
        return gradient_v, gradient_h, texture_weights_v, texture_weights_h

    convolution_v_abs, convolution_h_abs = convolve(gradient_v, gradient_h, kernel_shape, return_level=-2)

    if texture_style == 'IV':
        texture_weights_v = convolution_v_abs/(np.abs(convolution_v) + sharpness)
        texture_weights_h = convolution_h_abs/(np.abs(convolution_h)+ sharpness)
        return gradient_v, gradient_h, texture_weights_v, texture_weights_h

    if texture_style == 'V':
        texture_weights_v = convolution_v_abs/(np.abs(convolution_v) + sharpness)
        texture_weights_h = convolution_h_abs/(np.abs(convolution_h)+ sharpness)
        return gradient_v, gradient_h, texture_weights_v, texture_weights_h


@st.experimental_memo(show_spinner=False)
def construct_map_cyclic(texture_weights_v, texture_weights_h, lamda):
    ''' all cyclic elements present '''
    #log_memory('construct_map||B')  
    r, c = texture_weights_h.shape        
    k = r * c
    texture_weights_h = texture_weights_h.astype('float32')
    texture_weights_v = texture_weights_v.astype('float32')
    lamda = np.float32(lamda)

    dh = -lamda * texture_weights_h.flatten(order='F')
    dv = -lamda * texture_weights_v.flatten(order='F')

    texture_weights_h_permuted_cols = np.roll(texture_weights_h,1,axis=1)
    dh_permuted_cols = -lamda * texture_weights_h_permuted_cols.flatten(order='F')
    texture_weights_v_permuted_rows = np.roll(texture_weights_v,1,axis=0)
    dv_permuted_rows = -lamda * texture_weights_v_permuted_rows.flatten(order='F')
       
    texture_weights_h_permuted_cols_head = np.zeros_like(texture_weights_h_permuted_cols, dtype='float32') 
    texture_weights_h_permuted_cols_head[:,0] = texture_weights_h_permuted_cols[:,0]
    dh_permuted_cols_head = -lamda * texture_weights_h_permuted_cols_head.flatten(order='F')
    
    texture_weights_v_permuted_rows_head = np.zeros_like(texture_weights_v_permuted_rows, dtype='float32')
    texture_weights_v_permuted_rows_head[0,:] = texture_weights_v_permuted_rows[0,:]
    dv_permuted_rows_head = -lamda * texture_weights_v_permuted_rows_head.flatten(order='F')

    texture_weights_h_no_tail = np.zeros_like(texture_weights_h, dtype='float32')
    texture_weights_h_no_tail[:,:-1] = texture_weights_h[:,:-1]
    dh_no_tail = -lamda * texture_weights_h_no_tail.flatten(order='F')

    texture_weights_v_no_tail = np.zeros_like(texture_weights_v, dtype='float32')
    texture_weights_v_no_tail[:-1,:] = texture_weights_v[:-1,:]
    dv_no_tail = -lamda * texture_weights_v_no_tail.flatten(order='F')
    
    Ah = spdiags([dh_permuted_cols_head, dh_no_tail], [-k+r, -r], k, k)
    
    Av = spdiags([dv_permuted_rows_head, dv_no_tail], [-r+1,-1],  k, k)
    
    A = 1 - (dh + dv + dh_permuted_cols + dv_permuted_rows)

    d = spdiags(A, 0, k, k)
    
    A = Ah + Av
    A = A + A.T + d
    #log_memory('construct_map||E')  
    return A
    #return csc_matrix(A, dtype=np.float32)



#### Sparse solver function
#@st.experimental_memo(show_spinner=False)
def solve_sparse_system(A, B, method='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, x0=None):
    """
    
    Consolidation of two functions into one:
    1) solve_linear_equation(A, B)
    2) solver_sparse(A, B)

    Solves for x = b/A  [[b is vector(B)]]
    A can be sparse (csc or csr) or dense
    b must be dense
    
   """    
    #log_memory('solver_sparse||B')  

    r, c = B.shape
    
    b = B.flatten(order='F').astype(np.float32)
    
    N = A.shape[0]
    if method == 'cg': 
        # solve using conjugate gradient descent
        if CG_prec == 'ILU':
            # with incomplete cholesky preconditioner
            # Find ILU preconditioner (constant in time)
            A_ilu = spilu(A.tocsc(), drop_tol=LU_TOL, fill_factor=FILL)
            #log_memory('solver_sparse|cg|1')  
            M = LinearOperator(shape=(N, N), matvec=A_ilu.solve, dtype='float32')
        else:
            M = None
        #log_memory('solver_sparse|cg|2')  
        if x0 is None:
            x0 = b  # input image more closely resembles its smoothed self than a draw from any distribution
        #  x0 = np.random.random(N).astype('float32')#, dtype='float32') # initialize with uniform distribution
        #log_memory('solver_sparse|cg|E')
        return cg(A, b, x0=x0, tol=CG_TOL, maxiter=MAX_ITER, M=M)[0].astype(np.float32).reshape(r,c, order='F')

    elif method == 'direct':
        #log_memory('solver_sparse|spsolve|E') 
        use_solver( useUmfpack = False ) # use single precision
        return spsolve(A, b).astype(np.float32).reshape(r,c, order='F')

#### Illumination Map Function 
@st.experimental_memo(show_spinner=False)
def smooth(image_01_maxRGB_reduced, restore_shape, texture_style='I', kernel_shape=(5,1), sharpness=0.001, lamda=0.5, solver='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, return_texture_weights=False):
    #log_memory('smooth||B')  
    ############ TEXTURE MAP  ###########################
    #log_memory('bimef|textures|B')
    gradient_v, gradient_h, texture_weights_v, texture_weights_h = calculate_texture_weights(image_01_maxRGB_reduced, kernel_shape=kernel_shape, sharpness=sharpness, texture_style=texture_style)
    #log_memory('bimef|textures|E')
    ######################################################
    
    ############ ILLUMINATION MAP  ###########################
    #log_memory('bimef|construct_map|B')
    A = construct_map_cyclic(texture_weights_v, texture_weights_h, lamda) 
    #log_memory('bimef|construct_map|E')
    ######################################################
    
    ############ SOLVE SPARSE SYSTEM:  ###########################  # 20220716:  solve_linear_equation replaced by solve_sparse_system
    #log_memory('bimef|solve_sparse_system|B')
    image_01_maxRGB_reduced_smooth = solve_sparse_system(A, image_01_maxRGB_reduced, method=solver, CG_prec=CG_prec, CG_TOL=CG_TOL, LU_TOL=LU_TOL, MAX_ITER=MAX_ITER, FILL=FILL, x0=None)
    #log_memory('bimef|solve_sparse_system|E')
    ######################################################
    
    ############ RESTORE REDUCED SIZE SMOOTH MATRIX TO FULL SIZE:  ###########################
    #log_memory('bimef|imresize|B')
    image_01_maxRGB_smooth = imresize(image_01_maxRGB_reduced_smooth, size=restore_shape)
    #log_memory('bimef|imresize|E')
    ######################################################
    #log_memory('smooth||E')  

    if return_texture_weights:
        texture_weights_v = imresize(texture_weights_v, size=restore_shape)
        texture_weights_h = imresize(texture_weights_h, size=restore_shape)
        gradient_v = imresize(gradient_v, size=restore_shape)
        gradient_h = imresize(gradient_h, size=restore_shape)
        return gradient_v, gradient_h, texture_weights_v, texture_weights_h, image_01_maxRGB_smooth

    return image_01_maxRGB_smooth 
