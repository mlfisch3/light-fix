import numpy as np
from matplotlib import image as img
from matplotlib import pyplot as plt


def imshow(array, autoscale=True, show_grid=False, show_axes=False):
    # should probably rewrite this function 

    def autoscale_image(array):

        lo = array.flatten().min()
        array -= lo    
        hi = array.flatten().max()
        try:
            assert hi > 0, f'autoscale_image cannot map null array to interval [0,1]'
        except AssertionError as msg:
            print(msg)
            return array

        array = array/hi  # "true divide" `array/=hi` causes error (broadcasting issue?) ["TypeError: No loop matching the specified signature and casting was found for ufunc true_divide"]
 
        return array

    if array.ndim==3:
        if array.std(axis=2).flatten().sum() == 0:
            array = array[:,:,0]
        else:
            if autoscale:
                plt.imshow(autoscale_image(array))
            else:
                plt.imshow(array)
    if array.ndim==2:
        if autoscale:
            a = autoscale_image(array)
            plt.imshow(a, cmap='gray', vmin=0, vmax=1)
        else:
            if (array.dtype == np.float32) | (array.dtype == np.float64):
                plt.imshow(array, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(array, cmap='gray', vmin=0, vmax=255)


def imread(filename):
    frame = img.imread(filename)
    # if frame.ndim == 3:
    #     #frame = frame[:,:,:2]
    #     if frame.std(axis=2).sum()==0.:
    #         frame = frame[:,:,0]

    # print(frame.shape)

    # return np.uint8(frame * 255 / frame.max())
    return frame



def show_image_array(figs, num_cols=-1, size=20, titles=None, autoscale=True, show_grid=False, show_axes=False, tight_layout=True):
  ''' 20220709: functionality added: improved automatic spacing 
    Choose overall figure size and number of image columns.
    Automatically formats layout to maximize size of individual images (i.e., minimize gaps)

    tight_layout     (bool, Default=False) [Optional]: Further reduce spacing between images

  '''
  if type(figs) != type([]):
      figs = [figs]         # if figs is/contains only one image, not necessary to pass as list (i.e., 'figs' can be either single array OR list of array(s))
  
  if num_cols == -1:
      num_cols = min(4, len(figs))

  M = num_cols
  
  height = figs[0].shape[0]
  width = figs[0].shape[1]

  #R = round(len(figs)/M + 0.5) # how many rows are needed, in order to have no more than M images per row
  N = round(M * width / (1.2 * height) + 0.5) # given M & individual image dimensions, N is the number of rows corresponding to filled cells. 
  # if N < R, split figures into groups of at most N*M each.
  B = int(len(figs)/(N*M)) # number of groups
  x = len(figs)/(N*M) - B
  if x > 0:
    B += 1
  
  if titles is None:
      for k in range(B):
        fig = plt.figure(figsize=(size,size))
        for j, figure in enumerate(figs[k*M*N:min((k+1)*M*N, len(figs))]):        
            plt.subplot(N,M,j+1)
            imshow(figure, autoscale=autoscale)
            if not show_axes:
                plt.axis("off")
            if not show_grid:
                plt.grid(visible=False)

        if tight_layout:  
          fig.tight_layout()
          
        plt.show()
  else:
      for k in range(B):
        fig = plt.figure(figsize=(size,size))
        for j, (figure, title) in enumerate(zip(figs[k*M*N:min((k+1)*M*N, len(figs))], titles[k*M*N:min((k+1)*M*N, len(figs))])):        
            plt.subplot(N,M,j+1)
            imshow(figure, autoscale=autoscale)
            if not show_axes:
                plt.axis("off")
            if not show_grid:
                plt.grid(visible=False)

            if titles is not None:
                plt.title(title)
            
        if tight_layout:  
          fig.tight_layout()
          
        plt.show()