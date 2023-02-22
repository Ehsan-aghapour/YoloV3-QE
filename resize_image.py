import numpy as np
import cv2
#from rknn.api import RKNN
import sys
from PIL import Image

from scipy.ndimage import zoom
from skimage.transform import resize
#conda install scikit-image
#import caffe
from keras.preprocessing.image import load_img

precompile=False
PC=True





#caffe_io_resize_image:
def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        #print(f'min and max {im_min},{im_max}')
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            #print(im_std)
            resized_std = resize(im_std, new_dims, order=interp_order, mode='constant')
            #print(im_std)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)




def keras_resize(im, new_dims):
    im=load_img(im,new_dims)
    return im


_dir='/media/ehsan/Data/UvA/ARM-CL/compute_library_alexnet/images/'
from PIL import Image
import os


def resize_PIL(im, new_dims, ppm=False):
    images=[]
    if im.find('/'):
        images=os.listdir(im)
        images=[im+img for img in images if img.endswith(('.jpg', '.png', 'jpeg'))]
        
    else:
        images=[im] 
    
    resized_dir=im+'/resized_images_'+str(new_dims[0])+'/'
    #ppm_dir=_dir+'/ppm_images/'
    os.makedirs(resized_dir,exist_ok=True)
    print(images)
    i=0
    for _im in images:
        
        print(f'reading image {images[i]}')
        image=Image.open(images[i])
        resized_image=image.resize(new_dims)
        
        name=images[i].split('/')[-1]
        if ppm:
            name=name[0:name.find('.')]
            name=name+'.ppm'
        resized_image.save(resized_dir+name)
        i=i+1
