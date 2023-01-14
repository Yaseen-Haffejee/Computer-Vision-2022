import numpy as np
from sklearn.model_selection import train_test_split
from natsort import natsorted
import imageio
from glob import glob
from skimage import img_as_float32 as as_float

def load_data(path_to_images,path_to_masks):
    '''
    Input:
      path_to_images: The path to the folder with the rgb images
      path_to_masks: The path to the folder with the binary masks
    Output:
        images: a list containing all the images
        masks: a list containing all the masks
    '''
    ## This code was given
    path_pairs = list(zip(
    natsorted(glob(path_to_images+ '*.png')),
    natsorted(glob(path_to_masks+'*.png')),
    ))

    images = np.array([as_float(imageio.imread(ipath)) for ipath, _ in path_pairs])
    masks = np.array([as_float(imageio.imread(mpath)) for _, mpath in path_pairs])
    return images,masks

def split_data(X,Y,Training_split,Validation_split,Test_split):
    '''
    Input:
      X: The rgb images
      Y: The binary masks
      Training_split: size of the training data. Decimal between 0-1
      Validation_split: size of the validation data. Decimal between 0-1
      Test_split: size of the test data. Decimal between 0-1
      NB: Training_split + Validation_split + Test_split = 1
    Output:
       The data split according to the stipulated numbers all with their own array
    '''
    size = len(X)
    num_train_samples = int(np.round(size*Training_split))
    train_x,temp_x,train_y,temp_y = train_test_split(X,Y,train_size = num_train_samples,random_state=1)
    temp_size = Validation_split + Test_split
    test_size = Test_split/temp_size
    validation_x,test_x,validation_y,test_y = train_test_split(temp_x,temp_y,test_size=test_size,random_state=42)

    return train_x,train_y,validation_x,validation_y,test_x,test_y