
import numpy as np
import math
import matplotlib.pyplot as plt 
import seaborn as sns
import cv2 as cv
from skimage.feature import local_binary_pattern
from skimage import img_as_float,color
from skimage import exposure
from scipy.stats import multivariate_normal
from sklearn.metrics import precision_score,recall_score


def normalize_image(img):
    img = (img-np.mean(img))/np.sqrt(np.var(img))
    return img

def equalize_img(img):
    shape = np.shape(img)
    
    if len(shape) == 3:
        r = exposure.rescale_intensity(img[:,:,0])
        g = exposure.rescale_intensity(img[:,:,1])        
        b = exposure.rescale_intensity(img[:,:,2])
        result = np.stack((r,g,b),axis=-1)
    else:
        result = exposure.equalize_hist()
    
    return result

def Gaussian_Filter(shape,sigma):
    r = np.floor(shape[0]/2)
    c = np.floor(shape[1]/2)
    g = lambda x,y : (1/(2*np.pi*(sigma**2)))*np.exp(-((x-r)**2 + (y-c)**2)/(2*(sigma**2)))
    kernel = np.fromfunction(np.vectorize(g),shape)
    return kernel

def LoG_Filter(shape,sigma):
    r = np.floor(shape[0]/2)
    c = np.floor(shape[1]/2)
    g = lambda x,y : (-1/(np.pi*(sigma**4))) * (1 - (((x-r)**2 + (y-c)**2)/(2*(sigma**2)))) * np.exp(-((x-r)**2 + (y-c)**2)/(2*(sigma**2)))
    kernel = np.fromfunction(np.vectorize(g),shape)
    return kernel

def DoG_Filter(shape,sigma,K=1):
    r = np.floor(shape[0]/2)
    c = np.floor(shape[1]/2)
    g = lambda x,y : (1/(2*np.pi*(sigma**2)))*np.exp(-((x-r)**2 + (y-c)**2)/(2*(sigma**2))) - (1/(2*np.pi*(sigma**2)*K**2))*np.exp(-((x-r)**2 + (y-c)**2)/(2*(sigma**2)*(K**2)))
    kernel = np.fromfunction(np.vectorize(g),shape)
    return kernel


HORIZONTAL_PREWITT = np.fliplr(np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
]))
VERTICAL_PREWITT = np.flipud(np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
]))
LAPLACIAN = np.array([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]
])

def create_2d_Gaussian(theta,sigma_x,sigma_y,size,type):
    
    f = lambda x,sigma: (1/(np.sqrt(2*np.pi*sigma)))*np.exp(-(x**2)/(2*(sigma**2)))
    x_prime =  lambda x,y: x*math.cos(theta) -  y*math.sin(theta)
    y_prime =  lambda x,y: x*math.sin(theta) +  y*math.cos(theta)
    center = size//2
    if type == "edge":
        ## we subtract center from x and y so that the (0,0) occurs in the center of the filter
        g = lambda x,y: f(x_prime(x-center,y-center),sigma_x)* f(y_prime(x-center,y-center),sigma_y)*(-y_prime(x-center,y-center)/(sigma_y**2))
        ## We vectorize g since we want g to be applied over each element in the array
        edge_filter = np.fromfunction(np.vectorize(g),(size,size))
        return edge_filter
    elif type =="bar":
        g = lambda x,y: f(x_prime(x-center,y-center),sigma_x)* f(y_prime(x-center,y-center),sigma_y)*(((y_prime(x-center,y-center)**2)-(sigma_y**2))/(sigma_y**2))
        bar_filter = np.fromfunction(np.vectorize(g),(size,size))
        return bar_filter

variances = [[3,1],[6,2],[12,4]]
thetas = [0, (np.pi/6),(np.pi/3),(np.pi/2),((2*np.pi)/3),((5*np.pi)/6)]
size = 49
def RFS_set(variances = variances,thetas = thetas,size = size):
    edge_filters = []
    bar_filters = []
    for variance in variances:
        
        for theta in thetas:
            edge = create_2d_Gaussian(theta,variance[0],variance[1],type="edge",size = size)
            bar = create_2d_Gaussian(theta,variance[0],variance[1],type="bar",size = size)
            edge_filters.append(np.rot90(edge))
            bar_filters.append(np.rot90(bar))

    gaussians = [Gaussian_Filter((size,size),np.sqrt(10)),LoG_Filter((size,size),math.sqrt(10))]
    return edge_filters,bar_filters,gaussians


def get_max_response(matrices,image):

    n = len(matrices)
    convolved_R = [cv.filter2D(image[:,:,0],-1,f) for f in matrices]
    convolved_G = [cv.filter2D(image[:,:,1],-1,f) for f in matrices]
    convolved_B = [cv.filter2D(image[:,:,2],-1,f) for f in matrices]
    t = [convolved_R,convolved_G,convolved_B]
    results = []

    for j in range(3):
        result = t[j][0]
        # result = convolved[0]
        for i in range(n-1):
            ## Compare the rsult with the next matrix
            ## This will return 2d array of True and false values indicating whether or not the value is greater than the value in the other matrix
            next_mat = t[j][i+1]
            # next_mat = convolved[i+1]
            tmp = np.greater_equal(result,next_mat)
            ## The false indices will correlate to indices that have a larger value than result at that index
            idx = np.where(tmp == False)
            ## SO we replace the false indices with the higher values
            result[idx] = next_mat[idx]
        results.append(result)

    final = np.stack((results[0],results[1],results[2]),axis=-1)
    return final

def MR8_set(image,variances = variances,thetas = thetas,size = size):

    rfs = RFS_set(variances,thetas,size)
    edge_results = []
    bar_results = []
    for i in range(0,18,6):
        r1 = get_max_response(rfs[0][i:i+6],image)
        edge_results.append(r1)
    
    for i in range(0,18,6):
        r1 = get_max_response(rfs[1][i:i+6],image)
        bar_results.append(r1)

    gaus = [cv.filter2D(image,-1,f) for f in rfs[2]]
    return edge_results,bar_results,gaus


def integral_image(img):
    r,c = np.shape(img)
    integral =  np.zeros((r,c))
    integral[0,0] = img[0,0]
    # computing the first row of the integral image
    for i in range(1,c):
        integral[0,i] = np.sum(img[0,0:i+1])
        
    # computing the first column of the integral image
    for i in range(1,r):
        integral[i,0] = np.sum(img[0:i+1,0])
    
    for i in range(1,r):
        for j in range(1,c):
            A = integral[i-1,j-1]
            B = integral[i-1,j]
            C = integral[i,j-1]
            D = img[i,j]
            integral[i,j] = C+B-A+D
            
    return integral


def haar_filter(size):
    kernel = np.zeros((size,size))
    
    s = size//2
    
    for i in range(size):
        a = [1,-1]*s
        if(i%2 == 0):
            kernel[i] = a
        else:
            kernel[i] = -1*np.array(a)
    
    return kernel



def add_img_to_features(img,features,background = None):
    r = np.shape(img)
    img = normalize_image(img)
    if(len(r)==3 and background != None):
        if(type(features) == list):
            features.append(img[background][:,0].flatten())
            features.append(img[background][:,1].flatten())    
            features.append(img[background][:,2].flatten())
        else:
            features  = np.append(features,img[background][:,0].flatten().reshape(1,-1),axis=0)
            features  = np.append(features,img[background][:,1].flatten().reshape(1,-1),axis=0)
            features  = np.append(features,img[background][:,2].flatten().reshape(1,-1),axis=0)
            
    elif(len(r) == 3 and background == None):
        if(type(features) == list):
            features.append(img[:,:,0].flatten())
            features.append(img[:,:,1].flatten())
            features.append(img[:,:,2].flatten())
        else:
            features  = np.append(features,img[:,:,0].flatten().reshape(1,-1),axis=0)
            features  = np.append(features,img[:,:,1].flatten().reshape(1,-1),axis=0)
            features  = np.append(features,img[:,:,2].flatten().reshape(1,-1),axis=0)

    elif(len(r)==2 and background!= None):
        if(type(features) == list):
            features.append(img[background].flatten())
        else:
            features  = np.append(features,img[background].flatten().reshape(1,-1),axis=0)
    else:
        if(type(features) == list):
            features.append(img.flatten())
        else:
            features  = np.append(features,img.flatten().reshape(1,-1),axis=0)
    return features

        
def create_feature_vector(image,background= None):
    
    img_hsv = color.rgb2hsv(image)
    if(background != None ):
        features = [image[background][:,0].flatten(),image[background][:,1].flatten(),image[background][:,2].flatten(),img_hsv[background][:,0].flatten(),img_hsv[background][:,1].flatten(),img_hsv[background][:,2].flatten()]

    else:
        features = [image[:,:,0].flatten(),image[:,:,1].flatten(),image[:,:,2].flatten(),img_hsv[:,:,0].flatten(),img_hsv[:,:,1].flatten(),img_hsv[:,:,2].flatten()]
        
        

    # Applying the gaussian filters from section 1
    r = cv.filter2D(image,-1,Gaussian_Filter((49,49),np.sqrt(10)))
    add_img_to_features(r,features,background)
    add_img_to_features(cv.filter2D(image,-1,LAPLACIAN),features,background)
    add_img_to_features(cv.filter2D(image,-1,HORIZONTAL_PREWITT),features,background)
    add_img_to_features(cv.filter2D(image,-1,VERTICAL_PREWITT),features,background)

    #############################################################################################################################################
    # Applying filters from section 2
    ## RFS filter bank
    rfs_filter_bank = RFS_set()
    for f in rfs_filter_bank:
        for filter in f:
            add_img_to_features(cv.filter2D(image,-1,filter),features,background)

    MR8_filter_bank = MR8_set(image)
    for f in MR8_filter_bank:
        for response in f:
            add_img_to_features(response,features,background)

    #############################################################################################################################################
    # Applying section 3 filters
    ## Applying the LBP features
    radius = [4,8,16,24,32]
    for i in range(len(radius)):
        r = radius[i]
        add_img_to_features(local_binary_pattern(image[:,:,0], 12, r),features,background)
        add_img_to_features(local_binary_pattern(image[:,:,1], 12, r),features,background)
        add_img_to_features(local_binary_pattern(image[:,:,2], 12, r),features,background)

    ## Adding the integral images
    r =  integral_image(image[:,:,0])
    g =  integral_image(image[:,:,1])
    b =  integral_image(image[:,:,2])

    add_img_to_features(r,features,background)
    add_img_to_features(g,features,background)
    add_img_to_features(b,features,background)

    # ## Adding Haar filter results
    sizes = [4,8,16]
    for s in sizes:
        f = haar_filter(s)
        r_16 = cv.filter2D(src = r,ddepth = -1, kernel = f)
        g_16 = cv.filter2D(src = g,ddepth = -1, kernel = f)
        b_16 = cv.filter2D(src = b,ddepth = -1, kernel = f)
        add_img_to_features(r_16,features,background)
        add_img_to_features(g_16,features,background)
        add_img_to_features(b_16,features,background)
    features = np.array(features)
    return features



def cluster_image(features,shape):
    k = 4
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    c,labels,centers = cv.kmeans(np.float32(features.T),k,None,criteria,10, flags = cv.KMEANS_PP_CENTERS )
    print("CLUSTERING COMPLETED !")
    return labels.reshape(shape)


def get_threshold_value(training_pdf_values):
    return np.min(training_pdf_values[np.nonzero(training_pdf_values)])


def evaluate_model(threshold_value,pdf_values,img_shape ):
    background_probabilities = pdf_values>threshold_value

    # Binary Image Prediction
    ## Find the indices that will be classified as background pixels
    background_pixels_indices = np.where(background_probabilities == True)
    ## Create an array of 1's the same size as the mask and flatten it so we can index it with the background pixels above
    binary_result = np.ones(img_shape).flatten()
    ## Set the indices we found to be background pixels as 0.
    binary_result[background_pixels_indices] = 0
    ## reshape array into 2-d so we can plot it as an image
    binary_result = np.reshape(binary_result,img_shape)
    
    return binary_result


def plot_binary_results(original,mask,model):
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(20,20))

    plt.subplot(1,3,1),plt.imshow(original), plt.title("Original")
    plt.subplot(1,3,2),plt.imshow(mask,cmap="gray"), plt.title("Mask")
    plt.subplot(1,3,3),plt.imshow(model,cmap="gray"), plt.title("Model mask")
    np.vectorize(lambda ax:ax.axis('off'))(ax)
    plt.show()

def accuracy(predicted,mask,type):
    N = mask.flatten().shape[0]
    diff =  np.abs(predicted-mask)
    acc = 1-((1/N)*(np.sum(diff)))
    precision = precision_score(mask.astype(np.int32).flatten(),predicted.astype(np.int32).flatten())
    recall = recall_score(mask.astype(np.int32).flatten(),predicted.astype(np.int32).flatten())
    acc = np.round(acc*100,3)
    precision = np.round(precision*100,3)
    recall = np.round(recall*100,3)
    print(f"The {type} accuracy is: {acc} %")
    print(f"The {type} precision is: {precision} %")
    print(f"The {type} recall is: {recall} %")