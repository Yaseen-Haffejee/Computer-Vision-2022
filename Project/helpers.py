import numpy as np
from sklearn import cluster
from sklearn.metrics import precision_score,recall_score,confusion_matrix
from skimage.filters import difference_of_gaussians
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.color import rgb2gray
from GMM_Class import *
import time


def get_priors(train_y):
    '''
    Input:
        The masks of the training images
    Output:
        Priors: A foreground and background prior based on the training images

    '''
    foreground_prior = 0
    background_prior = 0
    total_pixels = len(train_y[0].flatten())*len(train_y)
    for img in train_y:
        foreground_prior += len(np.where(img != 0)[0])

    foreground_prior = foreground_prior/total_pixels
    background_prior = 1 - foreground_prior

    return foreground_prior,background_prior

def get_features(train_x,train_y,DOG = False,Texton = False):
    '''
    Input:
        Train_X: RGB images being used for training
        Train_Y: The corresponding training masks
    Output:
        Foreground_x : feature array of all the foreground pixels
        Background_x : feature array of all the background pixels
    '''
    n = len(train_x)
    foreground_x = None
    background_x = None
    for i in range(n):
        image = train_x[i]
        mask = train_y[i]

        fg_indices = np.where(mask != 0)
        bg_indices = np.where(mask == 0)

        tmp_fg = image[fg_indices]
        tmp_bg = image[bg_indices]
        if(DOG):
            dog_response = get_DOG(image)
            tmp_fg =  np.column_stack((tmp_fg,dog_response[fg_indices]))
            tmp_bg =  np.column_stack((tmp_bg,dog_response[bg_indices]))
        if(Texton):
            texton = cluster_image(image)
            tmp_fg =  np.column_stack((tmp_fg,texton[fg_indices]))
            tmp_bg =  np.column_stack((tmp_bg,texton[bg_indices]))

        if(i ==0):
            foreground_x =  np.copy(tmp_fg)
            background_x = np.copy(tmp_bg)
        else:
            foreground_x = np.vstack((foreground_x,tmp_fg))
            background_x = np.vstack((background_x,tmp_bg))

    del tmp_fg,tmp_bg
    return foreground_x,background_x


def get_DOG(X):
    '''
    Input:
       X: The image we would like to compute the Difference of Gaussians for
    Output:
        DOG image
    '''
    X = rgb2gray(X)
    DOG_IMAGE = difference_of_gaussians(X, 2, 30,channel_axis=2)

    return DOG_IMAGE


def cluster_image(X,shape = (768,1024)):
    '''
    Input:
       X: The image we would like to compute the Texton values for
       shape: The output shape required; Usually the shape of the masks
    Output:
        DOG image
    '''
    k = 3
    X = X.reshape(-1,X.shape[-1])
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    c,labels,centers = cv.kmeans(np.float32(X),k,None,criteria,10, flags = cv.KMEANS_PP_CENTERS )
    return labels.reshape(shape)


def get_features_extended(X,DOG=False,Texton=False):
    n = len(X)
    new_features = []
    for i in range(n):
        image = X[i]

        if(DOG):
            r = get_DOG(image)
            image = image.reshape(-1,image.shape[-1])
            tmp = np.column_stack((image,r.flatten()))
            tmp = tmp.reshape((768,1024,4))
            new_features.append(tmp)
        
        if(Texton):
            r = cluster_image(image)
            image = image.reshape(-1,image.shape[-1])
            tmp = np.column_stack((image,r.flatten()))
            tmp = tmp.reshape((768,1024,4))
            new_features.append(tmp)
    
    return new_features

def predict_output(FG_GMM,BG_GMM,FG_PRIOR,BG_PRIOR,data):
    '''
    Input:
        FG_GMM: GMM trained on foreground features
        BG_GMM: GMM trained on background features
        FG_PRIOR: Prior for a foreground pixel
        BG_PRIOR: Prior for a background pixel
        data: The set of data we are doing inference on
    Output:
        results: A list of 2-D arrays which are the predicted masks using the FG and BG GMMs

    '''
    n = len(data)
    results = []
    for i in range(n):
        img = data[i]
        img = img.reshape(-1,img.shape[-1])
        foreground_probs = FG_GMM.predict(img)[1]
        background_probs = BG_GMM.predict(img)[1]
        fg_indices = np.argsort(foreground_probs,axis=1)
        bg_indices = np.argsort(background_probs,axis=1)
        foreground_probs = np.take_along_axis(foreground_probs,fg_indices,axis=1)
        background_probs = np.take_along_axis(background_probs,bg_indices,axis=1)
        foreground_probs = foreground_probs[:,-1]
        background_probs = background_probs[:,-1]

        result = (foreground_probs*FG_PRIOR)/((foreground_probs*FG_PRIOR)+(background_probs*BG_PRIOR))
        result = result.reshape(768,1024)
        result[result>np.mean(result)] = 1
        result[result<=np.mean(result)] = 0
        results.append(result)
    return results

def metrics_eval(predicted_output,masks,type=None):
    '''
    Input:
        predicted_output: List of masks predicted by the GMM's
        masks: The actual/true masks 
        type: String denoting whether it is a validation or test set
    Output:
       Precision, Recall, and Accuracy
       The precision, recall and accuracy is printed

    '''
    n = len(predicted_output)
    # accuracies = []
    # precisions = []
    # recalls = []
    dscs = []
    ious = []
    for i in range(n):
        predicted = predicted_output[i]
        mask = masks[i]
        # N = mask.flatten().shape[0]
        # diff =  np.abs(predicted-mask)
        # acc = 1-((1/N)*(np.sum(diff)))
        # precision = precision_score(mask.astype(np.int32).flatten(),predicted.astype(np.int32).flatten())
        # recall = recall_score(mask.astype(np.int32).flatten(),predicted.astype(np.int32).flatten())
        # acc = np.round(acc*100,3)
        tn, fp, fn, tp = confusion_matrix(mask.astype(np.int32).flatten(),predicted.astype(np.int32).flatten()).ravel()
        DSC = 2*tp/(2*tp+fp+fn)
        IOU = tp/(tp+fp+fn)
        dscs.append(np.round(DSC,3))
        ious.append(np.round(IOU,3))
        # precision = np.round(precision*100,3)
        # recall = np.round(recall*100,3)
        # accuracies.append(acc)
        # precisions.append(precision)
        # recalls.append(recall)
    # avg_acc = np.round(np.mean(accuracies),3)
    # avg_precision = np.round(np.mean(precisions),3)
    # avg_recall = np.round(np.mean(recalls),3)
    avg_iou = np.round(np.mean(ious),3)
    avg_dscs = np.round(np.mean(dscs),3)
    if(type is not None):
        # print(f"The {type} accuracy is: {avg_acc} %")
        # print(f"The {type} precision is: {avg_precision} %")
        # print(f"The {type} recall is: {avg_recall} %")
        print(f"The {type} IOU is: {avg_iou} %")
        print(f"The {type} DICE is: {avg_dscs} %")
    return avg_iou, avg_dscs

def plot(predicted_masks,actual_masks,rgb_images,type,features):
    '''
    Input:
       predicted_output: List of masks predicted by the GMM's
       masks: The actual/true masks
       rgb_images: the images that were classified earlier 
       type: String denoting whether it is a validation or test set
       features: the set of features that were used to train on
    Output:
        plot: shows the RGB image, the actual mask and true mask
    '''
    iterator = zip(rgb_images,actual_masks,predicted_masks)
    fig = plt.figure()
    plt.axis('off')
    plt.title(f"{type} plot using {features} features")
    fig.set_figheight(15)
    fig.set_figwidth(15)
    n = len(predicted_masks)
    count = 1
    for a,b,c in iterator:

        fig.add_subplot(n,3,count)
        plt.axis('off')
        plt.imshow(a)
        plt.title("RGB image")


        fig.add_subplot(n,3,count+1)
        plt.axis('off')
        plt.imshow(b,cmap='gray')
        plt.title("Actual Mask")

        fig.add_subplot(n,3,count+2)
        plt.axis('off')
        plt.imshow(c,cmap='gray')
        plt.title("Predicted Mask")


        count +=3
    plt.tight_layout()
    return

def concat(X):
    res = X[0]
    n = len(X)
    for i in range(1,n):
        tmp = X[i]
        res = np.vstack((res,tmp))
    # res = res.reshape(-1,4)
    return res

def K_Fold_Validation(Train_X,Train_Y,n_components,max_iterations=20,K = 6):
    # Train_X = Train_X.reshape(41,768,1024,4)
    x1 = Train_X[:7]
    x2 = Train_X[7:14]
    x3 = Train_X[14:21]
    x4 = Train_X[21:28]
    x5 = Train_X[28:35]
    x6 = Train_X[35:41]
    x = [x1,x2,x3,x4,x5,x6]
    y1 = Train_Y[:7]
    y2 = Train_Y[7:14]
    y3 = Train_Y[14:21]
    y4 = Train_Y[21:28]
    y5 = Train_Y[28:35]
    y6 = Train_Y[35:41]
    y = [y1,y2,y3,y4,y5,y6]
    final = []
    for iteration in range(K):
        results = {}
        x_test = x[iteration]
        y_test = y[iteration]
        x_test = get_features_extended(x_test,DOG=True)
        x_train = x[iteration+1:] + x[:iteration]
        y_train = y[iteration+1:] + y[:iteration]
        x_train = concat(x_train)
        y_train = concat(y_train)
        fg_prior,bg_prior = get_priors(y_train)
        fg_x,bg_x = get_features(x_train,y_train,DOG=True)
        start = time.time()
        Foreground_GMM_dog = GMM(n_components = n_components,max_iterations=max_iterations,tolerance=1e-15)
        Foreground_GMM_dog.fit(fg_x)


        Background_GMM_dog = GMM(n_components = n_components ,max_iterations=max_iterations,tolerance=1e-15)
        Background_GMM_dog.fit(bg_x)
        end = time.time()
        train_time = (end-start)/60
        results["training_time"] = np.round(train_time,4)

        start = time.time()
        model_output = predict_output(Foreground_GMM_dog,Background_GMM_dog,fg_prior,bg_prior,x_test)
        end = time.time()
        inference_time = (end-start)/60
        results["inference_time"] = np.round(inference_time,4)
        iou,dice = metrics_eval(model_output,y_test,None)
        results['iou'] = iou
        results['dice'] = dice
        final.append(results)


    return final
