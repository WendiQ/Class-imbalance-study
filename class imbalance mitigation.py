# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:05:21 2018

@author: MedImaging7271
"""

import numpy as np
from numpy import random as random
import scipy as sp
import csv
import os
import glob
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import sys
import cv2
import skimage
import random
from random import *
from sklearn.utils.class_weight import compute_sample_weight

# Provide randomly sampled images for training and testing  
def sample_images(n_total, proportions, image_size=(256,256)):
    root_dir = 'C:\\Users\\MedImaging7271\\Desktop\\IRMA'
    os.chdir(root_dir)
    
    balance = 30 # Number of images in each class in testing set
    sub_dir = ['\\train', '\\validation']
    counts = {} #
    # create a named list "counts" with "train" as the number of training images for each class
    # eg. [50, 50, 50, 0, 0, 0]
    counts['\\train'] = np.round(np.array(int(n_total)*proportions)) 
    


# Validation set has number of images for all non-zero proportions set to fixed number "balanced"    
    validation_numbers = []
    for element in proportions:
        if element == 0:
            validation_numbers.append(0)
        else:
            validation_numbers.append(balance)
    # create a named list "count" with "validation" as the number of testing 
    # images for each class
    counts['\\validation'] = np.array(validation_numbers)
    
    # list of classes(labels) of images
    im_train_lab = []
    im_test_lab = []
    
    # list of images in pixel form
    im_train_pixel = []
    im_test_pixel = []
    
    for a in sub_dir:
        # select only classes being used in the sampling
        for cat in np.arange(1, 58)[proportions != 0]:
            # put images of different classes into their folders
            cat_dir = root_dir + a + '\\' + str(cat)
            os.chdir(cat_dir)
            # randomly select images (of the correct #) in the training/testing set
            for im_dir in np.random.permutation(os.listdir(cat_dir))[:int(counts[a][cat-1])]:
                im = Image.open(im_dir)
                im = im.convert('L')     #Convert all images to greyscale (some are RGB or RGBA)
                im = im.resize(image_size)  #Resize all images to standard sizing
                # get pixel values for each image
                #im_pixel = np.array(im.getdata(), dtype='int16')/255
                im_pixel = np.array(im)
                             
                
                if a == '\\train':
                    # append labels ordered as the training image order to 'im_train_lab'
                    im_train_lab.append(cat)
                    # append training images in their pixel form to 'im_train_og'
                    im_train_pixel.append(im_pixel)
                else:
                    # create a list of labels ordered as the testing image order
                    im_pixel = np.hstack(im_pixel)/255
                    im_test_lab.append(cat)
                    im_test_pixel.append(im_pixel)
                    
    sampled_image_pixel = [im_train_pixel, im_test_pixel]
    sampled_image_label = [im_train_lab, im_test_lab]
    return([sampled_image_label, sampled_image_pixel])

# set element = 0 if extracting training and element = 1 if test 
# shuffle images and corresponding labels from selected set                            
def shuffle(element, image_set):
    # select out either training or testing set from pixel image list
    set = image_set[1][element]
    # select out label for training/testing set
    labels = image_set[0][element]
    
    # shuffle the images and update the shuffled label sequence
    shuffle_indices = np.random.permutation(len(labels))                       
    set = list(np.array(set)[shuffle_indices])                       
    labels = list(np.array(labels)[shuffle_indices])                   
    
    return([labels, set])                         
                             
def flip_lr(image_list_pixel, labels, group):
    # 'image_list_og' is a list of images extracted from the sample set
    # 'labels' is the corresponding labels for the images
    # 'group' is the class to be modified
    image_flip_lr_im = []
    image_flip_lr_lab = []
    image_flip_lr_pixel = []
    for image in np.array(image_list_pixel)[np.array(labels)==group]:
        image = Image.fromarray(image) # convert image back from array to image form
        image_flip_lr_im.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        image_flip_lr_lab.append(group)
    for image in image_flip_lr_im:
        image = np.array(image)
        image_flip_lr_pixel.append(image)
    return([image_flip_lr_pixel, image_flip_lr_lab])

def flip_ud(image_list_pixel, labels, group):
    # 'image_list_og' is a list of images extracted from the sample set
    # 'labels' is the corresponding labels for the images
    # 'group' is the class to be modified
    image_flip_ud_im = []
    image_flip_ud_lab = []
    image_flip_ud_pixel = []
    for image in np.array(image_list_pixel)[np.array(labels)==group]:
        image = Image.fromarray(image) # convert image back from array to image form
        image_flip_ud_im.append(image.transpose(Image.FLIP_TOP_BOTTOM))
        image_flip_ud_lab.append(group)
    for image in image_flip_ud_im:
        image = np.array(image)
        image_flip_ud_pixel.append(image)
    return([image_flip_ud_pixel, image_flip_ud_lab])

def rotate(image_list_pixel, labels, group):
    image_rotate_im = []
    image_rotate_lab = []
    image_rotate_pixel = []
    for image in np.array(image_list_pixel)[np.array(labels)==group]:
        image = Image.fromarray(image)
        image_rotate_im.append(image.rotate(90))
        image_rotate_lab.append(group)
        image_rotate_im.append(image.rotate(270))
        image_rotate_lab.append(group)
    for image in image_rotate_im:
        image = np.array(image)
        image_rotate_pixel.append(image)
    shuffle_indices = np.random.permutation(len(image_rotate_lab))                       
    image_rotate_pixel = list(np.array(image_rotate_pixel)[shuffle_indices])
    return([image_rotate_pixel, image_rotate_lab])

def gaussian_filter(image_list_pixel, labels, group):
    image_gau_lab = []
    image_gau_pixel = []
    for image in np.array(image_list_pixel)[np.array(labels)==group]:
        image_gau_pixel.append(skimage.util.random_noise(image, mode='gaussian'))
        image_gau_lab.append(group)
    image_gau_pixel = list(np.array(image_gau_pixel)*255)
    return([image_gau_pixel, image_gau_lab])

def under_sample(image_list_pixel, labels, imb_group, change_imb):
    # Undersample the imbalanced class if change_imb = 1;
    # Undersample the other classes if change_img = 0
    
    imb_count = labels.count(imb_group)
    norm_group = np.array(labels)[np.array(labels) != imb_group][0] # find the number of images in a normal group
    num_norm_group = labels.count(norm_group)
    final_im = []
    final_lab = []
    lab_num = [2,3,4,7,12] # labels of 5 classes
    
    if imb_count == num_norm_group:
       for im in image_list_pixel:
           im = np.hstack(im)/255
           final_im.append(im) 
       return([final_im, labels])
              
    if change_imb == 0:       
        for lab in lab_num:
            lab_img = np.array(image_list_pixel)[np.array(labels)==lab] # extract images with the label 'lab'
            if lab != imb_group:
                # randomly select 'imb_count' # of images from normal groups
                ran_ind = sample(range(num_norm_group), imb_count)
                img_selected = lab_img[np.array(ran_ind)]
                
                for im in img_selected:
                    im = np.array(np.hstack(im)/255)
                    final_im.append(im)
                    final_lab.append(lab)
                    
            if lab == imb_group:
                # for the imbalanced(minority) group, simply attach the original image & label
                for im in lab_img:
                    im = np.hstack(im)/255
                    final_im.append(im) 
                    final_lab.append(lab)
                
    if change_imb == 1:
        for lab in lab_num:
            lab_img = np.array(image_list_pixel)[np.array(labels)==lab] # extract images with the label 'lab'
            if lab == imb_group:
                # randomly select 'norm_count' # of images from imbalanced class
                ran_ind = sample(range(imb_count), num_norm_group)
                img_selected = lab_img[np.array(ran_ind)]
                for im in img_selected:
                    im = np.hstack(im)/255
                    final_im.append(im)
                    final_lab.append(lab) 
            if lab != imb_group:
                # for the normal groups, simply attach the original image & label
                for im in lab_img:
                    im = np.hstack(im)/255
                    final_im.append(im) 
                    final_lab.append(lab)
                
    # shuffle image set
    shuffle_indices = np.random.permutation(len(final_lab))                       
    sampled_image_pixel = list(np.array(final_im)[shuffle_indices])                         
    labels = list(np.array(final_lab)[shuffle_indices])
    
    return([sampled_image_pixel, labels])
                                    
                                    
def combine(smp_img_pix, modified_image, lbls):
    # combine sample set (pixel) with modified set
    # make sure number of minority class images equal to other classes by 
    # adding the modified images (duplicates allowed) till the number is reached
    # sampled_image_pixel is the pixel image list from 'sample_images' function 
        # i.e. sample_images()[2][0]
    # 'modified_image' is a image set that underwent modification (eg. 'image_flip_lr')
    # 'labels' is the labels in the sample set
    # 'group' is the label for minority group (i.e. class number)                           
                             
    # extract & shuffle image & labels from modified_image
    sampled_image_pixel = smp_img_pix.copy()
    labels = lbls.copy()
    mod_im = np.random.permutation(modified_image[0])  # images
    mod_lab = modified_image[1]  # labels
    group = mod_lab[0] # pick out the minority class label
    combined_image = []
                   
    num_min_group = labels.count(group)
    norm_group = np.array(labels)[np.array(labels) != group][0] # find the number of images in a normal group
    num_norm_group = labels.count(norm_group)
    
    if num_min_group != num_norm_group:
        for im in mod_im:    
            if num_min_group < num_norm_group:
                sampled_image_pixel.append(im)
                labels.append(group)
                num_min_group += 1
        
        indexes = list(range(len(mod_im)))     
        
        while num_min_group < num_norm_group:
            index = np.random.choice(indexes) # randomly choose index of image to be added 
            sampled_image_pixel.append(mod_im[index])
            labels.append(group)
            num_min_group += 1
         
        # shuffle combined set
        shuffle_indices = np.random.permutation(len(labels))                       
        sampled_image_pixel = np.array(sampled_image_pixel)[shuffle_indices]                         
        labels = list(np.array(labels)[shuffle_indices])
    
    for im in sampled_image_pixel:
        im = np.hstack(im)/255
        combined_image.append(im)
    combined_image = list(combined_image)
    
    return([combined_image, labels])                                   
                             
def simulator(proportions, runs, n, method, class_index):
    
    # modified list is the output of any modification function (eg. flip_lr)
    runs_holder_acc_over = []
    runs_holder_f1_over = []
    
    runs_holder_acc_og = np.array([])
    runs_holder_f1_og = np.array([])
    
    runs_holder_acc_w = np.array([])
    runs_holder_f1_w = np.array([])
    
    runs_holder_acc_under = np.array([])
    runs_holder_f1_under = np.array([])
    
    runs_holder_imb = []
    
    f1_score_imb_og = []
    f1_score_imb_w = []
    f1_score_imb_under = []
    
    acc_list_og3 = []
    acc_list_w3 = []   
    acc_list_under3 = []
    runs_holder_imb_acc = []
    
    
    for i in range (runs):
        im_sample = sample_images(n, proportions, image_size=(256,256))
        im_train_sample = shuffle(0, im_sample)
        im_test_sample = shuffle(1, im_sample)
        
        
        # Train using original data w/ unweighted RF
        X_train = []
        for im in im_train_sample[1]:
            im = np.hstack(im)/255
            X_train.append(im)
        Y_train = im_train_sample[0] # extract training labels
        X_test = im_test_sample[1]
        Y_test = im_test_sample[0]
        model_RF_og = RandomForestClassifier(n_estimators = 15, max_depth = None, n_jobs=-1, random_state=123)
        model_RF_og.fit(X_train, Y_train)
        Y_Pred_og = model_RF_og.predict(X_test)
        CM_og = np.matrix(confusion_matrix(Y_test, Y_Pred_og))
        
        single_rec_og = CM[class_index,class_index]/sum(CM[class_index])  # recall score for imb class 
        single_prec_og = CM[class_index,class_index]/sum(CM[:,class_index]) # precision score for imb class 
        single_spec_og = (sum(CM)-sum(CM[class_index])-sum(CM[:,class_index])+ \
                       CM[class_index,class_index])/(sum(CM)-sum(CM[class_index]))
#        if np.isnan(single_prec):
#            precision_nan += 1
        overall_score_og = metrics.accuracy_score(Y_test, Y_Pred) # Overall accuracy score
        f1_og = metrics.f1_score(Y_test, Y_Pred, average='macro') # Overall f1 score
        
        # Train using weighted RF
#        f1_score_og = np.array([])
#        score_og = np.array([])
#        f1_score_w = np.array([])
#        score_w = np.array([])
        
        overall_acc.append(overall_score)
        single_precision.append(single_prec)
        single_recall.append(single_rec)
        overall_f1.append(f1)
        single_specificity.append(single_spec)
        
        model_RF_w = RandomForestClassifier(n_estimators = 15, max_depth = None, n_jobs=-1, class_weight='balanced', random_state=123)
        model_RF_w.fit(X_train, Y_train)
        Y_Pred_w = model_RF_w.predict(X_test)
        CM_w = np.matrix(confusion_matrix(Y_test, Y_Pred_w))
        
        # Score for unweighted & weighted RF
        for c in range(5):
            precision_og = CM_og[c,c]/sum(CM_og[:,c])
            recall_og = CM_og[c,c]/sum(CM_og[c])
            precision_w = CM_w[c,c]/sum(CM_w[:,c])
            recall_w = CM_w[c,c]/sum(CM_w[c])
            f1_score_og=np.append(f1_score_og,[2*((precision_og*recall_og)/(precision_og+recall_og))])
            score_og=np.append(score_og,[CM_og[c,c]/sum(CM_og[c])])
            f1_score_w=np.append(f1_score_w,[2*((precision_w*recall_w)/(precision_w+recall_w))])
            score_w=np.append(score_w,[CM_w[c,c]/sum(CM_w[c])])
            
        f1_score_imb_og.append(f1_score_og[1])
        f1_score_imb_w.append(f1_score_w[1])
        
        
        runs_holder_f1_og=np.append(runs_holder_f1_og,f1_score_og)
        runs_holder_acc_og=np.append(runs_holder_acc_og,score_og) 
        runs_holder_f1_w=np.append(runs_holder_f1_w,f1_score_w)
        runs_holder_acc_w=np.append(runs_holder_acc_w,score_w)
        acc_list_og3.append([CM_og[1,1]/sum(CM_og[1])])
        acc_list_w3.append([CM_w[1,1]/sum(CM_w[1])])
                
        # Train using undersampling if class 3 is majority
        if proportions[2] >= 0.2:
            train_under = under_sample(im_train_sample[1], im_train_sample[0], 3, 1)
            X_train_under = train_under[0]
            Y_train_under = train_under[1]
            model_RF_under = RandomForestClassifier(n_estimators = 15, max_depth = None, n_jobs=-1, random_state=123)
            model_RF_under.fit(X_train_under, Y_train_under)
            Y_Pred_under = model_RF_under.predict(X_test)
            CM_under = np.matrix(confusion_matrix(Y_test, Y_Pred_under))
            
            # calculate score
            f1_score_under = np.array([])
            score_under = np.array([])

            for c in range(5):
                precision_under = CM_under[c,c]/sum(CM_under[:,c])
                recall_under = CM_under[c,c]/sum(CM_under[c])
                f1_score_under = np.append(f1_score_under,[2*((precision_under*recall_under)/(precision_under+recall_under))])
                score_under = np.append(score_under,[CM_under[c,c]/sum(CM_under[c])])
            f1_score_imb_under.append(f1_score_under[1])
            acc_list_under3.append(score_under[1])
          
            runs_holder_f1_under = np.append(runs_holder_f1_under,f1_score_under)
            runs_holder_acc_under = np.append(runs_holder_acc_under,score_under)
                    
        
        # Train using oversampled data if the class is minority
        if proportions[2] <= 0.2:
            if method == 'fliplr':
                mod_list = [flip_lr(im_train_sample[1], im_train_sample[0], 3)]
            if method == 'flipud':
                mod_list = [flip_ud(im_train_sample[1], im_train_sample[0], 3)]
            if method == 'rotate':
                mod_list = [rotate(im_train_sample[1], im_train_sample[0], 3)]
            if method == 'gaussian':
                mod_list = [gaussian_filter(im_train_sample[1], im_train_sample[0], 3)]
            if method == 'all':
                mod_list = [flip_lr(im_train_sample[1], im_train_sample[0], 3),
                            flip_ud(im_train_sample[1], im_train_sample[0], 3), 
                            rotate(im_train_sample[1], im_train_sample[0], 3),
                             gaussian_filter(im_train_sample[1], im_train_sample[0], 3)]
            
            
            f1_over_all = []
            acc_over_all = []
            f1_score_imb_over_all = []
            acc_score_imb_over = []
            
            for modified_list in mod_list:
                train_over = combine(im_train_sample[1], modified_list, im_train_sample[0]) # combine original and modified image sets
                #print(train_over[1])
                X_train_over = train_over[0] # extract training images
                #im = Image.fromarray(np.array(X_train_over[0]).reshape((256,256))*255)
                #im.show()
                Y_train_over = train_over[1] # extract training labels
                X_test_over = im_test_sample[1]
                Y_test_over = im_test_sample[0]
                model_RF_over = RandomForestClassifier(n_estimators = 15, max_depth = None, n_jobs=-1, random_state=123)
                model_RF_over.fit(X_train_over, Y_train_over)
                Y_Pred_over = model_RF_over.predict(X_test_over)
                CM_over = np.matrix(confusion_matrix(Y_test_over, Y_Pred_over))  
    
                
                f1_score = []
                score = []            
                for c in range(5):
                    precision = CM_over[c,c]/sum(CM_over[:,c])
                    recall = CM_over[c,c]/sum(CM_over[c])
                    f1_score.append(2*((precision*recall)/(precision+recall)))
                    score.append(CM_over[c,c]/sum(CM_over[c]))
                
                f1_score_imb_over_all.append(f1_score[1])
                f1_over_all.append(np.mean(np.array(f1_score)))
                acc_over_all.append(np.mean(np.array(score)))
                acc_score_imb_over.append(score[1])
                
            runs_holder_f1_over.append(f1_over_all)
            #print(runs_holder_f1_over)
            #print(np.mean(runs_holder_f1_over))
            runs_holder_acc_over.append(acc_over_all)
            runs_holder_imb.append(f1_score_imb_over_all)
            runs_holder_imb_acc.append(acc_score_imb_over)
    
    print({'original': [np.mean(runs_holder_acc_og), np.mean(runs_holder_f1_og), mean(f1_score_imb_og), mean(acc_list_og3)]})
    print({'weighted': [np.mean(runs_holder_acc_w), np.mean(runs_holder_f1_w), mean(f1_score_imb_w), mean(acc_list_w3)]})
    
    if proportions[2] >= 0.2:
        print({'under sampling': [np.mean(runs_holder_acc_under), np.mean(runs_holder_f1_under), mean(f1_score_imb_under), mean(acc_list_under3)]})
    
    #if proportions[2] <= 0.2 and len(f1_over_all) == 1:
        #print({'over sampling-'+ method: [np.mean(acc_over_all), np.mean(f1_over_all), mean(f1_score_imb_over_all)], mean(acc_score_imb_over)})
    if proportions[2] <= 0.2 and len(f1_over_all) == 4:
        means_f1 = np.mean(np.array(runs_holder_f1_over),axis=0)
        means_acc = np.mean(np.array(runs_holder_acc_over),axis=0)
        means_imb = np.mean(np.array(runs_holder_imb),axis=0)
        means_acc_imb = np.mean(np.array(runs_holder_imb_acc),axis=0)
        print({'over sampling-fliplr': [means_acc[0],means_f1[0], means_imb[0], means_acc_imb[0]]})
        print({'over sampling-flipud': [means_acc[1],means_f1[1],  means_imb[1], means_acc_imb[1]]})
        print({'over sampling-rotate': [means_acc[2],means_f1[2],   means_imb[2], means_acc_imb[2]]})
        print({'over sampling-gaussian': [means_acc[3],means_f1[3],   means_imb[3], means_acc_imb[3]]})

        
    #{'over sampling-fliplr': [np.mean(runs_holder_acc_over), np.mean(runs_holder_f1_over), mean(f1_score_imb_over)]}
        #if proportions[2] == 0.2:
           # return({'under sampling': [np.mean(runs_holder_acc_under), np.mean(runs_holder_f1_under), mean(f1_score_imb_under)]})


def proportion_generator(list_of_lists):
    temp = np.zeros(57)
    for listt in list_of_lists:
        temp[(listt[0] - 1)] = listt[1]
    return temp

#varying class 3
null = proportion_generator([[2,0.2],[3,0.2],[4,0.2],[7,0.2],[12,0.2]])
x2_null = proportion_generator([[2,1/6],[3,1/3],[4,1/6],[7,1/6],[12,1/6]])
x4_null = proportion_generator([[2,1/8],[3,1/2],[4,1/8],[7,1/8],[12,1/8]])
x6_null = proportion_generator([[2,1/10],[3,3/5],[4,1/10],[7,1/10],[12,1/10]])
x8_null = proportion_generator([[2,1/12],[3,2/3],[4,1/12],[7,1/12],[12,1/12]])
x10_null = proportion_generator([[2,1/14],[3,10/14],[4,1/14],[7,1/14],[12,1/14]])


x1by2_null = proportion_generator([[2,2/9],[3,1/9],[4,2/9],[7,2/9],[12,2/9]])
x1by4_null = proportion_generator([[2,4/17],[3,1/17],[4,4/17],[7,4/17],[12,4/17]])
x1by6_null = proportion_generator([[2,6/25],[3,1/25],[4,6/25],[7,6/25],[12,6/25]])
x1by8_null = proportion_generator([[2,8/33],[3,1/33],[4,8/33],[7,8/33],[12,8/33]])
x1by10_null = proportion_generator([[2,10/41],[3,1/41],[4,10/41],[7,10/41],[12,10/41]])

#if __name__ == "__main__":
#    simulator(null, 5, 50, 'all')










             
                             
                             
                            