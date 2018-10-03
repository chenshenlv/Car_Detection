# -*- coding: utf-8 -*-
"""
File:   hw01.py
Author: 
Date:   
Desc:   
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import math as mat
import matplotlib.image as mpimg
from collections import Counter
from sklearn.model_selection import train_test_split

plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """
data_train=np.load('data_train.npy')
data_train_copy=data_train
ground_truth=np.load('ground_truth.npy')
data_easy_train=data_train[1800:2200,500:1500]  #generate easy train image 
#fig=plt.figure()
#plt.imshow(data_easy_train,cmap="gray")
#plt.savefig('data_easy_train.png') 

#for i in range(ground_truth.shape[0]):  # label red car
#    data_train_copy[ground_truth[i,1]-5:ground_truth[i,1]+5,ground_truth[i,0]-5:ground_truth[i,0]+5]=[203,216,120]
#plt.imsave('data_train_copy.pdf', data_train_copy)

#fig=plt.figure()
#plt.imshow(data_train,cmap="gray")
#plt.scatter(ground_truth[:,0],ground_truth[:,1],marker='.',c='r')
#plt.savefig('data_train_dot.jpg')
ground_truth_pixel_value=[]

for i in range(ground_truth.shape[0]):
    ground_truth_pixel_value.append(data_train[ground_truth[i,1]-3:ground_truth[i,1]+3,ground_truth[i,0]-3:ground_truth[i,0]+3])
    fig=plt.figure()
    plt.imshow(ground_truth_pixel_value[i],cmap="gray")

vehicle=data_train[1356:1366,1762:1772]
fig=plt.figure()
plt.imshow(vehicle,cmap="gray")

#more_redcar ground truth
more_redcar=np.array([[176,2047],[189,2052],[282,1979],[495,2034],[506,2033],[526,2010],[701,2105],[736,2153],[752,2148],[871,2142],[1043,2131],[771,2107],[930,2099],[998,2091],[1018,2090],[779,2061],[784,2042],[797,2057],[809,2066],[885,2068],[899,2048],[960,2046],[1008,2005],[1020,1998],[1022,2012],[1046,1957],[1048,1945],[1069,1938],[1116,1948],[1128,1916],[1118,2046],[1317,1990],[1318,2023],[1318,2015],[1405,2053],[1453,1973],[1430,1944],[786,1908],[812,1913],[817,1898],[832,1886],[850,1897],[861,1910],[877,1960],[4437,5416],[4433,5433],[4486,5421],[4506,5422],[4497,5437],[4491,5445],[4278,5588]])
np.save('more_redcar.npy', more_redcar)
ground_truth[:,0:2].shape
more_redcar.shape
#extend ground truth
ex_ground_truth=np.concatenate((ground_truth[:,0:2],more_redcar))
np.save('ex_ground_truth.npy', ex_ground_truth)
#non red car ground cords:
non_redcar_cords=np.array([[336,560],[730,510],[854,1213],[1025,1724],[1238,1969],[1761,2482],[3554,2667],[5062,1245],[5840,553],[6009,2439],[5224,5938],[5923,5122],[2540,4839],[5627,4856],[5489,5511],[1703,4894],[2019,4243],[1582,4119],[1088,6077],[4381,6081]])
np.save('non_redcar_cords.npy', non_redcar_cords)
non_redcar_pixel_value=[]
for i in range(non_redcar_cords.shape[0]):
    non_redcar_pixel_value.append(data_train[non_redcar_cords[i,1]-3:non_redcar_cords[i,1]+3,non_redcar_cords[i,0]-3:non_redcar_cords[i,0]+3])
    fig=plt.figure()
    plt.imshow(non_redcar_pixel_value[i],cmap="gray")



#pre_data_train=[]
#for row_im in range(3,6246,3):
#    for col_im in range(3,6246,3):
#        sum_value=np.array([0,0,0])
#        for row_kernel in range (-1,1):
#            for col_kernel in range(-1,1):
#                sum_value+=np.int32((data_train[row_im+row_kernel,col_im+col_kernel])/9)
#        pre_data_train.append([row_im,col_im,sum_value])
#        
#print(pre_data_train[624695])
#a=[0,0,0]
#for i in range(3):
#    a+=(np.int32((ground_truth_pixel_value[i])/3))
"""
   This seems like a good place to write a function to learn your regression
   weights!
   
   """
""" ======================  Compute Ground Truth box mean ========================== """       
ground_truth_center_mean=[]
for i in range(ex_ground_truth.shape[0]):
    sum_value=np.array([0,0,0])
    for row in range(-3,3,1):
        for col in range(-3,3,1):
            sum_value+=np.int32(data_train[ex_ground_truth[i,1]+row,ex_ground_truth[i,0]+col])
    ground_truth_center_mean.append(np.around(sum_value/36))
ground_truth_center_mean=np.asarray(ground_truth_center_mean)
ground_truth_center_mean=np.column_stack((ground_truth_center_mean, np.zeros(ground_truth_center_mean.shape[0])))


""" ======================  Compute Nonred car box mean ========================== """       
nonredcar_center_mean=[]
for i in range(non_redcar_cords.shape[0]):
    sum_value=np.array([0,0,0])
    for row in range(-3,3,1):
        for col in range(-3,3,1):
            sum_value+=np.int32(data_train[non_redcar_cords[i,1]+row,non_redcar_cords[i,0]+col])
    nonredcar_center_mean.append(np.around(sum_value/36))
nonredcar_center_mean=np.asarray(nonredcar_center_mean)
nonredcar_center_mean=np.column_stack((nonredcar_center_mean, np.ones(nonredcar_center_mean.shape[0])))    

#np.sqrt(np.sum(np.square(ground_truth_center_mean[5,0:2]-nonredcar_center_mean[8,0:2])))

plt.imshow(ground_truth_center_mean,cmap="gray")
""" ======================  Generate the trainning data set ========================== """   
Train_set=np.append(ground_truth_center_mean,nonredcar_center_mean,axis=0)
labels = Train_set[:,Train_set.shape[1]-1]
Train_set=np.delete(Train_set,Train_set.shape[1]-1,axis = 1)
Classes = np.sort(np.unique(labels))
# Here you can change M to get different validation data
M = 3
X_train, X_valid, label_train, label_valid = train_test_split(Train_set, labels, test_size = 0.8, random_state = M)

"""
===============================================================================
===============================================================================
============================ KNN Classifier ===================================
===============================================================================
===============================================================================
"""

""" Here you can write functions to achieve your KNN classifier. """

def is_odd(n):    #determine if the number is odd
    return n%2==1

def euclidian_dis(data1,data2):    #define distance function
	return np.sqrt(np.sum(np.square(data1-data2)/3))

def KNN_predictor(x_valida,X_train,label_train,K):    #define KNN predictor
	distance=[]
	Kth_target=[]

	for i in range(X_train.shape[0]):
		dis=euclidian_dis(x_valida,X_train[i,:])
		distance.append([dis,i])
	distance=sorted(distance, key=lambda s: s[0])    #sorted distance from small to big
	
	for i in range(K):
		lookup_index=distance[i][1]
		Kth_target.append(label_train[lookup_index])    #return Kth target form the distance list
	return Counter(Kth_target).most_common(1)[0][0]    #return the mostly counted target as the data target

def KNN_Classifier(X_train,label_train,X_valid,K_th_neigbor):    #define Knn classifier to validate a set of data
	prediction=[]
	for i in range(len(X_valid)):
		prediction.append(KNN_predictor(X_valid[i,:],X_train,label_train,K_th_neigbor))
	return prediction	


""" =======================  Preprocessing the image ======================= """
preprocessing_data_cord=[]
preprocessing_data_RGB=[]
for i in range(3,data_easy_train.shape[0]-3,3):
    for j in range(3,data_easy_train.shape[1]-3,3):
        sum_value=np.array([0,0,0])
        for row in range(-3,3,1):
            for col in range(-3,3,1):
                sum_value+=np.int32(data_easy_train[i+row,j+col])
        preprocessing_data_cord.append(i)
        preprocessing_data_cord.append(j)
        preprocessing_data_RGB.append(np.around(sum_value/36))
preprocessing_data_cord=np.reshape(preprocessing_data_cord,(-1,2))
preprocessing_data_RGB=np.reshape(preprocessing_data_RGB,(-1,3))
""" ========================  Train the Model ============================= """

prediction=KNN_Classifier(Train_set,labels,preprocessing_data_RGB,5)
#accuracy_KNN = accuracy_score(label_valid, prediction)
prediction=np.asarray(prediction)
prediction=np.reshape(prediction,(400,-1))
result=[]
for i in range(prediction.shape[0]):
    result.append(preprocessing_data_cord)
result=np.asarray(result)
result=np.reshape(result,(result.shape[0],2))
data_train=np.load('data_train.npy')
data_train_copy=data_train.copy
fig=plt.figure()
plt.imshow(data_easy_train,cmap="gray")
plt.scatter(result[:,1],result[:,0],marker='.',c='g')
        
""" ======================== Generate Test Data =========================== """


"""This is where you should generate a validation testing data set.  This 
should be generated with different parameters than the training data!   """


""" ========================  Test the Model ============================== """

""" This is where you should test the validation set with the trained model """


"""
===============================================================================
===============================================================================
============================ Question 2 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Variable Declaration ========================== """

#True distribution mean and variance 



"""========================== Plot the true distribution =================="""
#plot true Gaussian function


"""========================= Perform ML and MAP Estimates =================="""
#Calculate posterior and update prior for the given number of draws

  

### plot ML and MAP steady state trends with variable PriorVar




