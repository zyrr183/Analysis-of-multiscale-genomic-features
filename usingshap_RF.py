# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:47:14 2020

@author: jingr

test shap
"""
import os, sys
helpStr = '''
Usage:
    python usingshap_RF.py trainFile.arff testFile.arff
Or:
    python usingshap_RF.py trainFile1.arff testFile1.arff trainFile2.arff testFile2.arff
    
The first command above will generate the shrinked training and test dataset respectively.
The second will merge the two datasets respectively (i.e. trainFile1 + trainFile2, testFile1 + testFile2) and shrink the mergered dataset.

NOTE: all the files should be in ARFF format.

No matter which format of the input used, the output will be 3 files:
    selectedDataSetTrain.arff:  The shrinked training dataset in ARFF format.
    selectedDataSetTest.arff:   The shrinked test dataset in ARFF format.
    selectedFeaName.txt:        The selected features.
    
'''
if '-h' in sys.argv or '--help' in sys.argv:
    print(helpStr)
    exit()
if not len(sys.argv) == 3 or len(sys.argv) == 5:
#    print('Please make sure the input val')
#    print(sys.argv)
    print(helpStr)
    exit()
    
    
import shap
shap.initjs()
#import xgboost
from scipy.io.arff import loadarff 
import arff

import pandas as pd
#X,y = shap.datasets.boston()
import numpy as np
#from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 
#import scipy
from sklearn.ensemble import RandomForestClassifier



def scaleMat(matIn,minVal=0,maxVal=1,axis=0):
    tmpMat = matIn.copy().astype(float)
    diffOut = maxVal - minVal
    if axis == 0:
        tmpMat = tmpMat.transpose()
    for rowNum,array in enumerate(tmpMat):
        diff = np.max(array) - np.min(array)
        if diff == 0:
            array[:] = np.nan
        else:
            array = (array - np.min(array)) / diff * diffOut + minVal
        tmpMat[rowNum] = array
    if axis == 0:
        return tmpMat.transpose()
    else:
        return tmpMat
    
def get_n_fold(matIn,fold_num,label=None):
    dataMat = np.array(matIn).copy()
    sample_num = matIn.shape[0]
    shuffled_index = np.arange(sample_num)
    np.random.shuffle(shuffled_index)
    outDict = {}
    sample_in_one_fold = int(sample_num / fold_num)
    shuffled_mat = dataMat[shuffled_index,:]
#    print(dataMat.shape)
#    print(shuffled_mat.shape)
    if not label is None:
        shuffled_label = np.array(label).copy()[shuffled_index]
    for fold_index in range(fold_num):
        startNum = fold_index * sample_in_one_fold
        endNum = (fold_index+1) * sample_in_one_fold
        if fold_index == fold_num-1:
            endNum = sample_num
#        print(startNum,endNum)
        testSet = shuffled_mat[startNum:endNum,:]
        trainSet = np.concatenate((shuffled_mat[:startNum,:],shuffled_mat[endNum:,:]))
        outDict[fold_index] = {}
        outDict[fold_index]['trainX'] = trainSet
        outDict[fold_index]['testX'] = testSet
        if not label is None:
            testL = shuffled_label[startNum:endNum]
            trainL = np.concatenate((shuffled_label[:startNum],shuffled_label[endNum:]))
            outDict[fold_index]['trainY'] = trainL
            outDict[fold_index]['testY'] = testL
    return outDict        

#%% training data load




#tmpdata=loadarff('./BRCAmiRNAseq_matrix_out_cv.arff')
#tmpdata=loadarff('D:/database/tcga/old/pml_1.12/BRCAmRNA_FPKM_matrix_out_cv.arff')

if len(sys.argv) == 3:
    tmpdata = loadarff(sys.argv[1])
    
    
    X = pd.DataFrame(tmpdata[0])
    X = X.drop(['label'],axis=1)
    y=[]
    for tmparr in tmpdata[0]:
        y.append(int(tmparr[-1]))
    y=np.array(y)   
else:
    #for mix
#    tmpdata=loadarff('D:/database/tcga/old/pml_1.12/BRCAmRNA_FPKM_matrix_out_cv.arff')
    tmpdata = loadarff(sys.argv[1])
    X1 = pd.DataFrame(tmpdata[0])
    X1 = X1.drop(['label'],axis=1)
    y1=[]
    for tmparr in tmpdata[0]:
        y1.append(int(tmparr[-1]))
    y1=np.array(y1)   
    
#    tmpdata=loadarff('D:/database/tcga/old/pml_1.12/BRCAmiRNAseq_matrix_out_cv.arff')
    tmpdata = loadarff(sys.argv[3])
    X2 = pd.DataFrame(tmpdata[0])
    X2 = X2.drop(['label'],axis=1)
    y2=[]
    for tmparr in tmpdata[0]:
        y2.append(int(tmparr[-1]))
    y2=np.array(y2)   
    
    X = pd.concat([X1,X2],axis=1)
    y = y1


#%% model built

#model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
model = RandomForestClassifier(max_depth=None, random_state=0)
model.fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)



shap_values = explainer.shap_values(X)

y_hat = model.predict(X)

#%% test data load


#for ind test

#tmpdata=loadarff('./BRCAmiRNAseq_matrix_out_ind.arff')
#tmpdata=loadarff('D:/database/tcga/old/pml_1.12/BRCAmRNA_FPKM_matrix_out_ind.arff')

if len(sys.argv) == 3:
    tmpdata = loadarff(sys.argv[2])    
    X_test = pd.DataFrame(tmpdata[0])
    X_test = X_test.drop(['label'],axis=1)
    y_test = []
    for tmparr in tmpdata[0]:
        y_test.append(int(tmparr[-1]))
    y_test = np.array(y_test)   
else:
    #for mix
#    tmpdata=loadarff('D:/database/tcga/old/pml_1.12/BRCAmRNA_FPKM_matrix_out_ind.arff')
    tmpdata = loadarff(sys.argv[2])    
    X_test1 = pd.DataFrame(tmpdata[0])
    X_test1 = X_test1.drop(['label'],axis=1)
    y_test1=[]
    for tmparr in tmpdata[0]:
        y_test1.append(int(tmparr[-1]))
    y_test1=np.array(y_test1)   
    
#    tmpdata=loadarff('D:/database/tcga/old/pml_1.12/BRCAmiRNAseq_matrix_out_ind.arff')
    tmpdata = loadarff(sys.argv[4])    
    X_test2 = pd.DataFrame(tmpdata[0])
    X_test2 = X_test2.drop(['label'],axis=1)
    y_test2=[]
    for tmparr in tmpdata[0]:
        y_test2.append(int(tmparr[-1]))
    y_test2=np.array(y_test2)   
    
    X_test = pd.concat([X_test1,X_test2],axis=1)
    y_test = y_test1


#%% first test
#y_predict = model.predict(X_test)
#y_predict_binary = y_predict
#
#
#testLabelArr = y_test
#prediction = y_predict_binary
#cm=confusion_matrix(testLabelArr,prediction)
#print(cm)
#print("ACC: %f "%accuracy_score(testLabelArr,prediction))
#print("F1: %f "%f1_score(testLabelArr,prediction))
#print("Recall: %f "%recall_score(testLabelArr,prediction))
#print("Pre: %f "%precision_score(testLabelArr,prediction))
#print("MCC: %f "%matthews_corrcoef(testLabelArr,prediction))
#print("AUC: %f "%roc_auc_score(testLabelArr,prediction))

#%% using shaps with 10 fold cross validation
fold_num = 10

#10 fold for SHAP
subMatDict = get_n_fold(X,fold_num,y)
featureNum = X.shape[1]
featureCount = np.array([0] * featureNum)
sub_shap_values = []
crossvalidationMetrics = []
cms = []
for fold_index in range(fold_num):
    X_sub = subMatDict[fold_index]['trainX']
    X_sub_test = subMatDict[fold_index]['testX']
    Y_sub = subMatDict[fold_index]['trainY']
    Y_sub_test = subMatDict[fold_index]['testY']
#    model1 =  RandomForestClassifier(max_depth=None, random_state=0)
#    model1.fit(X_sub, Y_sub)

    tmp_shap_values = explainer.shap_values(X_sub)
    sub_shap_values.append(tmp_shap_values)
    feature_importance = np.sum(np.abs(tmp_shap_values[0]),axis=0) + np.sum(np.abs(tmp_shap_values[1]),axis=0)
    
    tmpCount = feature_importance > 0
    featureCount += tmpCount
    
    #metrics
#    prediction = model1.predict(X_sub_test)
#    testLabelArr = Y_sub_test
#    cm=confusion_matrix(testLabelArr,prediction)
##    print(cm)
#    acc = accuracy_score(testLabelArr,prediction)
#    f1 = f1_score(testLabelArr,prediction)
#    recall = recall_score(testLabelArr,prediction)
#    pre = precision_score(testLabelArr,prediction)
#    mcc = matthews_corrcoef(testLabelArr,prediction)
#    auc = roc_auc_score(testLabelArr,prediction)
#    subMetrics = [acc,f1,recall,pre,mcc,auc]
#    crossvalidationMetrics.append(subMetrics)
#    cms.append(cm)
#    print("ACC: %f "%accuracy_score(testLabelArr,prediction))
#    print("F1: %f "%f1_score(testLabelArr,prediction))
#    print("Recall: %f "%recall_score(testLabelArr,prediction))
#    print("Pre: %f "%precision_score(testLabelArr,prediction))
#    print("MCC: %f "%matthews_corrcoef(testLabelArr,prediction))
#    print("AUC: %f "%roc_auc_score(testLabelArr,prediction))
#for i in range(fold_num):
#    print(i)
#    print(cms[i])
#tmp = np.transpose(crossvalidationMetrics)
#tmpavg=np.average(np.transpose(crossvalidationMetrics),axis=1)   

featureCount = np.array([0] * featureNum) 
#feature_filtered_num = 8000
feature_filtered_num = 300
for tmp_shap_values in sub_shap_values:
    feature_importance = np.sum(np.abs(tmp_shap_values[0]),axis=0) + np.sum(np.abs(tmp_shap_values[1]),axis=0)
    tmp_sorted_feature_importance = np.sort(feature_importance)
    feature_importance[feature_importance<tmp_sorted_feature_importance[-feature_filtered_num]] = 0
    tmpCount = feature_importance > 0
    featureCount += tmpCount
#for i in range(fold_num):
#    print(np.sum(featureCount>i))


fold_count_thres = 10
selected_feature_index = np.arange(len(featureCount))[featureCount >= fold_count_thres]
X_sub = X.iloc[:,selected_feature_index]
X_test_sub = X_test.iloc[:,selected_feature_index]
selected_feature_name = X_test_sub.columns.values.tolist()

#%% make the output

arff.dump('selectedDataSetTrain.arff'
      , X_sub.values
      , relation='features'
      , names=X_sub.columns)

arff.dump('selectedDataSetTest.arff'
      , X_test_sub.values
      , relation='features'
      , names=X_test_sub.columns)

with open('selectedFeaName.txt','w') as FIDO:
    FIDO.write('\n'.join(selected_feature_name))
    


































