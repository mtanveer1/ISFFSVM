# -*- coding: utf-8 -*-

"""
run arguments:
    -data : string
    |   Specify a dataset.
    -n : integer
    |   Specify the number of n-fold cross-validation

"""
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import os
import argparse
import warnings
import statistics
warnings.filterwarnings("ignore") # Ignore warning messages during execution
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,confusion_matrix
from sklearn import preprocessing
import tools.dataprocess as dp
from sklearn.metrics import make_scorer, precision_score, recall_score, confusion_matrix, f1_score, matthews_corrcoef, auc, precision_recall_curve
from tools.imbalancedmetrics import ImBinaryMetric
from SlackFactorFSVM import *
RANDOM_STATE = None   # Random state for reproducibility


def parse():
    '''Parse system arguments.'''
    parse=argparse.ArgumentParser(
        description='General excuting ISFFSVM', 
        usage='demorun.py -data <datasetpath> -n <n-fold cross-validation>'
        )
    parse.add_argument("-data",dest="dataset",help="the path of a dataset")
    parse.add_argument("-n",dest="n",type=int,default=5,help="n-fold cross-validation")
    return parse.parse_args()

def metric(y,y_pre):
        return ImBinaryMetric(y,y_pre).AP() 

def validateCSVM(X, y):
    '''
    Perform cross-validation to determine the best parameters (C and gamma) for the SVM.
    '''    
        IR=max(np.bincount(y))*1.0/min(np.bincount(y))
        weights = {0:1.0, 1:IR}
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=RANDOM_STATE)
        C_range = np.logspace(-5, 11, 9,base=2)
        gamma_range = np.logspace(-10, 3, 14,base=2)
        tuned_params = {"gamma":gamma_range,"C" : C_range}
    # Perform grid search to find the best combination of C and gamma
        
        model = GridSearchCV(svm.SVC(probability=True, class_weight= weights),
                             tuned_params,cv=sss,
                             scoring=make_scorer(metric))#
        model.fit(X,y)
        return model.best_params_


def calculate_fnr_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    return fnr, fnr

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    #fnr, fpr = calculate_fnr_fpr(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred)
    #mcc = matthews_corrcoef(y_true, y_pred)
    #precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    #auc_pr = auc(recall_vals, precision_vals)
    return precision, recall

F1s=[]
MCCs=[]
Aucs=[]
Ps = [] 
Rs = []
FNRs = []
FPRs = []
for i in range(0,5):
    def main():
         para = parse()
         dataset=para.dataset
         scores = []
         X,y=dp.readDateSet(dataset)
         X=preprocessing.scale(X)
         print(f"Dataset:%s,#attribute:%s [neg pos]:%s\n "%(dataset,X.shape[1],str(np.bincount(y))))
         sss = StratifiedShuffleSplit(n_splits=para.n, test_size=0.2,random_state=RANDOM_STATE)
         fcnt=0
         for train_index, test_index in sss.split(X, y):
              fcnt+=1
              print('{} fold'.format(fcnt))
              X_train, X_test = X[train_index], X[test_index]
              y_train, y_test = y[train_index], y[test_index]
              #search best parameters of DEC classifier to calculate the slack variables by CV
              print("Searching best parameters (i.e., C and gamma) of  DEC classifier...")
              best_params=validateCSVM(X_train,y_train)
        #search the parameter C of SFSSVM by CV
              print("Searching the parameter (i.e., beta) of ISFFSVM...")
              tuned_params={}
              sss2 = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
              tuned_params["beta"]= [(i)*0.1 for i in range(0,11,1)]
              
              tuned_params["a"]=[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]

              model = GridSearchCV(SFFSVM(C=best_params['C'], gamma=best_params['gamma']),tuned_params,cv=sss2, n_jobs=-1)
              model.fit(X_train,y_train)

              y_pre=model.predict(X_test)
              print(y_pre)
              y_pred = model.predict_proba(X_test)[:, -1]
              metrics=ImBinaryMetric(y_test,y_pre)
              fnr, fpr = calculate_fnr_fpr(y_test, y_pre)
              precision, recall= calculate_metrics(y_test, y_pre)

              scores.append([metrics.f1(),metrics.MCC(),metrics.aucprc(y_pred),precision,recall,fnr,fpr])
              F1s.append(metrics.f1())
              print(F1s)
              MCCs.append(metrics.MCC())
              Aucs.append(metrics.aucprc(y_pred))
              Ps.append(precision)
              Rs.append(recall)
              FNRs.append(fnr)
              FPRs.append(fpr)
              


    if __name__ == '__main__':
         main()
         
print("Final Results")         
print("F1")
print(statistics.mean(F1s))
print("Standard Deviation of sample is % s "
                % (statistics.stdev(F1s)))
print("MCC")
print(statistics.mean(MCCs))
print("Standard Deviation of sample is % s "
                % (statistics.stdev(MCCs)))
print("Auc")
print(statistics.mean(Aucs))
print("Standard Deviation of sample is % s "
                % (statistics.stdev(Aucs)))

print("Precision")
print(statistics.mean(Ps))
print("Standard Deviation of sample is % s "
                % (statistics.stdev(Ps)))
print("Recall")
print(statistics.mean(Rs))
print("Standard Deviation of sample is % s "
                % (statistics.stdev(Rs)))

print("FNR")
print(statistics.mean(FNRs))
print("Standard Deviation of sample is % s " % (statistics.stdev(FNRs)))
print("FPR")
print(statistics.mean(FPRs))
print("Standard Deviation of sample is % s " % (statistics.stdev(FPRs)))


