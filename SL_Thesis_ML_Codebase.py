# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 08:16:39 2022

@author: staslist
"""

import pandas as pd
import numpy as np
import csv
import sys
import math
import time
import unittest
import seaborn
import itertools
import random
import threading
import gseapy as gp
import re
from copy import deepcopy
import os
import sys
import requests

import matplotlib.pyplot as plt
import seaborn

from io import StringIO

from itertools import product

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import mutual_info_classif

from scipy.stats import shapiro
from scipy.stats import normaltest

models_ALL = ['LR', 'kNN', 'SVM', 'RF']

p_grid_Ada = {"n_estimators": {5, 10, 15, 20, 25, 50}, "learning_rate": {0.5, 0.75, 0.9, 1, 1.15, 1.3}}
p_grid_RF = {"n_estimators": [10,25,50,100,200], "criterion": ['gini', 'entropy'], 
             "max_features": ["auto", "sqrt", "log2", None]}
p_grid_DT = {"criterion": ["gini", "entropy"], "max_depth": {5,10,None}, "min_samples_split": {2,3,4},
             "min_samples_leaf": [1,2,3],"max_features": ["auto", "sqrt", "log2", None], 
             "min_impurity_decrease": [0, 0.1], "ccp_alpha": [0, 0.1]}
p_grid_SVM = {"kernel":['linear', 'poly', 'rbf'], "C": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
              "class_weight": [None, 'balanced'], "probability": [True], "degree": [1, 3, 5],
              "gamma": ['scale', 'auto']}
# p_grid_SVM = {"kernel":['linear'], "C": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
#                 "class_weight": [None, 'balanced'], "probability": [True]}
p_grid_kNN = {"n_neighbors" : [3,5,7,9], "weights": ['uniform', 'distance'], 
              "metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
p_grid_LR = {"C":[0.5, 1.0, 2.0, 3.0, 4.0, 5.0], "class_weight": [None, 'balanced'],
             "solver": ['newton-cg', 'lbfgs', 'liblinear', 'saga']}

p_grid_dict = {'LR': p_grid_LR, 'kNN': p_grid_kNN, 'SVM': p_grid_SVM, 'DT': p_grid_DT,
               'RF': p_grid_RF, 'Ada': p_grid_Ada, 'GNB': {}}

def detect_outlier_features_by_std2(X, Y, feature_names:list, treshold:float = 3.5, flag_zeros:bool = True):
    '''Given X and Y matrices and a correspondingly ordered list of feature names identify 
    highly variant features. Return the list of feature names that are highly variant.
    A feature is highly variant if its counts for at least one sample are > than (mean + std * treshold) or < 
    than (mean - std * treshold). Also, optionally, detect features that are mostly 0.
    Rows are samples. Columns are features.
    Recommend using lower tresholds with smaller sample sizes. 5-10 samples: treshold = 2. 10+ samples: treshold 2.5+.
    '''
    
    outlier_features = set()
    #print("X:", X)
    #print("Y:", Y)
    
    class_labels = set(Y)
    for class_label in class_labels:
        #print('Class label: ', class_label)
        class_indeces_tuple = np.where(Y == class_label)
        class_indeces_array = class_indeces_tuple[0]
        sub_X = X[class_indeces_array, :]
        #print("sub_X:", sub_X)
        sub_X_t = sub_X.transpose()
        i = 0
        for feature in sub_X_t:
            # print('The value of i: ', i)
            # print("Feature in sub_X:", feature)
            if(flag_zeros):
                if(np.count_nonzero(feature) < (len(feature)/2)):
                    outlier_features.add(feature_names[i])
                    i += 1
                    continue
            fea_mean = np.mean(feature)
            fea_std = np.std(feature)
            # print("Feature mean:", fea_mean)
            # print("Feature std:", fea_std)
            for count_value in feature:
                if (count_value > (fea_mean + fea_std*treshold)) or (count_value < (fea_mean - fea_std*treshold)):
                    outlier_features.add(feature_names[i])
            
            i += 1
    return outlier_features

def generate_top_RF_features(X, Y, feature_names, num_features, fname, taboo_features = []):
    model = RandomForestClassifier()
    clf = model.fit(X, Y)
    RF = clf.feature_importances_
    
    data_rf = {}
    i = 0
    while i < len(RF):
        data_rf[feature_names[i]] = RF[i]
        i += 1
    s_data_rf = sorted(data_rf.items(), key = lambda x: abs(x[1]), reverse = True)
    
    top_features = []
    
    i = 0
    for kv_tup in s_data_rf:
        if(str(kv_tup[0]) not in taboo_features):
            top_features.append(str(kv_tup[0]))
            i += 1
            if(i == num_features):
                break
            
    with open(fname, 'w') as writer:
        for feature in top_features:
            writer.write(feature + '\n')
            
    return top_features

def generate_top_IG_features(X, Y, feature_names, num_features, fname, taboo_features = [], write_out = True):
    '''For fname provide full directory.'''
    # IG Feature Ranking Block
    IG = mutual_info_classif(X, Y)
    
    data_ig = {}
    i = 0
    while i < len(IG):
        data_ig[feature_names[i]] = IG[i]
        i += 1
    s_data_ig = sorted(data_ig.items(), key = lambda x: abs(x[1]), reverse = True)
    top_features = []
    
    #print(s_data_ig)
    feature_rank = {}
    
    i = 0
    for kv_tup in s_data_ig:
        if(str(kv_tup[0]) not in taboo_features):
            top_features.append(str(kv_tup[0]))
            feature_rank[str(kv_tup[0])] = kv_tup[1]
            i += 1
            if(i == num_features):
                break
    
    if(write_out):
        with open(fname, 'w') as writer:
            for feature in top_features:
                writer.write(feature + '\t' + str(feature_rank[feature]) + '\n')
            
    return top_features
    
def select_features_from_matrix(X, feature_names:list, feature_selection:list):
    '''Given a matrix, a corresponding list of feature names, and a list of features to 
    select, return a matrix with only the selected features included.'''
    feature_name_indeces = []
    for feature in feature_selection:
        if (feature_names.count(feature) != 1):
            print(feature)
            raise ValueError("Top feature was either not read in or read in multiple times.")
        feature_name_indeces.append(feature_names.index(feature))

    result = X[:, feature_name_indeces]
    return result

def read_in_csv_file_one_column(filename:str, column:int, delim:str, skip_n = 0, limit = None)->list:
    temp = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        i = 0
        for row in csv_reader:
            if(limit != None):
                if(i >= limit):
                    break
            if (i < skip_n):
                i += 1
                continue
            try:
                temp.append(row[column])
            except ValueError:
                pass
            i += 1
    return temp

def count_elements_in_2dlist(a:list):
    result = {}
    for collection in a:
        for ele in collection:
            try:
                result[ele] += 1
            except KeyError:
                result[ele] = 1
                
    return result

def generate_kfolds(samples:list, k:int):
    ''' Separates samples into k-folds (lists). 
    Returns a list of lists.'''
    if(k > len(samples)):
        raise ValueError("The number of folds - ", str(k), " - must be <= than the number of samples - ", str(samples), ".")
    folds = []
    i = 0
    while i < k:
        folds.append([])
        i = i + 1
        
    i = 0
    for sample in samples:
        folds[i % k].append(sample)
        i = i + 1
            
    return folds

def divide_folds_in_training_and_validation(folds:list):
    '''Folds is a list of lists. Each inner list is a collection of samples (fold). We want to training and validation sets using 
    these folds. The number of training/validation sets must equal to number of folds. 
    Training and validation sets order must correspond. Training and validation 'sets' are 1-d lists. '''
    assert(len(folds)>1)
    training_sets = []
    validation_sets = []
    fold_index = 0
    while fold_index < len(folds):
        training_set = []
        validation_set = []
        i = 0
        while i < len(folds):
            if(i != fold_index):
                training_set += folds[i]
            else:
                validation_st = folds[i]
            i += 1
        fold_index += 1
        training_sets.append(training_set)
        validation_sets.append(validation_set)
    return training_sets, validation_sets

def generate_nested_kfolds(samples:list, k_inner:int, k_outer:int):
    ''' A 3-d list. Each inner list represents folds formed using the outer loop's training sets. '''
    if(k_inner > len(samples) or k_outer > len(samples)):
        raise ValueError("The number of folds must be <= than the number of samples.")
    
    assert(k_outer > 1 and k_inner > 1)
    
    outer_folds = generate_kfolds(samples, k_outer)
    nested_kfolds = []
    i = 0
    training_sets, validation_sets = divide_folds_in_training_and_validation(outer_folds)
    #print("training_sets:", training_sets)
    while i < k_outer:
        training_set = training_sets[i]
        #print('training_set:', training_set)
        nested_folds = generate_kfolds(training_set, k_inner)
        nested_kfolds.append(nested_folds)
        i += 1
        
    return nested_kfolds

def perform_nested_cv(X, Y, feature_names:list, sample_order:list, model:str, conditions:list, cv_k_inner:int, cv_k_outer:int, num_features:int,
                      num_runs:int, FS_method:str, out_dir:str, dataset_name:str, FS_filter:str = None, Filter_Threshold:float = 3.0,
                      plot_ROC:bool = True):
    '''X and Y are numpy array matrices.
    feature_names: names of features within X matrix (columns), must match the order of columns in X matrix.
    model: name of the ML model.
    conditions: names of the conditions involved in the analysis.
    cv_k_innner, cv_k_outer: the number of folds for inner and outer loops of nested cross validation.
    num_features: number of features to used during nested cross-validation.
    num_runs: number of runs.
    FS_method: IG, RF.
    out_dir: directory into which all output files are placed.
    FS_filter: Variance'''
    
    assert(FS_filter in ['Variance', None])
    assert(FS_method in ['RF', 'IG', 'Custom', None])
    assert(num_runs >= 1)
    assert(num_features > 0)
    assert(cv_k_inner > 1)
    assert(cv_k_outer > 1)
    assert(model in models_ALL)
    
    p_grid = p_grid_dict[model]
    
    r = 0
    
    # Will store the performance metrics for each run in variables below. Then average them across runs.
    nested_accs = []
    nested_baccs = []
    nested_precisions = []
    nested_recalls = []
    nested_conf_matrices = []
    
    # Used to keep track of misclassification
    y_hats = []
    y_expecteds = []
    misclass = {}
    while r < num_runs:
        
        # To plot ROC
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0,1,100)
        
        
        # Each run samples are randomly assigned to folds for outer loop.
        r_outer = 0
        # To be used within GridSearch
        inner_cv = KFold(n_splits=cv_k_inner, shuffle=True, random_state=0)
        # To be used in outer CV 
        outer_cv = KFold(n_splits=cv_k_outer, shuffle=True)
        
        hyper_param_names = []
        hyper_param_values = []
        for k,v in p_grid.items():
            hyper_param_names.append(k)
            hyper_param_values.append(v)
        
        hyper_param_grid = list(product(*hyper_param_values))
        
        # Nested cross-validation
        # Use outer cross-validation for model evaluation
        scores_outer = []
        best_y_hat_full = []
        best_y_out_full = []
        for tr_va_te_indeces in outer_cv.split(X):
            r_inner = 0
            #print('R_Outer: ', str(r_outer))
            X_train_validate = X[tr_va_te_indeces[0], :]
            Y_train_validate = Y[tr_va_te_indeces[0]]
            X_test = X[tr_va_te_indeces[1], :]
            Y_test = Y[tr_va_te_indeces[1]]
            
            X_trains, Y_trains, X_validates, Y_validates = [],[],[],[]
            for tr_va_indeces in inner_cv.split(X_train_validate):
                #print('Perform inner loop feature selection.')
                X_train = X[tr_va_indeces[0], :]
                Y_train = Y[tr_va_indeces[0]]
                X_validate = X[tr_va_indeces[1], :]
                Y_validate = Y[tr_va_indeces[1]]
                
                # Perform feature selection for inner training sets.
                if(FS_method != None):
                    if(FS_filter == 'Variance'):
                        outlier_features = detect_outlier_features_by_std2(X_train, Y_train, feature_names, Filter_Threshold)
                    else:
                        outlier_features = []
                    if(FS_method == 'IG'):
                        out_fname = out_dir + 'top_IG_features' + '_' + str(r) + '_' + str(r_outer) + '_' + str(r_inner) + '.txt'
                        top_features = generate_top_IG_features(X_train, Y_train, feature_names, num_features, out_fname, outlier_features, False)
                    elif(FS_method == 'RF'):
                        out_fname = out_dir + 'top_RF_features' + '_' + str(r) + '_' + str(r_outer) + '_' + str(r_inner) + '.txt'
                        top_features = generate_top_RF_features(X_train, Y_train, feature_names, num_features, out_fname, outlier_features, False)
                    elif(FS_method == 'Custom'):
                        top_features = ['...']
                        
                    X_train = select_features_from_matrix(X_train, feature_names, top_features)
                    X_validate = select_features_from_matrix(X_validate, feature_names, top_features)
                
                X_trains.append(X_train)
                Y_trains.append(Y_train)
                X_validates.append(X_validate)
                Y_validates.append(Y_validate)
                
                r_inner += 1
            
            max_model_select_score = 0
            best_hyper_params = {}
            for hyper_param_tup in hyper_param_grid:
                hyper_param_dict = {}
                i = 0
                for hyper_param_name in hyper_param_names:
                    hyper_param_dict[hyper_param_name] = hyper_param_tup[i]
                    i += 1
                
                if(model == 'kNN'):classifier = KNeighborsClassifier(**hyper_param_dict)
                elif(model == 'LR'):classifier = LogisticRegression(**hyper_param_dict)
                elif(model == 'SVM'):classifier = SVC(**hyper_param_dict)
                elif(model == 'RF'):classifier = RandomForestClassifier(**hyper_param_dict)
                
                # Use inner cross-validation for model selection
                scores_inner = []
                r_inner = 0
                while r_inner < cv_k_inner:
                    clf = classifier.fit(X_trains[r_inner], Y_trains[r_inner])
                
                    Y_hat = clf.predict(X_validates[r_inner])
                    score = clf.score(X_validates[r_inner], Y_validates[r_inner])
                    scores_inner.append(score)
                    
                    r_inner += 1
                    
                scores_inner = np.array(scores_inner)
                mean_score_inner = np.mean(scores_inner)
                if(mean_score_inner > max_model_select_score):
                    max_model_select_score = mean_score_inner
                    best_hyper_params = hyper_param_dict
                    # print("Best hyper parameter set: ", best_hyper_params)
                    # print("Corresponding Score: ", str(max_model_select_score))
            
            if(model == 'kNN'):classifier = KNeighborsClassifier(**best_hyper_params)
            elif(model == 'LR'):classifier = LogisticRegression(**best_hyper_params)
            elif(model == 'SVM'):classifier = SVC(**best_hyper_params)
            elif(model == 'RF'):classifier = RandomForestClassifier(**best_hyper_params)
            
            # Perform Feature Selection for Outer Training Sets
            if(FS_method != None):
                if(FS_filter == 'Variance'):
                    outlier_features = detect_outlier_features_by_std2(X_train_validate, Y_train_validate, feature_names, Filter_Threshold)
                else:
                    outlier_features = []
                if(FS_method == 'IG'):
                    out_fname = out_dir + 'top_IG_features' + '_' + str(r) + '_' + str(r_outer) + '.txt'
                    top_features = generate_top_IG_features(X_train_validate, Y_train_validate, feature_names, num_features, out_fname, outlier_features)
                elif(FS_method == 'RF'):
                    out_fname = out_dir + 'top_RF_features' + '_' + str(r) + '_' + str(r_outer) + '.txt'
                    top_features = generate_top_RF_features(X_train_validate, Y_train_validate, feature_names, num_features, out_fname, outlier_features)
                elif(FS_method == 'Custom'):
                    top_features = ['...']
                    
                # print('Outer Training Set Features: ', top_features)    
                
                X_train_validate = select_features_from_matrix(X_train_validate, feature_names, top_features)
                X_test = select_features_from_matrix(X_test, feature_names, top_features)    
            
            clf = classifier.fit(X_train_validate, Y_train_validate)
                
            test_samples = np.array(sample_order)[tr_va_te_indeces[1]]
            Y_hat = clf.predict(X_test)
            num_test_samples = Y_hat.shape[0]
            q = 0
            while q < num_test_samples:
                if(Y_hat[q] != Y_test[q]):
                    # print("Misclassified: ", test_samples[q])
                    try:
                        misclass[test_samples[q]] = misclass[test_samples[q]] + 1
                    except KeyError:
                        misclass[test_samples[q]] = 1
                q += 1
                        
            # print("Current misclassification dictionary: ", misclass)
            
            best_y_hat_full.append(Y_hat)
            best_y_out_full.append(Y_test)
            
            score = clf.score(X_test, Y_test)
            probabilities = clf.predict_proba(X_test)
            # ROC BLOCK
            if(plot_ROC):
                fpr, tpr, t = roc_curve(Y_test, probabilities[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (r_outer, roc_auc))
            # ROC BLOCK OVER
            scores_outer.append(score)
            r_outer += 1
        
        if(plot_ROC):
            plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, color='blue',
                     label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC')
            plt.legend(loc="lower right")
            
            out_file = out_dir + dataset_name + '_ROC_' + model + '_run_' + str(r) +'.png'
            plt.savefig(out_file, bbox_inches="tight")
            
            plt.show()    
            
        
        best_y_hat_full = np.concatenate(best_y_hat_full, axis = 0)
        best_y_out_full = np.concatenate(best_y_out_full, axis = 0)
        mean_score = np.mean(scores_outer)
        
        y_hats.append(best_y_hat_full)
        y_expecteds.append(best_y_out_full)
        
        # Also attain precision, recall, and confusion matrix.
        balanced_acc = balanced_accuracy_score(best_y_out_full, best_y_hat_full)
        precision = precision_score(best_y_out_full, best_y_hat_full, average = None)
        recall = recall_score(best_y_out_full, best_y_hat_full, average = None)
        conf_matrix = confusion_matrix(best_y_out_full, best_y_hat_full)
        
        nested_accs.append(mean_score)
        nested_baccs.append(balanced_acc)
        nested_precisions.append(precision)
        nested_recalls.append(recall)
        nested_conf_matrices.append(conf_matrix)
        
        r += 1
    # Average performance metrics across runs.
    nested_acc = np.mean(np.array(nested_accs))
    nested_bacc = np.mean(np.array(nested_baccs))
    print("Nested cross validation score: ", nested_acc)
    #print("Balanced Accuracy: ", nested_bacc)
    #print("Precision: ",  np.mean(np.array(nested_precisions), axis=0 ))
    #print("Recall: ",  np.mean(np.array(nested_recalls), axis=0 ))
    avg_conf_matrix = np.mean(np.array(nested_conf_matrices), axis=0 )
    #print("Confusion Matrix: ",  avg_conf_matrix)
    print("Final misclassification dictionary: ", misclass)
    
    gen_conf_matrix(out_dir, dataset_name, avg_conf_matrix, conditions, nested_acc, nested_bacc, model)
    plot_misclass(out_dir, dataset_name, misclass, model)
    plot_feature_importance(out_dir, dataset_name, num_runs, cv_k_outer, False, model)
    
    return nested_acc

def plot_feature_importance_permutation(out_path:str, feature_names:list, feature_rankings:list, limit = 20):
    
    fig, ax = plt.subplots()
    ax.barh(feature_names[0:20], feature_rankings[0:20], align='center')
    ax.set_xlabel('Feature Importance', fontsize = 14)
    ax.set_title('Genomic Data')
    ax.invert_yaxis()
    
    out_file = out_path + '_FRank.png'
    plt.savefig(out_file, bbox_inches="tight")
    
    plt.show()
    

def plot_feature_importance(out_dir:str, run_name:str, num_runs:int, k_outer:int, model:str, validation:bool = False, limit = 20):
    # Average feature rankings across outer test sets / runs.
    def set_feature_rank():
        feature_rank = {}
        i = 0
        j = 0
        while i < num_runs:
            while j < k_outer:
                # filename = out_dir + 'top_model_features_set' + str(i) + '_' + str(j) + '.txt'
                # filename = out_dir + 'top_IG_features_' + str(i) + '_' + str(j) + '.txt'
                filename = out_dir + 'top_proteomic_features_set' + str(j) + '.txt'
                with open(filename) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter='\t')
                    for row in csv_reader:
                        if(len(feature_rank) > limit):
                            return feature_rank
                        try:
                            feature_rank[row[0]].append(float(row[1]))
                        except KeyError:
                            feature_rank[row[0]] = [float(row[1])]
                j += 1
            i += 1
            
        return feature_rank
            
    def set_feature_rank_validation():
        feature_rank = {}
        filename = out_dir + 'top_model_features.txt'
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if(len(feature_rank) > limit):
                    return feature_rank
                feature_rank[row[0]] = [float(row[1])]
                
        return feature_rank
    
    if(not validation):
        feature_rank = set_feature_rank()
    else:
        feature_rank = set_feature_rank_validation()
            
    feature_rank_mean = {}
    feature_names = []
    feature_rankings = []
    for k,v in feature_rank.items():
        mean = np.mean(np.array(v))
        feature_rank_mean[k] = mean
        
    sorted_features = sorted(feature_rank_mean.items(), key = lambda x: abs(x[1]), reverse = True)
    for k in sorted_features:
        feature_names.append(k[0])
        feature_rankings.append(k[1])
        
    feature_rankings = np.array(feature_rankings)
    #y_pos = np.arange(len(feature_names))
    print("Feature Rank Importance: ", feature_rank_mean)
    
    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_rankings, align='center')
    ax.set_xlabel('Feature Importance', fontsize = 14)
    ax.set_title('Genomic Data')
    ax.invert_yaxis()
    
    out_file = out_dir + run_name + '_FRank_' + model +'.png'
    plt.savefig(out_file, bbox_inches="tight")
    
    plt.show()

def plot_misclass(out_dir:str, dataset_name:str, misclass:dict, model:str):
    out_file = out_dir + dataset_name + '_misclass_' + model +'.txt'
    with open(out_file, 'w') as writer:
        for k,v in misclass.items():
            writer.write(k + '\t' + str(v) + '\n')
    
    
    sorted_samples = sorted(misclass.items(), key = lambda x: abs(x[1]), reverse = True)
    sample_names = []
    misclass_freq = []
    for k in sorted_samples:
        sample_names.append(k[0])
        misclass_freq.append(k[1])
    
    fig, ax = plt.subplots()
    
    ax.barh(sample_names, misclass_freq, align='center')
    ax.set_xlabel('Misclassification Frequency', fontsize = 14)
    ax.set_title('Proteomic Data')
    ax.invert_yaxis()
    
    fig.set_figheight(12)
    fig.set_figwidth(8)
    fig.show()
    out_file = out_dir + dataset_name + '_misclass_' + model +'.png'
    fig.savefig(out_file, bbox_inches="tight")

def plot_per_condition_counts_heatmap(counts:dict, top_features:list, conditions:list, out_dir:str, run_name:str, num_features:int = 30):
    # Draw a per-replicate heatmap    
    array = []
    for feature_name,means in counts.items():
        array.append(means)
    array = np.array(array)
    
    # We limit number of top genes to num_genes for the heatmap.
    if(array.shape[0] > num_features):
        array = array[0:num_features, :]
        top_features = top_features[0:num_features]
    
    # print(array)
    # print(array.shape)
    # print(top_features)
    # print(len(top_features))
    # print(conditions)
    
    df_cm = pd.DataFrame(array, index = top_features,
                         columns = conditions)
    
    width = len(conditions) * 3
    height = len(top_features) // 3
    plt.figure(figsize=(width,height), dpi=600)
    # 'BrBG'
    seaborn.heatmap(df_cm, cmap = 'Blues', vmin=0, vmax=2.5)
    plt.xlabel('Conditions', fontsize = 14)
    plt.ylabel('Features', fontsize = 14)
    plt.title('Counts', fontsize = 16)
    out_file = out_dir + run_name + '.png'
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    # plt.savefig(out_file, dpi=300)
    # plt.show()
    
def plot_per_sample_counts_heatmap(counts:dict, top_features:list, out_dir:str, run_name:str, num_features:int = 30):
    # Draw a per-replicate heatmap
    array = []
    rep_names = []
    for rep_name,values in counts.items():
        rep_names.append(rep_name)
        array.append(values)
    sample_names = rep_names
    array = np.array(array)
    
    # We limit number of top genes to num_genes for the heatmap.
    if(array.shape[1] > num_features):
        array = array[:, 0:num_features]
        top_features = top_features[0:num_features]
        
    df_cm = pd.DataFrame(array, index = sample_names,
                         columns = top_features)
    
    width = len(top_features) // 3
    height = len(rep_names) // 3
    print(width)
    print(height)
    plt.figure(figsize=(width,height), dpi=600)
    # 'BrBG'
    seaborn.heatmap(df_cm, cmap = 'Blues', vmin=0, vmax=2.5)
    plt.xlabel('Features', fontsize = 14)
    plt.ylabel('Replicates', fontsize = 14)
    plt.title('Counts', fontsize = 16)
    out_file = out_dir + run_name + '.png'
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    # plt.savefig(out_file, dpi=300)
    # plt.show()


def gen_conf_matrix(out_dir:str, dataset_name:str, conf_matrix:np.array, cm_labels:list, accuracy:float, b_accuracy:float, model:str):
    # Generating custom confusion matrices
    acc_text = 'Total Accuracy: ' + str(int(accuracy*100)) + '%'
    bacc_text = 'Total Balanced Accuracy: ' + str(int(b_accuracy*100)) + '%'
    
    annotation = conf_matrix.tolist()
    #print("Annotation: ", annotation)
    #print("Confusion Matrix: ", conf_matrix)
    # Add per class accuracy on diagonal entries.
    i = 0
    while i < conf_matrix.shape[0]:
        j = 0
        while j < conf_matrix.shape[1]:
            annotation[i][j] = str(round(conf_matrix[i][j], 2))
            if(i == j):
                if(np.sum(conf_matrix[i,:]) == 0):
                    annotation[i][j] += '\n0%'
                else:
                    annotation[i][j] += '\n' + str( int((conf_matrix[i][j]/np.sum(conf_matrix[i,:]))*100) ) + '%'
            j += 1
        i += 1
    
    #print("Annotation: ", annotation)
    
    df_cm = pd.DataFrame(conf_matrix, index = cm_labels, columns = cm_labels)
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    df_cm = df_cm.fillna(0)
    
    #print(df_cm)
    
    plt.figure()
    
    seaborn.heatmap(df_cm, cmap = 'Blues', annot=annotation, fmt='', annot_kws={"fontsize":12}, vmin = 0, vmax = 1.0)
    plt.xlabel('Predicted Classes', fontsize = 14)
    plt.ylabel('Actual Classes', fontsize = 14)
    #plt.title('Proteomic ' + model, fontsize = 16)
    plt.title('PBMC 3-Way Matched Balanced Integrated', fontsize = 16)

    plt.figtext(0.1, -0.05, acc_text, horizontalalignment='left', fontsize = 12) 
    plt.figtext(0.1, -0.10, bacc_text, horizontalalignment='left', fontsize = 12) 
    
    out_file = out_dir + dataset_name + '_proteomic_' + model +'.png'
    plt.savefig(out_file, bbox_inches="tight")