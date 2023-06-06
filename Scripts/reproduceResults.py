# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:40:15 2023

@author: 73273
"""


import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import KFold, train_test_split

def computePairwiseSeqsIdentity(seq1, seq2, size=20):
    """
    compute the sequence identity between seq1 and seq2
    """
    same_count = 0
    pointer = 0
    while pointer <= 2*size :
        aa_1 = seq1[pointer]
        aa_2 = seq2[pointer]
        if aa_1 == aa_2:
            same_count += 1
        pointer += 1
    return same_count / (2*size+1)


def performanceAssessments(y_true, y_pred):
    """evaluation indicators"""
    c = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = c[0, 0], c[0, 1], c[1, 0], c[1, 1]
    
    S_p = tn / (tn + fp)  # specificity
    S_n = tp / (tp + fn)  # sensitivity
    acc = accuracy_score(y_true, y_pred)
    Pre = precision_score(y_true, y_pred)  # precision
    Rec = recall_score(y_true, y_pred)  # recall
    mcc = matthews_corrcoef(y_true, y_pred)  # Matthens correlation coefficient   
    return S_p, S_n, acc, mcc, Pre, Rec


def nearmiss_undersampled_neg_samples(all_LabelFeature, all_neg_features, all_neg_seqs, savepath=None, save=0):
    """balance the binary classes through nearmiss;
       generate information of seleceted negative samples"""
    
    # choose one feature can distinguish neg samples
    uniquecount_arr = np.array([])
    m, n = all_neg_features.values.shape
    for i in range(n):
        uniquecount_arr = np.append(uniquecount_arr, 
                        np.unique(all_neg_features.values[:, i]).shape[0])
    check0 = np.where(uniquecount_arr == m)[0]
    if check0.shape[0] > 0:
        spe_id = check0[0]
    else:
        print("you need an another method!")
        
    # nearmiss undersampling
    ls, fs = all_LabelFeature['label'].values, all_LabelFeature.values[:, 1:]
    nm1 = NearMiss(version=1)
    features_resampled, labels_resampled = nm1.fit_resample(fs, ls)
    
    # generate the sampled samples from the majority class
    n_fea_resampled = features_resampled[np.where(labels_resampled==0)] 
    sampled_ids = []
    for i in range(n_fea_resampled.shape[0]):
        check1 = n_fea_resampled[i][spe_id] == all_neg_features.values[:, spe_id]
        id_array = np.where(check1 == True)  # tuple:(array,)
        if id_array[0].shape[0] == 1:
            sampled_ids.append(id_array[0][0])
    sampled_ids.sort()
    sampled_samples = all_neg_seqs.iloc[sampled_ids]
    # newid_sampled_samples = sampled_samples.set_index('ue')
    if save == 1:
        sampled_samples.to_excel(savepath)
    return features_resampled, labels_resampled, sampled_samples


def kFoldTraining(num_folds:list, binary_classifier, shuffled_resampled_features, shuffled_resampled_labels, 
                  savepath=None, save=0):
    """
    K-fold Cross Validation
    """
    df_result_list = []
    all_pred_probas = {}  # in order to plot ROC curve
    for num_fold in num_folds:
        kf = KFold(n_splits=num_fold, shuffle=True)
        s_p, s_n, acc, mcc, auc, pred_probas = [], [], [], [], [], []
        pred_probas = {'y_true':[], 'y_pred_proba':[]}
        for train_index, test_index in list(kf.split(shuffled_resampled_features)):
            train_features, test_features = shuffled_resampled_features[train_index], shuffled_resampled_features[test_index]
            train_label, test_label = shuffled_resampled_labels[train_index], shuffled_resampled_labels[test_index]
            binary_classifier.fit(train_features, train_label)
            test_pred = binary_classifier.predict(test_features)
            temp_sp, temp_sn, temp_acc, temp_mcc = performanceAssessments(test_label, test_pred)[:4]
            s_p.append(temp_sp); s_n.append(temp_sn); acc.append(temp_acc); mcc.append(temp_mcc); 
            
            test_pred_proba = binary_classifier.predict_proba(test_features)[:, 1] 
            auc.append(roc_auc_score(test_label, test_pred_proba))
            pred_probas['y_true'].append(test_label)
            pred_probas['y_pred_proba'].append(test_pred_proba)
        avg_s_p, std_s_p = np.mean(np.array(s_p)), np.std(np.array(s_p))
        avg_s_n, std_s_n = np.mean(np.array(s_n)), np.std(np.array(s_n))
        avg_acc, std_acc = np.mean(np.array(acc)), np.std(np.array(acc))
        avg_mcc, std_mcc = np.mean(np.array(mcc)), np.std(np.array(mcc))
        avg_auc, std_auc = np.mean(np.array(auc)), np.std(np.array(auc))
        s_p.insert(0, avg_s_p)
        s_p.insert(1, std_s_p)
        s_n.insert(0, avg_s_n)
        s_n.insert(1, std_s_n)
        acc.insert(0, avg_acc)
        acc.insert(1, std_acc)
        mcc.insert(0, avg_mcc)
        mcc.insert(1, std_mcc)
        auc.insert(0, avg_auc)
        auc.insert(1, std_auc)
        # s_n.insert(0, np.mean(np.array(s_n)))
        # acc.insert(0, np.mean(np.array(acc)))
        # mcc.insert(0, np.mean(np.array(mcc)))
        # auc.insert(0, np.mean(np.array(auc)))
        row_indices1 = [str(num_fold) for i in range(5)]
        row_indices2 = ['Sp', 'Sn', 'ACC', 'MCC', 'AUC']
        df_result = pd.DataFrame(np.c_[s_p, s_n, acc, mcc, auc].T, index=[row_indices1, row_indices2], 
                                 columns = ['avg', 'std']+[str(i) for i in range(1, num_fold+1)])
        
        df_result_list.append(df_result)
        all_pred_probas[num_fold] = pred_probas
        
    df_results = pd.concat(df_result_list, axis=0)
    if save == 1:
        df_results.to_csv(savepath)
    return df_results, all_pred_probas


def independentTest(ratio, classifier, fea, label, save=0, savepath={}):
    """
    independent set test with a given dividing ratio
    (savepath: {'train_savepath': path; 'test_savepath': path})
    """
    x_train, x_test, y_train, y_test = train_test_split(fea, label, test_size=ratio)
    classifier.fit(x_train, y_train.astype('int'))
    y_pred = classifier.predict(x_test)
    y_pred_proba = classifier.predict_proba(x_test)[:, 1]
    sp, sn, acc, mcc = performanceAssessments(y_test, y_pred)[:4]
    auc = roc_auc_score( y_test, y_pred_proba)
    
    if save == 1:
        train_info = pd.DataFrame(np.column_stack((y_train, x_train)), columns=['label']+[str(i) for i in range(x_train.shape[1])])
        test_info = pd.DataFrame(np.column_stack((y_test, x_test)), columns=['label']+[str(i) for i in range(x_test.shape[1])])
        train_info.to_csv(savepath['train_path'])
        test_info.to_csv(savepath['test_path'])
        pd.Series([sp, sn, acc, mcc, auc]).to_csv(savepath['assessment_path'])
 
    return sp, sn, acc, mcc, auc, y_pred_proba









