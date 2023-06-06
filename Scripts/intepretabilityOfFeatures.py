import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from binary_classifiers import *
from sklearn.metrics import roc_auc_score, roc_curve
from reproduceResults import *
import matplotlib.pyplot as plt
import seaborn as sns


def sorted_FScoresofAllFeature(shuffled_resampled_LabelFeature_df_filepath, 
                               savepath=None, save=0):
    """
    F-Score value of each topological features  
    """
    LabelFeature_df = pd.read_csv(shuffled_resampled_LabelFeature_df_filepath,
                                  engine='python')
    P_features = LabelFeature_df[LabelFeature_df['label'] == 1].values[:, 1:]  # 所有正样本的特征矩阵(第一列是标签)
    N_features = LabelFeature_df[LabelFeature_df['label'] == 0].values[:, 1:]  # 所有负样本的特征矩阵
    
    f_scores = np.zeros(P_features.shape[1], )  
    for i in range(P_features.shape[1]):
        phi_i_plus = np.mean(P_features[:, i])  # all_P_features[:, i]是一个(n_Psamples,)的数组
        phi_i_minus = np.mean(N_features[:, i])
        phi_i_ = np.mean(np.concatenate([P_features[:, i], N_features[:, i]]))

        numerator_part1 = np.power(phi_i_plus - phi_i_, 2)
        numerator_part2 = np.power(phi_i_minus - phi_i_, 2)
        denominator_part1 = np.std(P_features[:, i], ddof=1)
        denominator_part2 = np.std(N_features[:, i], ddof=1)
        if denominator_part1 + denominator_part2 != 0:
            f_scores[i] = (numerator_part1 + numerator_part2)/(denominator_part1 + denominator_part2)
            
    dict_f_scores = {}
    for i in range(f_scores.shape[0]):
        dict_f_scores[i] = f_scores[i]
    df_f_scores = pd.DataFrame.from_dict(dict_f_scores, orient='index', columns=['F-Score'])
    df_f_scores.sort_values(by='F-Score', inplace=True, ascending=False)
    if save == 1:
        df_f_scores.to_csv(savepath)
    return df_f_scores


def computeOptimalNumFeatures_plotIFS(shuffled_resampled_LabelFeature_df_filepath, classifier, show_fig=1, save_fig=0, savepath=None):
    """
    compute the optimal number of features in terms of F-Scores
    """
    df_f_scores = sorted_FScoresofAllFeature(shuffled_resampled_LabelFeature_df_filepath, save=0)
    shuffled_resampled_LabelFeature = pd.read_csv(shuffled_resampled_LabelFeature_df_filepath, engine='python')
    features, labels = shuffled_resampled_LabelFeature.values[:, 1:], shuffled_resampled_LabelFeature['label'].values
    
    kf = KFold(n_splits=10, shuffle=True)
    # record each performance measurements of each feature set (mean+std)
    all_probas = {}
    mean_sps, mean_sns, mean_accs, mean_mccs, mean_aucs, mean_pres = [], [], [], [], [], []
    std_sps, std_sns, std_accs, std_mccs, std_aucs, std_pres= [], [], [], [], [], []
    
    for i in range(df_f_scores.index.shape[0]):
        selected_features_ids = df_f_scores.index[:(i+1)].values
        selected_features = np.r_[[features[:, id] for id in selected_features_ids]].T
        sp, sn, acc, mcc, auc, pre, proba = [], [], [], [], [], [], {}
        j = 0
        for train_index, test_index in list(kf.split(selected_features)):
            j+=1
            train_features, test_features = selected_features[train_index], selected_features[test_index]
            train_label, test_label = labels[train_index], labels[test_index]
            classifier.fit(train_features, train_label)
            test_pred = classifier.predict(test_features)
            test_pred_proba = classifier.predict_proba(test_features)[:, 1]
            # print(y_pred, y_test)
            
            temp_auc = roc_auc_score(test_label, test_pred_proba)
            auc.append(temp_auc)
            temp_sp, temp_sn, temp_acc, temp_mcc, temp_pre = performanceAssessments(test_pred, test_label)[:5]
            sp.append(temp_sp); sn.append(temp_sn); acc.append(temp_acc); mcc.append(temp_mcc); pre.append(temp_pre)
            proba[str(j)]={'y_true': test_label.tolist(), 'y_pred_proba': test_pred_proba.tolist()}
        mean_sps.append(np.mean(np.array(sp))); std_sps.append(np.std(np.array(sp)))
        mean_sns.append(np.mean(np.array(sn))); std_sns.append(np.std(np.array(sn)))
        mean_accs.append(np.mean(np.array(acc))); std_accs.append(np.std(np.array(acc)))
        mean_mccs.append(np.mean(np.array(mcc))); std_mccs.append(np.std(np.array(mcc)))
        mean_aucs.append(np.mean(np.array(auc))); std_aucs.append(np.std(np.array(auc)))
        mean_pres.append(np.mean(np.array(pre))); std_pres.append(np.std(np.array(pre)))
        all_probas[str(i)] = proba
        # all_mccs.append(np.mean(np.array(mcc)))
    mean_dic = {'sp': mean_sps, 'sn': mean_sns, 'acc': mean_accs, 'mcc': mean_mccs, 'auc': mean_aucs, 'pre': mean_pres}
    std_dic = {'sp': std_sps, 'sn': std_sns, 'acc': std_accs, 'mcc': std_mccs, 'auc': std_aucs, 'pre': std_pres}
    optimal_feature_nums = np.where(np.array(mean_mccs) == np.max(np.array(mean_mccs)))
    
    plt.rcParams['figure.figsize']=(9, 9)
    plt.rcParams['savefig.dpi'] = 300
    plt.plot(np.arange(1, features.shape[1]+1), mean_mccs)
    x0, y0 = optimal_feature_nums[0][0], np.around(np.max(np.array(mean_mccs)), 6)  
    plt.scatter(x0+1, y0, marker='.', c="r", s=150)
    plt.text(x0+1, y0, (x0+1, y0), size=18)
    plt.xlabel('Numbers of feature', size=20)
    plt.ylabel('MCC Value', size=20)
    if show_fig == 1:
        plt.show()
    if save_fig == 1:
        plt.savefig(savepath)
    return optimal_feature_nums, mean_dic, std_dic, all_probas


def plot_multi_rocs(n, pre_proba, colors, save=False):

    plt.figure(figsize=(10, 10), dpi=300)
 
    for i in range(n):
        y_true = pre_proba[str(i)+"_true"].values
        y_proba = pre_proba[str(i)+"_proba"].values
        colorname = colors[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = format(roc_auc_score(y_true, y_proba), '.4f')
        
        plt.plot(fpr, tpr, lw=1, label= str(i)+'-fold('+str(auc_score)+')', color = colorname)
        plt.plot([0, 1], [0, 1], '--', lw=1, color = 'grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(fontsize=18); plt.yticks(fontsize=18)
        plt.xlabel('False Positive Rate', fontsize=22)
        plt.ylabel('True Positive Rate', fontsize=22)
        # plt.title('ROC Curve', fontsize=25)
        plt.legend(loc='lower right',fontsize=18)
 
    if save:
        plt.savefig('multi_models_roc.png')
        
    return plt.show()


def PiePlot(type_fractions:dict, optimal_features, F_score_filepath, save_fig=0, savepath=None):
    """
    pie plot
    """
    count_list = [0 for key in type_fractions.keys()]
    key_list = [key for key in type_fractions.keys()]
    selected_F_scores = pd.read_csv(F_score_filepath, engine='python').iloc[:optimal_features+1]
    for id in selected_F_scores.index:
        for i in range(len(key_list)):
            key = key_list[i]
            vals = type_fractions[key]
            if (id >= vals[0]) and (id <= vals[1]):
                count_list[i] += 1
    
    plt.figure(figsize=(10, 10), dpi=300)

    patches,l_text,p_text = plt.pie(np.array(count_list), autopct='%1.1f%%', 
                                    labels=[key for key in type_fractions.keys()],
                                    colors=['yellowgreen', 'gold', 'lightskyblue', 'lightcoral'])
    for t in l_text:
        t.set_size(8)
    for t in p_text:
        t.set_size(8)

    plt.axis('equal')
    if save_fig == 1:  
        plt.savefig(savepath)
    return plt.show()


def BarPlot(type_fractions:dict, optimal_features, F_score_filepath, xticks:list, save_fig=0, savepath=None):
    """
    bar plot
    """

    total_num_features = np.array([type_fractions[key][1]-type_fractions[key][0]+1 
                                    for key in type_fractions.keys()])  # 每类特征的总个数
    count_list = [0 for key in type_fractions.keys()]
    key_list = [key for key in type_fractions.keys()]
    selected_F_scores = pd.read_csv(F_score_filepath, engine='python').iloc[:optimal_features+1]
    for id in selected_F_scores.index:
        for i in range(len(key_list)):
            key = key_list[i]
            vals = type_fractions[key]
            if (id >= vals[0]) and (id <= vals[1]):
                count_list[i] += 1
                
    bar_x1 = [1.5+3*i for i in range(len(type_fractions.keys()))]
    bar_x2 = [x+1 for x in bar_x1]
    
    fig = plt.figure(figsize=(12, 8), dpi=300)
    bar1 = plt.bar(bar_x1, total_num_features, width=1, color=sns.xkcd_rgb['dark sky blue'], label="total num", alpha=0.6)
    bar2 = plt.bar(bar_x2, count_list, width=1, color=sns.xkcd_rgb['coral'], label='optimal num', alpha=0.6)
    plt.bar_label(bar1, fontsize=8)
    plt.bar_label(bar2, fontsize=8)
    plt.xticks([x+.5 for x in bar_x1], xticks, fontsize=8)
    plt.yticks(np.arange(0, 300, 70), fontsize=6)
    plt.ylabel('number of features', size=8)
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=8)
    if save_fig == 1:  
        plt.savefig(savepath)
    
    return plt.show()