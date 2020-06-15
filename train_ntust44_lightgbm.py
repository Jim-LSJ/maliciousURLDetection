import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, precision_score
import lightgbm as lgb
import lightgbm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.utils import shuffle, resample
import os

csv_path = "AllFeatures.csv"
cross_valid = 5
lightgbm_model_name = "Weight/ntust44"
lightgbm_importance_name = "Feature_Importance/ntust44"
result_name = "Result/ntust44.csv"
trainning_loss_name = "Result/ntust44_loss.png"
feature_rank_name = "Result/ntust44_feature_rank_split.csv"
feature_rank_gain = "Result/ntust44_feature_rank_gain.csv"
Label_name = "Label"
feature_name = ['RequestHeader', 'ResponseHeader', 'RedirectRequests',
                'Rank', 'CountryRank', 'ASN', 'ASNCountry', 'DomainCountry',
                'RequestsCount', 'Cookies', 'HTTPS', 'IPv6', 'Countries', 'Domains',
                'OutGoingLinks', 'JSGlobalVar', 'NameServers', 'CreationTime',
                'ExpirationTime', 'UpdateTime', 'UnsymmetricalTag', 'StringConcat',
                'DangerFunction', 'FunctionEval', 'FunctionExec', 'FunctionUnescape',
                'FunctionSearch', 'FunctionEscape', 'FunctionLink', 'WindowLocation',
                'iFrame', 'URLLength', 'URLDigitLetterRatio', 'URLSymbolRatio',
                'URLNonAlphanumRatio', 'DomainLength', 'DomainDigitRatio', 'DomainDot',
                'DomainHyphen', 'DomainNonAlphanumRatio', 'URLPathSlash', 'URLPathZero',
                'URLQueries', 'URLDoubleSlash'
                ]
seed = 10

def create_result_dir():
    if os.path.isdir("Weight/") == False:
        os.makedirs("Weight/")
    
    if os.path.isdir("Result/") == False:
        os.makedirs("Result/")
    
    if os.path.isdir("Feature_Importance/") == False:
        os.makedirs("Feature_Importance/")
        

def read_csv(csv_path):
    df = pd.read_csv(csv_path)    
    df = df.dropna()
    
    df = shuffle(df, random_state=0)
    df = df.reset_index(drop=True)
    
    return df

def split_train_and_test(df, cross_valid = 5):
    kf = KFold(n_splits=cross_valid)
    df_train_list = []
    df_test_list  = []

    df_features_train_list = []
    df_labels_train_list   = []
    df_features_test_list  = []
    df_labels_test_list    = []    
    
    for train_index, test_index in kf.split(df):
        df_train_list.append(df.iloc[train_index])
        df_test_list.append(df.iloc[test_index])
    
    # upsampling
    for i in range(len(df_train_list)):
        benign = df_train_list[i][df_train_list[i]['IsMalicious'] == False]
        malicious = df_train_list[i][df_train_list[i]['IsMalicious'] == True]
        print('Benign: {}, Malicious: {}'.format(len(benign), len(malicious)))
        malicious = resample(malicious, 
                            replace=True,     # sample with replacement
                            n_samples=len(benign),    # to match majority class
                            random_state=123) # reproducible results
        df_train_list[i] = pd.concat([benign, malicious], axis=0)

        df_features_train_list.append(df_train_list[i][feature_name])
        df_labels_train_list.append(df_train_list[i][Label_name])
        df_features_test_list.append(df_test_list[i][feature_name])
        df_labels_test_list.append(df_test_list[i][Label_name])

    return df_features_train_list, df_labels_train_list, df_features_test_list, df_labels_test_list

def lgbm_train(train_X, train_y, test_X, test_y):

    lgb_train = lgb.Dataset(train_X, train_y)  
    lgb_eval  = lgb.Dataset(test_X, test_y, reference=lgb_train) 
    eval_result = {}

    # specify your configurations as a dict  
    params = {
        'boosting_type': 'gbdt',
        'objective': 'cross_entropy',
        #'num_class': class_num,
        #'device': 'gpu',
        'metric': 'binary_logloss',    
        'num_leaves': 50,
        'max_depth': 10,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'min_data_in_leaf': 20,
        'bagging_freq': 5,
        'verbose': -1,
        'verbose_eval': False,
        'is_unbalance':True,
        #'scale_pos_weight': 10
    }
       
    gbm = lgb.train(params,  
                    lgb_train,  
                    num_boost_round=1000,  
                    valid_sets=[lgb_train, lgb_eval],
                    evals_result=eval_result,              
                    early_stopping_rounds=200
                   )    
    
    return gbm, eval_result

def save_result(precision_list, recall_list, fscore_list):
    with open(result_name, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(["num", "precision", "recall", "fscore"])
        for i in range(cross_valid):           
            spamwriter.writerow([str(i+1), precision_list[i], recall_list[i], fscore_list[i]])
            
        spamwriter.writerow(["ave", np.average(precision_list), np.average(recall_list), np.average(fscore_list)])
        spamwriter.writerow(["std", np.std(precision_list), np.std(recall_list), np.std(fscore_list)])       

def save_feature_importance(sorted_feature_importance_series, name=feature_rank_name):
    with open(name, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(["feature name", "number"])
        for i in range(len(sorted_feature_importance_series.index)):
            val = sorted_feature_importance_series.values[i]
            name  = sorted_feature_importance_series.index[i]
            spamwriter.writerow([name, val])             
        
        
def main():
    create_result_dir()
    
    df = read_csv(csv_path)
    df_features_train_list, df_labels_train_list, df_features_test_list, df_labels_test_list = split_train_and_test(df, cross_valid = 5)
    
    precision_list = []
    recall_list = []
    fscore_list = []
    feature_importance_list = []
    feature_importance_list2 = []
    
    for i in range(cross_valid):
        train_X = df_features_train_list[i]
        train_y = df_labels_train_list[i]
        test_X  = df_features_test_list[i]
        test_y  = df_labels_test_list[i]
        
        print("train num:" + str(len(train_y)))
        print("test  num:" + str(len(test_y)))
        print("train malicious num: " + str(sum(train_y)))
        print("test  malicious num: " + str(sum(test_y)))
        
        gbm, evals_result = lgbm_train(train_X, train_y, test_X, test_y)
        weight_path = lightgbm_model_name + "_" + str(i)
        gbm.save_model(weight_path)
        gbm = lgb.Booster(model_file = weight_path)
        pred_y = gbm.predict(test_X, num_iteration=gbm.best_iteration)
        pred_y = np.round_(pred_y, 0)
    
        precision = precision_score(test_y, pred_y)
        recall = recall_score(test_y, pred_y)
        fscore = f1_score(test_y, pred_y)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        
        ax = lightgbm.plot_importance(gbm, max_num_features=15, figsize=(12,12))
        file_path = lightgbm_importance_name + "_split_" + str(i) + ".png"
        plt.savefig(file_path)
        feature_importance_list.append(gbm.feature_importance())

        ax = lightgbm.plot_importance(gbm, max_num_features=15, figsize=(12,12), importance_type="gain")
        file_path = lightgbm_importance_name + "_gain_" + str(i) + ".png"
        plt.savefig(file_path)
        feature_importance_list2.append(gbm.feature_importance(importance_type='gain'))
        
        ax = lightgbm.plot_metric(evals_result, metric='binary_logloss')
        file_path = trainning_loss_name
        plt.savefig(file_path)
    
    feature_importance_series = pd.Series(np.sum(feature_importance_list, axis=0), index=train_X.columns)
    sorted_feature_importance_series = feature_importance_series.sort_values(ascending=False)
    """
    for i in range(cross_valid):
        feature_list = []
        for feature in sorted_feature_importance_series.index:
            feature_list.append(feature)
            train_X = df_features_train_list[i]
            train_y = df_labels_train_list[i]
            test_X  = df_features_test_list[i]
            test_y  = df_labels_test_list[i]
            
            train_X = [feature_list]
            test_X  = [feature_list]
    """     
    save_result(precision_list, recall_list, fscore_list)
    save_feature_importance(sorted_feature_importance_series)
    print("ave precision:" + str(np.average(precision_list)))
    print("ave recall:"    + str(np.average(recall_list)))
    print("ave fscore:"    + str(np.average(fscore_list)))
    print(sorted_feature_importance_series)
    
    feature_importance_series = pd.Series(np.sum(feature_importance_list2, axis=0), index=train_X.columns)
    sorted_feature_importance_series = feature_importance_series.sort_values(ascending=False)
    save_feature_importance(sorted_feature_importance_series, name=feature_rank_gain)
    print(sorted_feature_importance_series)
main()
        
