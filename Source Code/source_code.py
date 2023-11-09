# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 23:42:54 2022

@author: HE
"""

# %% importing 
import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import xgboost as xgb
import timeit 
import datetime 
import os 
import json 

# %% function defining
# clock function 
def clock(func):
    def clocked(*args, **kwargs):
        name = func.__name__
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'----------------------------Start {name:s} at {now:s}. -------------------------------') 
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        print(f'----------------------------Finish {name:s} in {elapsed: .6f}s! -------------------------------') 
        return result
    return clocked


# prepare data as DMatrix type 
@clock
def prepare_data_dm(data: pd.DataFrame) -> xgb.DMatrix: 
    drop_list = [
        'log_id', 'user_id', 'pt_d'
        # 'ad_click_list_v001','ad_click_list_v002','ad_click_list_v003',
        # 'ad_close_list_v001','ad_close_list_v002','ad_close_list_v003',
        # 'label', 'u_newsCatInterestsST'
        ]
    data=data.drop(columns=drop_list)
    
    to_cate_list = [
        'ad_click_list_v001','ad_click_list_v002','ad_click_list_v003',
        'ad_close_list_v001','ad_close_list_v002','ad_close_list_v003',
        'u_newsCatInterestsST', 
        # 'user_id', 
        'gender', 'residence', 'city', 'series_dev', 'series_group', 
        'device_name', 'net_type', 'task_id', 'adv_id', 'creat_type_cd', 
        'adv_prim_id', 'inter_type_cd', 'slot_id', 'site_id', 'spread_app_id', 
        'hispace_app_tags', 'app_second_class'
        ]
    data[to_cate_list] = data[to_cate_list].astype('category')
    
    
    if 'label' in data.columns: 
        drop_list = [
            'label'
            ] 
    
    
        return xgb.DMatrix(
            data=data.drop(columns=drop_list), 
            label=data['label'], 
            # weight=,  
            enable_categorical=True
            )
    else: 
        return xgb.DMatrix(
            data=data, 
            enable_categorical=True 
            )


# Cross Validation 
@clock 
def cross_validation(param: dict, train_data_dm: xgb.DMatrix, num_round: int): 
    cv_res_path = os.path.join(
        'cv_result', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        )
    os.makedirs(cv_res_path) 
    with open(os.path.join(cv_res_path, 'params.json'), 'w') as filepath: 
        json.dump(param, filepath) 
    cv_result = xgb.cv(
        param, train_data_dm, nfold=10, seed=0, num_boost_round=num_round
        )
    cv_result.to_csv(os.path.join(cv_res_path, 'result.csv')) 
   

# compare auc results  
@clock 
def comp_cv_results(ifplot: bool) -> list: 
    cv_auc_comp = pd.DataFrame() 
    for cv_res_dir in os.listdir('cv_result'): 
        if os.path.isdir(os.path.join('cv_result', cv_res_dir)): 
            try: 
                cv_auc = pd.read_csv(
                    os.path.join('cv_result', cv_res_dir, 'result.csv')
                    ) 
            except FileNotFoundError: 
                continue 
            cv_auc = cv_auc['test-auc-mean'] 
            cv_auc.name = cv_res_dir 
            cv_auc_comp = cv_auc_comp.append(cv_auc) 
    cv_auc_comp = cv_auc_comp.transpose() 

    if ifplot: 
        cv_auc_comp.plot()
        plt.savefig(os.path.join(
            'cv_result', 
            'comp_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            )) 
    
    max_param = cv_auc_comp.max().idxmax() 
    max_num_round = cv_auc_comp[max_param].idxmax() 
    max_auc = cv_auc_comp[max_param][max_num_round] 
    with open(os.path.join(
            'cv_result', max_param, 'params.json'
            ), 'r') as filepath: 
        max_param = json.load(filepath)
    return [max_param, max_num_round, max_auc] 


# train boost model 
@clock 
def train_xgb_model(param:dict, train_data_dm: xgb.DMatrix, num_round: int) -> xgb.Booster: 
    bst = xgb.train(params=param, dtrain=train_data_dm, num_boost_round=num_round)
    return bst 


# predict 
@clock 
def predict(valid_data_DMatrix: xgb.DMatrix, bst:xgb.Booster) -> np.ndarray: 
    return  bst.predict(valid_data_DMatrix) 


# %% main
if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_dir', type=str, help='input file dir') 
    parser.add_argument('--output_dir', type=str, help='output file dir') 
    args, unknown = parser.parse_known_args() 
    if len(unknown) != 0: 
        print('unknown args:%s', unknown) 
    
    input_dir = args.input_dir 
    output_dir = args.output_dir 

    train_data_ads = pd.read_csv(os.path.join(input_dir, 'train/train_data_ads.csv'))
    # prepare data
    # train_data_ads = train_data_ads.sample(n=3500000) 
    train_data_ads_dm = prepare_data_dm(train_data_ads) 
    
    # del train_data_ads 

    # cross validation 
    # set booster parameters 
    _ = xgb.config_context()
    count_pos_neg = train_data_ads.groupby('label').count()['log_id']
    param = {
        # General Parameters 
        'booster':'gbtree', # ! 'dart' for long time training 
        'verbosity':1, # ! 0 silent, 1 warning, 2 info, 3 debug 
        'validate_parameters':True, # True to perform validation of input parameters to check whether a parameter is used or not
        'disable_default_eval_metric':True, # Flag to disable default metric. Set to 1 or true to disable.
        # Parameters to Tree Booster 
        'eta':0.3, # ! learning rate 调小以降速
        'gamma':0, # Minimum loss reduction required to make a further partition on a leaf node of the tree. 调大以降速
        'max_depth':1, # ! max tree depth 调小以降速
        'min_child_weight':1, # ? Minimum sum of instance weight (hessian) needed in a child. 
        'subsample':1, # ! ratio of subsampling data in every iteraiton. prefer >0.5 调小以降速
        'colsample_bynode':1, # 1/train_data_ads_dm.num_col()**0.5, # ratio of subsampling columns in every node(split) like RF 调小以降速
        'lambda':1, # ! parameter of L2 regularization on weights 调大以降速
        'alpha':0, # ! L1, weights 
        'tree_method':'auto', # ! 'exact', 'approx', 'hist', 'gpu_hist'. try 'hist' to get faster training 
        'scale_pos_weight':count_pos_neg[0]/count_pos_neg[1], # !  control the balance of pos and neg weights
        'max_bin':256, # ! max number of bin in histogram methods, larger for more accurate result at the cost of longer computation time 
        'num_parallel_tree':1, # number of trees in RF as base predicitor 
        # Learning Task Parameters 
        'objective':'binary:logistic', # output probabilities 
        'eval_metric':['logloss', 'auc'] # Evaluation metrics for validation data
    }
    num_round = 31

    # cross_validation(param, train_data_ads_dm, num_round)  
    
    # max_param, max_num_round, max_auc = comp_cv_results(ifplot=True) 

    # train 
    # best_bst = train_xgb_model(max_param, train_data_ads_dm, max_num_round) 
    best_bst = train_xgb_model(param, train_data_ads_dm, num_round) 
    
    # predict 
    del train_data_ads_dm 
    test_data_ads = pd.read_csv(os.path.join(input_dir, 'test/test_data_ads.csv'))
    test_data_ads_dm = prepare_data_dm(test_data_ads) 
    pctr = predict(test_data_ads_dm, best_bst) 
    
    result = pd.DataFrame() 
    result['log_id'] = test_data_ads['log_id'] 
    result['pctr'] = pctr 
    result['log_id'] = result['log_id'].astype(int) 
    result.to_csv(os.path.join(output_dir, 'submission.csv'), float_format='%.6f', index=False) 
