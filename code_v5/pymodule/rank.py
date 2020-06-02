from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import lightgbm as lgb
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# def rank_xgb(X, Y):
#     clf = XGBClassifier()
#     clf.fit(X, Y)
#
#     return clf

def rank_rf(X, Y):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=(max(1, cpu_count() - 5)))
    clf.fit(X, Y)

    return clf

def train_model_rf(feature_all, recall_rate, hot_list, valid=0.2, topk=50):
    from sklearn.ensemble import RandomForestClassifier
    print('------- 训练模型 -----------')
    # todo 只有正样本？
    train_data = feature_all[feature_all['train_flag'] == 'train']
    test_data = feature_all[feature_all['train_flag'] == 'test']

    '''训练集切成两个集合，一个训练 一个测试'''
    df_user = pd.DataFrame(list(set(train_data['user_id'])))
    df_user.columns = ['user_id']

    df = df_user.sample(frac=1.0)
    cut_idx = int(round(valid * df.shape[0]))
    df_train_0, df_train_1 = df.iloc[:cut_idx], df.iloc[cut_idx:]

    train_data_0 = df_train_0.merge(train_data, on=['user_id'], how='left')
    train_data_1 = df_train_1.merge(train_data, on=['user_id'], how='left')

    # f0-f128特征：embedding的特征
    f_col = [c for c in feature_all.columns if c not in ['train_flag', 'label', 'user_id', 'item_similar']]
    f_label = 'label'

    X0 = train_data_0[f_col].values
    y0 = train_data_0[f_label].values

    X1 = train_data_1[f_col].values
    y1 = train_data_1[f_label].values

    '''
    测试集，待分类模型进行是否购买预测
    TODO 只需要对qtime中的user进行购买预测即可，不需要对整个test集合进行预测
    '''
    X_pred = test_data[f_col].values

    # clf = RandomForestClassifier(n_estimators=100)
    clf = RandomForestClassifier(n_estimators=100)
    # clf = clf.fit(X0, y0)
    clf.fit(X0, y0)

    # --------- train error --------------
    y0_pre = clf.predict(X0)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print('lr train error:', accuracy_score(y0, y0_pre), precision_score(y0, y0_pre), recall_score(y0, y0_pre))

    print("------------- eval -------------")
    train_eval = train_data_1[['user_id', 'item_similar', 'label']]
    len_hot = len(hot_list)
    high_half_item, low_half_item = hot_list[:len_hot // 2], hot_list[len_hot // 2:]
    train_eval['half'] = train_eval['item_similar'].map(lambda x: 1 if x in low_half_item else 0)

    y1_pred = clf.predict_proba(X1)[:, 1]
    train_eval['pred_prob'] = y1_pred

    train_eval['rank'] = train_eval.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
    train_eval_out = train_eval[train_eval['rank'] <= topk]

    len_user_id = len(set(train_eval.user_id))

    # todo 指标计算可能有问题
    hitrate_50_full = np.sum(train_eval_out['label']) / len_user_id * recall_rate
    hitrate_50_half = np.sum(train_eval_out['label'] * train_eval_out['half']) / len_user_id * recall_rate
    ndcg_50_full = np.sum(train_eval_out['label'] / np.log2(train_eval_out['rank'] + 2.0) * recall_rate)
    ndcg_50_half = np.sum(
        train_eval_out['label'] * train_eval_out['half'] / np.log2(train_eval_out['rank'] + 2.0) * recall_rate)

    print("------------- eval result -------------")
    print("hitrate_50_full : ", hitrate_50_full, 'ndcg_50_full : ', ndcg_50_full, '\n')
    print("hitrate_50_half : ", hitrate_50_half, 'ndcg_50_half : ', ndcg_50_half, '\n')
    print("------------- eval result -------------")

    print("------------- predict -------------")
    test_data_out = test_data[['user_id', 'item_similar']]
    test_y_pred = clf.predict_proba(X_pred)[:, 1]
    test_data_out['pred_prob'] = test_y_pred

    test_data_out['rank'] = test_data_out.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
    test_data_out = test_data_out[test_data_out['rank'] <= topk]
    test_data_out = test_data_out.sort_values(['rank'])

    submit = test_data_out.groupby(['user_id'])['item_similar'].agg(lambda x: ','.join(list(x))).reset_index()

    print("------------- assert -------------")
    for i, row in submit.iterrows():
        txt_item = row['item_similar'].split(',')
        assert len(txt_item) == topk
    return submit


def train_model_lgb(feature_all, recall_rate, hot_list, valid=0.2, topk=50, num_boost_round=1500,
                    early_stopping_rounds=100):
    print('------- 训练模型 -----------')
    train_data = feature_all[feature_all['train_flag'] == 'train']
    test_data = feature_all[feature_all['train_flag'] == 'test']

    #     print('------- 数据 -----------')
    #     print(len(train_data))
    #     print(len(test_data))
    #     print(train_data)
    #     print(test_data)
    #     print('------- 数据 -----------')

    df_user = pd.DataFrame(list(set(train_data['user_id'])))
    df_user.columns = ['user_id']

    df = df_user.sample(frac=1.0)
    cut_idx = int(round(valid * df.shape[0]))
    df_train_0, df_train_1 = df.iloc[:cut_idx], df.iloc[cut_idx:]

    train_data_0 = df_train_0.merge(train_data, on=['user_id'], how='left')
    train_data_1 = df_train_1.merge(train_data, on=['user_id'], how='left')

    train_data_0_group = list(train_data_0.groupby(['user_id']).size())
    train_data_1_group = list(train_data_1.groupby(['user_id']).size())

    f_col = [c for c in feature_all.columns if c not in ['train_flag', 'label', 'user_id', 'item_similar']]
    f_label = 'label'

    X0 = train_data_0[f_col].values
    y0 = train_data_0[f_label].values

    X1 = train_data_1[f_col].values
    y1 = train_data_1[f_label].values

    X_pred = test_data[f_col].values

    lgb_train = lgb.Dataset(X0, y0, group=train_data_0_group, free_raw_data=False)
    lgb_valid = lgb.Dataset(X1, y1, group=train_data_1_group, free_raw_data=False)

    params = {
        'objective': 'lambdarank',
        'boosting_type': 'gbdt',
        # 'num_trees' : -1,
        'num_leaves': 128,
        'feature_fraction': 1,
        'bagging_fraction': 1,
        # 'max_bin' : 256,
        'learning_rate': 0.8,
        'metric': 'MAP'
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_valid],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=10,

                    )

    print("------------- eval -------------")
    train_eval = train_data_1[['user_id', 'item_similar', 'label']]
    len_hot = len(hot_list)
    high_half_item, low_half_item = hot_list[:len_hot // 2], hot_list[len_hot // 2:]
    train_eval['half'] = train_eval['item_similar'].map(lambda x: 1 if x in low_half_item else 0)

    y1_pred = gbm.predict(X1)
    train_eval['pred_prob'] = y1_pred

    train_eval['rank'] = train_eval.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
    train_eval_out = train_eval[train_eval['rank'] <= topk]

    len_user_id = len(set(train_eval.user_id))

    hitrate_50_full = np.sum(train_eval_out['label']) / len_user_id * recall_rate
    hitrate_50_half = np.sum(train_eval_out['label'] * train_eval_out['half']) / len_user_id * recall_rate
    ndcg_50_full = np.sum(train_eval_out['label'] / np.log2(train_eval_out['rank'] + 2.0) * recall_rate)
    ndcg_50_half = np.sum(
        train_eval_out['label'] * train_eval_out['half'] / np.log2(train_eval_out['rank'] + 2.0) * recall_rate)

    print("------------- eval result -------------")
    print("hitrate_50_full : ", hitrate_50_full, 'ndcg_50_full : ', ndcg_50_full, '\n')
    print("hitrate_50_half : ", hitrate_50_half, 'ndcg_50_half : ', ndcg_50_half, '\n')
    print("------------- eval result -------------")

    print("------------- predict -------------")
    test_data_out = test_data[['user_id', 'item_similar']]
    test_y_pred = gbm.predict(X_pred)
    test_data_out['pred_prob'] = test_y_pred

    test_data_out['rank'] = test_data_out.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
    test_data_out = test_data_out[test_data_out['rank'] <= topk]
    test_data_out = test_data_out.sort_values(['rank'])

    submit = test_data_out.groupby(['user_id'])['item_similar'].agg(lambda x: ','.join(list(x))).reset_index()

    print("------------- assert -------------")
    for i, row in submit.iterrows():
        txt_item = row['item_similar'].split(',')
        assert len(txt_item) == topk
    return submit