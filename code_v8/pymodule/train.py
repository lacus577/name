import os
import time
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm
# from sklearn.externals import joblib
from sklearn.decomposition import PCA
import warnings
import pickle
from sklearn.metrics import roc_auc_score

from pymodule import conf
from pymodule.rank import train_model_rf, train_model_lgb, rank_rf, rank_xgb
from pymodule.eval import metrics_recall
from pymodule import recall
from pymodule.featuring import matrix_word2vec_embedding, get_train_test_data, \
    get_user_features, train_test_split, cal_user_item_sim, cal_txt_img_sim, \
    cal_click_sim, cal_item_of_user_def, cal_statistic_features, \
    cal_item_distance, cal_user_click_num, cal_total_statistic_features, process_after_featuring, \
    get_samples_v1, do_featuring, get_recall_sample
from pymodule import utils
from pymodule.eval import evaluate, make_answer, my_eval
from pymodule.recall import get_sim_item, recommend

warnings.filterwarnings("ignore")

''' 全局变量 '''
item_txt_embedding_dim = 128
item_img_embedding_dim = 128


def read_features(phase, is_open_train_recall):
    train_data = pd.read_csv(
        './cache/features_cache/train_features_phase_{}.csv'.format(phase),
        dtype={'user_id': np.str, 'item_id': np.str}
    )
    valid_data = pd.read_csv(
        './cache/features_cache/valid_features_phase_{}.csv'.format(phase),
        dtype={'user_id': np.str, 'item_id': np.str}
    )
    test_user_recall_df = pd.read_csv(
        './cache/features_cache/test_user_recall_features_phase_{}.csv'.format(phase),
        dtype={'user_id': np.str, 'item_id': np.str}
    )
    if is_open_train_recall:
        train_user_recall_df = pd.read_csv(
            './cache/features_cache/train_user_recall_features_phase_{}.csv'.format(phase),
            dtype={'user_id': np.str, 'item_id': np.str}
        )
        return train_data, valid_data, train_user_recall_df, test_user_recall_df

    return train_data, valid_data, test_user_recall_df


if __name__ == '__main__':
    # 数据组织
    item_info_df = utils.read_item_user_info()
    user_info_df = utils.read_user_info()

    all_phase_click = utils.read_all_phase_click()
    all_phase_click_no_qtime = all_phase_click[all_phase_click['train_or_test'] != 'predict']

    # 取k个点击作为候选正样本，当召回结果命中的时候才是正样本(控制了随机种子，每次采样结果都一样)
    candidate_positive_sample_df = utils.get_candidate_positive_samples(all_phase_click_no_qtime)

    # 全量点击删除候选正样本 -- 模拟qtime预测
    all_phase_click_no_qtime = all_phase_click_no_qtime.append(candidate_positive_sample_df)
    all_phase_click_no_qtime = all_phase_click_no_qtime.drop_duplicates(['user_id', 'item_id', conf.org_time_name], keep=False)
    all_phase_click_no_qtime = all_phase_click_no_qtime.sort_values(['user_id', 'item_id']).reset_index(drop=True)

    hot_df = all_phase_click_no_qtime.groupby('item_id')['user_id'].count().reset_index()
    hot_df.columns = ['item_id', 'item_deg']
    hot_df = hot_df.sort_values('item_deg', ascending=False).reset_index()

    # 相似度矩阵构建
    if os.path.exists(conf.sim_list_path):
        item_sim_list = pickle.load(open(conf.sim_list_path, 'rb'))
    else:
        item_sim_list, user_item = \
            recall.get_sim_item_5164(all_phase_click_no_qtime, 'user_id', 'item_id', use_iif=False)

        pickle.dump(item_sim_list, open(conf.sim_list_path, 'wb'))
        # pickle.dump(user_item, open('./cache/features_cache/user_item', 'wb'))

    if conf.is_samples_cached:
        sample_df = pd.read_csv(conf.samples_cache_path, dtype={'user_id': np.str, 'item_id': np.str})
        if conf.subsampling:
            sample_df = sample_df[sample_df['user_id'].isin(all_phase_click_no_qtime['user_id'])]

        print('loaded samples, shape:{}'.format(sample_df.shape))
    else:
        print('getting samples ...')
        if os.path.exists(conf.total_user_recall_path):
            candidate_recall_df = pd.read_csv(conf.total_user_recall_path, dtype={'user_id': np.str, 'item_id': np.str})
        else:
            # 候选正样本召回
            # tmp_df = candidate_positive_sample_df.copy(deep=True)
            # tmp_df.loc[:, 'user_id'] = tmp_df['user_id'].astype(np.int)
            # tmp_df.loc[:, 'item_id'] = tmp_df['item_id'].astype(np.int)
            _, recom_item = recall.items_recommod_5164(
                candidate_positive_sample_df, item_sim_list, all_phase_click_no_qtime, list(hot_df['item_id'])
            )
            candidate_recall_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
            candidate_recall_df.to_csv(conf.total_user_recall_path, index=False)
            # 取top50，待top50调试OK之后再放开
        candidate_recall_df = candidate_recall_df.sort_values('sim').reset_index(drop=True)
        candidate_recall_df = candidate_recall_df.groupby('user_id').head(conf.recall_num).reset_index(drop=True)

        # 正样本确认， 负样本构建
        candidate_positive_sample_df['label'] = 1
        tmp_total_df = candidate_recall_df.merge(candidate_positive_sample_df, on=['user_id', 'item_id'], how='left')
            # 命中user及对应命中点击item
        positive_sample_df = tmp_total_df[tmp_total_df['label'] == 1]
            # TODO 暂时不对负样本数量做控制，正负样本比可能达到1：50
        sample_df = tmp_total_df[tmp_total_df['user_id'].isin(positive_sample_df['user_id'])]
        sample_df.loc[sample_df['label'] != 1, 'label'] = 0
        if sample_df.shape[0] == 0:
            raise Exception('召回结果没有任何命中！')
        sample_df.to_csv(conf.samples_cache_path, index=False)

    ''' 训练/验证集 特征 '''
    if conf.is_feature_cached:
        feature_df = pd.read_csv(
            conf.features_cache_path,
            dtype={'user_id': np.str, 'item_id': np.str}
        )

        if conf.subsampling:
            feature_df = feature_df[feature_df['user_id'].isin(all_phase_click_no_qtime['user_id'])]
        print('loaded features, shape:{}'.format(feature_df.shape))
    else:
        feature_df = do_featuring(
            all_phase_click_no_qtime, sample_df, hot_df, conf.process_num,
            item_txt_embedding_dim, is_recall=False, feature_caching_path=conf.features_cache_path,
            itemcf_score_maxtrix=item_sim_list
        )

    assert sample_df.shape[0] == feature_df.shape[0]
    assert len(set(sample_df['user_id'])) == len(set(feature_df['user_id']))

    # 加入user属性
    feature_df = feature_df.merge(user_info_df, on='user_id', how='left')

    train_auc = valid_auc = 0
    pre_score_arr = np.zeros(5).reshape(-1, )
    rank_score_arr = np.zeros(5).reshape(-1, )
    for i in range(conf.k):
        ''' 训练集/验证集划分 '''
        train_df, valid_df = train_test_split(feature_df)

        qtime_user_df = all_phase_click[all_phase_click['train_or_test'] == 'predict']
        print(
            '训练集命中{}个qtime的user.'.format(
                len(set(train_df[train_df['user_id'].isin(qtime_user_df['user_id'])]['user_id']))
            )
        )

        train_x = train_df[train_df.columns.difference(['user_id', 'item_id', 'label'])].values
        train_y = train_df['label'].values

        valid_df = valid_df.sort_values('sim').reset_index(drop=True)
        valid_x = valid_df[valid_df.columns.difference(['user_id', 'item_id', 'label'])].values
        valid_y = valid_df['label'].values

        ''' 模型训练 '''
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('------------------------ 模型训练 start time:{}'.format(time_str))
        # submit = train_model_lgb(feature_all, recall_rate=hit_rate, hot_list=hot_list, valid=0.2, topk=50, num_boost_round=1, early_stopping_rounds=1)
        # submit = train_model_rf(train_test, recall_rate=1, hot_list=hot_list, valid=0.2, topk=50)
        # model = rank_rf(train_x, train_y)
        model = rank_xgb(train_x, train_y)
        one_train_auc = roc_auc_score(train_y, model.predict_proba(train_x)[:, 1])
        train_auc += one_train_auc
        print('train set: auc:{}'.format(one_train_auc))
        with open('./cache/rf.pickle', 'wb') as f:
            pickle.dump(model, f)

        ''' 模型验证 '''
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('------------------------ 模型验证 start time:{}'.format(time_str))
        pre_y = model.predict_proba(valid_x)[:, 1]
        one_valid_auc = roc_auc_score(valid_y, pre_y)
        valid_auc += one_valid_auc
        print('valid set: auc:{}'.format(one_valid_auc))
        answer = make_answer(valid_df[valid_df['label'] == 1], hot_df, phase=1)

        print('排序前score:')
        pre_score_arr += my_eval(list(valid_df['sim']), valid_df, answer)
        print('排序后score：')
        rank_score_arr += my_eval(pre_y, valid_df, answer)

    print('{}次留出验证，平均train auc:{}，平均valid auc:{}'.format(conf.k, train_auc / conf.k, valid_auc / conf.k))
    print('平均排序前score:{}'.format(pre_score_arr / conf.k))
    print('平均排序后score:{}'.format(rank_score_arr / conf.k))

    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('------------------------ underexpose_test_qtime 预测 start time:{}'.format(time_str))
    submit_all = pd.DataFrame()
    for phase in range(0, conf.now_phase + 1):
        print('----------------------- phase:{} -------------------------'.format(phase))
        if conf.is_recall_cached:
            one_phase_recall_item_df = \
                pd.read_csv(conf.recall_cache_path.format(phase), dtype={'user_id': np.int, 'item_id': np.int})
            one_phase_recall_item_df.loc[:, 'user_id'] = one_phase_recall_item_df['user_id'].astype(np.str)
            one_phase_recall_item_df.loc[:, 'item_id'] = one_phase_recall_item_df['item_id'].astype(np.str)
            if conf.subsampling:
                one_phase_recall_item_df = utils.subsampling_user(one_phase_recall_item_df, conf.subsampling)
            print('load recall items: phase:{} shape:{}'.format(phase, one_phase_recall_item_df.shape[0]))
        else:
            raise Exception('qtime召回结果文件不存在')

        if conf.is_recall_sample_cached:
            recall_sample_df = pd.read_csv(conf.recall_sample_path.format(phase), dtype={'user_id': np.str, 'item_id': np.str})
            if conf.subsampling:
                recall_sample_df = recall_sample_df[recall_sample_df['user_id'].isin(one_phase_recall_item_df['user_id'])]

            # recall_sample_df.loc[:, 'user_txt_vec'] = recall_sample_df.apply(
            #     lambda x: np.array([np.float(i) for i in x['user_txt_vec'].split('[')[1].split(']')[0].split()])
            #     if x['user_txt_vec'] is not np.nan and x['user_txt_vec'] else x['user_txt_vec'],
            #     axis=1
            # )
            # recall_sample_df.loc[:, 'user_img_vec'] = recall_sample_df.apply(
            #     lambda x: np.array([np.float(i) for i in x['user_img_vec'].split('[')[1].split(']')[0].split()])
            #     if x['user_img_vec'] is not np.nan and x['user_img_vec'] else x['user_img_vec'],
            #     axis=1
            # )
            recall_sample_df.loc[:, 'item_txt_vec'] = recall_sample_df.apply(
                lambda x: np.array([np.float(i) for i in x['item_txt_vec'].split('[')[1].split(']')[0].split()])
                if x['item_txt_vec'] is not np.nan and x['item_txt_vec'] else x['item_txt_vec'],
                axis=1
            )
            recall_sample_df.loc[:, 'item_img_vec'] = recall_sample_df.apply(
                lambda x: np.array([np.float(i) for i in x['item_img_vec'].split('[')[1].split(']')[0].split()])
                if x['item_img_vec'] is not np.nan and x['item_img_vec'] else x['item_img_vec'],
                axis=1
            )

            print('load recall samples: phase:{} shape:{}'.format(phase, recall_sample_df.shape[0]))
        else:
            # 取前50rank
            one_phase_recall_item_df = \
                one_phase_recall_item_df.sort_values(['user_id', conf.ITEM_CF_SCORE], ascending=False).reset_index(drop=True)
            one_phase_recall_item_df = one_phase_recall_item_df.groupby('user_id').head(50).reset_index(drop=True)

            print(one_phase_recall_item_df.shape)
            # sample 构造
            recall_sample_df = get_recall_sample(one_phase_recall_item_df, item_info_df, item_txt_embedding_dim)
            recall_sample_df.to_csv(conf.recall_sample_path.format(phase), index=False)

        if conf.is_recall_feature_cached:
            recall_feature_df = pd.read_csv(conf.recall_feature_path.format(phase), dtype={'user_id': np.str, 'item_id': np.str})
            if conf.subsampling:
                recall_feature_df = recall_feature_df[recall_feature_df['user_id'].isin(one_phase_recall_item_df['user_id'])]
            print('load recall features: phase:{} shape:{}'.format(phase, recall_feature_df.shape[0]))
        else:
            # featuring
            recall_feature_df = do_featuring(
                all_phase_click_no_qtime, recall_sample_df, hot_df, conf.process_num,
                item_txt_embedding_dim, is_recall=True, feature_caching_path=conf.recall_feature_path.format(phase),
                itemcf_score_maxtrix=item_sim_list
            )

        submit_x = recall_feature_df[recall_feature_df.columns.difference(['user_id', 'item_id', 'label'])].values
        submit_pre_y = model.predict_proba(submit_x)[:, 1]
        submit = utils.save_pre_as_submit_format_csv(recall_sample_df, submit_pre_y)
        submit_all = submit_all.append(submit)

    print('--------------------------- 保存预测文件 --------------------------')
    utils.save(submit_all, 50)

    # todo 统计不同阶段时间分段情况
    # todo user点击深度超过10的情况怎么处理