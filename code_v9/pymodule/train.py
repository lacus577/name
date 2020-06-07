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
    do_featuring, get_recall_sample
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

    all_phase_click = utils.read_all_phase_click()
    all_phase_click_no_qtime = all_phase_click[all_phase_click['train_or_test'] != 'predict']

    # 将数据按天切分成14天，从第七天开始构建样本
    min_time = int(np.min(all_phase_click_no_qtime[conf.new_time_name]))
    max_time = int(np.max(all_phase_click_no_qtime[conf.new_time_name])) + 1
    step = (max_time - min_time) // conf.days

    total_feature_df = pd.DataFrame()
    # 从第七天开始
    for end_time in range(min_time + 7 * step, max_time, step):
        print('period {} ...'.format(end_time))
        period_click_df = all_phase_click_no_qtime[
            (all_phase_click_no_qtime[conf.new_time_name] <= end_time) &
            (all_phase_click_no_qtime[conf.new_time_name] >= min_time)
        ]

        # 对每个user取最后一个点击作为候选正样本，召回命中才是正式正样本
        period_click_df = period_click_df.sort_values(['user_id', conf.new_time_name]).reset_index(drop=True)
        candidate_positive_sample_df = period_click_df.groupby('user_id').tail(1).reset_index(drop=True)
        assert candidate_positive_sample_df.shape[0] == len(set(candidate_positive_sample_df['user_id']))
            # 正样本删除，后面要用于构建相似度矩阵
        period_click_df = period_click_df.append(candidate_positive_sample_df)
        period_click_df = \
            period_click_df.drop_duplicates(['user_id', 'item_id', conf.new_time_name], keep=False)
        period_click_df = period_click_df.sort_values(['user_id', conf.new_time_name]).reset_index(drop=True)
        assert period_click_df.merge(
            candidate_positive_sample_df, on=['user_id', 'item_id', conf.new_time_name], how='inner'
        ).shape[0] == 0

        hot_df = period_click_df.groupby('item_id')['user_id'].count().reset_index()
        hot_df.columns = ['item_id', 'item_deg']
        hot_df = hot_df.sort_values('item_deg', ascending=False).reset_index(drop=True)

        # 相似度矩阵构建
        if os.path.exists(conf.sim_list_path.format(end_time)):
            item_sim_list = pickle.load(open(conf.sim_list_path.format(end_time), 'rb'))
        else:
            item_sim_list, user_item = \
                recall.get_sim_item_5164(period_click_df, 'user_id', 'item_id', use_iif=False)

            pickle.dump(item_sim_list, open(conf.sim_list_path.format(end_time), 'wb'))
        assert len(set(item_sim_list.keys())) == len(set(period_click_df['item_id']))

        if conf.is_samples_cached:
            sample_df = pd.read_csv(conf.samples_cache_path.format(end_time), dtype={'user_id': np.str, 'item_id': np.str})
            if conf.subsampling:
                sample_df = sample_df[sample_df['user_id'].isin(period_click_df['user_id'])]

            tmp = sample_df[sample_df['label'] == 1][['user_id', 'item_id']]
            tmp.columns = ['user_id', 'truth_item_id']
            sample_df = sample_df.merge(tmp, on='user_id', how='left')

            print('loaded samples, shape:{}'.format(sample_df.shape))
        else:
            print('getting samples ...')
            if os.path.exists(conf.total_user_recall_path.format(end_time)):
                candidate_recall_df = pd.read_csv(conf.total_user_recall_path.format(end_time),
                                                  dtype={'user_id': np.str, 'item_id': np.str})
            else:
                _, recom_item = recall.items_recommod_5164(
                    candidate_positive_sample_df, item_sim_list, period_click_df, list(hot_df['item_id'])
                )
                candidate_recall_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
                candidate_recall_df.to_csv(conf.total_user_recall_path.format(end_time), index=False)
                assert len(set(candidate_recall_df['user_id'])) == len(set(candidate_positive_sample_df['user_id']))
                # 取top50，待top50调试OK之后再放开
            candidate_recall_df = candidate_recall_df.sort_values('sim').reset_index(drop=True)
            candidate_recall_df = candidate_recall_df.sort_values(['user_id', conf.new_time_name], ascending=False).reset_index(drop=True)
            candidate_recall_df = candidate_recall_df.groupby('user_id').head(conf.recall_num).reset_index(drop=True)

            # 正样本确认， 负样本构建
            candidate_positive_sample_df['label'] = 1
            tmp_total_df = candidate_recall_df.merge(candidate_positive_sample_df, on=['user_id', 'item_id'],
                                                     how='left')
            # 命中user及对应命中点击item
            positive_sample_df = tmp_total_df[tmp_total_df['label'] == 1]
            # TODO 暂时不对负样本数量做控制，正负样本比可能达到1：50
            sample_df = tmp_total_df[tmp_total_df['user_id'].isin(positive_sample_df['user_id'])]
            sample_df.loc[sample_df['label'] != 1, 'label'] = 0
            tmp = positive_sample_df[['user_id', 'item_id']]
            tmp.columns = ['user_id', 'truth_item_id']
            assert len(set(tmp['user_id'])) == tmp.shape[0]
            sample_df = sample_df.merge(tmp, on='user_id', how='left')
            if sample_df.shape[0] == 0:
                raise Exception('召回结果没有任何命中！')
            sample_df.to_csv(conf.samples_cache_path.format(end_time), index=False)
            assert sample_df[sample_df['label'] == 0].shape[0] / sample_df[sample_df['label'] == 1].shape[0] == conf.recall_num - 1

        ''' 特征工程 '''
        if conf.is_feature_cached:
            feature_df = pd.read_csv(
                conf.features_cache_path.format(end_time),
                dtype={'user_id': np.str, 'item_id': np.str}
            )
            feature_df['truth_item_id'] = sample_df['truth_item_id']

            if conf.subsampling:
                feature_df = feature_df[feature_df['user_id'].isin(period_click_df['user_id'])]
            print('loaded features, shape:{}'.format(feature_df.shape))
        else:
            feature_df = do_featuring(
                period_click_df, sample_df, hot_df, conf.process_num,
                item_txt_embedding_dim, is_recall=False,
                feature_caching_path=conf.features_cache_path.format(end_time),
                itemcf_score_maxtrix=item_sim_list
            )

        assert sample_df.shape[0] == feature_df.shape[0]
        assert len(set(sample_df['user_id'])) == len(set(feature_df['user_id']))
        assert sample_df[sample_df['label'] == 1].shape[0] == len(set(sample_df['user_id']))
        assert feature_df[feature_df['label'] == 1].shape[0] == len(set(feature_df['user_id']))

        total_feature_df = total_feature_df.append(feature_df)

    print('feature shape:{}, positive feature num:{}'.format(total_feature_df.shape, total_feature_df[total_feature_df['label'] == 1].shape[0]))
    # 这里的hot_df与训练集不是同步的，暂时凑合着用
    hot_df = all_phase_click_no_qtime.groupby('item_id')['user_id'].count().reset_index()
    hot_df.columns = ['item_id', 'item_deg']
    hot_df = hot_df.sort_values('item_deg', ascending=False).reset_index(drop=True)
    train_auc = valid_auc = 0
    pre_score_arr = np.zeros(5).reshape(-1, )
    rank_score_arr = np.zeros(5).reshape(-1, )
    for i in range(conf.k):
        ''' 训练集/验证集划分 '''
        train_df, valid_df = train_test_split(total_feature_df)

        qtime_user_df = all_phase_click[all_phase_click['train_or_test'] == 'predict']
        print(
            '训练集命中{}个qtime的user.'.format(
                len(set(train_df[train_df['user_id'].isin(qtime_user_df['user_id'])]['user_id']))
            )
        )

        train_x = train_df[train_df.columns.difference(['user_id', 'item_id', 'label', 'truth_item_id'])].values
        train_y = train_df['label'].values

        valid_df = valid_df.sort_values('sim').reset_index(drop=True)
        valid_x = valid_df[valid_df.columns.difference(['user_id', 'item_id', 'label', 'truth_item_id'])].values
        valid_y = valid_df['label'].values

        ''' 模型训练 '''
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('------------------------ 模型训练 start time:{}'.format(time_str))
        # model = rank_rf(train_x, train_y)
        model = rank_xgb(train_x, train_y)
        one_train_auc = roc_auc_score(train_y, model.predict_proba(train_x)[:, 1])
        train_auc += one_train_auc
        print('train set: auc:{}'.format(one_train_auc))
        with open('./cache/model.pickle', 'wb') as f:
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

    print('{}次留出验证，平均train auc:{}，平均valid auc:{}'.format(conf.k, train_auc/conf.k, valid_auc/conf.k))
    print('平均排序前score:{}'.format(pre_score_arr/conf.k))
    print('平均排序后score:{}'.format(rank_score_arr/conf.k))

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
            if os.path.exists(conf.total_sim_list_path):
                item_sim_list = pickle.load(open(conf.total_sim_list_path, 'rb'))
            else:
                raise Exception('no total item_sim_list')
            qitme_df = utils.read_qtime(conf.test_path, phase)
            # raise Exception('qtime召回结果文件不存在')
            _, recom_item = recall.items_recommod_5164(
                qitme_df, item_sim_list, all_phase_click_no_qtime, list(hot_df['item_id'])
            )
            one_phase_recall_item_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
            one_phase_recall_item_df.to_csv(conf.recall_cache_path.format(phase), index=False)

        if conf.is_recall_sample_cached:
            recall_sample_df = pd.read_csv(conf.recall_sample_path.format(phase), dtype={'user_id': np.str, 'item_id': np.str})
            if conf.subsampling:
                recall_sample_df = recall_sample_df[recall_sample_df['user_id'].isin(one_phase_recall_item_df['user_id'])]
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
            if os.path.exists(conf.total_sim_list_path):
                item_sim_list = pickle.load(open(conf.total_sim_list_path, 'rb'))
            else:
                raise Exception('no total item_sim_list')
            recall_feature_df = do_featuring(
                all_phase_click_no_qtime, recall_sample_df, hot_df, conf.process_num,
                item_txt_embedding_dim, is_recall=True, feature_caching_path=conf.recall_feature_path.format(phase),
                itemcf_score_maxtrix=item_sim_list
            )

        submit_x = recall_feature_df[recall_feature_df.columns.difference(['user_id', 'item_id', 'label'])].values
        # TODO k此留出验证模型输出结果加权作为最终结果
        with open('./cache/model.pickle', 'rb') as f:
            model = pickle.load(f)
        submit_pre_y = model.predict_proba(submit_x)[:, 1]
        submit = utils.save_pre_as_submit_format_csv(recall_sample_df, submit_pre_y)
        submit_all = submit_all.append(submit)

    print('--------------------------- 保存预测文件 --------------------------')
    utils.save(submit_all, 50)

    # todo 统计不同阶段时间分段情况
    # todo user点击深度超过10的情况怎么处理