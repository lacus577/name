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
from pymodule.rank import train_model_rf, train_model_lgb, rank_rf
from pymodule.eval import metrics_recall
from pymodule.recall import topk_recall_association_rules_open_source
from pymodule.featuring import matrix_word2vec_embedding, get_train_test_data, \
    get_user_features, train_test_split, cal_user_item_sim, cal_txt_img_sim, \
    cal_click_sim, cal_time_interval, cal_item_of_user_def, cal_statistic_features, \
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
    ''' 读取item和user属性 '''
    train_underexpose_item_feat_path = os.path.join(conf.train_path, 'underexpose_item_feat.csv')
    train_underexpose_user_feat_path = os.path.join(conf.train_path, 'underexpose_user_feat.csv')

    train_underexpose_item_feat_df_columns = \
        ['item_id'] + \
        ['txt_vec' + str(i) for i in range(item_txt_embedding_dim)] + \
        ['img_vec' + str(i) for i in range(item_img_embedding_dim)]
    train_underexpose_user_feat_df_columns = \
        ['user_id', 'user_age_level', 'user_gender', 'user_city_level']

    item_info_df = pd.read_csv(
        train_underexpose_item_feat_path,
        names=train_underexpose_item_feat_df_columns,
        dtype={'item_id': np.str}
    )
    user_info_df = pd.read_csv(
        train_underexpose_user_feat_path,
        names=train_underexpose_user_feat_df_columns,
        dtype={'user_id': np.str, 'item_id': np.str, 'time': np.float}
    )

    # 删除[ 和 ]
    item_info_df['txt_vec0'] = \
        item_info_df['txt_vec0'].apply(lambda x: float(str(x)[1:]) if x is not np.nan else x)
    item_info_df['img_vec0'] = \
        item_info_df['img_vec0'].apply(lambda x: float(str(x)[1:]) if x is not np.nan else x)
    item_info_df['txt_vec127'] = \
        item_info_df['txt_vec127'].apply(lambda x: float(str(x)[:-1]) if x is not np.nan else x)
    item_info_df['img_vec127'] = \
        item_info_df['img_vec127'].apply(lambda x: float(str(x)[:-1]) if x is not np.nan else x)

    txt_embedding_df = item_info_df[['txt_vec{}'.format(i) for i in range(item_txt_embedding_dim)]]
    img_embedding_df = item_info_df[['img_vec{}'.format(i) for i in range(item_img_embedding_dim)]]

    # todo 降维后信息丢失情况
    item_txt_embedding_dim = item_img_embedding_dim = 32  # 降维后维度
    short_txt_embedding = \
        PCA(n_components=item_txt_embedding_dim).fit_transform(txt_embedding_df.values)
    short_img_embedding = \
        PCA(n_components=item_img_embedding_dim).fit_transform(img_embedding_df.values)

    item_info_df = item_info_df[['item_id']]
    item_info_df = pd.concat(
        [item_info_df,
         pd.DataFrame(data=short_txt_embedding, columns=['txt_vec{}'.format(i) for i in range(item_txt_embedding_dim)])],
        axis=1
    )
    item_info_df = pd.concat(
        [item_info_df,
         pd.DataFrame(data=short_img_embedding, columns=['img_vec{}'.format(i) for i in range(item_img_embedding_dim)])],
        axis=1
    )

    if conf.is_click_cached:
        all_phase_click_666 = pd.read_csv(conf.click_cache_path, dtype={'user_id': np.str, 'item_id': np.str})
        ''' sampling '''
        if conf.subsampling:
            all_phase_click_666 = utils.subsampling_user(all_phase_click_666, conf.subsampling)
        print('load all click, shape:{}'.format(all_phase_click_666.shape))
    else:
        all_phase_click_org = pd.DataFrame()
        for phase in range(0, conf.now_phase + 1):
            one_phase_train_click = utils.read_train_click(conf.train_path, phase)
            one_phase_test_click = utils.read_test_click(conf.test_path, phase)
            one_phase_qtime = utils.read_qtime(conf.test_path, phase)

            one_phase_test_click['phase'] = str(phase)
            one_phase_test_click['train_or_test'] = 'test'
            one_phase_train_click['phase'] = str(phase)
            one_phase_train_click['train_or_test'] = 'train'
            one_phase_qtime['phase'] = str(phase)
            one_phase_qtime['train_or_test'] = 'predict'
            one_phase_qtime['item_id'] = np.nan

            all_phase_click_org = all_phase_click_org.append(one_phase_train_click).reset_index(drop=True)
            all_phase_click_org = all_phase_click_org.append(one_phase_test_click).reset_index(drop=True)
            all_phase_click_org = all_phase_click_org.append(one_phase_qtime).reset_index(drop=True)

        ''' sampling '''
        if conf.subsampling:
            all_phase_click_org = utils.subsampling_user(all_phase_click_org, conf.subsampling)

        # 删除重复点击
        all_phase_click = utils.del_dup(all_phase_click_org)
        # 删除待预测时间点 之后的点击数据 防止数据泄露
        all_phase_click_666 = utils.del_qtime_future_click(all_phase_click)
        # 时间处理 乘上 1591891140
        all_phase_click_666 = utils.process_time(all_phase_click_666, 1591891140)

        all_phase_click_666 = all_phase_click_666.sort_values(['user_id', 'time']).reset_index(drop=True)
        all_phase_click_666.to_csv(conf.click_cache_path, index=False)

    all_phase_click_no_qtime = all_phase_click_666[all_phase_click_666['train_or_test'] != 'predict']
    hot_df = all_phase_click_666.groupby('item_id')['user_id'].count().reset_index()
    hot_df.columns = ['item_id', 'item_deg']

    if conf.is_samples_cached:
        sample_df = pd.read_csv(conf.samples_cache_path, dtype={'user_id': np.str, 'item_id': np.str})
        if conf.subsampling:
            sample_df = sample_df[sample_df['user_id'].isin(all_phase_click_no_qtime['user_id'])]

        sample_df.loc[:, 'user_txt_vec'] = sample_df.apply(
            lambda x: np.array([np.float(i) for i in x['user_txt_vec'].split('[')[1].split(']')[0].split()])
            if x['user_txt_vec'] is not np.nan and x['user_txt_vec'] else x['user_txt_vec'],
            axis=1
        )
        sample_df.loc[:, 'user_img_vec'] = sample_df.apply(
            lambda x: np.array([np.float(i) for i in x['user_img_vec'].split('[')[1].split(']')[0].split()])
            if x['user_img_vec'] is not np.nan and x['user_img_vec'] else x['user_img_vec'],
            axis=1
        )
        sample_df.loc[:, 'item_txt_vec'] = sample_df.apply(
            lambda x: np.array([np.float(i) for i in x['item_txt_vec'].split('[')[1].split(']')[0].split()])
            if x['item_txt_vec'] is not np.nan and x['item_txt_vec'] else x['item_txt_vec'],
            axis=1
        )
        sample_df.loc[:, 'item_img_vec'] = sample_df.apply(
            lambda x: np.array([np.float(i) for i in x['item_img_vec'].split('[')[1].split(']')[0].split()])
            if x['item_img_vec']  is not np.nan and x['item_img_vec'] else x['item_img_vec'],
            axis=1
        )
        print('load samples, shape:{}'.format(sample_df.shape))
    else:
        sample_df = get_samples_v1(all_phase_click_666, item_info_df, 280, 5, item_txt_embedding_dim, conf.process_num)
        sample_df.to_csv(conf.samples_cache_path, index=False)

    ''' 训练/验证集 特征 '''
    if conf.is_feature_cached:
        feature_df = pd.read_csv(
            conf.features_cache_path,
            dtype={'user_id': np.str, 'item_id': np.str}
        )
        print('load features, shape:{}'.format(feature_df.shape))
    else:
        feature_df = do_featuring(
            all_phase_click_no_qtime, sample_df, hot_df, conf.process_num,
            item_txt_embedding_dim, is_recall=False, feature_caching_path=conf.features_cache_path
        )

    assert sample_df.shape[0] == feature_df.shape[0]
    assert len(set(sample_df['user_id'])) == len(set(feature_df['user_id']))
    ''' 训练集/验证集划分 '''
    train_df, valid_df = train_test_split(feature_df)
    train_x = train_df[train_df.columns.difference(['user_id', 'item_id', 'label'])].values
    train_y = train_df['label'].values

    valid_x = valid_df[valid_df.columns.difference(['user_id', 'item_id', 'label'])].values
    valid_y = valid_df['label'].values

    ''' 模型训练 '''
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('------------------------ 模型训练 start time:{}'.format(time_str))
    # submit = train_model_lgb(feature_all, recall_rate=hit_rate, hot_list=hot_list, valid=0.2, topk=50, num_boost_round=1, early_stopping_rounds=1)
    # submit = train_model_rf(train_test, recall_rate=1, hot_list=hot_list, valid=0.2, topk=50)
    model = rank_rf(train_x, train_y)
    # model = rank_xgb(train_x, train_y)
    print('train set: auc:{}'.format(roc_auc_score(train_y, model.predict_proba(train_x)[:, 1])))
    with open('./cache/rf.pickle', 'wb') as f:
        pickle.dump(model, f)

    ''' 模型验证 '''
    print('------------------------ 模型验证 start time:{}'.format(time_str))
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    pre_y = model.predict_proba(valid_x)[:, 1]
    print('valid set: auc:{}'.format(roc_auc_score(valid_y, pre_y)))
    answer = make_answer(valid_df[valid_df['label'] == 1], hot_df, phase=1)
    my_eval(pre_y, valid_df, answer)


    # ''' qtime 提交集 特征 '''
    # for phase in range(0, conf.now_phase + 1):
    #     pass

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
            raise Exception('召回结果文件不存在')

        if conf.is_recall_sample_cached:
            recall_sample_df = pd.read_csv(conf.recall_sample_path.format(phase), dtype={'user_id': np.str, 'item_id': np.str})

            sample_df.loc[:, 'user_txt_vec'] = sample_df.apply(
                lambda x: np.array([np.float(i) for i in x['user_txt_vec'].split('[')[1].split(']')[0].split()])
                if x['user_txt_vec'] is not np.nan and x['user_txt_vec'] else x['user_txt_vec'],
                axis=1
            )
            sample_df.loc[:, 'user_img_vec'] = sample_df.apply(
                lambda x: np.array([np.float(i) for i in x['user_img_vec'].split('[')[1].split(']')[0].split()])
                if x['user_img_vec'] is not np.nan and x['user_img_vec'] else x['user_img_vec'],
                axis=1
            )
            sample_df.loc[:, 'item_txt_vec'] = sample_df.apply(
                lambda x: np.array([np.float(i) for i in x['item_txt_vec'].split('[')[1].split(']')[0].split()])
                if x['item_txt_vec'] is not np.nan and x['item_txt_vec'] else x['item_txt_vec'],
                axis=1
            )
            sample_df.loc[:, 'item_img_vec'] = sample_df.apply(
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
            recall_sample_df = get_recall_sample(sample_df, one_phase_recall_item_df, item_info_df, item_txt_embedding_dim)
            recall_sample_df.to_csv(conf.recall_sample_path.format(phase), index=False)

        if conf.is_recall_feature_cached:
            recall_feature_df = pd.read_csv(conf.recall_feature_path.format(phase), dtype={'user_id': np.str, 'item_id': np.str})
            print('load recall features: phase:{} shape:{}'.format(phase, recall_feature_df.shape[0]))
        else:
            # featuring
            recall_feature_df = do_featuring(
                all_phase_click_no_qtime, recall_sample_df, hot_df, conf.process_num,
                item_txt_embedding_dim, is_recall=True, feature_caching_path=conf.recall_feature_path.format(phase)
            )

        submit_x = recall_feature_df[recall_feature_df.columns.difference(['user_id', 'item_id', 'label'])].values
        submit_pre_y = model.predict_proba(submit_x)[:, 1]
        submit = utils.save_pre_as_submit_format_csv(recall_sample_df, submit_pre_y)
        submit_all = submit_all.append(submit)

    print('--------------------------- 保存预测文件 --------------------------')
    utils.save(submit_all, 50)

    # todo 统计不同阶段时间分段情况
    # todo user点击深度超过10的情况怎么处理