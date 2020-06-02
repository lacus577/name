import multiprocessing
import time

import pickle
from tqdm import tqdm
import psutil
import pandas as pd
import numpy as np
from gensim import models

from pymodule import conf, utils

def matrix_word2vec_embedding(click_all, flag, mode, threshold=0, dim=100, epochs=30, learning_rate=0.5):
    """
        word2vec 原理 skip bow：
            窗口内 预测
        # 注释：doc2vec 有bug，建议不使用

        四种向量化方式：
            flag='item' mode='all':
                sku1 sku2 sku3 sku4 sku5 user
            flag='user' mode='all':
                user1 user2 user3 user4 user5 sku
            flag='item',mode='only':
                item1 item2 item3 item4 item5
            flag='user' mode='only'
                user1 user2 user3 user4 user5
    """

    if flag == 'user':
        group_by_col, agg_col = 'item_id', 'user_id'
    if flag == 'item':
        group_by_col, agg_col = 'user_id', 'item_id'

    data_ = click_all.groupby([group_by_col])[agg_col].agg(lambda x: ','.join(list(x))).reset_index()
    if mode == 'only':
        list_data = list(data_[agg_col].map(lambda x: x.split(',')))
    if mode == 'all':
        data_['concat'] = data_[agg_col] + ',' + data_[group_by_col].map(lambda x: 'all_' + x)
        list_data = data_['concat'].map(lambda x: x.split(','))

    model = models.Word2Vec(
        list_data,
        size=dim,
        alpha=learning_rate,
        # window=999999,
        min_count=1,
        workers=psutil.cpu_count(),
        # compute_loss=True,
        iter=epochs,
        hs=0,
        sg=1,
        seed=42
    )
    # model = models.Word2Vec(list_data, size=dim)

    # model.build_vocab(list_data, update=True)
    # model.train(list_data, total_examples=model.corpus_count, epochs=model.iter)


    keys = model.wv.vocab.keys()
    # print(len(keys))
    if mode == 'only':
        word2vec_embedding = {flag: {}}
    if mode == 'all':
        word2vec_embedding = {'user': {}, 'item': {}}
    for k in keys:
        if 'all' not in k:
            word2vec_embedding[flag][k] = model.wv[k]
        if 'all' in k:
            flag_ = group_by_col.split('_')[0]
            k_ = k.split('_')[1]
            word2vec_embedding[flag_][k_] = model.wv[k]

    return word2vec_embedding


def get_train_test_data(
        topk_recall,
        dict_embedding_all_ui_item,
        dict_embedding_all_ui_user,
        dict_embedding_item_only,
        dict_embedding_user_only,
        flag_test=False
):
    from tqdm import tqdm

    data_list = []

    print('------- 构建样本 -----------')
    temp_ = topk_recall
    """
        测试
    """
    if flag_test == True:
        len_temp = len(temp_)
        len_temp_2 = len_temp // 2
        temp_['label'] = [1] * len_temp_2 + [0] * (len_temp - len_temp_2)
    if flag_test == False:
        '''
        label:
        1表示召回命中
        0表示召回不命中
        同一个user会召回多行，这里没有对一个user进行合并，所以一个user如果命中的话，其中一行是1，其他都是0
        '''
        temp_['label'] = [1 if next_item_id == item_similar else 0 for (next_item_id, item_similar) in
                          zip(temp_['next_item_id'], temp_['item_similar'])]

    ''' 命中的user_id集合 '''
    set_user_label_1 = set(temp_[temp_['label'] == 1]['user_id'])

    ''' 
    只取命中的用户构建精排分类模型的训练集
    其中命中的item作为正样本，未命中的item作为负样本
    TODO 存在样本不均衡问题: 1: recall_num
    '''
    temp_['keep'] = temp_['user_id'].map(lambda x: 1 if x in set_user_label_1 else 0)
    train_data = temp_[temp_['keep'] == 1][['user_id', 'item_similar', 'score_similar', 'label']]

    # temp_['pred'] = temp_['user_id'].map(lambda x: 'test' if x in set_pred else 'train')
    # 测试集
    test_data = temp_[temp_['pred'] == 'test'][['user_id', 'item_similar', 'score_similar']]

    list_train_test = [('train', train_data), ('test', test_data)]
    for flag, data in list_train_test:

        print('------- 加入特征 {flag} -----------'.format(flag=flag))

        list_train_flag, list_user_id, list_item_similar, list_label, list_features = [], [], [], [], []

        for i, row in tqdm(data.iterrows()):

            user_id, item_id, score_similar = str(row['user_id']), str(row['item_similar']), float(row['score_similar'])
            # similarity(a,b) = a/|a| * b/|b|
            dim1_user = dict_embedding_all_ui_item['user'][user_id]
            dim1_item = dict_embedding_all_ui_item['item'][item_id]
            # 余弦相似度/大小罚项
            similarity_d1 = np.sum(
                dim1_user / np.sqrt(np.sum(dim1_user ** 2)) * dim1_item / np.sqrt(np.sum(dim1_item ** 2)))

            dim2_user = dict_embedding_all_ui_user['user'][user_id]
            dim2_item = dict_embedding_all_ui_user['item'][item_id]
            similarity_d2 = np.sum(
                dim2_user / np.sqrt(np.sum(dim2_user ** 2)) * dim2_item / np.sqrt(np.sum(dim2_item ** 2)))

            dim3_item = dict_embedding_item_only['item'][item_id]

            dim4_user = dict_embedding_user_only['user'][user_id]

            feature = list(dim1_user) + \
                      list(dim1_item) + \
                      list(dim2_user) + \
                      list(dim2_item) + \
                      list(dim3_item) + \
                      list(dim4_user) + \
                      [similarity_d1] + [similarity_d2] + [score_similar]

            list_features.append(feature)

            list_train_flag.append(flag)
            list_user_id.append(user_id)
            list_item_similar.append(item_id)

            if flag == 'train':
                label = int(row['label'])
                list_label.append(label)

            if flag == 'test':
                label = -1
                list_label.append(label)

        feature_all = pd.DataFrame(list_features)
        feature_all.columns = ['f_' + str(i) for i in range(len(feature_all.columns))]

        feature_all['train_flag'] = list_train_flag
        feature_all['user_id'] = list_user_id
        feature_all['item_similar'] = list_item_similar
        feature_all['label'] = list_label

        data_list.append(feature_all)

    feature_all_train_test = pd.concat(data_list)

    print('--------------------------- 特征数据 ---------------------')
    len_f = len(feature_all_train_test)
    len_train = len(feature_all_train_test[feature_all_train_test['train_flag'] == 'train'])
    len_test = len(feature_all_train_test[feature_all_train_test['train_flag'] == 'test'])
    len_train_1 = len(feature_all_train_test[
                          (feature_all_train_test['train_flag'] == 'train') & (feature_all_train_test['label'] == 1)])
    print('所有数据条数', len_f)
    print('训练数据 : ', len_train)
    print('训练数据 label 1 : ', len_train_1)
    print('训练数据 1 / 0 rate : ', len_train_1 * 1.0 / len_f)
    print('测试数据 : ', len_test)
    print('flag : ', set(feature_all_train_test['train_flag']))
    print('--------------------------- 特征数据 ---------------------')

    return feature_all_train_test


def get_user_features(click_info, item_info_df, txt_dim, img_dim):
    # TODO user_id、item_id存在价值，后面研究加入
    # TODO 时间特征如何融入
    # 当时仅仅使用vec特征

    # todo 缺失值处理，不处理就被后面的sum掩盖了
    user_features = click_info[['user_id', 'item_id']]
    # features = features.merge(user_info_df, on='user_id', how='left')
    # TODO 基于上述分析，用户特征太稀疏 先不使用， 先只是使用item特征
    user_features = user_features.merge(item_info_df, on='item_id', how='left').drop(['item_id'], axis=1)
    # TODO normolizaiton
    user_features = user_features.groupby(by='user_id').sum().reset_index()


    # txt_dict = dict(zip(['user_txt_vec' + str(i) for i in range(128)], [lambda x: np.nansum(x) for i in range(128)]))
    # img_dict = dict(zip(['user_img_vec' + str(i) for i in range(128)], [lambda x: np.nansum(x) for i in range(128)]))
    # txt_dict.update(img_dict)
    # user_features = user_features[
    #     ['user_id'] + ['txt_vec' + str(i) for i in range(128)] + ['img_vec' + str(i) for i in range(128)]
    # ]
    user_features.columns = ['user_id'] + ['user_txt_vec' + str(i) for i in range(txt_dim)] + ['user_img_vec' + str(i) for i in range(img_dim)]
    # user_features = user_features.groupby(['user_id']).agg(txt_dict).reset_index()

    return user_features

def my_cos_sim(vec1, vec2):
    if vec1 is None or vec1 is np.nan or vec2 is None or vec2 is np.nan or isinstance(vec1, float) or isinstance(vec2, float):
        return np.nan

    assert len(vec1) == len(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cal_sim_(user_features, item_features):
    if user_features.shape[0] == 0 or item_features.shape[0] == 0:
        # todo 最小值待修订
        return np.nan

    assert 1 == user_features.shape[0] and 1 == item_features.shape[0]

    user_vector = np.array(list(user_features.values)).reshape(-1, )
    item_vector = np.array(list(item_features.values)).reshape(-1, )

    return my_cos_sim(user_vector, item_vector)

def cal_user_item_sim(user_item_df):
    print(user_item_df.columns)
    user_item_df['txt_embedding_sim'] = np.nan
    user_item_df.loc[:, 'txt_embedding_sim'] = user_item_df.apply(
        lambda x: my_cos_sim(
            x['user_txt_vec'],
            x['item_txt_vec']
        ),
        axis=1
    )

    # todo 最小值待修订
    user_item_df['img_embedding_sim'] = np.nan
    user_item_df.loc[:, 'img_embedding_sim'] = user_item_df.apply(
        lambda x: my_cos_sim(
            x['user_img_vec'],
            x['item_img_vec']
        ),
        axis=1
    )

    return user_item_df

def cal_txt_img_sim(df, process_num):
    pool = multiprocessing.Pool(processes=process_num)
    process_result = []
    for i in range(process_num):
        train_data_len = df.shape[0]
        step = train_data_len // process_num
        if i + 1 != process_num:
            input_train_data = df.iloc[i * step: (i + 1) * step, :]
        else:
            input_train_data = df.iloc[i * step:, :]
        process_result.append(
            pool.apply_async(cal_user_item_sim, (input_train_data, ))
        )

    pool.close()
    pool.join()
    temp_train_data = pd.DataFrame()
    for res in process_result:
        # print(res)
        # temp_train_data.append(res.get())
        temp_train_data = pd.concat([temp_train_data, res.get()])

    return temp_train_data

def cal_user_item_sim_v1(user_item_df, item_user_emb, user_item_emb):
    user_item_df['click_item_user_sim'] = np.nan
    user_item_df.loc[:, 'click_item_user_sim'] = user_item_df.apply(
        lambda x: my_cos_sim(
            item_user_emb['user'].get(x['user_id']),
            item_user_emb['item'].get(x['item_id'])
            # user_features[user_features['user_id'] == x['user_id']].iloc[:, -dim - dim:-dim],
            # item_features[item_features['item_id'] == x['item_id']].iloc[:, -dim - dim:-dim]
        ),
        axis=1
    )

    # todo 最小值待修订
    user_item_df['click_user_item_sim'] = np.nan
    user_item_df.loc[:, 'click_user_item_sim'] = user_item_df.apply(
        lambda x: my_cos_sim(
            user_item_emb['user'].get(x['user_id']),
            user_item_emb['item'].get(x['item_id'])
        ),
        axis=1
    )

    return user_item_df

def cal_click_sim(df, item_user_emb, user_item_emb, process_num):
    pool = multiprocessing.Pool(processes=process_num)
    process_result = []
    for i in range(process_num):
        train_data_len = df.shape[0]
        step = train_data_len // process_num
        if i + 1 != process_num:
            input_train_data = df.iloc[i * step: (i + 1) * step, :]
        else:
            input_train_data = df.iloc[i * step:, :]
        process_result.append(
            pool.apply_async(cal_user_item_sim_v1, (input_train_data, item_user_emb, user_item_emb,))
        )

    pool.close()
    pool.join()
    temp_train_data = pd.DataFrame()
    for res in process_result:
        # print(res)
        # temp_train_data.append(res.get())
        temp_train_data = pd.concat([temp_train_data, res.get()])

    return temp_train_data

def train_test_split(total_features, percentage=0.7):
    # 训练集、验证集、提交集划分
    # # todo test集合没用于训练，后续可以考虑加上一起训练
    # train_valid_data = total_features[total_features['train_or_test'] == 'train']
    # # todo 这个集合不对，应该是test click中user召回的结果
    # submition_data = total_features[total_features['train_or_test'] == 'test']

    '''训练集切成两个集合，一个训练 一个验证测试'''
    df_user = pd.DataFrame(list(set(total_features['user_id'])))
    df_user.columns = ['user_id']

    df = df_user.sample(frac=1.0, random_state=1)  # 打散
    # TODO 划分比例valid
    cut_idx = int(round(percentage * df.shape[0]))
    df_train_0, df_train_1 = df.iloc[:cut_idx], df.iloc[cut_idx:]

    train_data = df_train_0.merge(total_features, on=['user_id'], how='left')
    tmp_valid_data = df_train_1.merge(total_features, on=['user_id'], how='left')

    # valid_data_len = tmp_valid_data.shape[0]
    # valid_data = tmp_valid_data[tmp_valid_data['label'] == 1]
    # negative_len = (valid_data_len - valid_data.shape[0]) // valid_data.shape[0]
    # valid_data = valid_data.drop_duplicates(['user_id'], keep='last')
    # valid_data = valid_data.append(
    #     tmp_valid_data[tmp_valid_data['label'] != 1].sample(n_samples=negative_len, random_state=1)
    # )

    # 验证集中每个user只保留一个label为1的正样本
    valid_data = tmp_valid_data.groupby('user_id').head(6)
    # assert 1 == np.sum(valid_data['label'] == 1)
    assert len(set(valid_data['user_id'])) == valid_data[valid_data['label'] == 1].shape[0]
    tmp_valid_data = tmp_valid_data.append(valid_data).drop_duplicates(['user_id', 'item_id', 'label'], keep=False)
    train_data = train_data.append(tmp_valid_data)

    return train_data, valid_data

def cal_time_interval(target_df, total_df):
    # for i in range(target_df.shape[0]):
    #     print(
    #         'xxxxxxxxxx',
    #         np.nanmean(
    #             np.array(list(total_df[total_df['user_id'] == target_df.loc[i, 'user_id']]['time'][:-1])) -
    #             np.array(list(total_df[total_df['user_id'] == target_df.loc[i, 'user_id']]['time'][1: ]))
    #         )
    #     )

    return {
        'user_click_interval_mean': target_df.apply(
            lambda x: np.nanmean(
                np.array(list(total_df[total_df['user_id'] == x['user_id']]['time'][:-1])) -
                np.array(list(total_df[total_df['user_id'] == x['user_id']]['time'][1: ]))
            ),
            axis=1
        ),
        'user_click_interval_min': target_df.apply(
            lambda x: np.nanmin(
                np.array(list(total_df[total_df['user_id'] == x['user_id']]['time'][:-1])) -
                np.array(list(total_df[total_df['user_id'] == x['user_id']]['time'][1: ]))
            ),
            axis=1
        ),
        'user_click_interval_max': target_df.apply(
            lambda x: np.nanmax(
                np.array(list(total_df[total_df['user_id'] == x['user_id']]['time'][:-1])) -
                np.array(list(total_df[total_df['user_id'] == x['user_id']]['time'][1: ]))
            ),
            axis=1
        )
    }


def cal_item_of_user_def(df, total_df):
    return {
        'user_item_mean_deg': df.apply(
            lambda x: np.nanmean(df[df['user_id'] == x['user_id']]['item_deg']),
            axis=1
        ),
        'user_item_min_deg':  df.apply(
            lambda x: np.nanmin(df[df['user_id'] == x['user_id']]['item_deg']),
            axis=1
        ),
        'user_item_max_deg': df.apply(
            lambda x: np.nanmax(df[df['user_id'] == x['user_id']]['item_deg']),
            axis=1
        )
    }

def cal_item_distance(df, total_df):
    '''
    点击序中item距离user的距离（最近一个点击距离0）
    :param df:
    :param total_df:
    :return:
    '''
    # for i in range(df.shape[0]):
    #     # # print(
    #     # #     'xxxxxxxxxx', np.sum(
        # #         total_df[total_df['user_id'] == df.loc[i, 'user_id']]['item_id'] == df.loc[i, 'item_id']
        # #     )
        # # )
        # print(
        #     'xxxxxxxxxx', np.sum(
        #         total_df[total_df['user_id'] == df.loc[i, 'user_id']]['item_id'] == df.loc[i, 'item_id']
        #     ),
        #     df.loc[i, 'item_id'] in total_df[total_df['user_id'] == df.loc[i, 'user_id']]['item_id'],
        #     list(total_df[total_df['user_id'] == df.loc[i, 'user_id']]['item_id'] == df.loc[i, 'item_id']).index(True)
        # )

    return {
        'item_distance': df.apply(
            lambda x: list(total_df[total_df['user_id'] == x['user_id']]['item_id'] == x['item_id']).index(True)
            if np.sum(
                total_df[total_df['user_id'] == x['user_id']]['item_id'] == x['item_id']
            ) > 0
            else conf.MAX_CLICK_LEN,
            axis=1
        )
    }

def cal_user_click_num(df, total_df):
    '''
    2. 训练集点击序中user点击次数（即 点击深度 TODO 去做个统计：点击深度和冷门物品偏好的关系）
    :param df:
    :param total_df:
    :return:
    '''

    return {
        'user_click_num': df.apply(
            lambda x:  len(total_df[total_df['user_id'] == x['user_id']]),
            axis=1
        )
    }

def cal_statistic_features(df_key, df, total_df, process_func_dict, process_num):
    pool = multiprocessing.Pool(processes=process_num)
    process_result = []
    for i in range(process_num):
        # train_data_len = df.shape[0]
        # step = train_data_len // process_num
        # if i + 1 != process_num:
        #     input_train_data = df.iloc[i * step: (i + 1) * step, :]
        # else:
        #     input_train_data = df.iloc[i * step:, :]
        process_result.append(
            pool.apply_async(process_func_dict[i], (df, total_df, ))
        )

    pool.close()
    pool.join()
    result_list = []
    for res in process_result:
        # print(res)
        # temp_train_data.append(res.get())
        # temp_train_data = pd.concat([temp_train_data, res.get()])
        result_list.append(res.get())

    return {df_key: result_list}

def cal_total_statistic_features(df_dict, total_df, process_num, process_func_dict):
    pool = multiprocessing.Pool(processes=process_num)
    process_result = []
    for i in range(process_num):
        process_result.append(
            pool.apply_async(
                cal_statistic_features,
                (i, df_dict[i], total_df, process_func_dict, len(list(process_func_dict.keys())), )
            )
        )

    pool.close()
    pool.join()
    result_list = []
    for res in process_result:
        # print(res)
        # temp_train_data.append(res.get())
        # temp_train_data = pd.concat([temp_train_data, res.get()])
        result_list.append(res.get())

    return result_list


def process_after_featuring(df, is_recall=False):
    '''

    :param train_data:
    :param valid_data:
    :param train_user_recall_df:
    :param test_user_recall_df:
    :return:
    '''
    ''' 缺失值处理 当前填0 '''
    df.fillna(value=0, axis=0, inplace=True)

    ''' 特征列顺序 重新 组织 '''
    # ['user_id', 'item_id', 'label', 'itemcf_score', 'txt_embedding_sim', 'img_embedding_sim', 'click_item_user_sim', 'click_user_item_sim', 'item_distance',
    #    'user_click_num', 'user_click_interval_mean', 'user_click_interval_min',
    #    'user_click_interval_max', 'item_deg', 'user_item_mean_deg',
    #    'user_item_min_deg', 'user_item_max_deg']
    features_columns = ['user_id', 'item_id', 'txt_embedding_sim', 'img_embedding_sim',
       'click_item_user_sim', 'click_user_item_sim', 'user_click_num',
       'user_click_interval_mean', 'user_click_interval_min',
       'user_click_interval_max', 'item_deg', 'user_item_mean_deg',
       'user_item_min_deg', 'user_item_max_deg',
       '0_item2item_itemcf_score', 'item20_item_itemcf_score',
       '1_item2item_itemcf_score', 'item21_item_itemcf_score',
       '2_item2item_itemcf_score', 'item22_item_itemcf_score',
       '3_item2item_itemcf_score', 'item23_item_itemcf_score',
       '4_item2item_itemcf_score', 'item24_item_itemcf_score',
       'user_avg_click', 'user_span_click', 'user_total_deg', 'user_avg_deg',
       '0_item_deg', '1_item_deg', 'top_1_item_deg', '2_item_deg',
       'top_2_item_deg', '3_item_deg', 'top_3_item_deg', '4_item_deg',
       'top_4_item_deg']

    if is_recall:
        df = df[features_columns]
    else:
       df = df[features_columns + ['label']]

    return df


def get_samples_v1(df, item_info_df, time_interval_thr, negative_num, dim, process_num):
    '''
    第一版样本如下：
    1. 最近一个session构建user_feature（一个session 280以内）
    2. qtime前面的那次点击也做为正样本
    3. 负样本随机采样10个（剔除掉此用户点击的）
    :param df:
    :return:
    '''
    # 正样本构建
    df = df[df['train_or_test'] != 'predict']
    user_set = set(df['user_id'])
    print(len(user_set))
    positive_sample_list = []
    for user in tqdm(user_set):
        user_df = df[df['user_id'] == user]
        user_df = user_df.sort_values(['time'], ascending=False).reset_index()
        user_df['time_interval'] = list(np.array(list(user_df['time']))[:-1] - np.array(list(user_df['time']))[1:]) + [np.inf]

        s = e = 0
        one_user_sample = 0
        for i in range(user_df.shape[0]):
            if user_df.loc[i, 'time_interval'] <= time_interval_thr:
                e += 1
            else:
                if e - s >= 3:
                    positive_sample_list.append([user] + list(user_df.loc[s: e, 'item_id']))
                    one_user_sample += 1
                s = e = i + 1

    # 对于每个待预测的user， 需要以其最近一次点击来构建一个正样本 -- 虽然不知道是否有用
    qtime_user_set = set(df[df['train_or_test'] == 'predict']['user_id'])
    for user in tqdm(qtime_user_set):
        user_df = df[df['user_id'] == user]
        positive_sample_list.append([user] + list(user_df.loc[0: 4, 'item_id']))

    print(len(positive_sample_list))

    # 负采样
    negative_sample_dict = {}
    one_user_tmp = None
    for i in tqdm(range(len(positive_sample_list))):
        user = positive_sample_list[i][0]
        if one_user_tmp is None or user not in one_user_tmp.keys():
            one_user_tmp = df[~df['item_id'].isin(df[df['user_id'] == user]['item_id'])]

        negative_tmp = one_user_tmp.sample(n=negative_num, random_state=1, axis=0)
        # negative_sample_list.append([user] + list(negative_tmp['item_id']))
        negative_sample_dict[i] = list(negative_tmp['item_id'])


    # positive_sample_list.extend(negative_sample_list)
    # 正负样本df构建 并 缓存
    item_info_dict = utils.transfer_item_features_df2dict(item_info_df, dim)

    pool = multiprocessing.Pool(processes=process_num)
    process_result = []
    for i in range(process_num):
        sample_len = len(positive_sample_list)
        step = sample_len // process_num
        if i + 1 != process_num:
            input_data = positive_sample_list[i * step: (i + 1) * step]
        else:
            input_data = positive_sample_list[i * step: ]
        process_result.append(
            pool.apply_async(make_samples, (i * step, input_data, item_info_dict, negative_sample_dict, ))
        )

    pool.close()
    pool.join()
    result_pd = pd.DataFrame()
    for res in process_result:
        result_pd = result_pd.append(res.get())

    return result_pd


def make_samples(start_index, sample_list, item_info_dict, negative_sample_dict):
    sample_df = pd.DataFrame()
    for i in tqdm(range(len(sample_list))):
        user = sample_list[i][0]
        item = sample_list[i][1]
        # user_txt_embedding = np.nansum([item_info_dict['txt_vec'].get(j) for j in sample_list[i][2: ]], axis=0)
        # user_img_embedding = np.nansum([item_info_dict['img_vec'].get(j) for j in sample_list[i][2: ]], axis=0)
        user_txt_embedding = np.zeros(list(item_info_dict['txt_vec'].values())[0].shape)
        user_img_embedding = np.zeros(list(item_info_dict['img_vec'].values())[0].shape)
        for item_id in sample_list[i][2:]:
            if item_info_dict['txt_vec'].get(str(item_id)) is not None:
                # assert type(item_info_dict['txt_vec'].get(str(item_id))) isinstance(x, np.ndarray)
                user_txt_embedding += item_info_dict['txt_vec'].get(str(item_id))
            if item_info_dict['img_vec'].get(str(item_id)) is not None:
                user_img_embedding += item_info_dict['img_vec'].get(str(item_id))

        one_user_df = [[
            user, item,
            user_txt_embedding, user_img_embedding,
            item_info_dict['txt_vec'].get(item),
            item_info_dict['img_vec'].get(item),
            1 # 正样本
        ]]

        for negative_item in negative_sample_dict[start_index + i]:
            one_user_df.append(
                [
                    user, negative_item,
                    user_txt_embedding, user_img_embedding,
                    item_info_dict['txt_vec'].get(negative_item),
                    item_info_dict['img_vec'].get(negative_item),
                    0 # 负样本
                ]
            )

        sample_df = sample_df.append(one_user_df)

    sample_df.columns = ['user_id', 'item_id', 'user_txt_vec', 'user_img_vec', 'item_txt_vec', 'item_img_vec', 'label']
    return sample_df

def do_featuring(
        all_phase_click_in,
        sample_df,
        hot_df_in,
        process_num,
        dim,
        is_recall,
        feature_caching_path
):
    """

    :param click_df:
    :param item_info_df:
    :param user_info_df:
    :return:
    """

    # '''
    # 官方特征:
    # 1. user和item之间txt相似度
    # 2. user和item之间img相似度
    # '''
    # time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print('官方特征 start time:{}'.format(time_str))
    #
    # # 每计算好一个特征就缓存下来
    # features_df = cal_txt_img_sim(sample_df, process_num)
    # features_df.to_csv(feature_caching_path, index=False)
    # # print(features_df)
    #
    #
    # '''
    # 点击序：
    # 1. 纯item序列  -- 砍掉
    # 2. item序列和对应user  -- 砍掉
    # 3. 纯user序列  -- 砍掉
    # 4. user序列和共同item  -- 砍掉
    # 5. 2 带来的user和item相似度
    # 6. 4 带来的user和item相似度
    # '''
    # time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print('点击序embedding特征 start time:{}'.format(time_str))
    # # todo 当前使用全量点击做的序列 --- > 一个session内的点击做序列
    # dict_embedding_all_ui_item, dict_embedding_all_ui_user = click_embedding(all_phase_click_in, dim)
    # features_df = cal_click_sim(
    #     features_df, dict_embedding_all_ui_item, dict_embedding_all_ui_user, process_num
    # )
    # features_df.to_csv(feature_caching_path, index=False)
    #
    # '''
    # 统计特征:
    # 一阶特征：
    #     user点击序中user点击次数（即 点击深度 TODO 去做个统计：点击深度和冷门物品偏好的关系） -- 全量数据集统计
    #     user点击序中item平均热度、最大热度、最小热度 -- 先不分train和test即使用全量数据集统计，调优的时候再分
    #     user平均点击间隔、最大点击间隔、最小点击间隔 -- 需要分train和test两个集合统计
    #     本item在全局的热度：先使用全量数据集统计，调优的时候分在train、test、item-feature中的热度
    # 二阶特征（样本中user和item交互）：
    #     样本中user和item的距离--如果item在user点击序中则根据时间排序当做距离，否则设为最大距离（最近一个点击距离0）
    #     ? 用户热度--用户点击序中所有item热度和
    # '''
    # time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print('统计特征 start time:{}'.format(time_str))
    #
    # all_phase_click_in = all_phase_click_in.sort_values(['user_id', 'time'], ascending=False).reset_index(drop=True)
    #
    # ''' user点击序中user点击次数（即 点击深度 TODO 去做个统计：点击深度和冷门物品偏好的关系） -- 全量数据集统计 '''
    # print('用户点击次数特征 doing')
    # user_click_num_df = all_phase_click_in.groupby('user_id')['item_id'].count().reset_index()
    # user_click_num_df.columns = ['user_id', 'user_click_num']
    # user_click_dict = utils.two_columns_df2dict(user_click_num_df)
    #
    # # todo qtime的user由于被删除掉未来的点击，所以统计出来的点击次数肯定少
    # features_df['user_click_num'] = features_df.apply(
    #     lambda x: user_click_dict[x['user_id']] if user_click_dict.get(x['user_id']) else 0, axis=1)
    # features_df.to_csv(feature_caching_path, index=False)
    #
    # ''' 本item在全局的热度：先使用全量数据集统计，调优的时候分在train、test、item-feature中的热度 '''
    # print('item在全局的热度 doing')
    # features_df = features_df.merge(hot_df_in, on='item_id', how='left')
    # features_df.to_csv(feature_caching_path, index=False)
    #
    # ''' user点击序中item平均热度、最大热度、最小热度 -- 先不分train和test即使用全量数据集统计，调优的时候再分 '''
    # print('user点击序中item平均热度、最大热度、最小热度 doing')
    # all_phase_click_in = all_phase_click_in.merge(hot_df_in, on='item_id', how='left')
    # user_item_hot_df = \
    #     all_phase_click_in.groupby('user_id').agg({'item_deg': lambda x: ','.join([str(i) for i in list(x)])}).reset_index()
    # user_item_hot_df.columns = ['user_id', 'item_hot_arr']
    # user_item_hot_df['item_hot_arr'] = user_item_hot_df.apply(
    #     lambda x: np.array(list(x['item_hot_arr'].split(',')), dtype=np.int), axis=1)
    # user_item_hot_dict = utils.two_columns_df2dict(user_item_hot_df)
    #
    # features_df['user_item_mean_deg'] = \
    #     features_df.apply(
    #         lambda x: np.nanmean(user_item_hot_dict.get(x['user_id'])) if user_item_hot_dict.get(x['user_id']) is not None else np.nan,
    #         axis=1
    #     )
    # features_df['user_item_min_deg'] = \
    #     features_df.apply(
    #         lambda x: np.nanmin(user_item_hot_dict.get(x['user_id'])) if user_item_hot_dict.get(
    #             x['user_id']) is not None else np.nan,
    #         axis=1
    #     )
    # features_df['user_item_max_deg'] = \
    #     features_df.apply(
    #         lambda x: np.nanmax(user_item_hot_dict.get(x['user_id'])) if user_item_hot_dict.get(
    #             x['user_id']) is not None else np.nan,
    #         axis=1
    #     )
    # features_df.to_csv(feature_caching_path, index=False)
    #
    #
    # ''' user平均点击间隔、最大点击间隔、最小点击间隔 -- 需要分train和test两个集合统计 '''
    # print('user平均点击间隔、最大点击间隔、最小点击间隔 doing')
    # train_time_interval_df = \
    #     all_phase_click_in.groupby('user_id').agg({'time': lambda x: ','.join([str(i) for i in list(x)])}).reset_index()
    # train_time_interval_df.columns = ['user_id', 'time_interval_arr']
    # train_time_interval_df['time_interval_arr'] = train_time_interval_df.apply(
    #     lambda x: np.array(list(x['time_interval_arr'].split(',')), dtype=np.float)[:-1] -
    #               np.array(list(x['time_interval_arr'].split(',')), dtype=np.float)[1:],
    #     axis=1
    # )
    # train_time_interval_dict = utils.two_columns_df2dict(train_time_interval_df)
    #
    # features_df['user_click_interval_mean'] = \
    #     features_df.apply(
    #         lambda x: np.nanmean(train_time_interval_dict.get(x['user_id'])) if train_time_interval_dict.get(
    #             x['user_id']) is not None else np.nan,
    #         axis=1
    #     )
    # features_df['user_click_interval_min'] = \
    #     features_df.apply(
    #         lambda x: np.nanmin(train_time_interval_dict.get(x['user_id'])) if train_time_interval_dict.get(
    #             x['user_id']) is not None else np.nan,
    #         axis=1
    #     )
    # features_df['user_click_interval_max'] = \
    #     features_df.apply(
    #         lambda x: np.nanmax(train_time_interval_dict.get(x['user_id'])) if train_time_interval_dict.get(
    #             x['user_id']) is not None else np.nan,
    #         axis=1
    #     )
    # features_df.to_csv(feature_caching_path, index=False)

    # todo
    features_df = pd.read_csv(feature_caching_path, dtype={'user_id': np.str, 'item_id': np.str})

    '''
    itemCF相似度：
    '''
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('itemCF相似度特征 start time:{}'.format(time_str))
    itemcf_score_maxtrix = pickle.load(open('./cache/features_cache/item_sim_list', 'rb'))
    # user和最近k个item字典
    user2kitem_dict = utils.get_user2kitem_dict(all_phase_click_in, conf.itemcf_num)
    for i in tqdm(range(conf.itemcf_num)):
        features_df['{}_item2item_itemcf_score'.format(i)] = features_df.apply(
            lambda x:
            itemcf_score_maxtrix.get(user2kitem_dict[x['user_id']][i]).get(x['item_id'])
            if user2kitem_dict.get(x['user_id']) is not None
               and len(user2kitem_dict.get(x['user_id'])) > i
               and itemcf_score_maxtrix.get(user2kitem_dict[x['user_id']][i]) is not None
               and itemcf_score_maxtrix.get(user2kitem_dict[x['user_id']][i]).get(x['item_id']) is not None
            else None,
            axis=1
        )

        features_df['item2{}_item_itemcf_score'.format(i)] = features_df.apply(
            lambda x:
            itemcf_score_maxtrix.get(x['item_id']).get(user2kitem_dict[x['user_id']][i])
            if itemcf_score_maxtrix.get(x['item_id']) is not None
               and itemcf_score_maxtrix.get(x['item_id']).get(user2kitem_dict[x['user_id']][i]) is not None
            else None,
            axis=1
        )
    features_df.to_csv(feature_caching_path, index=False)

    '''
    新增统计特征
    用户点击深度/平均时间间隔 -- 间接反应用户活跃度：更活跃的用户应该值较大，因为经常发生点击行为时间间隔更小
    用户点击深度/时间跨度    -- 应该没有什么用
    用户点击所有item的总热度 -- 用户对热门item的偏好程度，一定程度上反应上条特征中用户是否活跃
    用户点击所有item的总热度/用户点击item数量 -- 用户点击平均热度，消除用户点击深度影响
    用户最近1/2/3个item热度（单独热度、总热度）
    '''
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('新增统计特征 start time:{}'.format(time_str))
    print('用户点击深度/平均时间间隔 特征 doing')
    features_df['user_avg_click'] = features_df['user_click_num'] / features_df['user_click_interval_mean']
    print('用户点击深度/时间跨度 特征 doing')
    user2click_span_dict = utils.get_user2click_span_dict(all_phase_click_in)
    features_df['user_span_click'] = features_df.apply(
        lambda x: x['user_click_num'] / user2click_span_dict.get(x['user_id'])
        if user2click_span_dict.get(x['user_id']) is not None else None,
        axis=1
    )
    features_df.to_csv(feature_caching_path, index=False)

    print('用户点击所有item的总热度 特征 doing')
    user2total_deg_dict = utils.get_user2total_deg_dict(all_phase_click_in, hot_df_in)
    features_df['user_total_deg'] = features_df.apply(
        lambda x: user2total_deg_dict.get(x['user_id']),
        axis=1
    )
    features_df.to_csv(feature_caching_path, index=False)

    print('用户点击所有item的总热度/用户点击item数量 特征 doing')
    features_df['user_avg_deg'] = features_df['user_total_deg'] / features_df['user_click_num']

    print('用户最近k个item热度 特征 doing')
    item2deg_dict = utils.two_columns_df2dict(hot_df_in)
    tmp = None
    for i in range(conf.itemcf_num):
        features_df['{}_item_deg'.format(i)] = features_df.apply(
            lambda x: item2deg_dict.get(x['item_id']),
            axis=1
        )
        if tmp is None:
            tmp = features_df['{}_item_deg'.format(i)]
        else:
            tmp += features_df['{}_item_deg'.format(i)]

        if i > 0:
            features_df['top_{}_item_deg'.format(i)] = tmp
    features_df.to_csv(feature_caching_path, index=False)

    features_df = process_after_featuring(features_df, is_recall)
    features_df.to_csv(feature_caching_path, index=False)
    print(features_df.iloc[:5, :])

    return features_df

def click_embedding(click_info_df, dim):
    # print('-------- sku1 sku2 sku3 sku4 sku5 user ----------')
    # dim, epochs, learning_rate = 32, 1, 0.5
    # dim = 32

    dict_embedding_all_ui_item = matrix_word2vec_embedding(
        click_all=click_info_df,
        flag='item',
        mode='all',
        dim=dim
        # epochs=epochs,
        # learning_rate=learning_rate
    )
    # print('------- user1 user2 user3 user4 user5 sku -------')
    dict_embedding_all_ui_user = matrix_word2vec_embedding(
        click_all=click_info_df,
        flag='user',
        mode='all',
        dim=dim
    )
    # print('------- item1 item2 item3 item4 item5 -------')
    # dict_embedding_item_only = matrix_word2vec_embedding(
    #     click_all=click_info_df,
    #     flag='item',
    #     mode='only',
    #     dim=dim
    # )
    # print('------- user1 user2 user3 user4 user5 -------')
    # dict_embedding_user_only = matrix_word2vec_embedding(
    #     click_all=click_info_df,
    #     flag='user',
    #     mode='only',
    #     dim=dim
    # )

    return dict_embedding_all_ui_item, dict_embedding_all_ui_user

def get_recall_sample(total_sample_df, df, item_info_df, dim):
    positive_sample_df = total_sample_df[total_sample_df['label'] == 1]
    positive_sample_df = positive_sample_df.drop_duplicates(['user_id']).reset_index(drop=True)
    # pd.DataFrame().drop_duplicates()
    user_txt_vec_dict = dict(zip(positive_sample_df['user_id'], positive_sample_df['user_txt_vec']))
    user_img_vec_dict = dict(zip(positive_sample_df['user_id'], positive_sample_df['user_img_vec']))

    df['user_txt_vec'] = df.apply(
        lambda x: user_txt_vec_dict.get(x['user_id']),
        axis=1
    )

    df['user_img_vec'] = df.apply(
        lambda x: user_img_vec_dict.get(x['user_id']),
        axis=1
    )

    item_info_dict = utils.transfer_item_features_df2dict(item_info_df, dim)
    df['item_txt_vec'] = df.apply(
        lambda x: item_info_dict['txt_vec'].get(x['item_id']),
        axis=1
    )
    df['item_img_vec'] = df.apply(
        lambda x: item_info_dict['img_vec'].get(x['item_id']),
        axis=1
    )

    return df

if __name__ == '__main__':
    pass