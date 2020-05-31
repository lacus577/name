import os
import time
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm
# from sklearn.externals import joblib
from sklearn.decomposition import PCA
import warnings

from pymodule import constant
from pymodule.rank import train_model_rf, train_model_lgb, rank_rf
from pymodule.eval import metrics_recall
from pymodule.recall import topk_recall_association_rules_open_source
from pymodule.featuring import matrix_word2vec_embedding, get_train_test_data, \
    get_user_features, train_test_split, cal_user_item_sim, cal_txt_img_sim, \
    cal_click_sim, cal_time_interval, cal_item_of_user_def, cal_statistic_features, \
    cal_item_distance, cal_user_click_num, cal_total_statistic_features, process_after_featuring, get_samples_v1
from pymodule import utils
from pymodule.eval import evaluate
from pymodule.recall import get_sim_item, recommend

warnings.filterwarnings("ignore")

''' 进程并发数 '''
process_num = 3

now_phase = 6
train_path = '../../../../data/underexpose_train'
test_path = '../../../../data/underexpose_test'
temp_result_path = './cache/tmp_phase_submit'

''' 全局变量 '''
item_txt_embedding_dim = 128
item_img_embedding_dim = 128

''' 配置项 '''
flag_append = False
flag_test = False
recall_num = 500
topk = 50
nrows = None
subsampling = 3
''' 特征是否已经缓存 '''
is_cached_features = False
''' 打开：缓存特征 '''
is_caching_features = True
''' 训练集、测试集是否已经缓存 '''
is_data_set_cached = True
''' 训练集、测试集是否正在缓存 '''
# is_data_set_caching = False
''' 召回的训练集item太多，达到800W 并且也不是必须，只是验证使用 先mask 后面在慢慢计算后缓存下来 '''
is_open_train_recall = False

'''
上述两个开关功能描述：
1. is_cached_features=True    特征已经缓存好了， 程序直接读取缓存特征
2. is_caching_features=True   程序训练同时将特征缓存下来
'''

''' 提交的话构建itemCF相似度矩阵的时候会使用全量点击数据 '''
is_submition = True

''' itemCF对1000个训练集中的user进行召回，用于精排模型验证 '''
itemCF_train_subsampling = 1000

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

def save_pre_as_submit_format_csv(data_df, out_y):
    # 构造submit格式csv
    valid_eval_data = data_df[['user_id', 'item_id']]
    valid_eval_data['pred_prob'] = out_y
    valid_eval_data['rank'] = valid_eval_data.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
    valid_eval_data.sort_values(['rank'], inplace=True)

    valid_submit = valid_eval_data.groupby(['user_id'])['item_id'].agg(lambda x: ','.join(list(x))).reset_index()
    return valid_submit

def get_answer(phase, user_item_dict, hot_df):
    phase_answer = pd.DataFrame()
    phase_answer['user_id'] = list(user_item_dict.keys())
    phase_answer['item_id'] = list(user_item_dict.values())
    phase_answer['phase_id'] = phase
    phase_answer = phase_answer.merge(hot_df, on='item_id', how='left')
    phase_answer = phase_answer[['phase_id', 'user_id', 'item_id', 'item_deg']]
    return phase_answer

def transfer_user_features_df2dict(user_features, dim):
    txt_vec = user_features.apply(lambda x: np.array(list(x.iloc[-dim-dim: -dim])).reshape(-1, ), axis=1)
    img_vec = user_features.apply(lambda x: np.array(list(x.iloc[-dim: ])).reshape(-1, ), axis=1)
    user_key = user_features['user_id']

    assert len(user_key) == len(txt_vec) and len(user_key) == len(img_vec)

    user_features_dict = {}
    user_features_dict['txt_vec'] = dict(zip(user_key, txt_vec))
    user_features_dict['img_vec'] = dict(zip(user_key, img_vec))

    return user_features_dict

def transfer_item_features_df2dict(item_features, dim):
    txt_vec = item_features.apply(lambda x: np.array(list(x.iloc[-dim-dim: -dim])).reshape(-1, ), axis=1)
    img_vec = item_features.apply(lambda x: np.array(list(x.iloc[-dim: ])).reshape(-1, ), axis=1)
    user_key = item_features['item_id']

    assert len(user_key) == len(txt_vec) and len(user_key) == len(img_vec)

    user_features_dict = {}
    user_features_dict['txt_vec'] = dict(zip(user_key, txt_vec))
    user_features_dict['img_vec'] = dict(zip(user_key, img_vec))

    return user_features_dict


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

def do_featuring(
        click_df,
        item_info_df,
        user_info_df,
        user_item_dict,
        train_user_recall_df,
        test_user_recall_df,
        sim_matrix,
        hot_df
):
    """

    :param click_df:
    :param item_info_df:
    :param user_info_df:
    :return:
    """
    ''' 集合划分 '''
    # 训练集 测试集
    # 负样本采样：从所有点击历史中采样非正样本item
    # TODO 从官方给的item表中采样、从点击+item表中采样
    # TODO  many todo in sampling_negtive_samples
    # todo 只用train去负采样，后面尝试train和test一起
    if is_data_set_cached:
        print('reading train/valid ... set')
        train_data = pd.read_csv(
            './cache/features_cache/train_data_{}.csv'.format(phase),
            dtype={'user_id': np.str, 'item_id': np.str, 'label': np.int}
        )
        valid_data = pd.read_csv(
            './cache/features_cache/valid_data_{}.csv'.format(phase),
            dtype={'user_id': np.str, 'item_id': np.str, 'label': np.int}
        )
        train_user_recall_df = pd.read_csv(
            './cache/features_cache/train_user_recall_{}.csv'.format(phase),
            dtype={'user_id': np.str, 'item_id': np.str, 'label': np.int, 'itemcf_score': np.float}
        )
        test_user_recall_df = pd.read_csv(
            './cache/features_cache/test_user_recall_{}.csv'.format(phase),
            dtype={'user_id': np.str, 'item_id': np.str, 'itemcf_score': np.float}
        )
        print(train_data.shape)
        print(valid_data.shape)
        print(train_user_recall_df.shape)
        print(test_user_recall_df.shape)
    else:
        train_test_df = click_df[click_df['train_or_test'] == 'train'][['user_id', 'item_id']]
        user_set = set(train_test_df['user_id'])
        negtive_features = utils.sampling_negtive_samples(user_set, train_test_df, sample_num=10)
        # features = features.merge(user_features, on='user_id', how='left')
        # features = features.merge(item_info_df, on='item_id', how='left')
        negtive_features['label'] = 0

        # time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # print('正样本特征 start time:{}'.format(time_str))
        # 正样本加入
        positive_features = pd.DataFrame()
        positive_features['user_id'] = list(user_item_dict.keys())
        positive_features['item_id'] = list(user_item_dict.values())
        # positive_features = positive_features.merge(user_features, on='user_id', how='left')
        # positive_features = positive_features.merge(item_info_df, on='item_id', how='left')
        positive_features['label'] = 1
        # positive_features['train_or_test'] = 'train'

        # 正负样本合并
        features = negtive_features.append(positive_features).reset_index(drop=True)
        # features.sort_values(by='user_id', inplace=True)
        # features.reset_index(drop=True, inplace=True)


        # time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # print('训练集验证集划分 start time:{}'.format(time_str))
        # TODO  many todo in train_test_split
        train_data, valid_data = train_test_split(features, 0.8)

        # 训练集召回结果，此部分用于验证上述训练集训练出来的模型
        train_user_recall_df = train_user_recall_df[['user_id', 'item_id', 'itemcf_score']]
        train_user_recall_df['label'] = 0
        train_user_recall_df.loc[
            train_user_recall_df['user_id'].isin(list(user_item_dict.keys())) & train_user_recall_df['item_id'].isin(list(user_item_dict.values())),
            'label'
        ] = 1

        # 测试集召回结果，此部分用于提交
        test_user_recall_df = test_user_recall_df[['user_id', 'item_id', 'itemcf_score']]

        # if is_data_set_caching:
        #     print('caching splited data.',
        #           train_data.shape, valid_data.shape, train_user_recall_df.shape, test_user_recall_df.shape)
        #     train_data.to_csv('./cache/features_cache/train_data_{}.csv'.format(phase), index=False)
        #     valid_data.to_csv('./cache/features_cache/valid_data_{}.csv'.format(phase), index=False)
        #     train_user_recall_df.to_csv('./cache/features_cache/train_user_recall_{}.csv'.format(phase), index=False)
        #     test_user_recall_df.to_csv('./cache/features_cache/test_user_recall_{}.csv'.format(phase), index=False)

    print(
        np.sum(train_data['user_id'].isin(click_df['user_id'])), ',',
        np.sum(click_df['user_id'].isin(train_data['user_id']))
    )
    '''
    itemCF相似度：
    '''
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('itemCF相似度特征 start time:{}'.format(time_str))
    train_data['itemcf_score'] = np.nan
    train_data.loc[:, 'itemcf_score'] = train_data.apply(
        lambda x: sim_matrix[x['user_id']][x['item_id']]
        if sim_matrix.get(x['user_id']) is not None and sim_matrix.get(x['user_id']).get(x['item_id']) is not None
        else np.nan,
        axis=1
    )
    print(train_data)

    valid_data['itemcf_score'] = np.nan
    valid_data.loc[:, 'itemcf_score'] = valid_data.apply(
        lambda x: sim_matrix[x['user_id']][x['item_id']]
        if sim_matrix.get(x['user_id']) is not None and sim_matrix.get(x['user_id']).get(x['item_id']) is not None
        else np.nan,
        axis=1
    )

    # 把负数统一洗成0 TODO 带来的问题：可能非常稀疏
    # train_user_recall_df.loc[train_user_recall_df['itemcf_score'] < 0, 'itemcf_score'] = 0
    # test_user_recall_df.loc[test_user_recall_df['itemcf_score'] < 0, 'itemcf_score'] = 0

    '''
    官方特征:
    1. user和item之间txt相似度
    2. user和item之间img相似度
    '''
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('官方特征 start time:{}'.format(time_str))
    # 注意，此处click_df中user_id必须包含上述四个集合user
    total_set_user = set(train_data['user_id']).union(set(valid_data['user_id'])).union(set(train_user_recall_df['user_id'])).union(set(test_user_recall_df['user_id']))
    assert (
        0 == len(set(click_df['user_id']).difference(total_set_user)) and
        0 == len(total_set_user.difference(set(click_df['user_id'])))
    )
    user_features = get_user_features(click_df, item_info_df, item_txt_embedding_dim, item_img_embedding_dim)
    user_features_dict = transfer_user_features_df2dict(user_features, item_txt_embedding_dim)
    item_features_dict = transfer_item_features_df2dict(item_info_df, item_txt_embedding_dim)

    assert item_txt_embedding_dim == item_img_embedding_dim
    # 每计算好一个数据集就缓存下来
    train_data = cal_txt_img_sim(train_data, user_features_dict, item_features_dict, item_img_embedding_dim, process_num)
    if is_caching_features:
        print('正在缓存train_data')
        train_data.to_csv('./cache/features_cache/part0_train_features_phase_{}.csv'.format(phase), index=False)
    print(train_data)

    valid_data = cal_txt_img_sim(valid_data, user_features_dict, item_features_dict, item_img_embedding_dim, process_num)
    if is_caching_features:
        print('正在缓存valid_data')
        valid_data.to_csv('./cache/features_cache/part0_valid_features_phase_{}.csv'.format(phase), index=False)

    if is_open_train_recall:
        train_user_recall_df = cal_txt_img_sim(train_user_recall_df, user_features_dict, item_features_dict, item_img_embedding_dim, process_num)
        if is_caching_features:
            print('正在缓存train_user_recall_df')
            train_user_recall_df.to_csv(
                './cache/features_cache/part0_train_user_recall_features_phase_{}.csv'.format(phase), index=False
            )

    test_user_recall_df = cal_txt_img_sim(test_user_recall_df, user_features_dict, item_features_dict, item_img_embedding_dim, process_num)
    if is_caching_features:
        print('正在缓存test_user_recall_df')
        test_user_recall_df.to_csv(
            './cache/features_cache/part0_test_user_recall_features_phase_{}.csv'.format(phase), index=False
        )


    '''
    点击序：
    1. 纯item序列  -- 砍掉
    2. item序列和对应user  -- 砍掉
    3. 纯user序列  -- 砍掉
    4. user序列和共同item  -- 砍掉
    5. 2 带来的user和item相似度
    6. 4 带来的user和item相似度
    '''
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('点击序embedding特征 start time:{}'.format(time_str))
    dict_embedding_all_ui_item, dict_embedding_all_ui_user = click_embedding(click_df, item_img_embedding_dim)
    train_data = cal_click_sim(
        train_data, dict_embedding_all_ui_item, dict_embedding_all_ui_user, process_num
    )
    if is_caching_features:
        print('正在缓存train_data')
        train_data.to_csv('./cache/features_cache/part0_train_features_phase_{}.csv'.format(phase), index=False)

    valid_data = cal_click_sim(
        valid_data, dict_embedding_all_ui_item, dict_embedding_all_ui_user, process_num
    )
    if is_caching_features:
        print('正在缓存valid_data')
        valid_data.to_csv('./cache/features_cache/part0_valid_features_phase_{}.csv'.format(phase), index=False)

    if is_open_train_recall:
        train_user_recall_df = cal_click_sim(
            train_user_recall_df, dict_embedding_all_ui_item, dict_embedding_all_ui_user, process_num
        )
        if is_caching_features:
            print('正在缓存train_user_recall_df')
            train_user_recall_df.to_csv(
                './cache/features_cache/part0_train_user_recall_features_phase_{}.csv'.format(phase), index=False
            )

    test_user_recall_df = cal_click_sim(
        test_user_recall_df, dict_embedding_all_ui_item, dict_embedding_all_ui_user, process_num
    )
    if is_caching_features:
        print('正在缓存test_user_recall_df')
        test_user_recall_df.to_csv(
            './cache/features_cache/part0_test_user_recall_features_phase_{}.csv'.format(phase), index=False
        )
    print(train_data.columns)
    print(train_data.iloc[:5, :])
    print(valid_data.iloc[:5, :])
    print(train_user_recall_df.iloc[:5, :])
    print(test_user_recall_df.iloc[:5, :])

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
    # click_df = click_df.sort_values(['user_id', 'time'], ascending=False).reset_index(drop=True)
    #
    # ''' user点击序中user点击次数（即 点击深度 TODO 去做个统计：点击深度和冷门物品偏好的关系） -- 全量数据集统计 '''
    # user_click_num_df = click_df.groupby('user_id')['item_id'].count().reset_index()
    # user_click_num_df.columns = ['user_id', 'user_click_num']
    # user_click_dict = utils.two_columns_df2dict(user_click_num_df)
    #
    # train_data['user_click_num'] = train_data.apply(
    #     lambda x: user_click_dict[x['user_id']] if user_click_dict.get(x['user_id']) else 0, axis=1)
    # if is_caching_features:
    #     print('用户点击次数特征--正在缓存train_data')
    #     train_data.to_csv('./cache/features_cache/part2_train_features_phase_{}.csv'.format(phase), index=False)
    #
    # valid_data['user_click_num'] = valid_data.apply(
    #     lambda x: user_click_dict[x['user_id']] if user_click_dict.get(x['user_id']) else 0, axis=1)
    # if is_caching_features:
    #     print('用户点击次数特征--正在缓存valid_data')
    #     valid_data.to_csv('./cache/features_cache/part2_valid_features_phase_{}.csv'.format(phase), index=False)
    #
    # if is_open_train_recall:
    #     train_user_recall_df['user_click_num'] = train_user_recall_df.apply(
    #         lambda x: user_click_dict[x['user_id']] if user_click_dict.get(x['user_id']) else 0, axis=1)
    #     if is_caching_features:
    #         print('用户点击次数特征--正在缓存train_user_recall_df')
    #         train_user_recall_df.to_csv(
    #             './cache/features_cache/part2_train_user_recall_features_phase_{}.csv'.format(phase), index=False)
    #
    # test_user_recall_df['user_click_num'] = test_user_recall_df.apply(
    #     lambda x: user_click_dict[x['user_id']] if user_click_dict.get(x['user_id']) else 0, axis=1)
    # if is_caching_features:
    #     print('用户点击次数特征--正在缓存test_user_recall_df')
    #     test_user_recall_df.to_csv(
    #         './cache/features_cache/part2_test_user_recall_features_phase_{}.csv'.format(phase), index=False)
    #
    #
    # ''' 本item在全局的热度：先使用全量数据集统计，调优的时候分在train、test、item-feature中的热度 '''
    # print('item在全局的热度 doing')
    # train_data = train_data.merge(hot_df, on='item_id', how='left')
    # valid_data = valid_data.merge(hot_df, on='item_id', how='left')
    # if is_open_train_recall:
    #     train_user_recall_df = train_user_recall_df.merge(hot_df, on='item_id', how='left')
    # test_user_recall_df = test_user_recall_df.merge(hot_df, on='item_id', how='left')
    #
    # ''' user点击序中item平均热度、最大热度、最小热度 -- 先不分train和test即使用全量数据集统计，调优的时候再分 '''
    # click_df = click_df.merge(hot_df, on='item_id', how='left')
    # user_item_hot_df = \
    #     click_df.groupby('user_id').agg({'item_deg': lambda x: ','.join([str(i) for i in list(x)])}).reset_index()
    # user_item_hot_df.columns = ['user_id', 'item_hot_arr']
    # user_item_hot_df['item_hot_arr'] = user_item_hot_df.apply(
    #     lambda x: np.array(list(x['item_hot_arr'].split(',')), dtype=np.int), axis=1)
    # user_item_hot_dict = utils.two_columns_df2dict(user_item_hot_df)
    #
    # train_data['user_item_mean_deg'] = \
    #     train_data.apply(lambda x: np.nanmean(user_item_hot_dict[x['user_id']]), axis=1)
    # train_data['user_item_min_deg'] = \
    #     train_data.apply(lambda x: np.nanmin(user_item_hot_dict[x['user_id']]), axis=1)
    # train_data['user_item_max_deg'] = \
    #     train_data.apply(lambda x: np.nanmax(user_item_hot_dict[x['user_id']]), axis=1)
    # if is_caching_features:
    #     print('user点击序中item热度统计特征--正在缓存train_data')
    #     train_data.to_csv('./cache/features_cache/part2_train_features_phase_{}.csv'.format(phase), index=False)
    #
    # valid_data['user_item_mean_deg'] = \
    #     valid_data.apply(lambda x: np.nanmean(user_item_hot_dict[x['user_id']]), axis=1)
    # valid_data['user_item_min_deg'] = \
    #     valid_data.apply(lambda x: np.nanmin(user_item_hot_dict[x['user_id']]), axis=1)
    # valid_data['user_item_max_deg'] = \
    #     valid_data.apply(lambda x: np.nanmax(user_item_hot_dict[x['user_id']]), axis=1)
    # if is_caching_features:
    #     print('user点击序中item热度统计特征--正在缓存valid_data')
    #     valid_data.to_csv('./cache/features_cache/part2_valid_features_phase_{}.csv'.format(phase), index=False)
    #
    # if is_open_train_recall:
    #     train_user_recall_df['user_item_mean_deg'] = \
    #         train_user_recall_df.apply(lambda x: np.nanmean(user_item_hot_dict[x['user_id']]), axis=1)
    #     train_user_recall_df['user_item_min_deg'] = \
    #         train_user_recall_df.apply(lambda x: np.nanmin(user_item_hot_dict[x['user_id']]), axis=1)
    #     train_user_recall_df['user_item_max_deg'] = \
    #         train_user_recall_df.apply(lambda x: np.nanmax(user_item_hot_dict[x['user_id']]), axis=1)
    #     if is_caching_features:
    #         print('user点击序中item热度统计特征--正在缓存train_user_recall_df')
    #         train_user_recall_df.to_csv(
    #             './cache/features_cache/part2_train_user_recall_features_phase_{}.csv'.format(phase), index=False)
    #
    # test_user_recall_df['user_item_mean_deg'] = \
    #     test_user_recall_df.apply(lambda x: np.nanmean(user_item_hot_dict[x['user_id']]), axis=1)
    # test_user_recall_df['user_item_min_deg'] = \
    #     test_user_recall_df.apply(lambda x: np.nanmin(user_item_hot_dict[x['user_id']]), axis=1)
    # test_user_recall_df['user_item_max_deg'] = \
    #     test_user_recall_df.apply(lambda x: np.nanmax(user_item_hot_dict[x['user_id']]), axis=1)
    # if is_caching_features:
    #     print('user点击序中item热度统计特征--正在缓存test_user_recall_df')
    #     test_user_recall_df.to_csv(
    #         './cache/features_cache/part2_test_user_recall_features_phase_{}.csv'.format(phase), index=False)
    #
    # ''' user平均点击间隔、最大点击间隔、最小点击间隔 -- 需要分train和test两个集合统计 '''
    # train_time_interval_df = \
    #     click_df[click_df['train_or_test'] == 'train'].groupby('user_id').agg({'time': lambda x: ','.join([str(i) for i in list(x)])}).reset_index()
    # train_time_interval_df.columns = ['user_id', 'time_interval_arr']
    # train_time_interval_df['time_interval_arr'] = train_time_interval_df.apply(
    #     lambda x: np.array(list(x['time_interval_arr'].split(',')), dtype=np.float)[:-1] -
    #               np.array(list(x['time_interval_arr'].split(',')), dtype=np.float)[1:],
    #     axis=1
    # )
    # train_time_interval_dict = utils.two_columns_df2dict(train_time_interval_df)
    #
    # train_data['user_click_interval_mean'] = \
    #     train_data.apply(lambda x: np.nanmean(train_time_interval_dict[x['user_id']]), axis=1)
    # train_data['user_click_interval_min'] = \
    #     train_data.apply(lambda x: np.nanmin(train_time_interval_dict[x['user_id']]), axis=1)
    # train_data['user_click_interval_max'] = \
    #     train_data.apply(lambda x: np.nanmax(train_time_interval_dict[x['user_id']]), axis=1)
    # if is_caching_features:
    #     print('用户点击时间间隔特征--正在缓存train_data')
    #     train_data.to_csv('./cache/features_cache/part2_train_features_phase_{}.csv'.format(phase), index=False)
    #
    # valid_data['user_click_interval_mean'] = \
    #     valid_data.apply(lambda x: np.nanmean(train_time_interval_dict[x['user_id']]), axis=1)
    # valid_data['user_click_interval_min'] = \
    #     valid_data.apply(lambda x: np.nanmin(train_time_interval_dict[x['user_id']]), axis=1)
    # valid_data['user_click_interval_max'] = \
    #     valid_data.apply(lambda x: np.nanmax(train_time_interval_dict[x['user_id']]), axis=1)
    # if is_caching_features:
    #     print('用户点击时间间隔特征--正在缓存valid_data')
    #     valid_data.to_csv('./cache/features_cache/part2_valid_features_phase_{}.csv'.format(phase), index=False)
    #
    # if is_open_train_recall:
    #     train_user_recall_df['user_click_interval_mean'] = \
    #         train_user_recall_df.apply(lambda x: np.nanmean(train_time_interval_dict[x['user_id']]), axis=1)
    #     train_user_recall_df['user_click_interval_min'] = \
    #         train_user_recall_df.apply(lambda x: np.nanmin(train_time_interval_dict[x['user_id']]), axis=1)
    #     train_user_recall_df['user_click_interval_max'] = \
    #         train_user_recall_df.apply(lambda x: np.nanmax(train_time_interval_dict[x['user_id']]), axis=1)
    #     if is_caching_features:
    #         print('用户点击时间间隔特征--正在缓存train_user_recall_df')
    #         train_user_recall_df.to_csv('./cache/features_cache/part2_train_user_recall_features_phase_{}.csv'.format(phase), index=False)
    #
    # test_time_interval_df = \
    #     click_df[click_df['train_or_test'] == 'test'].groupby('user_id').agg({'time': lambda x: ','.join([str(i) for i in list(x)])}).reset_index()
    # test_time_interval_df.columns = ['user_id', 'time_interval_arr']
    # test_time_interval_df['time_interval_arr'] = test_time_interval_df.apply(
    #     lambda x: np.array(list(x['time_interval_arr'].split(',')), dtype=np.float)[:-1] -
    #               np.array(list(x['time_interval_arr'].split(',')), dtype=np.float)[1:],
    #     axis=1
    # )
    # test_time_interval_dict = utils.two_columns_df2dict(test_time_interval_df)
    #
    # test_user_recall_df['user_click_interval_mean'] = \
    #     test_user_recall_df.apply(
    #         lambda x: np.nanmean(test_time_interval_dict[x['user_id']]) if 0 != len(test_time_interval_dict[x['user_id']]) else np.nan,
    #         axis=1
    #     )
    # test_user_recall_df['user_click_interval_min'] = \
    #     test_user_recall_df.apply(
    #         lambda x: np.nanmin(test_time_interval_dict[x['user_id']]) if 0 != len(test_time_interval_dict[x['user_id']]) else np.nan,
    #         axis=1
    #     )
    # test_user_recall_df['user_click_interval_max'] = \
    #     test_user_recall_df.apply(
    #         lambda x: np.nanmax(test_time_interval_dict[x['user_id']]) if 0 != len(test_time_interval_dict[x['user_id']]) else np.nan,
    #         axis=1
    #     )
    # if is_caching_features:
    #     print('用户点击时间间隔特征--正在缓存test_user_recall_df')
    #     test_user_recall_df.to_csv('./cache/features_cache/part2_test_user_recall_features_phase_{}.csv'.format(phase), index=False)

    # '''
    # 暂时关系， 此特征有问题，存在数据泄露
    # 样本中user和item的距离--如果item在user点击序中则根据时间排序当做距离，否则设为最大距离（最近一个点击距离0）
    # 由于train和test集合user_id不重复，所以不需要分开
    # '''
    # user_clicked_items_df = click_df.groupby('user_id').agg({'item_id': lambda x: ','.join(list(x))}).reset_index()
    # user_clicked_items_df.columns = ['user_id', 'item_id_arr']
    # user_clicked_items_df['item_id_arr'] = user_clicked_items_df.apply(
    #     lambda x: list(x['item_id_arr'].split(',')), axis=1)
    # user_clicked_items_dict = utils.two_columns_df2dict(user_clicked_items_df)
    #
    # # TODO  巨大问题， 训练集中仅有500和0 而实际上还有1 2 3 等
    # train_data['item_distance'] = train_data.apply(
    #     lambda x: list(user_clicked_items_dict[x['user_id']]).index(x['item_id'])
    #     if x['item_id'] in user_clicked_items_dict[x['user_id']] else constant.MAX_CLICK_LEN,
    #     axis=1
    # )
    # if is_caching_features:
    #     print('样本中user和item的距离--正在缓存train_data')
    #     train_data.to_csv('./cache/features_cache/part2_train_features_phase_{}.csv'.format(phase), index=False)
    #
    # valid_data['item_distance'] = valid_data.apply(
    #     lambda x: list(user_clicked_items_dict[x['user_id']]).index(x['item_id'])
    #     if x['item_id'] in user_clicked_items_dict[x['user_id']] else constant.MAX_CLICK_LEN,
    #     axis=1
    # )
    # if is_caching_features:
    #     print('样本中user和item的距离--正在缓存valid_data')
    #     valid_data.to_csv('./cache/features_cache/part2_valid_features_phase_{}.csv'.format(phase), index=False)
    #
    # if is_open_train_recall:
    #     train_user_recall_df['item_distance'] = train_user_recall_df.apply(
    #         lambda x: list(user_clicked_items_dict[x['user_id']]).index(x['item_id'])
    #         if x['item_id'] in user_clicked_items_dict[x['user_id']] else constant.MAX_CLICK_LEN,
    #         axis=1
    #     )
    #     if is_caching_features:
    #         print('样本中user和item的距离--正在缓存train_user_recall_df')
    #         train_user_recall_df.to_csv('./cache/features_cache/part2_train_user_recall_features_phase_{}.csv'.format(phase), index=False)
    #
    # # TODO 这个好像都没有匹配上 都是最大值500？
    # test_user_recall_df['item_distance'] = test_user_recall_df.apply(
    # lambda x: list(user_clicked_items_dict[x['user_id']]).index(x['item_id'])
    # if x['item_id'] in user_clicked_items_dict[x['user_id']] else constant.MAX_CLICK_LEN,
    # axis=1
    # )
    # if is_caching_features:
    #     print('样本中user和item的距离--正在缓存test_user_recall_df')
    #     test_user_recall_df.to_csv('./cache/features_cache/part2_test_user_recall_features_phase_{}.csv'.format(phase), index=False)


    # data_dict = {
    #     0: train_data, 1: valid_data, 2: train_user_recall_df, 3: test_user_recall_df
    # }
    # process_func_dict = {
    #     0: cal_item_distance
    # }
    # total_statistic_features_dict_list = \
    #     cal_total_statistic_features(data_dict, click_df, len(list(data_dict.keys())), process_func_dict)
    #
    # for total_statistic_features_dict in total_statistic_features_dict_list:
    #     if total_statistic_features_dict.get(0):
    #         for statistic_feature in total_statistic_features_dict[0]:
    #             for k, v in statistic_feature.items():
    #                 train_data[k] = v
    #     elif total_statistic_features_dict.get(1):
    #         for statistic_feature in total_statistic_features_dict[1]:
    #             for k, v in statistic_feature.items():
    #                 valid_data[k] = v
    #     elif total_statistic_features_dict.get(2):
    #         for statistic_feature in total_statistic_features_dict[2]:
    #             for k, v in statistic_feature.items():
    #                 train_user_recall_df[k] = v
    #     elif total_statistic_features_dict.get(3):
    #         for statistic_feature in total_statistic_features_dict[3]:
    #             for k, v in statistic_feature.items():
    #                 test_user_recall_df[k] = v


    # print(train_data.iloc[:5, :])
    # print(valid_data.iloc[:5, :])
    # print(train_user_recall_df.iloc[:5, :])
    # print(test_user_recall_df.iloc[:5, :])

    if is_open_train_recall:
        train_data, valid_data, train_user_recall_df, test_user_recall_df = process_after_featuring(
            train_data, valid_data, train_user_recall_df, test_user_recall_df, is_open_train_recall
        )
    else:
        train_data, valid_data, test_user_recall_df = process_after_featuring(
            train_data, valid_data, None, test_user_recall_df, is_open_train_recall
        )
        train_user_recall_df = None

    print(train_data.iloc[:5, :])
    print(valid_data.iloc[:5, :])
    if is_open_train_recall:
        print(train_user_recall_df.iloc[:5, :])
    print(test_user_recall_df.iloc[:5, :])

    return train_data, valid_data, train_user_recall_df, test_user_recall_df


if __name__ == '__main__':
    ''' 读取item和user属性 '''
    train_underexpose_item_feat_path = os.path.join(train_path, 'underexpose_item_feat.csv')
    train_underexpose_user_feat_path = os.path.join(train_path, 'underexpose_user_feat.csv')

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

    all_phase_click_org = pd.DataFrame()
    for phase in range(0, now_phase + 1):
        one_phase_train_click = utils.read_train_click(train_path, phase)
        one_phase_test_click = utils.read_test_click(test_path, phase)
        one_phase_qtime = utils.read_qtime(test_path, phase)

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
    if subsampling:
        all_phase_click_org = utils.subsampling_user(all_phase_click_org, subsampling)

    # 删除重复点击
    all_phase_click = utils.del_dup(all_phase_click_org)
    # 删除待预测时间点 之后的点击数据 防止数据泄露
    all_phase_click_666 = utils.del_qtime_future_click(all_phase_click)
    # 时间处理 乘上 1591891140
    all_phase_click_666 = utils.process_time(all_phase_click_666, 1591891140)

    all_phase_click = all_phase_click.sort_values(['user_id', 'time']).reset_index(drop=True)


    sample_df = get_samples_v1(all_phase_click_666, item_info_df, 280)

    # submit_all = pd.DataFrame()
    # # one_phase_click = pd.DataFrame()
    # whole_click = pd.DataFrame()
    # for phase in range(0, now_phase + 1):
    #     print('----------------------- phase:{} -------------------------'.format(phase))
    #
    #     one_phase_click = all_phase_click_org[all_phase_click_org['phase'] == str(phase)]
    #
    #     ''' sampling '''
    #     if subsampling:
    #         one_phase_click = utils.subsampling_user(one_phase_click, subsampling)
    #
    #     click = click_train.append(click_test)
    #
    #
    #     one_phase_click = one_phase_click.sort_values('time')
    #     one_phase_click = one_phase_click.drop_duplicates(['user_id', 'item_id', 'time'], keep='last')
    #
    #     # train、test重新划分，并去重
    #     set_pred = set(click_test['user_id'])
    #     set_train = set(one_phase_click['user_id']) - set_pred
    #
    #     temp_ = one_phase_click
    #     temp_['train_or_test'] = temp_['user_id'].map(lambda x: 'test' if x in set_pred else 'train')
    #
    #     '''train集合中每个user点击序中最后一个点击item作为正样本标签'''
    #     temp_ = temp_[temp_['train_or_test'] == 'train'].drop_duplicates(['user_id'], keep='last')
    #     temp_['remove'] = 'remove'
    #
    #     # 去掉remove列标签，剩下的作为训练集
    #     train_test = one_phase_click
    #     train_test = train_test.merge(temp_, on=['user_id', 'item_id', 'time', 'train_or_test'], how='left').sort_values('user_id')
    #     train_test = train_test[train_test['remove'] != 'remove']
    #
    #     dict_label_user_item = dict(zip(temp_['user_id'], temp_['item_id']))
    #
    #     ''' item频率统计，作为热度 '''
    #     # 计算热门，用于评估 -- 根据官方评估中关于truth热门的计算，热门应该是本阶段内train和test热门统计
    #     hot_df = one_phase_click.groupby(['item_id'])['user_id'].count().reset_index().rename(columns={'user_id': 'item_deg'})
    #     hot_df = hot_df.sort_values(['item_deg'], ascending=False).reset_index(drop=True)
    #
    #     # TODO 这里的answer构建和正样本构建强相关，正样本构建方式修改，这里也要跟着修改
    #     debias_track_answer = get_answer(phase, dict_label_user_item, hot_df)
    #
    #
    #     if is_cached_features:
    #         time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #         print('------------------------ 排序：读缓存特征 start time:{}'.format(time_str))
    #         if is_open_train_recall:
    #             train_data, valid_data, train_user_recall_df, test_user_recall_df = read_features(phase, is_open_train_recall)
    #             train_data, valid_data, train_user_recall_df, test_user_recall_df = process_after_featuring(
    #                 train_data, valid_data, train_user_recall_df, test_user_recall_df, is_open_train_recall
    #             )
    #
    #             print(train_data.shape, valid_data.shape, train_user_recall_df.shape, test_user_recall_df.shape)
    #         else:
    #             train_data, valid_data, test_user_recall_df = read_features(phase, is_open_train_recall)
    #             train_data, valid_data, test_user_recall_df = process_after_featuring(
    #                 train_data, valid_data, None, test_user_recall_df, is_open_train_recall
    #             )
    #
    #             print(train_data.shape, valid_data.shape, test_user_recall_df.shape)
    #
    #     else:
    #         # todo 应该使用阶段内的热门
    #         whole_click = whole_click.append(train_test)
    #         whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    #         whole_click = whole_click.sort_values('time')
    #         hot_list = list(whole_click['item_id'].value_counts().index[:200].values)
    #
    #         print('------------------------ 召回')
    #         print('------------------------ 召回：相似度矩阵构建')
    #         # todo 注意输入的集合，根据之前实验验证：输入完整训练集+测试集 比 输入mask掉最后点击的训练集和测试集效果要好不少
    #         # if is_submition:
    #         #     item_sim_list, user_item = get_sim_item(click_all, 'user_id', 'item_id', use_iif=False)
    #         # else:
    #         #     item_sim_list, user_item = get_sim_item(train_test, 'user_id', 'item_id', use_iif=False)
    #         item_sim_list, user_item = get_sim_item(one_phase_click, 'user_id', 'item_id', use_iif=False)
    #
    #         print('------------------------ 召回：测试集召回，用于提交')
    #         ''' 测试集user召回，用于精排后提交 '''
    #         phase_recom_item = []
    #         for i in tqdm(click_test['user_id'].unique()):
    #             rank_item = recommend(item_sim_list, user_item, i, 500, 500)
    #             rank_item = rank_item[:400]
    #             for j in rank_item:
    #                 phase_recom_item.append([i, j[0], j[1]])
    #             hot_cover = 500 - len(rank_item)
    #             if hot_cover > 0:
    #                 for hot_index, hot_item in enumerate(hot_list):
    #                     if hot_item not in user_item[i] and hot_item not in [x[0] for x in rank_item]:
    #                         phase_recom_item.append([i, hot_item, -1 * hot_index])
    #                         hot_cover -= 1
    #                     if hot_cover <= 0: break
    #         test_user_recall_result = pd.DataFrame(data=phase_recom_item,
    #                                                columns=['user_id', 'item_id', 'itemcf_score'])
    #         print('------------------------ 召回：训练集正样本召回，用于验证精排模型')
    #         ''' 训练集user召回，用于精排模型评估 '''
    #         train_recom_item = []
    #         count = 0
    #         for i in tqdm(dict_label_user_item.keys()):
    #             count += 1
    #             # 全量召回16000+ * 500，太多了；这里采样召回部分user
    #             if count > itemCF_train_subsampling:
    #                 break
    #
    #             rank_item = recommend(item_sim_list, user_item, i, 500, 500)
    #             rank_item = rank_item[:400]
    #             for j in rank_item:
    #                 train_recom_item.append([i, j[0], j[1]])
    #             hot_cover = 500 - len(rank_item)
    #             if hot_cover > 0:
    #                 for hot_index, hot_item in enumerate(hot_list):
    #                     if hot_item not in user_item[i] and hot_item not in [x[0] for x in rank_item]:
    #                         train_recom_item.append([i, hot_item, -1 * hot_index])
    #                         hot_cover -= 1
    #                     if hot_cover <= 0: break
    #         train_user_recall_result = pd.DataFrame(data=train_recom_item,
    #                                                 columns=['user_id', 'item_id', 'itemcf_score'])
    #
    #         time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #         print('------------------------ 排序：构建特征 start time:{}'.format(time_str))
    #         train_data, valid_data, train_user_recall_df, test_user_recall_df = do_featuring(
    #             one_phase_click,
    #             item_info_df=item_info_df,
    #             user_info_df=user_info_df,
    #             user_item_dict=dict_label_user_item,
    #             train_user_recall_df=train_user_recall_result,
    #             test_user_recall_df=test_user_recall_result,
    #             sim_matrix=item_sim_list,
    #             hot_df=hot_df
    #         )
    #
    #     print(phase, '  ', len(set(test_user_recall_df['user_id'])), len(set(click_test['user_id'])))
    #     assert len(set(test_user_recall_df['user_id'])) == len(set(click_test['user_id']))
    #
    #     ''' 模型输入 准备 '''
    #     train_x = train_data[train_data.columns.difference(['user_id', 'item_id', 'label'])].values
    #     train_y = train_data['label'].values
    #
    #     valid_x = valid_data[valid_data.columns.difference(['user_id', 'item_id', 'label'])].values
    #     valid_y = valid_data['label'].values
    #
    #     if is_open_train_recall:
    #         train_user_recall_x = train_user_recall_df[train_user_recall_df.columns.difference(['user_id', 'item_id', 'label'])].values
    #         train_user_recall_y = train_user_recall_df['label'].values
    #
    #     # 取前50rank
    #     test_user_recall_df = \
    #         test_user_recall_df.sort_values(['user_id', 'itemcf_score'], ascending=False).reset_index(drop=True)
    #     test_user_recall_df = test_user_recall_df.groupby('user_id').head(50).reset_index(drop=True)
    #
    #     test_user_recall_x = test_user_recall_df[test_user_recall_df.columns.difference(['user_id', 'item_id', 'label'])].values
    #
    #     time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     print('------------------------ 模型训练 start time:{}'.format(time_str))
    #     # submit = train_model_lgb(feature_all, recall_rate=hit_rate, hot_list=hot_list, valid=0.2, topk=50, num_boost_round=1, early_stopping_rounds=1)
    #     # submit = train_model_rf(train_test, recall_rate=1, hot_list=hot_list, valid=0.2, topk=50)
    #     model = rank_rf(train_x, train_y)
    #     # model = rank_xgb(train_x, train_y)
    #     # joblib.dump(model, './cache/rf.pkl')
    #     print('------------------------ 模型验证 ----------------------------')
    #     time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     print('------------------------ 模型验证 part1:采样集合验证 start time:{}'.format(time_str))
    #     pre_y = model.predict_proba(valid_x)[:, 1]
    #
    #     # 构造submit csv
    #     valid_submit = save_pre_as_submit_format_csv(valid_data, pre_y)
    #     submit_csv_path = utils.save(valid_submit, file_dir=temp_result_path)
    #
    #     # 构造truth csv
    #     valid_answer = valid_data.loc[:, ['user_id']].drop_duplicates(['user_id'], keep='first')
    #     # answer中user列唯一、item列也是唯一
    #     valid_answer = valid_answer.merge(debias_track_answer, on='user_id', how='left')
    #     valid_answer_save_path = './cache/tmp_phase_submit/valid_answer.csv'
    #     valid_answer = valid_answer[['phase_id', 'user_id', 'item_id', 'item_deg']]
    #     valid_answer.to_csv(valid_answer_save_path, index=False, header=False)
    #
    #     score, \
    #     ndcg_50_full, ndcg_50_half, \
    #     hitrate_50_full, hitrate_50_half = evaluate(submit_csv_path, valid_answer_save_path, recall_num=50)  # todo 跑全量数据改成50
    #     # print('------------------------ valid set from sampling --------------------')
    #     print(
    #         'phase:{}, score:{}, ndcg_50_full:{}, ndcg_50_half:{}, hitrate_50_full:{}, hitrate_50_half:{}'.format(
    #             phase, score, ndcg_50_full, ndcg_50_half, hitrate_50_full, hitrate_50_half
    #         )
    #     )
    #
    #     if is_open_train_recall:
    #         time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #         print('------------------------ 模型验证 part2:训练集召回集合验证 start time:{}'.format(time_str))
    #         train_recall_top50 = train_user_recall_df.sort_values(
    #             ['user_id', 'itemcf_score'], ascending=False
    #         ).groupby(by='user_id').head(50).reset_index(drop=True)[['user_id', 'item_id']]
    #
    #         train_recall_top50 = \
    #             train_recall_top50.groupby(by='user_id').agg({'item_id':lambda x: ','.join(list(x))}).reset_index()
    #         train_recall_top50_path = utils.save(train_recall_top50, file_dir='./cache/tmp_phase_submit')
    #
    #         phase_answer_save_path = './cache/tmp_phase_submit/debias_track_answer.csv'
    #         # 由于召回训练集的时候做了采样，这里也要跟着只取采样那部分的user
    #         tmp_debias_track_answer = \
    #             debias_track_answer[debias_track_answer['user_id'].isin(train_recall_top50['user_id'])]
    #         tmp_debias_track_answer.to_csv(phase_answer_save_path, index=False, header=False)
    #
    #         score, \
    #         ndcg_50_full, ndcg_50_half, \
    #         hitrate_50_full, hitrate_50_half = evaluate(train_recall_top50_path, phase_answer_save_path, recall_num=50)
    #         print('------------------------ 召回结果未精排top50 --------------------')
    #         print(
    #             'phase:{}, score:{}, ndcg_50_full:{}, ndcg_50_half:{}, hitrate_50_full:{}, hitrate_50_half:{}'.format(
    #                 phase, score, ndcg_50_full, ndcg_50_half, hitrate_50_full, hitrate_50_half
    #             )
    #         )
    #
    #         # 预测
    #         recall_valid_pre_y = model.predict_proba(train_user_recall_x)[:, 1]
    #
    #         # 构造submit csv
    #         train_rank_result_csv = save_pre_as_submit_format_csv(train_user_recall_df, recall_valid_pre_y)
    #         train_rank_result_csv_path = utils.save(train_rank_result_csv, file_dir=temp_result_path)
    #
    #         score, \
    #         ndcg_50_full, ndcg_50_half, \
    #         hitrate_50_full, hitrate_50_half = evaluate(train_rank_result_csv_path, phase_answer_save_path, recall_num=50)
    #         print('------------------------ 召回结果精排后top50 --------------------')
    #         print(
    #             'phase:{}, score:{}, ndcg_50_full:{}, ndcg_50_half:{}, hitrate_50_full:{}, hitrate_50_half:{}'.format(
    #                 phase, score, ndcg_50_full, ndcg_50_half, hitrate_50_full, hitrate_50_half
    #             )
    #         )
    #
    #     time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     print('------------------------ underexpose_test_qtime 预测 start time:{}'.format(time_str))
    #     submit_pre_y = model.predict_proba(test_user_recall_x)[:, 1]
    #
    #     submit = save_pre_as_submit_format_csv(test_user_recall_df, submit_pre_y)
    #     submit_all = submit_all.append(submit)
    #
    # print('--------------------------- 保存预测文件 --------------------------')
    # utils.save(submit_all, 50)

    # todo 统计不同阶段时间分段情况
    # todo user点击深度超过10的情况怎么处理