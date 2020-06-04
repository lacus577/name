import os
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from pymodule import conf

def sampling_negtive_samples(user_set, negtive_df, sample_num=10):
    features = pd.DataFrame()
    for user in user_set:
        # TODO 采样比例调优
        # temp = negtive_df[negtive_df['user_id'] == user]  # 第一版错误，使用了user的历时点击item作为负样本
        temp = negtive_df[negtive_df['user_id'] != user]    # 第二版：使用全量点击序中非本user的item作为负样本
        sample_n = temp.shape[0] if temp.shape[0] < sample_num else sample_num
        temp = temp.sample(n=sample_n, replace=False, random_state=1)
        temp.loc[:, 'user_id'] = user
        features = features.append(temp)

    return features

def subsampling_user(sample_df, sample_num):
    # TODO 修改成打散的？
    sub_user_list = sorted(list(set(sample_df['user_id'])))[:sample_num]
    sample_df = sample_df[sample_df['user_id'].isin(sub_user_list)]

    # 用户行为序中点击item少于等于2个的，都删除 由于需要将最后一个item作为标签，所以只剩下一个无法构建item相似度矩阵
    single_item_user_list = list(
        sample_df['user_id'].value_counts()[sample_df['user_id'].value_counts() <= 2].index)
    click_train = sample_df[~sample_df['user_id'].isin(single_item_user_list)]

    return click_train

def save(submit_all, topk=None, file_dir=None):
    import time

    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    file_name = 'submit{time_str}.csv'.format(time_str=int(time.time()))
    if file_dir:
        file_path = os.path.join(file_dir, file_name)
    else:
        file_path = os.path.join('./', file_name)

    with open(file_path, 'w') as f:
        for i, row in submit_all.iterrows():
            user_id = str(row['user_id'])

            item_list = str(row['item_id']).split(',')
            if topk:
                item_list = item_list[:topk]

            if topk:
                assert len(set(item_list)) == topk

            line = user_id + ',' + ','.join(item_list) + '\n'
            if topk:
                assert len(line.strip().split(',')) == (topk + 1)

            f.write(line)

    return file_path


def read_train_click(train_path, phase):
    return pd.read_csv(
        train_path + '/underexpose_train_click-{phase}.csv'.format(phase=phase)
        , header=None
        # , nrows=nrows
        , names=['user_id', 'item_id', 'time']
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.float}
    )

def read_test_click(test_path, phase):
    return pd.read_csv(
        test_path + '/underexpose_test_click-{phase}/underexpose_test_click-{phase}.csv'.format(phase=phase)
        , header=None
        # , nrows=nrows
        , names=['user_id', 'item_id', 'time']
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.float}
    )

def read_qtime(test_path, phase):
    return pd.read_csv(
        test_path + '/underexpose_test_click-{phase}/underexpose_test_qtime-{phase}.csv'.format(phase=phase)
        , header=None
        # , nrows=nrows
        , names=['user_id', 'time']
        , sep=','
        , dtype={'user_id': np.str, 'time': np.float}
    )

def two_columns_df2dict(df):
    result_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return result_dict


def del_qtime_future_click(df):
    qtime_user_set = set(df[df['train_or_test'] == 'predict']['user_id'])

    result_df = df[~df['user_id'].isin(qtime_user_set)]
    for qtime_user in tqdm(qtime_user_set):
        tmp = df[df['user_id'] == qtime_user]
        tmp = tmp[tmp['time'] <= tmp[tmp['train_or_test'] == 'predict']['time'].iloc[0]]

        result_df = result_df.append(tmp)

    return result_df


def del_dup(df):
    df = df.drop_duplicates(['user_id', 'item_id', 'time'])
    return df

def process_time(df, time_stamp):
    df.loc[:, 'time'] = df['time'] * time_stamp
    return df

def transfer_item_features_df2dict(item_features, dim):
    txt_vec = item_features.apply(lambda x: np.array(list(x.iloc[-dim-dim: -dim])).reshape(-1, ), axis=1)
    img_vec = item_features.apply(lambda x: np.array(list(x.iloc[-dim: ])).reshape(-1, ), axis=1)
    user_key = item_features['item_id']

    assert len(user_key) == len(txt_vec) and len(user_key) == len(img_vec)

    user_features_dict = {}
    user_features_dict['txt_vec'] = dict(zip(user_key, txt_vec))
    user_features_dict['img_vec'] = dict(zip(user_key, img_vec))

    return user_features_dict

def transfer_user_features_df2dict(user_features, dim):
    txt_vec = user_features.apply(lambda x: np.array(list(x.iloc[-dim-dim: -dim])).reshape(-1, ), axis=1)
    img_vec = user_features.apply(lambda x: np.array(list(x.iloc[-dim: ])).reshape(-1, ), axis=1)
    user_key = user_features['user_id']

    assert len(user_key) == len(txt_vec) and len(user_key) == len(img_vec)

    user_features_dict = {}
    user_features_dict['txt_vec'] = dict(zip(user_key, txt_vec))
    user_features_dict['img_vec'] = dict(zip(user_key, img_vec))

    return user_features_dict

def save_pre_as_submit_format_csv(data_df, out_y):
    # 构造submit格式csv
    valid_eval_data = data_df[['user_id', 'item_id']]
    valid_eval_data['pred_prob'] = out_y
    valid_eval_data['rank'] = valid_eval_data.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
    valid_eval_data.sort_values(['rank'], inplace=True)

    valid_submit = valid_eval_data.groupby(['user_id'])['item_id'].agg(lambda x: ','.join(list(x))).reset_index()
    return valid_submit


def get_user2kitem_dict(df, k):
    df = df.sort_values(['user_id', 'time']).reset_index()
    df = df.groupby('user_id').head(k)
    df = df.groupby('user_id').agg({'item_id': lambda x: ','.join(list(x))}).reset_index()
    df.loc[:, 'item_id'] = df.apply(
        lambda x: list(x['item_id'].split(',')),
        axis=1
    )

    result_dict = dict(zip(df['user_id'], df['item_id']))
    return result_dict

def get_user2click_span_dict(df):
    # df = df.sort_values(['user_id', 'time'])
    df = df.groupby('user_id').agg({'time': lambda x: np.max(list(x)) - np.min(list(x))}).reset_index()
    result_dict = dict(zip(df['user_id'], df['time']))

    return result_dict

def get_user2total_deg_dict(df):
    tmp = df.groupby('user_id').agg({'item_deg': lambda x: np.sum(list(x))}).reset_index()

    result_dict = dict(zip(tmp['user_id'], tmp['item_deg']))

    return result_dict

def read_item_user_info():
    ''' 读取item和user属性 '''
    train_underexpose_item_feat_path = os.path.join(conf.train_path, 'underexpose_item_feat.csv')
    train_underexpose_user_feat_path = os.path.join(conf.train_path, 'underexpose_user_feat.csv')

    train_underexpose_item_feat_df_columns = \
        ['item_id'] + \
        ['txt_vec' + str(i) for i in range(conf.org_embedding_dim)] + \
        ['img_vec' + str(i) for i in range(conf.org_embedding_dim)]
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

    txt_embedding_df = item_info_df[['txt_vec{}'.format(i) for i in range(conf.org_embedding_dim)]]
    img_embedding_df = item_info_df[['img_vec{}'.format(i) for i in range(conf.org_embedding_dim)]]

    # todo 降维后信息丢失情况
    # item_txt_embedding_dim = item_img_embedding_dim = 32  # 降维后维度
    short_txt_embedding = \
        PCA(n_components=conf.new_embedding_dim).fit_transform(txt_embedding_df.values)
    short_img_embedding = \
        PCA(n_components=conf.new_embedding_dim).fit_transform(img_embedding_df.values)

    item_info_df = item_info_df[['item_id']]
    item_info_df = pd.concat(
        [item_info_df,
         pd.DataFrame(data=short_txt_embedding,
                      columns=['txt_vec{}'.format(i) for i in range(conf.new_embedding_dim)])],
        axis=1
    )
    item_info_df = pd.concat(
        [item_info_df,
         pd.DataFrame(data=short_img_embedding,
                      columns=['img_vec{}'.format(i) for i in range(conf.new_embedding_dim)])],
        axis=1
    )

    return item_info_df

def read_all_phase_click():
    if conf.is_click_cached:
        all_phase_click_666 = pd.read_csv(conf.click_cache_path, dtype={'user_id': np.str, 'item_id': np.str})
        ''' sampling '''
        if conf.subsampling:
            all_phase_click_666 = subsampling_user(all_phase_click_666, conf.subsampling)
        print('load all click, shape:{}'.format(all_phase_click_666.shape))
    else:
        all_phase_click_org = pd.DataFrame()
        for phase in range(0, conf.now_phase + 1):
            one_phase_train_click = read_train_click(conf.train_path, phase)
            one_phase_test_click = read_test_click(conf.test_path, phase)
            one_phase_qtime = read_qtime(conf.test_path, phase)

            one_phase_test_click['phase'] = str(phase)
            one_phase_test_click['train_or_test'] = 'test'
            one_phase_train_click['phase'] = str(phase)
            one_phase_train_click['train_or_test'] = 'train'
            one_phase_qtime['phase'] = str(phase)
            one_phase_qtime['train_or_test'] = 'predict'
            one_phase_qtime['item_id'] = None

            all_phase_click_org = all_phase_click_org.append(one_phase_train_click).reset_index(drop=True)
            all_phase_click_org = all_phase_click_org.append(one_phase_test_click).reset_index(drop=True)
            all_phase_click_org = all_phase_click_org.append(one_phase_qtime).reset_index(drop=True)

        ''' sampling '''
        if conf.subsampling:
            all_phase_click_org = subsampling_user(all_phase_click_org, conf.subsampling)

        # 删除重复点击
        all_phase_click = del_dup(all_phase_click_org)
        # 删除待预测时间点 之后的点击数据 防止数据泄露
        # all_phase_click_666 = utils.del_qtime_future_click(all_phase_click)
        # 时间处理 乘上 1591891140， 否则时间做操作结果太小，防止溢出
        all_phase_click_666 = process_time(all_phase_click, 1591891140)

        all_phase_click_666 = all_phase_click_666.sort_values(['user_id', 'time']).reset_index(drop=True)
        all_phase_click_666.to_csv(conf.click_cache_path, index=False)

    return all_phase_click_666


def get_candidate_positive_samples(df):
    tmp = df.groupby('user_id')['item_id'].count().reset_index()
    tmp = tmp[tmp['item_id'] > conf.candidate_positive_num + 2]     # 删掉候选正样本后训练集里面至少有两个点击记录用于训练
    df = df[df['user_id'].isin(tmp['user_id'])].sample(frac=1, random_state=1)
    return df.groupby('user_id').head(conf.candidate_positive_num).reset_index(drop=True)