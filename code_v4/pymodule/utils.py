import os
from tqdm import tqdm

import pandas as pd
import numpy as np

def sampling_negtive_samples(user_set, negtive_df, sample_num=10):
    features = pd.DataFrame()
    for user in user_set:
        # TODO 采样比例调优
        # temp = negtive_df[negtive_df['user_id'] == user]  # 第一版错误，使用了user的历时点击item作为负样本
        temp = negtive_df[negtive_df['user_id'] != user]    # 第二版：使用全量点击序中非本user的item作为负样本
        sample_n = temp.shape[0] if temp.shape[0] < sample_num else sample_num
        temp = temp.sample(n=sample_n, replace=False, random_state=1, axis=0)
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
