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
        , names=['user_id', 'item_id', conf.org_time_name]
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.float}
    )

def read_test_click(test_path, phase):
    return pd.read_csv(
        test_path + '/underexpose_test_click-{phase}/underexpose_test_click-{phase}.csv'.format(phase=phase)
        , header=None
        # , nrows=nrows
        , names=['user_id', 'item_id', conf.org_time_name]
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.float}
    )

def read_qtime(test_path, phase):
    return pd.read_csv(
        test_path + '/underexpose_test_click-{phase}/underexpose_test_qtime-{phase}.csv'.format(phase=phase)
        , header=None
        # , nrows=nrows
        , names=['user_id', conf.org_time_name]
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
    df[conf.new_time_name] = df[conf.org_time_name] * time_stamp
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
    df = df.sort_values(['user_id', 'time'], ascending=False).reset_index()
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

def get_user2total_deg_dict(df, day):
    tmp = df.groupby('user_id').agg({'{}day_item_deg'.format(day): lambda x: np.sum(list(x))}).reset_index()

    result_dict = dict(zip(tmp['user_id'], tmp['{}day_item_deg'.format(day)]))

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

def read_user_info():
    ''' 读取item和user属性 '''
    train_underexpose_user_feat_path = os.path.join(conf.train_path, 'underexpose_user_feat.csv')
    train_underexpose_user_feat_df_columns = \
        ['user_id', 'user_age_level', 'user_gender', 'user_city_level']

    user_info_df = pd.read_csv(
        train_underexpose_user_feat_path,
        names=train_underexpose_user_feat_df_columns,
        dtype={'user_id': np.str, 'item_id': np.str, 'time': np.float}
    )

    return user_info_df

def clean_user_info(df):
    df.loc[:, 'user_gender'] = df['user_gender'].replace({'F': 1, 'M': 2})
    return df

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
        all_phase_click_666 = process_time(all_phase_click, conf.time_puls)

        all_phase_click_666 = all_phase_click_666.sort_values(['user_id', 'time']).reset_index(drop=True)
        all_phase_click_666.to_csv(conf.click_cache_path, index=False)

    return all_phase_click_666


def get_candidate_positive_samples(df):
    tmp = df.groupby('user_id')['item_id'].count().reset_index()
    tmp = tmp[tmp['item_id'] > conf.candidate_positive_num + 2]     # 删掉候选正样本后训练集里面至少有两个点击记录用于训练
    df = df[df['user_id'].isin(tmp['user_id'])].sample(frac=1, random_state=1)
    return df.groupby('user_id').head(conf.candidate_positive_num).reset_index(drop=True)


def sava_user_features_dict(user_features_dict, save_path):
    result_df = pd.DataFrame(data=user_features_dict).reset_index()
    result_df = result_df.rename(columns={'index': 'user_id'})
    print(result_df)
    result_df.to_csv(save_path, index=False)

def user_features_df2dict(df):
    df = df.set_index('user_id', drop=True)
    df.index.name = ''
    result_dict = df.to_dict()
    return result_dict

def get_features(df, is_label, type):
    ''' 特征列顺序 重新 组织 '''
    pre_columns = ['user_id', 'item_id']
    recall_columns = [conf.ITEM_CF_SCORE]
    # part1_columns = ['click_item_user_sim', 'click_user_item_sim', 'user_click_num',
    #                 'user_click_interval_mean', 'user_click_interval_min',
    #                 'user_click_interval_max', 'item_deg', 'user_item_mean_deg',
    #                 'user_item_min_deg', 'user_item_max_deg',
    #                 '0_item2item_itemcf_score', 'item20_item_itemcf_score',
    #                 '1_item2item_itemcf_score', 'item21_item_itemcf_score',
    #                 '2_item2item_itemcf_score', 'item22_item_itemcf_score',
    #                 '3_item2item_itemcf_score', 'item23_item_itemcf_score',
    #                 '4_item2item_itemcf_score', 'item24_item_itemcf_score',
    #                 'user_avg_click', 'user_span_click', 'user_total_deg', 'user_avg_deg',
    #                 '0_item_deg', '1_item_deg', 'top_1_item_deg', '2_item_deg',
    #                 'top_2_item_deg', '3_item_deg', 'top_3_item_deg', '4_item_deg',
    #                 'top_4_item_deg']
    # part2_columns = ['1_day_user_txt_sim', '1_day_user_img_sim',
    #                 '2_day_user_txt_sim', '2_day_user_img_sim', '3_day_user_txt_sim',
    #                 '3_day_user_img_sim', '7_day_user_txt_sim', '7_day_user_img_sim',
    #                 'all_day_user_txt_sim', 'all_day_user_img_sim']
    part1_columns = ['1_day_user_txt_sim', '1_day_user_img_sim', '2_day_user_txt_sim', '2_day_user_img_sim', '3_day_user_txt_sim', '3_day_user_img_sim', '7_day_user_txt_sim', '7_day_user_img_sim', 'all_day_user_txt_sim', 'all_day_user_img_sim', 'click_item_user_sim', 'click_user_item_sim', '1day_user_click_num', '1day_item_deg', '1day_user_item_mean_deg', '1day_user_item_min_deg', '1day_user_item_max_deg', '1day_user_item_var_deg', '1day_user_item_median_deg', '1day_user_total_deg', '1day_user_avg_deg', '1day_0_item_deg', '1day_1_item_deg', '1day_top_1_item_deg', '1day_2_item_deg', '1day_top_2_item_deg', '1day_3_item_deg', '1day_top_3_item_deg', '1day_4_item_deg', '1day_top_4_item_deg', '1day_user_click_interval_mean', '1day_user_click_interval_min', '1day_user_click_interval_max', '1day_user_click_interval_var', '1day_user_click_interval_median', '1day_user_mean_click', '1day_user_median_click', '1day_user_span_click', '2day_user_click_num', '2day_item_deg', '2day_user_item_mean_deg', '2day_user_item_min_deg', '2day_user_item_max_deg', '2day_user_item_var_deg', '2day_user_item_median_deg', '2day_user_total_deg', '2day_user_avg_deg', '2day_0_item_deg', '2day_1_item_deg', '2day_top_1_item_deg', '2day_2_item_deg', '2day_top_2_item_deg', '2day_3_item_deg', '2day_top_3_item_deg', '2day_4_item_deg', '2day_top_4_item_deg', '2day_user_click_interval_mean', '2day_user_click_interval_min', '2day_user_click_interval_max', '2day_user_click_interval_var', '2day_user_click_interval_median', '2day_user_mean_click', '2day_user_median_click', '2day_user_span_click', '3day_user_click_num', '3day_item_deg', '3day_user_item_mean_deg', '3day_user_item_min_deg', '3day_user_item_max_deg', '3day_user_item_var_deg', '3day_user_item_median_deg', '3day_user_total_deg', '3day_user_avg_deg', '3day_0_item_deg', '3day_1_item_deg', '3day_top_1_item_deg', '3day_2_item_deg', '3day_top_2_item_deg', '3day_3_item_deg', '3day_top_3_item_deg', '3day_4_item_deg', '3day_top_4_item_deg', '3day_user_click_interval_mean', '3day_user_click_interval_min', '3day_user_click_interval_max', '3day_user_click_interval_var', '3day_user_click_interval_median', '3day_user_mean_click', '3day_user_median_click', '3day_user_span_click', '7day_user_click_num', '7day_item_deg', '7day_user_item_mean_deg', '7day_user_item_min_deg', '7day_user_item_max_deg', '7day_user_item_var_deg', '7day_user_item_median_deg', '7day_user_total_deg', '7day_user_avg_deg', '7day_0_item_deg', '7day_1_item_deg', '7day_top_1_item_deg', '7day_2_item_deg', '7day_top_2_item_deg', '7day_3_item_deg', '7day_top_3_item_deg', '7day_4_item_deg', '7day_top_4_item_deg', '7day_user_click_interval_mean', '7day_user_click_interval_min', '7day_user_click_interval_max', '7day_user_click_interval_var', '7day_user_click_interval_median', '7day_user_mean_click', '7day_user_median_click', '7day_user_span_click', 'earlierday_user_click_num', 'earlierday_item_deg', 'earlierday_user_item_mean_deg', 'earlierday_user_item_min_deg', 'earlierday_user_item_max_deg', 'earlierday_user_item_var_deg', 'earlierday_user_item_median_deg', 'earlierday_user_total_deg', 'earlierday_user_avg_deg', 'earlierday_0_item_deg', 'earlierday_1_item_deg', 'earlierday_top_1_item_deg', 'earlierday_2_item_deg', 'earlierday_top_2_item_deg', 'earlierday_3_item_deg', 'earlierday_top_3_item_deg', 'earlierday_4_item_deg', 'earlierday_top_4_item_deg', 'earlierday_user_click_interval_mean', 'earlierday_user_click_interval_min', 'earlierday_user_click_interval_max', 'earlierday_user_click_interval_var', 'earlierday_user_click_interval_median', 'earlierday_user_mean_click', 'earlierday_user_median_click', 'earlierday_user_span_click', 'allday_user_click_num', 'allday_item_deg', 'allday_user_item_mean_deg', 'allday_user_item_min_deg', 'allday_user_item_max_deg', 'allday_user_item_var_deg', 'allday_user_item_median_deg', 'allday_user_total_deg', 'allday_user_avg_deg', 'allday_0_item_deg', 'allday_1_item_deg', 'allday_top_1_item_deg', 'allday_2_item_deg', 'allday_top_2_item_deg', 'allday_3_item_deg', 'allday_top_3_item_deg', 'allday_4_item_deg', 'allday_top_4_item_deg', 'allday_user_click_interval_mean', 'allday_user_click_interval_min', 'allday_user_click_interval_max', 'allday_user_click_interval_var', 'allday_user_click_interval_median', 'allday_user_mean_click', 'allday_user_median_click', 'allday_user_span_click', 'allday_0_item2item_itemcf_score', 'allday_item20_item_itemcf_score', 'allday_1_item2item_itemcf_score', 'allday_item21_item_itemcf_score', 'allday_2_item2item_itemcf_score', 'allday_item22_item_itemcf_score', 'allday_3_item2item_itemcf_score', 'allday_item23_item_itemcf_score', 'allday_4_item2item_itemcf_score', 'allday_item24_item_itemcf_score']
    part2_columns = ['1_day_user_txt_sim', '1_day_user_img_sim', '2_day_user_txt_sim', '2_day_user_img_sim', '3_day_user_txt_sim', '3_day_user_img_sim', '7_day_user_txt_sim', '7_day_user_img_sim', 'all_day_user_txt_sim', 'all_day_user_img_sim', 'click_item_user_sim', 'click_user_item_sim', '1day_user_click_num', '1day_item_deg', '1day_user_item_mean_deg', '1day_user_item_min_deg', '1day_user_item_max_deg', '1day_user_item_var_deg', '1day_user_item_median_deg', '1day_user_total_deg', '1day_user_avg_deg', '1day_0_item_deg', '1day_1_item_deg', '1day_top_1_item_deg', '1day_2_item_deg', '1day_top_2_item_deg', '1day_3_item_deg', '1day_top_3_item_deg', '1day_4_item_deg', '1day_top_4_item_deg', '1day_user_click_interval_mean', '1day_user_click_interval_min', '1day_user_click_interval_max', '1day_user_click_interval_var', '1day_user_click_interval_median', '1day_user_mean_click', '1day_user_median_click', '1day_user_span_click', '2day_user_click_num', '2day_item_deg', '2day_user_item_mean_deg', '2day_user_item_min_deg', '2day_user_item_max_deg', '2day_user_item_var_deg', '2day_user_item_median_deg', '2day_user_total_deg', '2day_user_avg_deg', '2day_0_item_deg', '2day_1_item_deg', '2day_top_1_item_deg', '2day_2_item_deg', '2day_top_2_item_deg', '2day_3_item_deg', '2day_top_3_item_deg', '2day_4_item_deg', '2day_top_4_item_deg', '2day_user_click_interval_mean', '2day_user_click_interval_min', '2day_user_click_interval_max', '2day_user_click_interval_var', '2day_user_click_interval_median', '2day_user_mean_click', '2day_user_median_click', '2day_user_span_click', '3day_user_click_num', '3day_item_deg', '3day_user_item_mean_deg', '3day_user_item_min_deg', '3day_user_item_max_deg', '3day_user_item_var_deg', '3day_user_item_median_deg', '3day_user_total_deg', '3day_user_avg_deg', '3day_0_item_deg', '3day_1_item_deg', '3day_top_1_item_deg', '3day_2_item_deg', '3day_top_2_item_deg', '3day_3_item_deg', '3day_top_3_item_deg', '3day_4_item_deg', '3day_top_4_item_deg', '3day_user_click_interval_mean', '3day_user_click_interval_min', '3day_user_click_interval_max', '3day_user_click_interval_var', '3day_user_click_interval_median', '3day_user_mean_click', '3day_user_median_click', '3day_user_span_click', '7day_user_click_num', '7day_item_deg', '7day_user_item_mean_deg', '7day_user_item_min_deg', '7day_user_item_max_deg', '7day_user_item_var_deg', '7day_user_item_median_deg', '7day_user_total_deg', '7day_user_avg_deg', '7day_0_item_deg', '7day_1_item_deg', '7day_top_1_item_deg', '7day_2_item_deg', '7day_top_2_item_deg', '7day_3_item_deg', '7day_top_3_item_deg', '7day_4_item_deg', '7day_top_4_item_deg', '7day_user_click_interval_mean', '7day_user_click_interval_min', '7day_user_click_interval_max', '7day_user_click_interval_var', '7day_user_click_interval_median', '7day_user_mean_click', '7day_user_median_click', '7day_user_span_click', 'earlierday_user_click_num', 'earlierday_item_deg', 'earlierday_user_item_mean_deg', 'earlierday_user_item_min_deg', 'earlierday_user_item_max_deg', 'earlierday_user_item_var_deg', 'earlierday_user_item_median_deg', 'earlierday_user_total_deg', 'earlierday_user_avg_deg', 'earlierday_0_item_deg', 'earlierday_1_item_deg', 'earlierday_top_1_item_deg', 'earlierday_2_item_deg', 'earlierday_top_2_item_deg', 'earlierday_3_item_deg', 'earlierday_top_3_item_deg', 'earlierday_4_item_deg', 'earlierday_top_4_item_deg', 'earlierday_user_click_interval_mean', 'earlierday_user_click_interval_min', 'earlierday_user_click_interval_max', 'earlierday_user_click_interval_var', 'earlierday_user_click_interval_median', 'earlierday_user_mean_click', 'earlierday_user_median_click', 'earlierday_user_span_click', 'allday_user_click_num', 'allday_item_deg', 'allday_user_item_mean_deg', 'allday_user_item_min_deg', 'allday_user_item_max_deg', 'allday_user_item_var_deg', 'allday_user_item_median_deg', 'allday_user_total_deg', 'allday_user_avg_deg', 'allday_0_item_deg', 'allday_1_item_deg', 'allday_top_1_item_deg', 'allday_2_item_deg', 'allday_top_2_item_deg', 'allday_3_item_deg', 'allday_top_3_item_deg', 'allday_4_item_deg', 'allday_top_4_item_deg', 'allday_user_click_interval_mean', 'allday_user_click_interval_min', 'allday_user_click_interval_max', 'allday_user_click_interval_var', 'allday_user_click_interval_median', 'allday_user_mean_click', 'allday_user_median_click', 'allday_user_span_click', 'allday_0_item2item_itemcf_score', 'allday_item20_item_itemcf_score', 'allday_1_item2item_itemcf_score', 'allday_item21_item_itemcf_score', 'allday_2_item2item_itemcf_score', 'allday_item22_item_itemcf_score', 'allday_3_item2item_itemcf_score', 'allday_item23_item_itemcf_score', 'allday_4_item2item_itemcf_score', 'allday_item24_item_itemcf_score']
    if type == 0:
        features_columns = pre_columns + recall_columns + part1_columns + part2_columns
    elif type == 1:
        features_columns = pre_columns + recall_columns + part1_columns
    elif type == 2:
        features_columns = pre_columns + recall_columns + part2_columns
    else:
        raise Exception('columns error.')

    if is_label:
        df = df[features_columns + ['label']]
    else:
        df = df[features_columns]

    return df