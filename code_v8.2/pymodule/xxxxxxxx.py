root_path = '../../data'
import pandas as pd
import numpy as np

from pymodule import utils

# part1_test_user_recall_features_phase_3.csv
submit_all = pd.DataFrame()

# for i in range(0, 7):
#     test_user = pd.read_csv(
#         './cache/features_cache/part1_test_user_recall_features_phase_{}.csv'.format(i),
#         dtype={'user_id': np.str, 'item_id': np.str}
#     )
#
#     test_user = test_user.sort_values(['user_id', 'itemcf_score'], ascending=False).reset_index(drop=True)
#     test_user = test_user.groupby('user_id').head(50).reset_index(drop=True)
#     test_user = test_user.groupby('user_id').agg({'item_id': lambda x: ','.join(list(x))}).reset_index()
#     submit_all = submit_all.append(test_user)
#
# utils.save(submit_all, 50)

# for i in range(0, 7):
#     part1_test_user = pd.read_csv(
#         './cache/features_cache/part1_test_user_recall_features_phase_{}.csv'.format(i),
#         dtype={'user_id': np.str, 'item_id': np.str}
#     )
#     part1_train_user = pd.read_csv(
#         './cache/features_cache/part1_train_features_phase_{}.csv'.format(i),
#         dtype={'user_id': np.str, 'item_id': np.str}
#     )
#     part1_xxtest_user = pd.read_csv(
#         './cache/features_cache/part1_valid_features_phase_{}.csv'.format(i),
#         dtype={'user_id': np.str, 'item_id': np.str}
#     )
#
#     part2_test_user = pd.read_csv(
#         './cache/features_cache/part2_test_user_recall_features_phase_{}.csv'.format(i),
#         dtype={'user_id': np.str, 'item_id': np.str}
#     )
#     part2_train_user = pd.read_csv(
#         './cache/features_cache/part2_train_features_phase_{}.csv'.format(i),
#         dtype={'user_id': np.str, 'item_id': np.str}
#     )
#     part2_xxtest_user = pd.read_csv(
#         './cache/features_cache/part2_valid_features_phase_{}.csv'.format(i),
#         dtype={'user_id': np.str, 'item_id': np.str}
#     )
#
#     test_user = part1_test_user.merge(part2_test_user, on=['user_id', 'item_id', 'itemcf_score'], how='left')
#     train_user = part1_train_user.merge(part2_train_user, on=['user_id', 'item_id', 'label', 'itemcf_score'], how='left')
#     xxtest_user = part1_xxtest_user.merge(part2_xxtest_user, on=['user_id', 'item_id', 'label', 'itemcf_score'], how='left')
#     print(test_user.columns)
#
#     # print(test_user)
#     test_user.to_csv('./cache/features_cache/test_user_recall_features_phase_{}.csv'.format(i), index=False)
#     train_user.to_csv('./cache/features_cache/train_features_phase_{}.csv'.format(i), index=False)
#     xxtest_user.to_csv('./cache/features_cache/valid_features_phase_{}.csv'.format(i), index=False)


df_columns = ['user_id', 'item_id', 'time']

import os

click_all = pd.DataFrame()
for phase in range(0, 7):
    test_underexpose_test_click_path = os.path.join(root_path,
                                                      'underexpose_test/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(
                                                          phase, phase))
    test_underexpose_test_qtime_0_path = os.path.join(root_path, 'underexpose_test/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(phase, phase))

    train_underexpose_train_click_path = os.path.join(root_path,
                                                        'underexpose_train/underexpose_train_click-{}.csv'.format(phase))

    train_underexpose_train_click_0_df = pd.read_csv(train_underexpose_train_click_path, names=df_columns,
        dtype={'user_id': np.str, 'item_id': np.str, 'time':np.float}
    )

    test_underexpose_test_click_0_df = pd.read_csv(test_underexpose_test_click_path, names=df_columns,
        dtype={'user_id': np.str, 'item_id': np.str, 'time':np.float}
    )

    qtime = pd.read_csv(test_underexpose_test_qtime_0_path, names=['user_id', 'time'], dtype={'user_id': np.str, 'time':np.float})

    test_underexpose_test_click_0_df['phase'] = str(phase)
    test_underexpose_test_click_0_df['train_or_test'] = 'test'
    train_underexpose_train_click_0_df['phase'] = str(phase)
    train_underexpose_train_click_0_df['train_or_test'] = 'train'

    # print(qtime)
    qtime['phase'] = str(phase)
    qtime['item_id'] = np.nan
    qtime['train_or_test'] = 'predict'


    click_all = click_all.append(test_underexpose_test_click_0_df).reset_index(drop=True)
    click_all = click_all.append(train_underexpose_train_click_0_df).reset_index(drop=True)
    click_all = click_all.append(qtime).reset_index(drop=True)

click_all = click_all.sort_values(['user_id', 'time'], ascending=True)
click_all = click_all.groupby(['user_id', 'item_id', 'time']).agg({'phase': lambda x: ','.join(list(x)), 'train_or_test': lambda x: ','.join(list(x))}).reset_index()

click_all.loc[:, 'new_time'] = click_all['time'] * 1591891140
click_all = click_all.sort_values(['user_id', 'time'], ascending=True)
click_all['time_interval'] = list(np.array(list(click_all['new_time'][1: ])) - np.array(list(click_all['new_time'][: -1])))  + [111]

print(
    np.nanmean(click_all[click_all['time_interval'] > 0]['time_interval']),
    np.nanmin(click_all[click_all['time_interval'] > 0]['time_interval']),
    np.nanmax(click_all[click_all['time_interval'] > 0]['time_interval']),
    np.nanmedian(click_all[click_all['time_interval'] > 0]['time_interval'])
)

# click_all = click_all.groupby(['user_id', 'item_id']).count().reset_index()


# click_all.to_csv('./click_all.csv', index=False)


train_path = '../../data/underexpose_train'
train_underexpose_item_feat_path = os.path.join(train_path, 'underexpose_item_feat.csv')
train_underexpose_item_feat_df_columns = ['item_id'] + ['txt_vec' + str(i) for i in range(128)] + ['img_vec' + str(i) for i in range(128)]
item_info_df = pd.read_csv(train_underexpose_item_feat_path, names=train_underexpose_item_feat_df_columns, dtype={'item_id': np.str})
# 删除[ 和 ]
item_info_df['txt_vec0'] = \
    item_info_df['txt_vec0'].apply(lambda x: float(str(x)[1:]) if x is not np.nan else x)
item_info_df['img_vec0'] = \
    item_info_df['img_vec0'].apply(lambda x: float(str(x)[1:]) if x is not np.nan else x)
item_info_df['txt_vec127'] = \
    item_info_df['txt_vec127'].apply(lambda x: float(str(x)[:-1]) if x is not np.nan else x)
item_info_df['img_vec127'] = \
    item_info_df['img_vec127'].apply(lambda x: float(str(x)[:-1]) if x is not np.nan else x)
from pymodule import train
item_info_dict = train.transfer_item_features_df2dict(item_info_df, 128)

from pymodule.featuring import my_cos_sim
# 同一个user点击的item，时间间隔小于中位数的时候，平均item相似度如何
txt_matched_sim = 0
img_matched_sim = 0
txt_total_sim = 0
img_total_sim = 0
txt_matched_sim_count = 0
img_matched_sim_count = 0
txt_total_sim_count = 0
img_total_sim_count = 0
total_num = 0
for i in range(click_all.shape[0]):
    if i+1 >= click_all.shape[0]:
        break

    tmp = my_cos_sim(item_info_dict['txt_vec'].get(click_all.loc[i, 'item_id']),
                                     item_info_dict['txt_vec'].get(click_all.loc[i + 1, 'item_id']))
    if tmp is not None and tmp is not np.inf and tmp is not np.nan:
        txt_total_sim += tmp
        txt_total_sim_count += 1


    tmp = my_cos_sim(item_info_dict['img_vec'].get(click_all.loc[i, 'item_id']),
                                     item_info_dict['img_vec'].get(click_all.loc[i + 1, 'item_id']))
    if tmp is not None and tmp is not np.inf and tmp is not np.nan:
        img_total_sim += tmp
        img_total_sim_count += 1

    time_interval = click_all.loc[i, 'time_interval']
    if time_interval >= 0 and time_interval <= 500:
        total_num += 1
        tmp = my_cos_sim(item_info_dict['txt_vec'].get(click_all.loc[i, 'item_id']), item_info_dict['txt_vec'].get(click_all.loc[i + 1, 'item_id']))
        if tmp is not None and tmp is not np.inf and tmp is not np.nan:
            txt_matched_sim += tmp
            txt_matched_sim_count += 1

        tmp = my_cos_sim(item_info_dict['img_vec'].get(click_all.loc[i, 'item_id']), item_info_dict['img_vec'].get(click_all.loc[i + 1, 'item_id']))
        if tmp is not None and tmp is not np.inf and tmp is not np.nan:
            img_matched_sim += tmp
            img_matched_sim_count += 1

        if total_num % 20000 == 0:
            print('same session: cur num:{}, cur txt sim:{}, cur img sim:{}'.format(total_num, txt_matched_sim / txt_matched_sim_count, img_matched_sim / img_matched_sim_count))
            print('total: cur num:{}, cur txt sim:{}, cur img sim:{}'.format(i, txt_total_sim / txt_total_sim_count, img_total_sim / img_total_sim_count))

txt_sim = 0
img_sim = 0
for i in range(item_info_df.shape[0] - 100):
    txt_sim += my_cos_sim(item_info_dict['txt_vec'].get(item_info_df.loc[i, 'item_id']), item_info_dict['txt_vec'].get(item_info_df.loc[i + 1, 'item_id']))
    img_sim += my_cos_sim(item_info_dict['img_vec'].get(item_info_df.loc[i, 'item_id']), item_info_dict['img_vec'].get(item_info_df.loc[i + 1, 'item_id']))

    if i % 2000 == 0:
        print('item: cur num:{}, cur txt sim:{}, cur img sim:{}'.format(i, txt_sim / i,
                                                                                img_sim / i))
