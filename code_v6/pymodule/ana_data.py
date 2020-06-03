import pandas as pd
import numpy as np

from pymodule import conf

# click_df = pd.read_csv(conf.click_cache_path, dtype={'user_id': np.str, 'item_id': np.str})
# print('train user num:{}'.format(
#     len(set(click_df[click_df['train_or_test'] == 'train']['user_id']))
# ))
# print('test user num:{}'.format(
#     len(set(click_df[click_df['train_or_test'] == 'test']['user_id']))
# ))
# print('qtime user num:{}'.format(
#     len(set(click_df[click_df['train_or_test'] == 'predict']['user_id']))
# ))
#
# print('cold user num:{}'.format(
#     len(set(click_df[click_df['train_or_test'] == 'predict']['user_id'])) -
#     len(set(click_df[click_df['train_or_test'] == 'test']['user_id']))
# ))
#
# print('time interval:{}'.format(
#     np.max(click_df['time']) - np.min(click_df['time'])
# ))
#
# # 不存在完全冷启动用户
# xx = click_df.groupby('user_id').agg({'train_or_test': lambda x: ','.join(set(list(x)))}).reset_index()
# xx = xx[xx['train_or_test'] != 'train']
# xx = xx[xx['train_or_test'] != 'predict,train,test']
# xx = xx[xx['train_or_test'] != 'train,test,predict']
# xx = xx[xx['train_or_test'] != 'test,predict,train']
# xx = xx[xx['train_or_test'] != 'test,train,predict']
# xx = xx[xx['train_or_test'] != 'train,predict,test']
# xx = xx[xx['train_or_test'] != 'test,predict']
# xx = xx[xx['train_or_test'] != 'predict,test']
# xx = xx[xx['train_or_test'] != 'train,predict']
# print(xx)


features = pd.read_csv(conf.features_cache_path)
print(features[features['user_id'] == 1][['user_id', 'item_id', 'label']])