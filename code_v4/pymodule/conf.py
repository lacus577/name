''''''
''' -------------------------- conf --------------------------- '''
''' 进程并发数 '''
process_num = 3

now_phase = 6


''' 样本采样个数 '''
subsampling = 3

''' 全量点击序 '''
is_click_cached = True

''' 训练集、测试集 '''
''' 训练集、测试集样本是否已经缓存 '''
is_samples_cached = True
''' 训练集、测试集特征是否已经缓存 '''
is_feature_cached = False

''' 召回结果 '''
''' 召回结果缓存 '''
is_recall_cached = True
''' 召回结果样本缓存 '''
is_recall_sample_cached = False
''' 召回结果特征缓存 '''
is_recall_feature_cached = False

''' itemcf相似度前k个'''
itemcf_num = 5


''' -------------------------- constant --------------------------- '''
train_path = '../../../../data/underexpose_train'
test_path = '../../../../data/underexpose_test'

''' 全量点击序 '''
click_cache_path = './cache/features_cache/click.csv'

''' 召回结果缓存路径 '''
# recall_cache_path = './cache/features_cache/test_user_recall_{}.csv'
recall_cache_path = './cache/features_cache/phase_{}_recall_100.csv'
recall_sample_path = './cache/features_cache/test_user_recall_sample_{}.csv'
recall_feature_path = './cache/features_cache/test_user_recall_feature_{}.csv'

''' 缓存路径合集 '''
samples_cache_path = './cache/features_cache/samples.csv'
features_cache_path = './cache/features_cache/features.csv'

# 用户点击最大深度
MAX_CLICK_LEN = 500

# 召回结果中相似度列名
# ITEM_CF_SCORE = 'itemcf_score'
ITEM_CF_SCORE = 'sim'