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
is_recall_sample_cached = True
''' 召回结果特征缓存 '''
is_recall_feature_cached = False

''' itemcf相似度前k个'''
itemcf_num = 5

''' 候选正样本个数 '''
candidate_positive_num = 1

''' 召回数量 '''
recall_num = 50

''' 负样本数量 '''
negative_num = 15

''' 官方embedding原始向量维度 '''
org_embedding_dim = 128
''' 处理后维度'''
new_embedding_dim = 32

''' 留出验证次数 '''
k = 5

''' 时间区间 '''
time_periods = [1, 2, 3, 7]


''' -------------------------- constant --------------------------- '''
train_path = '../../../../data/underexpose_train'
test_path = '../../../../data/underexpose_test'

''' 全量点击序 '''
click_cache_path = './cache/features_cache/click.csv'

''' 召回结果缓存路径 '''
sim_list_path = './cache/features_cache/item_sim_list_{}'
total_sim_list_path = './cache/features_cache/item_sim_list'
# recall_cache_path = './cache/features_cache/test_user_recall_{}.csv'
recall_cache_path = './cache/features_cache/phase_{}_recall_500.csv'
recall_sample_path = './cache/features_cache/test_user_recall_sample_{}.csv'
recall_feature_path = './cache/features_cache/test_user_recall_feature_{}.csv'

''' 缓存路径合集 '''
samples_cache_path = './cache/features_cache/samples_{}.csv'
features_cache_path = './cache/features_cache/features_{}.csv'

''' 训练集user召回结果缓存 '''
total_user_recall_path = './cache/features_cache/total_user_recall_500_{}.csv'

''' 用户画像特征 '''
user_features_path = './cache/features_cache/user_features.csv'
user_emb_features_path = './cache/features_cache/user_emb_features.csv'

# 用户点击最大深度
MAX_CLICK_LEN = 500

# 召回结果中相似度列名
# ITEM_CF_SCORE = 'itemcf_score'
ITEM_CF_SCORE = 'sim'

''' 数据分段， 官方给的时间是10+天， 先假设给的是14天， 将全量数据分成14段， 一段表示一天 '''
days = 14

''' 时间预处理 '''
org_time_name = 'time'
new_time_name = 'new_time'
time_puls = 1591891140