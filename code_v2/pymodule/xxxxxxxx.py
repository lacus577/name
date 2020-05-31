import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# warnings.filterwarnings("ignore")

now_phase = 6
train_path = '../../data/underexpose_train'
test_path = '../../data/underexpose_test'

# train
flag_append = False
flag_test = False
recall_num = 50
topk = 50
nrows = None
subsampling = 100

# test
# flag_append = False
# flag_test = True
# recall_num = 50
# topk = 50
# nrows = 1000

def metrics_recall(topk_recall, phase, k, sep=10):
    data_ = topk_recall[topk_recall['pred'] == 'train'].sort_values(['user_id', 'score_similar'], ascending=False)

    # 处理之后，一个user一行
    data_ = data_.groupby(['user_id']).agg(
        {'item_similar': lambda x: list(x), 'next_item_id': lambda x: ''.join(set(x))})

    '''
    -1 表示此user召回没有命中
       其他表示此user召回命中，并且数值表示命中的位置
    '''
    data_['index'] = [recall_.index(label_) if label_ in recall_ else -1 for (label_, recall_) in
                      zip(data_['next_item_id'], data_['item_similar'])]

    print('-------- 召回效果 -------------')
    print('--------:phase: ', phase, ' -------------')
    data_num = len(data_)
    for topk in range(0, k + 1, sep):
        hit_num = len(data_[(data_['index'] != -1) & (data_['index'] <= topk)])
        hit_rate = hit_num * 1.0 / data_num
        print('phase: ', phase, ' top_', topk, ' : ', 'hit_num : ', hit_num, 'hit_rate : ', hit_rate, ' data_num : ',
              data_num)
        print()

    hit_rate = len(data_[data_['index'] != -1]) * 1.0 / data_num
    return hit_rate


def train_model_rf(feature_all, recall_rate, hot_list, valid=0.2, topk=50):
    from sklearn.ensemble import RandomForestClassifier
    print('------- 训练模型 -----------')
    # todo 只有正样本？
    train_data = feature_all[feature_all['train_flag'] == 'train']
    test_data = feature_all[feature_all['train_flag'] == 'test']

    '''训练集切成两个集合，一个训练 一个测试'''
    df_user = pd.DataFrame(list(set(train_data['user_id'])))
    df_user.columns = ['user_id']

    df = df_user.sample(frac=1.0)
    cut_idx = int(round(valid * df.shape[0]))
    df_train_0, df_train_1 = df.iloc[:cut_idx], df.iloc[cut_idx:]

    train_data_0 = df_train_0.merge(train_data, on=['user_id'], how='left')
    train_data_1 = df_train_1.merge(train_data, on=['user_id'], how='left')

    # f0-f128特征：embedding的特征
    f_col = [c for c in feature_all.columns if c not in ['train_flag', 'label', 'user_id', 'item_similar']]
    f_label = 'label'

    X0 = train_data_0[f_col].values
    y0 = train_data_0[f_label].values

    X1 = train_data_1[f_col].values
    y1 = train_data_1[f_label].values

    '''
    测试集，待分类模型进行是否购买预测
    TODO 只需要对qtime中的user进行购买预测即可，不需要对整个test集合进行预测
    '''
    X_pred = test_data[f_col].values

    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X0, y0)

    print("------------- eval -------------")
    train_eval = train_data_1[['user_id', 'item_similar', 'label']]
    len_hot = len(hot_list)
    high_half_item, low_half_item = hot_list[:len_hot // 2], hot_list[len_hot // 2:]
    train_eval['half'] = train_eval['item_similar'].map(lambda x: 1 if x in low_half_item else 0)

    y1_pred = clf.predict_proba(X1)[:, 1]
    train_eval['pred_prob'] = y1_pred

    train_eval['rank'] = train_eval.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
    train_eval_out = train_eval[train_eval['rank'] <= topk]

    len_user_id = len(set(train_eval.user_id))

    # todo 指标计算可能有问题
    hitrate_50_full = np.sum(train_eval_out['label']) / len_user_id * recall_rate
    hitrate_50_half = np.sum(train_eval_out['label'] * train_eval_out['half']) / len_user_id * recall_rate
    ndcg_50_full = np.sum(train_eval_out['label'] / np.log2(train_eval_out['rank'] + 2.0) * recall_rate)
    ndcg_50_half = np.sum(
        train_eval_out['label'] * train_eval_out['half'] / np.log2(train_eval_out['rank'] + 2.0) * recall_rate)

    print("------------- eval result -------------")
    print("hitrate_50_full : ", hitrate_50_full, 'ndcg_50_full : ', ndcg_50_full, '\n')
    print("hitrate_50_half : ", hitrate_50_half, 'ndcg_50_half : ', ndcg_50_half, '\n')
    print("------------- eval result -------------")

    print("------------- predict -------------")
    test_data_out = test_data[['user_id', 'item_similar']]
    test_y_pred = clf.predict_proba(X_pred)[:, 1]
    test_data_out['pred_prob'] = test_y_pred

    test_data_out['rank'] = test_data_out.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
    test_data_out = test_data_out[test_data_out['rank'] <= topk]
    test_data_out = test_data_out.sort_values(['rank'])

    submit = test_data_out.groupby(['user_id'])['item_similar'].agg(lambda x: ','.join(list(x))).reset_index()

    print("------------- assert -------------")
    for i, row in submit.iterrows():
        txt_item = row['item_similar'].split(',')
        assert len(txt_item) == topk
    return submit

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
    import psutil
    from gensim import models

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
    print(len(keys))
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

def topk_recall_association_rules_open_source(click_all, dict_label, k=200):
    """
        author: 青禹小生 鱼遇雨欲语与余
        修改：Cookly
        关联矩阵：

    """

    '''
    train和test集合都用于构建相似度矩阵
    只有train集合 有  正样本标签
    召回的时候train和test集合user都进行了召回，但是只有train有正样本标签，所以test集合user没有必要进行召回，或者原作者在构建正样本的时候有其他想法，test集合没有利用上
    TODO 需要改写
    '''
    from collections import Counter

    data_ = click_all.groupby(['user_id'])[['item_id', 'time']].agg(
        {'item_id': lambda x: ','.join(list(x)), 'time': lambda x: ','.join(list(x))}).reset_index()

    hot_list = list(click_all['item_id'].value_counts().index[:].values)
    stat_cnt = Counter(list(click_all['item_id']))
    stat_length = np.mean([len(item_txt.split(',')) for item_txt in data_['item_id']])

    matrix_association_rules = {}
    print('------- association rules matrix 生成 ---------')
    for i, row in tqdm(data_.iterrows()):

        list_item_id = row['item_id'].split(',')
        list_time = row['time'].split(',')
        len_list_item = len(list_item_id)
        #
        for i, (item_i, time_i) in enumerate(zip(list_item_id, list_time)):
            for j, (item_j, time_j) in enumerate(zip(list_item_id, list_time)):

                t = np.abs(float(time_i) - float(time_j))
                d = np.abs(i - j)

                if i < j:
                    if item_i not in matrix_association_rules:
                        matrix_association_rules[item_i] = {}
                    if item_j not in matrix_association_rules[item_i]:
                        matrix_association_rules[item_i][item_j] = 0

                    matrix_association_rules[item_i][item_j] += 1 * 0.7 * (0.8 ** (d - 1)) * (1 - t * 10000) / np.log(
                        1 + len_list_item)

                if i > j:
                    if item_i not in matrix_association_rules:
                        matrix_association_rules[item_i] = {}
                    if item_j not in matrix_association_rules[item_i]:
                        matrix_association_rules[item_i][item_j] = 0

                    matrix_association_rules[item_i][item_j] += 1 * 1.0 * (0.8 ** (d - 1)) * (1 - t * 10000) / np.log(
                        1 + len_list_item)

    assert len(matrix_association_rules.keys()) == len(set(click_all['item_id']))

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- association rules 召回 ---------')
    for i, row in tqdm(data_.iterrows()):
        list_item_id = row['item_id'].split(',')

        dict_item_id_score = {}
        for i, item_i in enumerate(list_item_id[::-1]):
            for item_j, score_similar in sorted(matrix_association_rules[item_i].items(), reverse=True)[0:k]:
                if item_j not in list_item_id:
                    if item_j not in dict_item_id_score:
                        dict_item_id_score[item_j] = 0

                    '''
                    对于一个user召回item的分数计算：
                    1. user的正反馈物品列表中的每个物品取其相似的topK个物品
                    2. 第1步取出的物品最后相似度得分：w*r -- w是topK集合物品和user正反馈物品的相似度即score_similar，
                            r是user对正反馈物品的评分，点击类的一般都是正反馈物品为1，这里根据点击的顺序进行了惩罚，点击间隔越多的物品，惩罚越大
                    '''
                    dict_item_id_score[item_j] += score_similar * (0.7 ** i)

        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])

        # 不足的热度补全
        if len(dict_item_id_score_topk) < k:
            for i, item in enumerate(hot_list):
                if (item not in list_item_id) and (item not in dict_item_id_set):
                    item_similar = item
                    score_similar = - i - 100
                    dict_item_id_score_topk.append((item_similar, score_similar))
                if len(dict_item_id_score_topk) == k:
                    break

        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        '''
        召回格式：
        user_id 相似item 相似分
        user_id召回k个相似item，就k行
        '''
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])

    topk_recall = pd.DataFrame(
        {'user_id': list_user_id, 'item_similar': list_item_similar, 'score_similar': list_score_similar})

    ''' 只有train集合的才有next_item_id标签，test集合没有next_item_id标签 '''
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall.sort_values('user_id', inplace=True)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall


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


def save(submit_all, topk):
    import time

    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_name = './submit{time_str}.csv'.format(time_str=time_str)
    with open(file_name, 'w') as f:
        for i, row in submit_all.iterrows():
            user_id = str(row['user_id'])
            item_list = str(row['item_similar']).split(',')[:topk]
            assert len(set(item_list)) == topk

            line = user_id + ',' + ','.join(item_list) + '\n'
            assert len(line.strip().split(',')) == (topk + 1)

            f.write(line)


if __name__ == '__main__':
    submit_all = pd.DataFrame()
    click_all = pd.DataFrame()
    for phase in range(0, now_phase + 1):
        print('phase:', phase)
        click_train = pd.read_csv(
            train_path + '/underexpose_train_click-{phase}.csv'.format(phase=phase)
            , header=None
            # , nrows=nrows
            , names=['user_id', 'item_id', 'time']
            , sep=','
            , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str}
        )
        click_test = pd.read_csv(
            test_path + '/underexpose_test_click-{phase}/underexpose_test_click-{phase}.csv'.format(phase=phase)
            , header=None
            # , nrows=nrows
            , names=['user_id', 'item_id', 'time']
            , sep=','
            , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str}
        )

        '''
            sampling
        '''
        print('total sample user:', len(set(click_train['user_id'])))
        if subsampling:
            click_train_user_sublist = sorted(list(set(click_train['user_id'])))[:subsampling]
            click_train = click_train[click_train['user_id'].isin(click_train_user_sublist)]

            # 用户行为序中点击item少于等于2个的，都删除 由于需要将最后一个item作为标签，所以只剩下一个无法构建item相似度矩阵
            single_item_user_list = list(click_train['user_id'].value_counts()[click_train['user_id'].value_counts() <= 2].index)
            click_train = click_train[~click_train['user_id'].isin(single_item_user_list)]


            click_test_user_sublist = sorted(list(set(click_test['user_id'])))[:subsampling]
            click_test = click_test[click_test['user_id'].isin(click_test_user_sublist)]
            single_item_user_list = list(
                click_test['user_id'].value_counts()[click_test['user_id'].value_counts() <= 2].index)
            click_test = click_test[~click_test['user_id'].isin(single_item_user_list)]

        click = click_train.append(click_test)

        if flag_append:
            click_all = click_all.append(click)
        else:
            click_all = click

        click_all = click_all.sort_values('time')
        click_all = click_all.drop_duplicates(['user_id', 'item_id', 'time'], keep='last')

        # train、test重新划分，并去重
        set_pred = set(click_test['user_id'])
        set_train = set(click_all['user_id']) - set_pred

        temp_ = click_all
        # 给train和test样本打上标签，pred列
        temp_['pred'] = temp_['user_id'].map(lambda x: 'test' if x in set_pred else 'train')

        '''train集合中每个user点击序中最后一个点击item作为正样本标签'''
        temp_ = temp_[temp_['pred'] == 'train'].drop_duplicates(['user_id'], keep='last')
        temp_['remove'] = 'remove'

        # 去掉remove列标签，剩下的作为训练集
        train_test = click_all
        train_test = train_test.merge(temp_, on=['user_id', 'item_id', 'time', 'pred'], how='left').sort_values('user_id')
        train_test = train_test[train_test['remove'] != 'remove']

        dict_label_user_item = dict(zip(temp_['user_id'], temp_['item_id']))

        # item频率统计，作为热度
        temp_ = train_test.groupby(['item_id'])['user_id'].count().reset_index()
        temp_ = temp_.sort_values(['item_id'])
        hot_list = list(temp_['item_id'])[::-1]

        print('-------- 召回 -------------')
    #     """
    #     # glove embedding 召回 太慢了 放弃
    #     topk_recall = topk_recall_glove_embedding(
    #                                                 click_all=train_test
    #                                                 ,dict_label=dict_label_user_item
    #                                                 ,k=recall_num
    #                                                 ,dim=88
    #                                                 ,epochs=1
    #                                                 ,learning_rate=0.5
    #                                                 )
    #     # word2vec embedding 召回 太慢了 放弃
    #     topk_recall = topk_recall_word2vec_embedding(
    #                                                 click_all
    #                                                 ,dict_label
    #                                                 ,k=100
    #                                                 ,dim=88
    #                                                 ,epochs=30
    #                                                 ,learning_rate=0.5
    #                                                 )
    #
    #     topk_recall = topk_recall_association_rules(
    #                                   click_all=train_test
    #                                  ,dict_label=dict_label_user_item
    #                                  ,k=recall_num
    #                                  )
    #     """

        # print(set(dict_label_user_item.keys()) - set(train_test['user_id']))
        # print(len(set(dict_label_user_item.keys())), len(set(train_test['user_id'])))
        # print(len(set(train_test['user_id']) - set(dict_label_user_item.keys())))
        # assert set(dict_label_user_item.keys()) - set(train_test['user_id'])
        # print(train_test[train_test['user_id'] == '10226'])
        # print('xxx', dict_label_user_item.get('10226'))
        topk_recall = topk_recall_association_rules_open_source(
            click_all=train_test,
            dict_label=dict_label_user_item,
            k=recall_num
        )
        print(topk_recall)

        print('-------- 评测召回效果 -------------')
        hit_rate = metrics_recall(topk_recall=topk_recall, phase=phase, k=recall_num, sep=int(recall_num / 10))
        print('-------- 召回TOP:{k}时, 命中百分比:{hit_rate} -------------'.format(k=recall_num, hit_rate=hit_rate))
    #
        print('-------- 排序 -------------')
        print('-------- 构建特征 ---------')
        print('-------- sku1 sku2 sku3 sku4 sku5 user ----------')
        dim, epochs, learning_rate = 30, 1, 0.5
    #
        dict_embedding_all_ui_item = matrix_word2vec_embedding(
            click_all=train_test,
            flag='item',
            mode='all',
            dim=dim,
            epochs=epochs,
            learning_rate=learning_rate
        )
        print('------- user1 user2 user3 user4 user5 sku -------')
        dict_embedding_all_ui_user = matrix_word2vec_embedding(
            click_all=train_test,
            flag='user',
            mode='all',
            dim=dim,
            epochs=epochs,
            learning_rate=learning_rate
        )
        print('------- item1 item2 item3 item4 item5 -------')
        dict_embedding_item_only = matrix_word2vec_embedding(
            click_all=train_test,
            flag='item',
            mode='only',
            dim=dim,
            epochs=epochs,
            learning_rate=learning_rate
        )
        print('------- user1 user2 user3 user4 user5 -------')
        dict_embedding_user_only = matrix_word2vec_embedding(
            click_all=train_test,
            flag='user',
            mode='only',
            dim=dim,
            epochs=epochs,
            learning_rate=learning_rate
        )

        print('------- 特征加工 -----------')
        feature_all = get_train_test_data(
            topk_recall,
            dict_embedding_all_ui_item,
            dict_embedding_all_ui_user,
            dict_embedding_item_only,
            dict_embedding_user_only,
            flag_test=flag_test
        )

        print('------- 训练模型 -----------')
        # submit = train_model_lgb(feature_all, recall_rate=hit_rate, hot_list=hot_list, valid=0.2, topk=50, num_boost_round=1, early_stopping_rounds=1)

        submit = train_model_rf(feature_all, recall_rate=hit_rate, hot_list=hot_list, valid=0.2, topk=50)

        submit_all = submit_all.append(submit)

    print('------- 保存预测文件 -----------')
    save(submit_all, 50)

    # todo 统计不同阶段时间分段情况