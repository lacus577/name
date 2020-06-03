from collections import defaultdict
import math, os

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

recall_num = 500

def recall_5146():
    now_phase = 6
    train_path = '../../../../data/underexpose_train'
    test_path = '../../../../data/underexpose_test'
    recom_item = []

    whole_click = pd.DataFrame()
    # 计算物品相似度
    print('导入数据')
    for c in range(now_phase + 1):
        #     phase_recom_item = []
        print('phase:', c)
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,
                                  names=['user_id', 'item_id', 'time'])
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c), header=None,
                                 names=['user_id', 'item_id', 'time'])

        whole_click = whole_click.append(click_train)
        whole_click = whole_click.append(click_test)

    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    whole_click = whole_click.sort_values(by=['user_id', 'time'])

    hot_click = list(whole_click['item_id'].value_counts().index[:200].values)
    print('计算物品相似度')
    dump_path = './cache/features_cache/item_sim_list'
    if os.path.exists(dump_path):
        item_sim_list = pickle.load(open(dump_path, 'rb'))
        user_item = pickle.load(open('./cache/features_cache/user_item', 'rb'))
    else:
        item_sim_list, user_item = get_sim_item_5164(whole_click, 'user_id', 'item_id', use_iif=False)

        pickle.dump(item_sim_list, open(dump_path, 'wb'))
        pickle.dump(user_item, open('./cache/features_cache/user_item', 'wb'))

    # 基于用户点击序列构建可能点击物品集
    print('基于用户点击序列构建可能点击物品集')
    for c in range(now_phase + 1):
        phase_recom_item = []
        print('phase:', c)
        click_pred = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c, c), header=None,
                                 names=['user_id', 'time'])

        #     pred_user_time_dict = dict(zip(click_pred['user_id'], user_time_['time']))

        for i, row in tqdm(click_pred.iterrows()):
            rank_item, interacted_items = recommend_time_5164(item_sim_list, whole_click, row, 500, 500)
            #         rank_item = recommend(item_sim_list, user_item, row['user_id'], 500, 500)
            rank_item = rank_item[:recall_num]
            for j in rank_item:
                recom_item.append([row['user_id'], j[0], j[1]])
                phase_recom_item.append([row['user_id'], j[0], j[1]])
            hot_cover = 100 - len(rank_item)
            #         while hot_cover>0:
            if hot_cover > 0:
                for hot_index, hot_item in enumerate(hot_click):
                    if hot_item not in interacted_items and hot_item not in [x[0] for x in rank_item]:
                        recom_item.append([row['user_id'], hot_item, -1 * hot_index])
                        phase_recom_item.append([row['user_id'], hot_item, -1 * hot_index])
                        hot_cover -= 1
                    if hot_cover <= 0:
                        break
        phase_recom_df = pd.DataFrame(phase_recom_item, columns=['user_id', 'item_id', 'sim'])
        phase_recom_df.to_csv('./cache/features_cache/phase_{}_recall_{}.csv'.format(c, recall_num), index=False)
    print('构建提交结果集')
    recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
    # result = get_predict(recom_df, 'sim', top50_click)
    result = get_predict_5164(recom_df, 'sim')
    result['user_id'] = result.astype({'user_id': 'int'})
    result.to_csv('baseline_whole_click_int.csv', index=False, header=None)

    # 基于用户点击序列构建可能点击物品集
    print('训练集user召回')
    phase_recom_item = []
    single_user_click = whole_click.drop_duplicates(['user_id'], keep='last').reset_index(drop=True)
    for i, row in tqdm(single_user_click.iterrows()):
        rank_item, interacted_items = recommend_time_5164(item_sim_list, whole_click, row, 500, 500)
        #         rank_item = recommend(item_sim_list, user_item, row['user_id'], 500, 500)
        rank_item = rank_item[:recall_num]
        for j in rank_item:
            recom_item.append([row['user_id'], j[0], j[1]])
            phase_recom_item.append([row['user_id'], j[0], j[1]])
        hot_cover = 100 - len(rank_item)
        #         while hot_cover>0:
        if hot_cover > 0:
            for hot_index, hot_item in enumerate(hot_click):
                if hot_item not in interacted_items and hot_item not in [x[0] for x in rank_item]:
                    recom_item.append([row['user_id'], hot_item, -1 * hot_index])
                    phase_recom_item.append([row['user_id'], hot_item, -1 * hot_index])
                    hot_cover -= 1
                if hot_cover <= 0:
                    break
    phase_recom_df = pd.DataFrame(phase_recom_item, columns=['user_id', 'item_id', 'sim'])
    phase_recom_df.to_csv('./cache/features_cache/total_user_recall_{}.csv'.format(recall_num), index=False)


def get_sim_item_5164(df_, user_col, item_col, use_iif=False):
    df = df_.copy()
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    sim_item = {}
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                t1 = user_time_dict[user][loc1]  # 点击时间提取
                t2 = user_time_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)
                if not use_iif:
                    if loc1 - loc2 > 0:
                        sim_item[item][relate_item] += 1 * 1.0 * (0.8 ** (loc1 - loc2 - 1)) * (
                        1.3 - (t1 - t2) * 10000) / math.log(1 + len(items))  # 逆向
                    else:
                        sim_item[item][relate_item] += 1 * 1.0 * (0.8 ** (loc2 - loc1 - 1)) * (
                        1.3 - (t2 - t1) * 10000) / math.log(1 + len(items))  # 正向
                else:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)

    return sim_item_corr, user_item_dict

# fill user to 50 items
# def get_predict(df, pred_col, top_fill):
def get_predict_5164(df, pred_col):
#     top_fill = [int(t) for t in top_fill.split(',')]
#     scores = [-1 * i for i in range(1, len(top_fill) + 1)]
    ids = list(df['user_id'].unique())
#     fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])
#     fill_df.sort_values('user_id', inplace=True)
#     fill_df['item_id'] = top_fill * len(ids)
#     fill_df[pred_col] = scores * len(ids)
#     df = df.append(fill_df)
    df.sort_values(pred_col, ascending=False, inplace=True)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)
    df = df[df['rank'] <= 50]
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',', expand=True).reset_index()
    return df

def recommend_time_5164(sim_item_corr, user_item_df, row, top_k, item_num):
    '''
    input:item_sim_list, user_item, uid, 500, 50
    # 用户历史序列中的所有商品均有关联商品,整合这些关联商品,进行相似性排序
    '''
    rank = {}
    interacted_items = set(user_item_df[user_item_df['user_id']==row['user_id']]['item_id'].values)
    #前面时间的要按照降序来，时间越大权重越大
    user_item_before = list(user_item_df[(user_item_df['user_id']==row['user_id']) & (user_item_df['time'] <= row['time'])].sort_values(by='time',ascending=False)['item_id'].values)
    #后面时间的要按照升序来，时间越小的权重越大
    user_item_after = list(user_item_df[(user_item_df['user_id']==row['user_id']) & (user_item_df['time'] > row['time'])].sort_values(by='time')['item_id'].values)

#     interacted_items = user_item_dict[user_id]
#     interacted_items = interacted_items[::-1]
#0.7  0.7 0.5140
#0.7 0.6 0.5164
#0.7 0.5 0.5164 但是full和half存在变化，full下降，half上升
    for loc,i in enumerate(user_item_before):
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.7**loc)

    for loc,i in enumerate(user_item_after):
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.5**loc)

#     for loc, i in enumerate(interacted_items):
#         for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True)[0:top_k]:
#             if j not in interacted_items:
#                 rank.setdefault(j, 0)
#                 rank[j] += wij * (0.7**loc)

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num],interacted_items

def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):
    '''
    input:item_sim_list, user_item, uid, 500, 50
    # 用户历史序列中的所有商品均有关联商品,整合这些关联商品,进行相似性排序
    '''
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.7**loc)

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]

def get_sim_item(df_, user_col, item_col, use_iif=False):
    df = df_.copy(deep=True)
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    sim_item = {}
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                t1 = user_time_dict[user][loc1]  # 点击时间提取
                t2 = user_time_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)
                if not use_iif:
                    if loc1 - loc2 > 0:
                        sim_item[item][relate_item] += 1 * 0.7 * (0.8 ** (loc1 - loc2 - 1)) * (
                        1 - (t1 - t2) * 10000) / math.log(1 + len(items))  # 逆向
                    else:
                        sim_item[item][relate_item] += 1 * 1.0 * (0.8 ** (loc2 - loc1 - 1)) * (
                        1 - (t2 - t1) * 10000) / math.log(1 + len(items))  # 正向
                else:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)

    return sim_item_corr, user_item_dict

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

if __name__ == '__main__':
    recall_5146()