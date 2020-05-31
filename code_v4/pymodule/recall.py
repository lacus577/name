from collections import defaultdict
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

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