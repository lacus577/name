# coding=utf-8
from __future__ import division
from __future__ import print_function

import datetime
import json
import sys
import time
from collections import defaultdict

from pymodule import utils

import numpy as np
import pandas as pd

def make_answer(df, hot_df, phase=-1):
    df = df[['user_id', 'item_id']]
    df['phase_id'] = phase
    df = df.merge(hot_df, on='item_id', how='left')
    df = df[['phase_id', 'user_id', 'item_id', 'item_deg']]
    return df

def my_eval(pre_y, valid_df, answer, phase=-1, print_mark=True):
    # 构造submit csv
    valid_submit = utils.save_pre_as_submit_format_csv(valid_df, pre_y)
    submit_csv_path = utils.save(valid_submit, file_dir='./cache/tmp_phase_submit')

    # 构造truth csv
    valid_answer = valid_df.loc[:, ['user_id']].drop_duplicates(['user_id'], keep='first')
    # answer中user列唯一、item列也是唯一
    valid_answer = valid_answer.merge(answer, on='user_id', how='left')
    valid_answer_save_path = './cache/tmp_phase_submit/valid_answer.csv'
    valid_answer = valid_answer[['phase_id', 'user_id', 'item_id', 'item_deg']]
    valid_answer.to_csv(valid_answer_save_path, index=False, header=False)

    score, \
    ndcg_50_full, ndcg_50_half, \
    hitrate_50_full, hitrate_50_half = evaluate(submit_csv_path, valid_answer_save_path,
                                                recall_num=None)

    print(
        'phase:{}, score:{}, ndcg_50_full:{}, ndcg_50_half:{}, hitrate_50_full:{}, hitrate_50_half:{}'.format(
            phase, score, ndcg_50_full, ndcg_50_half, hitrate_50_full, hitrate_50_half
        )
    )

    return np.array([score, ndcg_50_full, ndcg_50_half, hitrate_50_full, hitrate_50_half]).reshape(-1, )

# submit_fname is the path to the file submitted by the participants.
# debias_track_answer.csv is the standard answer, which is not released.
def evaluate(submit_fname,
             answer_fname='debias_track_answer.csv',
             recall_num=50,
             current_time=None):
    schedule_in_unix_time = [
        0,  # ........ 1970-01-01 08:00:00 (T=0)
        1586534399,  # 2020-04-10 23:59:59 (T=1)
        1587139199,  # 2020-04-17 23:59:59 (T=2)
        1587743999,  # 2020-04-24 23:59:59 (T=3)
        1588348799,  # 2020-05-01 23:59:59 (T=4)
        1588953599,  # 2020-05-08 23:59:59 (T=5)
        1589558399,  # 2020-05-15 23:59:59 (T=6)
        1590163199,  # 2020-05-22 23:59:59 (T=7)
        1590767999,  # 2020-05-29 23:59:59 (T=8)
        1591372799  # .2020-06-05 23:59:59 (T=9)
    ]
    assert len(schedule_in_unix_time) == 10
    for i in range(1, len(schedule_in_unix_time) - 1):
        # 604800 == one week
        assert schedule_in_unix_time[i] + 604800 == schedule_in_unix_time[i + 1]

    if current_time is None:
        current_time = int(time.time())
    # print('current_time:', current_time)
    # print('date_time:', datetime.datetime.fromtimestamp(current_time))
    current_phase = 0
    while (current_phase < 9) and (
                current_time > schedule_in_unix_time[current_phase + 1]):
        current_phase += 1
    # print('current_phase:', current_phase)

    '''读truth文件'''
    # try:
    answers = [{} for _ in range(10)]
    with open(answer_fname, 'r') as fin:
        for line in fin:
            line = [int(x) for x in line.split(',')]
            phase_id, user_id, item_id, item_degree = line
            # assert user_id % 11 == phase_id  # todo
            # exactly one test case for each user_id
            answers[phase_id][user_id] = (item_id, item_degree)
    # except Exception as _:
        # raise Exception('server-side error: answer file incorrect')

    try:
        predictions = {}
        with open(submit_fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(',')
                user_id = int(line[0])
                if user_id in predictions:
                    raise Exception('submitted duplicate user_ids')
                item_ids = [int(i) for i in line[1:]]
                # if len(item_ids) != recall_num:
                #     raise Exception('each row need have recall_num items')
                # if len(set(item_ids)) != 50:
                #     raise Exception('each row need have 50 DISTINCT items')
                predictions[user_id] = item_ids
    except Exception as _:
        raise Exception('submission not in correct format')

    scores = np.zeros(4, dtype=np.float32)

    # The final winning teams will be decided based on phase T=7,8,9 only.
    # We thus fix the scores to 1.0 for phase 0,1,2,...,6 at the final stage.
    # if current_phase >= 7:  # if at the final stage, i.e., T=7,8,9
    #     scores += 7.0  # then fix the scores to 1.0 for phase 0,1,2,...,6
    # phase_beg = (7 if (current_phase >= 7) else 0)
    phase_beg = 0 # todo 目前自己测试，都是从阶段0开始
    phase_end = current_phase + 1
    for phase_id in range(phase_beg, phase_end):
        '''这里要求qtime文件的里每个user都需要给出预测值'''
        for user_id in answers[phase_id]:
            if user_id not in predictions:
                raise Exception('user_id %d of phase %d not in submission' % (
                    user_id, phase_id))
        # try:
            # We sum the scores from all the phases, instead of averaging them.
        if not answers[phase_id]:
            continue
        scores += evaluate_each_phase(predictions, answers[phase_id], recall_num=recall_num)
        # except Exception as _:
        #     raise Exception('error occurred during evaluation')

    score = float(scores[0])
    ndcg_50_full = float(scores[0])
    ndcg_50_half = float(scores[1])
    hitrate_50_full = float(scores[2])
    hitrate_50_half = float(scores[3])

    return score, \
           ndcg_50_full, ndcg_50_half, \
           hitrate_50_full, hitrate_50_half

# the higher scores, the better performance
def evaluate_each_phase(predictions, answers, recall_num=50):
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        if not recall_num:      # 不指定recall_num，对整个召回序列都做评估
            recall_num = len(predictions[user_id])
        while rank < recall_num and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0
        if rank < recall_num:   # 命中情况
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)  # ndcg 使用dcg公式，但是gain都置1
            hitrate_50_full += 1.0    # hitrate +1
        if item_degree <= median_item_degree:
            num_cases_half += 1.0           # low half总数
            if rank < recall_num:       # 命中 且 rare
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    # 均值
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)

# FYI. You can create a fake answer file for validation based on this. For example,
# you can mask the latest ONE click made by each user in underexpose_test_click-T.csv,
# and use those masked clicks to create your own validation set, i.e.,
# a fake underexpose_test_qtime_with_answer-T.csv for validation.
def _create_answer_file_for_evaluation(answer_fname='debias_track_answer.csv'):
    train = 'underexpose_train_click-%d.csv'
    test = 'underexpose_test_click-%d.csv'

    # underexpose_test_qtime-T.csv contains only <user_id, time>
    # underexpose_test_qtime_with_answer-T.csv contains <user_id, item_id, time>
    answer = 'underexpose_test_qtime_with_answer-%d.csv'  # not released

    item_deg = defaultdict(lambda: 0)
    with open(answer_fname, 'w') as fout:
        for phase_id in range(10):
            with open(train % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(test % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(answer % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    assert user_id % 11 == phase_id
                    print(phase_id, user_id, item_id, item_deg[item_id],
                          sep=',', file=fout)

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