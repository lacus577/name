import os
import time
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm
# from sklearn.externals import joblib
from sklearn.decomposition import PCA
import warnings
import pickle
from sklearn.metrics import roc_auc_score

from pymodule import conf
from pymodule.rank import train_model_rf, train_model_lgb, rank_rf, rank_xgb
from pymodule.eval import metrics_recall
from pymodule import recall
from pymodule.featuring import matrix_word2vec_embedding, get_train_test_data, \
    get_user_features, train_test_split, cal_user_item_sim, cal_txt_img_sim, \
    cal_click_sim, cal_item_of_user_def, cal_statistic_features, \
    cal_item_distance, cal_user_click_num, cal_total_statistic_features, process_after_featuring, \
    do_featuring, get_recall_sample
from pymodule import utils
from pymodule.eval import evaluate, make_answer, my_eval
from pymodule.recall import get_sim_item, recommend


item_info_df = utils.read_item_user_info()
user_info_df = utils.read_user_info()

all_phase_click = utils.read_all_phase_click()
all_phase_click_no_qtime = all_phase_click[all_phase_click['train_or_test'] != 'predict']


user_df = all_phase_click_no_qtime.drop_duplicates(['user_id'], keep='last').reset_index(drop=True)
all_click = all_phase_click_no_qtime
recom_item, phase_recom_item = recall.items_recommod_5164_by_cos(user_df, all_click, item_info_df)

one_phase_recall_item_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'cossim'])
one_phase_recall_item_df.to_csv('./cos_recall.csv', index=False)