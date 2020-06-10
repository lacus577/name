
import pandas as pd
import numpy as np
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from pymodule import utils, conf

# all_phase_click = utils.read_all_phase_click()
# all_phase_click_no_qtime = all_phase_click[all_phase_click['train_or_test'] != 'predict']
item_info_df = utils.read_item_user_info()

item_info_df['txt_norm2'] = item_info_df.apply(
    lambda x: np.linalg.norm(np.array(list(x.iloc[-conf.new_embedding_dim-conf.new_embedding_dim: -conf.new_embedding_dim])).reshape(-1, )),
    axis=1
)

item_info_df = item_info_df[['item_id', 'txt_norm2']]
item_info_df.to_csv('./txt_norm2.csv', index=False)