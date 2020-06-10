
import pandas as pd
import numpy as np
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from pymodule import utils, conf

# all_phase_click = utils.read_all_phase_click()
# all_phase_click_no_qtime = all_phase_click[all_phase_click['train_or_test'] != 'predict']
item_info_df = utils.read_item_user_info()
cluster_num = item_info_df.shape[0] // 10

item_vec_list = list(
    item_info_df.apply(
        lambda x: np.array(list(x.iloc[-conf.new_embedding_dim-conf.new_embedding_dim: -conf.new_embedding_dim])).reshape(-1, ),
        axis=1
    )
)
print(len(item_vec_list), len(item_vec_list[0]))

# Manually override euclidean
def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
    return cosine_similarity(X, Y)
k_means_.euclidean_distances = euc_dist

scaler = StandardScaler(with_mean=False)
sparse_data = scaler.fit_transform(item_vec_list)

ac=AgglomerativeClustering(n_clusters=cluster_num,affinity='cosine',linkage='ward')
labels = ac.fit_predict(sparse_data)


# kmeans = k_means_.KMeans(n_clusters = cluster_num, n_jobs = 20, random_state = 3425)
# _ = kmeans.fit(sparse_data)
# print(kmeans.labels_)
# print(kmeans.labels_)
item_info_df['cluster'] = labels
item_info_df = item_info_df[['item_id', 'cluster']]
item_info_df.to_csv('./clustered_item.csv', index=False)