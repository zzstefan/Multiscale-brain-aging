import pickle
import pandas as pd
import numpy as np
#from nilearn.connectome import ConnectivityMeasure
from tangent_geometric import *
import bisect
import matplotlib.pyplot as plt


def generate_sym(n):
    A = np.zeros((n, n))
    iu = np.mask_indices(n, np.tril, k=0)
    A[iu] = range(int(n*(n+1)/2))
    return A



f = open('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/final_gmean_within.pkl', 'rb')
final_gmean = pickle.load(f)  # dict_keys(['TC_17', 'TC_25', 'TC_32', 'TC_50', 'TC_75', 'TC_100', 'TC_125', 'TC_150'])
f.close()

permutation_weight= np.load('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/fea_weight_1000time.npy')
#############
###project the weight derived from the 1000 permutation tests into original space

K=[17,25,50,75,100,124,150]
sum_index = np.zeros(len(K))
for i,k in enumerate(K):
    temp = int(k*(k+1)/2)
    if i!=0:
     sum_index[i]=temp+sum_index[i-1]
    else:
        sum_index[i]=temp

weight_ambient = np.zeros((permutation_weight.shape))
index=np.zeros((7,2)).astype('int')
for i in range(index.shape[0]):
    if i==0:
        index[i,0]=0
        index[i,1]=sum_index[i]
    else:
        index[i,0]=sum_index[i-1]
        index[i, 1] = sum_index[i]

for i in range(permutation_weight.shape[0]):
    for j in range(permutation_weight.shape[1]):
        for k in range(index.shape[0]):
            temp=permutation_weight[i,j,index[k,0]:index[k,1]]
            conn = vec_to_sym_matrix(temp)
            mean_sqrt = map_eigenvalues(lambda x: np.sqrt(x), final_gmean[k])
            temp_A=mean_sqrt.dot(map_eigenvalues(np.exp, conn)).dot(mean_sqrt)
            weight_ambient[i,j,index[k,0]:index[k,1]]=sym_matrix_to_vec(temp_A, discard_diagonal=False)

np.save('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/weight_permu_ambient.npy', weight_ambient)


###########
### project the weight from the best model back to original space
weight = np.load('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/feature_importance_1.npy')
weight_ambient = np.zeros((weight.shape))
index=np.zeros((7,2)).astype('int')
for i in range(index.shape[0]):
    if i==0:
        index[i,0]=0
        index[i,1]=sum_index[i]
    else:
        index[i,0]=sum_index[i-1]
        index[i, 1] = sum_index[i]
for i in range(weight.shape[0]):
    for k in range(index.shape[0]):
        temp=weight[i,index[k,0]:index[k,1]]
        conn = vec_to_sym_matrix(temp)
        mean_sqrt = map_eigenvalues(lambda x: np.sqrt(x), final_gmean[k])
        temp_A=mean_sqrt.dot(map_eigenvalues(np.exp, conn)).dot(mean_sqrt)
        weight_ambient[i,index[k,0]:index[k,1]]=sym_matrix_to_vec(temp_A, discard_diagonal=False)


weight = np.abs(weight_ambient)
A = np.mean(weight, axis=0)

# plt.hist(A, bins='auto')
# plt.show()

sum_weight=A

#############
### select those connections with significant importance p<0.05
import bisect


weight_per_ambient =np.load('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/weight_permu_ambient.npy')
weight_per_ambient = np.abs(np.mean(weight_per_ambient, axis=1))
p_value=np.zeros(weight_per_ambient.shape[1])
for i in range(weight_per_ambient.shape[1]):
    final_weight= sum_weight[i]
    p_value[i]=np.sum(weight_per_ambient[:,i]>final_weight)/1000


# plt.hist(p_value, bins='auto')
# plt.show()

# top_k=100
# top_k_index=p_value.argsort()[::-1][0:top_k]
#top_k_index=np.delete(top_k_index,np.where(top_k_index==153)[0])

top_k_index=np.where(p_value<0.05)[0]
K=[17,25,50,75,100,124,150]
result_network = np.zeros((top_k_index.shape[0], 3), dtype='int')
for i,j in enumerate(top_k_index):
    k_index = bisect.bisect_left(sum_index,j)
    result_network[i, 0] = K[k_index]
    sym_matrix = generate_sym(K[k_index])
    if k_index==0:
        result_network[i, 1] = np.where(sym_matrix == j)[0]
        result_network[i, 2] = np.where(sym_matrix == j)[1]
    else:
        j=j-sum_index[k_index-1]-1
        result_network[i, 1] = np.where(sym_matrix == j)[0]
        result_network[i, 2] = np.where(sym_matrix == j)[1]


###########save draw.npy for circular plot
df_category=pd.read_pickle('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/parallel_matrix.pkl')
df_category-=1
net_color=np.zeros((result_network.shape[0],2))
for i in range(result_network.shape[0]):
    net_id = result_network[i,0]
    net_index='network_'+str(net_id)
    A = df_category.loc[df_category[net_index]==result_network[i,1],'Yeo_7'].values[0]
    B = df_category.loc[df_category[net_index]==result_network[i,2],'Yeo_7'].values[0]
    net_color[i,0]=min(A,B)
    net_color[i,1]=max(A,B)
draw = np.hstack((result_network, net_color))


link_color=np.zeros((result_network.shape[0],1))
df_netcolor=pd.DataFrame(net_color)
link_color[df_netcolor[df_netcolor.duplicated(keep=False)].index.values]=1
draw= np.hstack((draw, link_color))
np.save('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/draw.npy',draw)
############


K=np.unique(result_network[:,0])
dict_connectmatrix={}
dict_projection={}

for i,k in enumerate(K):

    temp = np.where(result_network[:, 0] == k)[0]
    net_id = set(np.hstack((result_network[temp, 1], result_network[temp, 2])))
    # set(list(net_id))
    node_list = sorted(net_id)

    print([i+1 for i in node_list])
    if k ==124:
        k=125
    network = [int(df_category.loc[df_category['network_' + str(k)] == a, 'Yeo_7'].values[0]) for a in node_list]

    print(network)

    node_project = np.zeros((len(node_list), 3), dtype='int')
    node_project[:, 0] = np.array(node_list)
    node_project[:, 2] = np.array(network)
    node_project = node_project[node_project[:, 2].argsort()]

    node_project[:, 1] = range(len(node_list))
    connect_matrix = np.zeros((len(node_list), len(node_list)))
    for j in range(len(temp)):
        A = node_project[node_project[:, 0] == result_network[temp[j]][1], 1]
        B = node_project[node_project[:, 0] == result_network[temp[j]][2], 1]
        connect_matrix[A, B] = 1
        connect_matrix[B, A] = 1
    dict_connectmatrix['connect_matrix_' + str(k)] = connect_matrix
    dict_projection['projection_' + str(k)] = node_project
    main_dir = '/cbica/home/zhouz/projects/istaging/LiHM_NMF/result/All/' + str(k) + '_network/fig/'
    # network_file = [main_dir+'icn_'+'{0:03}'.format(x)+'.nii.gz' for x in node_list ]
    # print(network_file)

f = open('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/connectmatrix.pkl','wb')
pickle.dump(dict_connectmatrix,f)
f.close()

f = open('/mnt/8TBSSD/zhouz/istaging_nonDL/Data/draw_figure/projection.pkl','wb')
pickle.dump(dict_projection,f)
f.close()
#






