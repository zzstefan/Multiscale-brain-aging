#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:31:59 2021

@author: zhouz
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nilearn import image
import pickle 

main_dir = '/cbica/home/zhouz/projects/istaging/LiHM_NMF2/result/All'

K=[25,50,75,100,125,150]
mask_image = '/cbica/home/zhouz/projects/istaging/LiHM_NMF/atlas/mask_thr0p5_wmparc_2_cc.nii.gz'
nonzero_index = np.nonzero(image.get_data(mask_image).flatten())


###############################################################
K=[17,25,50,75,100,125,150]
result_matrix=[]
for i in range(7):
    if i==6:
        break
    corr_matrix = np.zeros((K[i],K[i+1]))
    for a in range(K[i]):
        for b in range(K[i+1]):
            network_1=image.get_data(main_dir+'/'+str(K[i])+'_network/fig/icn_'+'{0:03}'.format(a+1)+'.nii.gz').flatten()
            network_2=image.get_data(main_dir+'/'+str(K[i+1])+'_network/fig/icn_'+'{0:03}'.format(b+1)+'.nii.gz').flatten()
            corr_matrix[a,b]=np.corrcoef(network_1[nonzero_index], network_2[nonzero_index])[0,1]
    result_matrix.append(corr_matrix)
    
f = open('network_correlation.pkl','wb')
pickle.dump(result_matrix,f)
f.close()

###############################################################
with open('network_correlation.pkl', 'rb') as f:
    result_matrix=pickle.load(f)

result=[]
for i in range(6):
    corr_mat = result_matrix[i]
    max_value=[]
    for k in range(corr_mat.shape[1]):
        max_value.append(np.max(corr_mat[:,k]))

        # temp=np.sort(corr_mat[:,k])
        # m=1
        # while (temp[-1]-temp[-m-1])/temp[-1]<0.15:
        #     max_value.append(temp[-m-1])
        #     m+=1
    # max_value_nodup=[]
    # [max_value_nodup.append(x) for x in max_value if x not in max_value_nodup]
    
    index=np.zeros((len(max_value),2))
    for a,b in enumerate(max_value):
        index[a,1]=np.where(corr_mat==b)[0]
        index[a,0]=np.where(corr_mat==b)[1]
    
    
    missing_net=[]
    [missing_net.append(x) for x in range(K[i]) if x not in list(index[:,1])]
    for j in missing_net:
        max_value.append(np.max(corr_mat[j,:]))
        
    index_new=np.zeros((len(max_value),2))
    for a,b in enumerate(max_value):
        index_new[a,1]=np.where(corr_mat==b)[0]
        index_new[a,0]=np.where(corr_mat==b)[1]
    
    result.append(np.array(index_new))
    

df_1=pd.DataFrame(result[5], columns=['150_network','125_network'])
df_2=pd.DataFrame(result[4], columns=['125_network','100_network'])
df_3=pd.DataFrame(result[3], columns=['100_network','75_network'])
df_4=pd.DataFrame(result[2], columns=['75_network','50_network'])
df_5=pd.DataFrame(result[1], columns=['50_network','25_network'])
df_6=pd.DataFrame(result[0], columns=['25_network','17_network'])

# df_7=df_1.merge(df_2,how='left')
# df_8=df_7.merge(df_3,how='left')
# df_9=df_8.merge(df_4,how='left')
# df_10=df_9.merge(df_5,how='left')
# df_final=df_10.merge(df_6,how='left')

df_7=df_6.merge(df_5, how='left')
df_8=df_7.merge(df_4,how='left')
df_9=df_8.merge(df_3,how='left')
df_10=df_9.merge(df_2,how='left')
df_final=df_10.merge(df_1,how='left')



    


############################################################

yeo_network = '/cbica/home/zhouz/projects/istaging/LiHM_NMF/atlas/Yeo_JNeurophysiol11_MNI152-2/Yeo7_networks.nii.gz'
new_mask_data = image.get_data(yeo_network).flatten()
yeo_network_voxel = new_mask_data[nonzero_index]
for k  in [17]:
    Yeo7_corr=np.zeros((7,k))
    for i in range(7):
        network_index = np.where(yeo_network_voxel==(i+1))
        for j in range(k):
            my_network=image.get_data(main_dir+'/17_network/fig/icn_'+'{0:03}'.format(j+1)+'.nii.gz').flatten()
            my_network_voxel=my_network[nonzero_index]
            
            Yeo7_corr[i,j]=np.sum(my_network_voxel[network_index])/np.sum(my_network_voxel)
            
            
net7_17=np.zeros((17,2))
net7_17[:,0]=range(17)
net7_17[:,1]=np.argmax(Yeo7_corr,axis=0)

df_7_17=pd.DataFrame(net7_17,columns=['17_network','Yeo_7'])
df_final=df_final.merge(df_7_17,how='left')
df_category=df_final.rename(columns={'150_network':'network_150','125_network':'network_125','100_network':'network_100','75_network':'network_75',
                         '50_network':'network_50','25_network':'network_25','17_network':'network_17'})

df_category=df_category.sort_values(by=['Yeo_7'])
##########################################################

import plotly.graph_objects as go
import pandas as pd

df_category+=1
# Create dimensions
net7_dim = go.parcats.Dimension(values=df_category.Yeo_7 ,label="Yeo 7 network")
net17_dim = go.parcats.Dimension(values=df_category.network_17 ,label="17 network")

net25_dim = go.parcats.Dimension(
    values=df_category.network_25,label="25 network")
net50_dim = go.parcats.Dimension(
    values=df_category.network_50,label="50 network")
net75_dim = go.parcats.Dimension(
    values=df_category.network_75,label="75 network")
net100_dim = go.parcats.Dimension(
    values=df_category.network_100,label="100 network")
net125_dim = go.parcats.Dimension(
    values=df_category.network_125,label="125 network")
net150_dim = go.parcats.Dimension(
    values=df_category.network_150,label="150 network")


color = df_category.Yeo_7
#colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen'],[2,],[3,],[4,],[5,]];
# colorscale = [[1,'rgb(120,18,124)'],[2,'rgb(70,130,180)'],[3,'rgb(0,118,114)'],
#               [4,'rgb(196,58,250)'],[5,'rgb(220,248,164)'],[6,'rgb(230,148,34)'],[7,'rgb(205,62,78)']]

#colorscale =[[1,'purp'],[2,'sunsetdark'],[3,'bluered'],[4,'solar'],[5,'gray'],[6,'blackbody'],[7,'balance']]
#color_discrete_sequence=['red','green','blue','black','goldenrod','magenta','yellow']
color_discrete_sequence=['rgb(120,18,124)','rgb(70,130,180)','rgb(0,118,114)','rgb(196,58,250)','rgb(220,248,164)','rgb(230,148,34)','rgb(205,62,78)']

fig = go.Figure(data = [go.Parcats(dimensions=[net7_dim, net17_dim,net25_dim,net50_dim,net75_dim,net100_dim,net125_dim,net150_dim],
        line={'color': color,'colorscale':color_discrete_sequence,'shape':'hspline'},
        #hoveron='color', hoverinfo='count+probability',
        labelfont={'size': 60, 'family': 'Times'},
        tickfont={'size': 1, 'family': 'Times'},
        arrangement='perpendicular')])
#fig.update_xaxes(visible=False)230 148 34
#fig.show(renderer='browser')
fig.update_layout(width=4000,height=3200)
#fig.update_xaxes(visible=False, showticklabels=False)
#fig.show(renderer='browser')
fig.write_image('parallel_plot_new6.png')
#df_category.to_csv('test_plotly.csv')

df_category.to_pickle('./plot/parallel_matrix.pkl')