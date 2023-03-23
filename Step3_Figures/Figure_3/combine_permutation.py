import numpy as np

dir='/cbica/home/zhouz/projects/istaging_2022/between/ridge/tangent/geometric/har/fea_permutation'

for i in range(10):
    temp_weight = np.load(dir+'/fea_weight_100time_%s.npy'%(str(i)))
    temp_MAE = np.load(dir+'/MAE_100time_%s.npy'%(str(i)))
    if i==0:
        final_weight = temp_weight
        final_MAE = temp_MAE
    else:
        final_weight = np.concatenate((final_weight, temp_weight), axis=0)
        final_MAE = np.concatenate((final_MAE, temp_MAE))

np.save(dir+'/fea_weight_1000time.npy', final_weight)
np.save(dir+'/MAE_1000time.npy', final_MAE)