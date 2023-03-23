import pickle
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0, help='cross validation random seed')
opt = parser.parse_args()


main_dir = '/cbica/home/zhouz/projects/istaging_2022'
NMF_fMRI = pd.read_pickle('./Data/final_NMF_fMRI.pkl')
Age = NMF_fMRI.Age.values
features = np.load('%s/within/ridge/tangent/geometric/har/features.npy'%(main_dir))
random_state=opt.seed
random_state=check_random_state(random_state)
shuffling_idx = np.arange(Age.shape[0])

MAE=np.zeros(100)
cor_coef=np.zeros(100)

alpha = np.exp2(-10)
fea_weight = np.zeros((100,5,features.shape[1]))

for i in range(100):
    print(i)
    random_state.shuffle(shuffling_idx)
    age = Age[shuffling_idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    y_pre = np.zeros(age.shape[0])
    m=0
    for train_index, test_index in kf.split(age):
        trainData = features[train_index, :]
        testData = features[test_index, :]
        normalize = preprocessing.StandardScaler()
        trainData = normalize.fit_transform(trainData)
        testData = normalize.transform(testData)
        model = Ridge(alpha=alpha).fit(trainData, age[train_index])
        y_pre[test_index] = model.predict(testData)
        fea_weight[i,m, :] = model.coef_
        m=m+1
    MAE[i] = mean_absolute_error(y_pre, age)
    cor_coef[i] = np.corrcoef(y_pre, age)[0,1]

fea_output = '%s/within/ridge/tangent/geometric/har/fea_permutation' % (main_dir)
if not os.path.exists(fea_output):
    os.makedirs(fea_output)
np.save('%s/MAE_100time_%s.npy' % (fea_output, str(opt.seed)), MAE)
np.save('%s/fea_weight_100time_%s.npy' % (fea_output, str(opt.seed)), fea_weight)



