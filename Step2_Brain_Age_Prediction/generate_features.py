from sklearn.svm import SVR
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
import pickle
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform, norm, randint
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances
from pprint import pprint
from sklearn import preprocessing
import seaborn as sns
from numpy.linalg import inv
from scipy.linalg import logm, expm, sqrtm, eig
import neuroHarmonize as nh
import groupyr as gpr
import argparse
import time

sns.set()
#from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from tangent_geometric import *
#from make_group import *
# from covbat import *
from joblib import Parallel, delayed
import multiprocessing
import os.path
import xgboost as xgb


def age_predict(feature, splits, cv_times, k, age, data, out_dir, method, group=None):
    ### feature  n_subject * n_featurs, splits means 5/10 fold, cv_times means number of cross validation time;
    ### k means the number of the networks, or the features you choose (17,25,32,50,75,100,125,200)
    ### age means all the subjects' age, data means the dataframe containing the info of all subjects
    ### method means the regression method 'SVR', 'RidgeRegression', 'GBM' and 'GBM_bagging'
    print(str(k) + ' networks')
    data['SEX'] = (data['Sex'] == 'M').astype(np.int)
    age_result = np.zeros(6)
    tag_fea = feature
    print(tag_fea.shape)
    print(age.shape)
    RMSE = np.zeros(cv_times)
    MAE = np.zeros(cv_times)
    Corr = np.zeros(cv_times)
    for i in range(cv_times):
        kf = KFold(n_splits=splits, shuffle=True, random_state=2021)
        m = 0
        train_index_all = np.zeros((splits, age.shape[0]))
        predicted_train_label = np.zeros((splits, age.shape[0]))
        y_pre = np.zeros(age.shape[0])
        fea_weight = np.zeros((splits, feature.shape[1]))
        for train_index, test_index in kf.split(age):
            trainData = tag_fea[train_index, :]
            testData = tag_fea[test_index, :]
            normalize = preprocessing.StandardScaler()
            trainData = normalize.fit_transform(trainData)
            testData = normalize.transform(testData)
            #                 plt.hist(age[train_index],bins='auto')
            #                 plt.show()
            cv_inner = KFold(n_splits=splits - 1, shuffle=True, random_state=1)

            if method == 'GroupLasso':
                rf, param_grid = predict_method(method, group)
            else:
                rf, param_grid = predict_method(method)
            #print(rf)
            #print(param_grid
            if method == 'xgboost':
                search = RandomizedSearchCV(rf, param_grid, cv=cv_inner, n_iter=200, scoring='neg_mean_absolute_error', refit=True, n_jobs=6)
            else:
                search = GridSearchCV(rf, param_grid, cv=cv_inner, scoring='neg_mean_absolute_error', refit=True, n_jobs=6)

            result = search.fit(trainData, age[train_index])
            best_grid = result.best_estimator_
            pprint(result.best_params_)
            train_age_pre = best_grid.predict(trainData)
            y_pre[test_index] = best_grid.predict(testData)
            if method !='xgboost' and method!= 'SVR':
                fea_weight[m, :] = best_grid.coef_
            train_index_all[m, train_index] = 1
            predicted_train_label[m, train_index] = train_age_pre
            m = m + 1

        RMSE[i] = mean_squared_error(y_pre, age, squared=False)
        MAE[i] = mean_absolute_error(y_pre, age)
        corr = np.corrcoef(y_pre, age)
        Corr[i] = corr[0, 1]
        # print('fold:'+str(m))
        # print(RMSE[m], MAE[m])
        # m=m+1
    plot_realvspredicted(age, y_pre, data, out_dir, k, round(np.mean(Corr), 2))
    age_result[0] = np.mean(RMSE)
    age_result[1] = np.std(RMSE)
    age_result[2] = np.mean(MAE)
    age_result[3] = np.std(MAE)
    age_result[4] = np.mean(Corr)
    age_result[5] = np.std(Corr)
    print(age_result)
    print("---------------------------------------")
    np.save(out_dir + '/' + str(k) + '_result.npy', age_result)
    np.save(out_dir + '/predicted_age_' + str(k) + '.npy', y_pre)
    np.save(out_dir + '/train_test_split_' + str(k) + '.npy', train_index_all)
    np.save(out_dir + '/train_predicted_age_' + str(k) + '.npy', predicted_train_label)
    if method != 'xgboost' and method != 'SVR':
        np.save(out_dir + '/feature_importance_' + str(k) + '.npy', fea_weight)
    return y_pre


def correct_bias(predicted_age, chronological_age, sex):
    lm = LinearRegression()
    X = np.hstack((chronological_age.reshape(-1, 1), sex.reshape(-1, 1)))
    # print(X.shape)
    model = lm.fit(X, predicted_age)
    return model


### Different regressors here.
def predict_method(method, group=None):
    if method == 'SVR':
        rf = SVR(kernel='rbf')
        random_grid = {"C": np.exp([0, 1, 2, 3, 4, 5]), "gamma": ['auto', 'scale']}
    elif method == 'ridge':
        rf = Ridge()
        #random_grid = {"alpha": np.exp2(np.arange(16) - 10)}
        random_grid = {"alpha": np.exp2([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])}
    elif method == 'GroupLasso':
        #print(group)
        rf = gpr.SGL(l1_ratio=0.05, groups=group) # l1=0 means group lasso, l1=1 means lasso
        random_grid = {"alpha": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]}
        #random_grid = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    elif method == "rf_2":
        rf = RandomForestRegressor(criterion='mse', n_estimators=250, n_jobs=4)
        random_grid = dict()
        random_grid['max_features'] = ['auto', 'sqrt', 'log2']
        random_grid['max_depth'] = [4, 5, 6, 7, 8, 9, 10]
    elif method == 'GB':
        rf = GradientBoostingRegressor(criterion='mae')
        random_grid = dict()
        random_grid['n_estimators'] = [20, 40, 60, 80, 100, 150, 200]
        random_grid['max_depth'] = [4, 5, 6, 7, 8, 9, 10]
    elif method == 'Ada':
        rf = AdaBoostRegressor()
        random_grid = dict()
        random_grid['n_estimators'] = [20, 40, 60, 80, 100, 150, 200]
        random_grid['loss'] = ['linear', 'square', 'exponential']
    elif method == 'xgboost':
        rf = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.05, random_state=2022,
                              n_estimators=500, max_depth=3)
        random_grid = dict()
        # random_grid['n_estimators'] = [100, 250, 500]
        # random_grid['max_depth'] = randint(2, 5)
        random_grid['min_child_weight'] = randint(1,6)
        random_grid['reg_lambda'] = loguniform(1e1,1e2)
        random_grid['subsample'] = uniform(0.5, 0.5)
        random_grid['colsample_bytree'] = uniform(0.5, 0.5)
    return rf, random_grid


### The correlation between the predicted age and the chronological age.
def plot_realvspredicted(real_age, predicted_age, data, out_dir, k, Corr):
    color_dict = dict({'BLSA-3T': 'blue', 'CARDIA-1': 'red', 'CARDIA-3': 'pink', 'CARDIA-4': 'purple',
                       'OASIS-3': 'yellow', 'ABC': 'brown', 'UKBIOBANK': 'green'})
    x = np.linspace(0, 100, 100)
    df_age = pd.DataFrame(np.hstack((real_age.reshape(-1, 1), predicted_age.reshape(-1, 1))),
                          columns=['Age', 'Predicted age'])
    df_age['SITE'] = data['SITE']
    fig, axes = plt.subplots(figsize=(8, 6))

    sns.scatterplot(x='Age', y='Predicted age', data=df_age, hue='SITE', palette=color_dict, alpha=0.5)
    plt.plot(x, x, 'k')
    plt.xlim([20, 100])
    plt.ylim([20, 100])

    plt.text(75, 95, '    k=%s' % (k))
    plt.text(75, 92.5, 'corr=%s' % (Corr))
    plt.tight_layout()
    axes.set_aspect('equal', adjustable='box')
    plt.savefig(out_dir + "/" + str(k) + "_correlation.png", dpi=300)
    # plt.show()
    plt.clf()


### tangent space using geometric mean reference, and the mean of the dataset
### with the minimum sum distance to the mean of all the other datasets, to harmonize or not harmonize
def geometric_site(All_TC, k, data, to_harmonize, useGAMs, mean_index):
    ### data means the dataframe containing the info of all subjects
    correlation_measure = ConnectivityMeasure(kind='covariance')
    shape = int(k * (k + 1) / 2)
    tangent_result = np.zeros((4259, shape))
    g_mean = np.zeros((7, k, k))
    g_mean_upper = np.zeros((7, shape))
    for i, name in enumerate(['BLSA', 'OASIS', 'ABC', 'CARDIA-1', 'CARDIA-3', 'CARDIA-4', 'UKBIOBANK']):
        index = data[data.SITE.str.contains(name)].index
        temp = [All_TC['TC_' + str(k)][i] for i in index]
        correlation_matrix = correlation_measure.fit_transform(temp)
        g_mean[i, :, :] = geometric_mean(correlation_matrix, max_iter=100, tol=1e-7)
        # g_mean_upper[i, :] = g_mean[i, :, :][np.triu_indices(k, k=0)]
    # distances = euclidean_distances(g_mean_upper)
    # index_mean = np.argmin(np.sum(distances, axis=0))
    # print(index_mean)
    final_gmean = g_mean[mean_index, :, :]
    whitening = map_eigenvalues(lambda x: 1. / np.sqrt(x), final_gmean)
    all_tc = All_TC['TC_' + str(k)]
    connectivities = correlation_measure.fit_transform(all_tc)
    connectivities = [map_eigenvalues(np.log, whitening.dot(cov).dot(whitening)) for cov in connectivities]
    connectivities = [sym_matrix_to_vec(con, discard_diagonal=False) for con in connectivities]

    if to_harmonize == 1:
        harmonized_features = harmonize_onebyone(np.array(connectivities), data, useGAMs)
    else:
        harmonized_features = np.array(connectivities)
    return harmonized_features


# tangent space using geometric mean reference, and the mean of the dataset
# with the minimum sum distance to the mean of all the other datasets
# concatenating features from all network from different scales into one vector
# to harmonized or not harmonize, harmonize is first harmonize on the features from different networks then concatenating into one vector
def geometric_allfeature_site(All_TC, K, data, to_harmonize, useGAMs, mean_index):
    for i, k in enumerate(K):
        features_eachk = geometric_site(All_TC, k, data, to_harmonize, useGAMs, mean_index)
        if i == 0:
            combined_fea = features_eachk
        else:
            combined_fea = np.hstack((combined_fea, features_eachk))

    harmonized_features = combined_fea
    return harmonized_features


### tangent space using harmonic mean reference, and the mean of the dataset
### with the minimum sum distance to the mean of all the other datasets, to harmonize or not harmonize
def harmonic_site(All_TC, k, data, correlation_method, to_harmonize, use_GAMs, mean_index):
    correlation_measure = ConnectivityMeasure(kind=correlation_method)
    harmonic_mean = np.zeros((7, k, k))
    shape = int(k * (k + 1) / 2)
    harmonic_mean_upper = np.zeros((7, shape))
    for i, name in enumerate(['BLSA', 'OASIS', 'ABC', 'CARDIA-1', 'CARDIA-3', 'CARDIA-4', 'UKBIOBANK']):
        index = data[data.SITE.str.contains(name)].index
        temp = [All_TC['TC_' + str(k)][i] for i in index]
        connectivity = correlation_measure.fit_transform(temp)
        harmonic_mean[i, :, :] = get_C_h(connectivity)
    #     harmonic_mean_upper[i, :] = harmonic_mean[i, :, :][np.triu_indices(k, k=0)]
    # distances = euclidean_distances(harmonic_mean_upper)
    # index_mean = np.argmin(np.sum(distances, axis=0))
    # print(index_mean)
    final_mean = harmonic_mean[mean_index, :, :]
    all_tc = All_TC['TC_' + str(k)]
    connectivities = correlation_measure.fit_transform(all_tc)
    connectivities = tangent_withref(connectivities, final_mean)

    if to_harmonize == 1:
        harmonized_features = harmonize_onebyone(connectivities, data, use_GAMs)
    else:
        harmonized_features = connectivities
    return harmonized_features


# tangent space using harmonic mean reference, and the mean of the dataset
# with the minimum sum distance to the mean of all the other datasets
# concatenating features from all network from different scales into one vector
# to harmonized or not harmonize, harmonize is first harmonize on the features from different networks then concatenating into one vector
def harmonic_allfeature_site(All_TC, K, data, correlation_method, to_harmonize, useGAMs, mean_index):
    for i, k in enumerate(K):
        features_eachk = harmonic_site(All_TC, k, data, correlation_method, to_harmonize, useGAMs, mean_index)
        if i == 0:
            combined_fea = features_eachk
        else:
            combined_fea = np.hstack((combined_fea, features_eachk))

    harmonized_features = combined_fea
    return harmonized_features


def tangent_withref(in_mat, ref_mean):
    w, v = eig(ref_mean)
    new_g = inv(sqrtm(ref_mean))
    w, v = eig(new_g)
    data_size = in_mat.shape[0]
    out = np.zeros((in_mat.shape[0], int(in_mat.shape[1] * (in_mat.shape[1] + 1) / 2)))
    for i in range(data_size):
        w, v = eig(in_mat[i, :, :])
        temp_mat = logm(np.matmul(np.matmul(new_g, in_mat[i, :, :]), new_g))
        out[i, :] = temp_mat[np.triu_indices(in_mat.shape[1], k=0)]
    return out


def tangent(in_mat, ref):
    if ref == 1:
        C_g = get_C_l(in_mat)
    elif ref == 2:
        C_g = get_C_e(in_mat)
    else:
        C_g = get_C_h(in_mat)
    w, v = eig(C_g)
    new_g = inv(sqrtm(C_g))
    w, v = eig(new_g)
    data_size = in_mat.shape[0]
    out = np.zeros((in_mat.shape[0], int(in_mat.shape[1] * (in_mat.shape[1] + 1) / 2)))
    for i in range(data_size):
        w, v = eig(in_mat[i, :, :])
        temp_mat = logm(np.matmul(np.matmul(new_g, in_mat[i, :, :]), new_g))
        out[i, :] = temp_mat[np.triu_indices(in_mat.shape[1], k=0)]
    return out


def fill_matrix(a):
    n = int(np.sqrt(len(a) * 2)) + 1
    mask = np.tri(n, dtype=bool, k=-1)  # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n, n))
    out[mask] = a
    out = out + np.transpose(out)
    np.fill_diagonal(out, 1)
    return out


def get_C_e(in_mat):
    data_size = in_mat.shape[0]
    C_e = np.zeros((in_mat.shape[1], in_mat.shape[1]))
    for i in range(data_size):
        C_e = C_e + in_mat[i, :, :]
    C_e = C_e / (data_size)
    return C_e


def get_C_h(in_mat):
    data_size = in_mat.shape[0]
    C_h = np.zeros((in_mat.shape[1], in_mat.shape[1]))
    for i in range(data_size):
        C_h = C_h + inv(in_mat[i, :, :])
    C_h = 1 / data_size * C_h
    C_h = inv(C_h)
    return C_h


def get_C_l(in_mat):
    data_size = in_mat.shape[0]
    C_l = np.zeros((in_mat.shape[1], in_mat.shape[1]))
    for i in range(data_size):
        C_l = C_l + (1 / (data_size)) * logm(in_mat[i, :, :])
    C_l = expm(C_l)
    return C_l


def full_correlation(All_TC, k, dataframe, to_harmonize, use_GAMs):
    correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
    temp = All_TC['TC_' + str(k)]
    correlation_matrix = correlation_measure.fit_transform(temp)

    if to_harmonize == 1:
        correlation_matrix = np.arctanh(correlation_matrix)
        harmonized_features = np.tanh(harmonize_onebyone(correlation_matrix, dataframe, use_GAMs))
    else:
        harmonized_features = correlation_matrix
    return harmonized_features


# when k=1 , first harmonize then concatenate
def full_correlation_allfeature(All_TC, K, dataframe, to_harmonize, use_GAMs):
    # k is a list ,[17,25,50,75,100,125,150]
    # concanaten
    for i, k in enumerate(K):
        harmonized_eachk = full_correlation(All_TC, k, dataframe, to_harmonize, use_GAMs)
        if i == 0:
            harmonized_all_features = harmonized_eachk
        else:
            harmonized_all_features = np.hstack((harmonized_all_features, harmonized_eachk))

    return harmonized_all_features


def harmonize_onebyone(features, dataframe, use_GAMs):
    dataframe['SEX'] = (dataframe['Sex'] == 'M').astype(np.int)
    covars = dataframe[['SITE', 'Age', 'SEX']]
    if use_GAMs == 1:
        model, harmonized_features = nh.harmonizationLearn(features, covars,
                                                           smooth_terms=['Age'],
                                                           smooth_term_bounds=(
                                                               np.floor(np.min(dataframe.Age)),
                                                               np.ceil(np.max(dataframe.Age))))
    else:
        model, harmonized_features = nh.harmonizationLearn(features, covars)
    return harmonized_features


# harmonization in parallel
def harmonized_parallel(features, feature_id, dataframe):
    dataframe['SEX'] = (dataframe['Sex'] == 'M').astype(int)
    covars = dataframe[['SITE', 'Age', 'SEX']]
    temp = features[:, feature_id].reshape(-1, 1)
    model, harmonized_feature_single = nh.harmonizationLearn(temp, covars, eb=False,
                                                             smooth_terms=['Age'],
                                                             smooth_term_bounds=(
                                                                 np.floor(np.min(dataframe.Age)),
                                                                 np.ceil(np.max(dataframe.Age))))
    return harmonized_feature_single.flatten()


###harmonized in combat paralleled
def full_correlation_parallel(TC, dataframe, to_harmonize, use_GAMs, num_cores):
    correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
    correlation_matrix = correlation_measure.fit_transform(TC)

    if to_harmonize == 1:
        correlation_matrix = np.arctanh(correlation_matrix)
        results = Parallel(n_jobs=num_cores)(
            delayed(harmonized_parallel)(correlation_matrix, i, dataframe) for i in range(correlation_matrix.shape[1]))
        harmonized_features = np.tanh(np.array(results)).T
    else:
        harmonized_features = correlation_matrix
    return harmonized_features


# def harmonize(features, dataframe,use_GAMs):
#     batch = dataframe[['SITE']].squeeze()
#     mod = patsy.dmatrix("~ Age + Sex", dataframe, return_type="dataframe")
#     tag_covbat = covbat(pd.DataFrame(np.transpose(features)), batch, mod, "Age", pct_var=0.95, n_pc=0)
#     harmonized_features = np.transpose(tag_covbat.to_numpy())
#     return harmonized_features


def get_between_scale_TC(K, All_TC):
    new_data = []
    for i in range(4259):
        temp = np.hstack(([All_TC['TC_' + str(k)][i] for k in K]))
        new_data.append(temp)
    k = np.sum(K)
    return new_data, k


### tangent space parameterization with harmonic mean (paralleled)
def harmonic_between_scale(TC, k, data, to_harmonize, use_GAMs, mean_index, num_cores, correlation_method='covariance'):
    correlation_measure = ConnectivityMeasure(kind=correlation_method)
    harmonic_mean = np.zeros((7, k, k))
    shape = int(k * (k + 1) / 2)
    harmonic_mean_upper = np.zeros((7, shape))
    for i, name in enumerate(['BLSA', 'OASIS', 'ABC', 'CARDIA-1', 'CARDIA-3', 'CARDIA-4', 'UKBIOBANK']):
        index = data[data.SITE.str.contains(name)].index
        temp = [TC[i] for i in index]
        connectivity = correlation_measure.fit_transform(temp)
        harmonic_mean[i, :, :] = get_C_h(connectivity)
    #     harmonic_mean_upper[i, :] = harmonic_mean[i, :, :][np.triu_indices(k, k=0)]
    # distances = euclidean_distances(harmonic_mean_upper)
    # index_mean = np.argmin(np.sum(distances, axis=0))
    # print(index_mean)
    final_mean = harmonic_mean[mean_index, :, :]
    all_tc = TC
    connectivities = correlation_measure.fit_transform(all_tc)
    connectivities = tangent_withref(connectivities, final_mean)

    if to_harmonize == 1:
        results = Parallel(n_jobs=num_cores)(
            delayed(harmonized_parallel)(connectivities, i, data) for i in range(connectivities.shape[1]))
        harmonized_features = np.array(results).T
    else:
        harmonized_features = connectivities
    return harmonized_features


### tangent space parameterization with geometric mean (paralleled)
def geometric_between_scale(TC, k, data, to_harmonize, use_GAMs, mean_index, num_cores):
    correlation_measure = ConnectivityMeasure(kind='covariance')
    shape = int(k * (k + 1) / 2)
    g_mean = np.zeros((7, k, k))
    for i, name in enumerate(['BLSA', 'OASIS', 'ABC', 'CARDIA-1', 'CARDIA-3', 'CARDIA-4', 'UKBIOBANK']):
        index = data[data.SITE.str.contains(name)].index
        temp = [TC[i] for i in index]
        correlation_matrix = correlation_measure.fit_transform(temp)
        g_mean[i, :, :] = geometric_mean(correlation_matrix, max_iter=100, tol=1e-7)
        # g_mean_upper[i, :] = g_mean[i, :, :][np.triu_indices(k, k=0)]
    # distances = euclidean_distances(g_mean_upper)
    # index_mean = np.argmin(np.sum(distances, axis=0))
    # print(index_mean)
    final_gmean = g_mean[mean_index, :, :]
    whitening = map_eigenvalues(lambda x: 1. / np.sqrt(x), final_gmean)
    all_tc = TC
    connectivities = correlation_measure.fit_transform(all_tc)
    connectivities = [map_eigenvalues(np.log, whitening.dot(cov).dot(whitening)) for cov in connectivities]
    connectivities = [sym_matrix_to_vec(con, discard_diagonal=False) for con in connectivities]
    connectivity_matrix = np.array(connectivities)

    if to_harmonize == 1:
        results = Parallel(n_jobs=num_cores)(
            delayed(harmonized_parallel)(connectivity_matrix, i, data) for i in range(connectivity_matrix.shape[1]))
        harmonized_features = np.array(results).T
    else:
        harmonized_features = np.array(connectivities)
    return harmonized_features


def predict_within_scale_fullcorr(n_splits, k, n_cv_times, to_harmonize, regression_method, fea_dir, main_dir):
    # correlation method is used when using harmonic tangent projection, correlation or covariance
    # to_harmonize = 0
    use_GAMs = 1
    f = open(main_dir+'/Data/All_TC_new.pkl', 'rb')
    All_TC = pickle.load(f)  # dict_keys(['TC_17', 'TC_25', 'TC_32', 'TC_50', 'TC_75', 'TC_100', 'TC_125', 'TC_150'])
    f.close()
    NMF_fMRI = pd.read_pickle(main_dir+'/Data/final_NMF_fMRI_QC.pkl')
    age = NMF_fMRI.Age.values
    if os.path.exists(fea_dir + '/features_%s.npy'%(str(k))):
        features = np.load(fea_dir + '/features_%s.npy'%(str(k)))
    else:
        if k == 1:
            K = [17, 25, 50, 75, 100, 125, 150]
            features = full_correlation_allfeature(All_TC, K, NMF_fMRI, to_harmonize, use_GAMs)
        else:
            features = full_correlation(All_TC, k, NMF_fMRI, to_harmonize, use_GAMs)
        np.save(fea_dir +'/features_%s.npy'%(str(k)), features)


    print(features.shape)
    out_dir = fea_dir + '/' + regression_method +'/scale_' +str(k)
    if regression_method == 'GroupLasso':
        group = make_group('within', features.shape[1])

        age_predict(features, n_splits, n_cv_times, k, age, NMF_fMRI, out_dir, regression_method, group=group)
    else:
        age_predict(features, n_splits, n_cv_times, k, age, NMF_fMRI, out_dir, regression_method)


def predict_within_scale_tangent(n_splits, k, n_cv_times, to_harmonize, regression_method, tangent_method,
                                 correlation_method, fea_dir, mean_index, main_dir):
    # correlation method is used when using harmonic tangent projection, correlation or covariance
    # to_harmonize = 0
    use_GAMs = 1
    f = open(main_dir+'/Data/All_TC_new.pkl', 'rb')
    All_TC = pickle.load(f)  # dict_keys(['TC_17', 'TC_25', 'TC_32', 'TC_50', 'TC_75', 'TC_100', 'TC_125', 'TC_150'])
    f.close()
    NMF_fMRI = pd.read_pickle(main_dir+'/Data/final_NMF_fMRI_QC.pkl')
    age = NMF_fMRI.Age.values
    print(age.shape)
    if os.path.exists(fea_dir + '/features_%s.npy'%(str(k))):
        features = np.load(fea_dir+ '/features_%s.npy'%(str(k)))
    else:
        if tangent_method == 'harmonic':
            # correlation_method = 'correlation' # or covariance
            if k == 1:
                K = [17, 25,50,75, 100, 125,150]
                features = harmonic_allfeature_site(All_TC, K, NMF_fMRI, correlation_method, to_harmonize,
                                                    use_GAMs, mean_index)
            else:
                features = harmonic_site(All_TC, k, NMF_fMRI, correlation_method, to_harmonize, use_GAMs, mean_index)
        elif tangent_method == 'geometric':
            if k == 1:
                K = [17, 25, 50, 75, 100, 125, 150]
                features = geometric_allfeature_site(All_TC, K, NMF_fMRI, to_harmonize, use_GAMs, mean_index)
            else:
                features = geometric_site(All_TC, k, NMF_fMRI, to_harmonize, use_GAMs, mean_index)
        np.save(fea_dir+'/features_%s.npy'%(str(k)), features)
    print(features.shape)
    out_dir = fea_dir + '/' + regression_method +'/scale_' +str(k)
    if regression_method == 'GroupLasso':
        group = make_group('within', features.shape[1])
        #print(group)
        age_predict(features, n_splits, n_cv_times, k, age, NMF_fMRI, out_dir, regression_method, group=group)
    else:
        age_predict(features, n_splits, n_cv_times, k, age, NMF_fMRI, out_dir, regression_method)



def predict_between_scale(n_splits, n_cv_times, to_harmonize, FC_measure, tangent_method, regression_method,
                          num_cores, mean_index, fea_dir, main_dir):
    use_GAMs = 1
    f = open(main_dir+'/Data/All_TC_new.pkl', 'rb')
    All_TC = pickle.load(f)  # dict_keys(['TC_17', 'TC_25', 'TC_32', 'TC_50', 'TC_75', 'TC_100', 'TC_125', 'TC_150'])
    f.close()
    NMF_fMRI = pd.read_pickle(main_dir+'/Data/final_NMF_fMRI_QC.pkl')
    age = NMF_fMRI.Age.values
    K = [17, 25, 50, 75, 100, 125, 150]

    new_TC, new_k = get_between_scale_TC(K, All_TC)
    

    if os.path.exists(fea_dir + '/features_%s.npy'%(str(new_k))):
        features = np.load(fea_dir + '/features_%s.npy'%(str(new_k)))
    else:
        if FC_measure == 'correlation':
            features = full_correlation_parallel(new_TC, NMF_fMRI, to_harmonize, use_GAMs, num_cores)
        elif FC_measure == 'tangent':
            if tangent_method == 'harmonic':
                features = harmonic_between_scale(new_TC, new_k, NMF_fMRI, to_harmonize, use_GAMs, mean_index,
                                                  num_cores)
            elif tangent_method == 'geometric':
                features = geometric_between_scale(new_TC, new_k, NMF_fMRI, to_harmonize, use_GAMs, mean_index,
                                                   num_cores)
        np.save('%s/features_%s.npy' % (os.path.dirname(out_dir), str(new_k)), features)
    out_dir = fea_dir + '/' + regression_method +'/scale_' +str(new_k)
    if regression_method == 'GroupLasso':
        group = make_group('between', features.shape[1])
        age_predict(features, n_splits, n_cv_times, new_k, age, NMF_fMRI, out_dir, regression_method, group=group)
    else:
        age_predict(features, n_splits, n_cv_times, new_k, age, NMF_fMRI, out_dir, regression_method)


########################################################################################

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--feature_type', type=str, default=None, help='within features or between features')
#     parser.add_argument('--to_harmonize', type=int, default=0, help='harmonize or not harmonize')
#     parser.add_argument('--FC_measure', type=str, default=None, help='full correlation or tangent space')
#     parser.add_argument('--tangent_method', type=str, default=None, help='number of leaves')
#     parser.add_argument('--regression_method', type=str, default=None, help='number of leaves')
#     parser.add_argument('--result_dir', type=str, default=None, help='the result directory')
#     return parser.parse_args()
st = time.time()

n_splits = 5
n_cv_times = 1
mean_index = 6


parser = argparse.ArgumentParser()
parser.add_argument('--feature_type', type=str, default=None, help='within features or between features')
parser.add_argument('--to_harmonize', type=int, default=0, help='harmonize or not harmonize')
parser.add_argument('--FC_measure', type=str, default=None, help='full correlation or tangent space')
parser.add_argument('--tangent_method', type=str, default=None, help='number of leaves')
parser.add_argument('--regression_method', type=str, default=None, help='number of leaves')
parser.add_argument('--result_dir', type=str, default=None, help='the result directory')
parser.add_argument('--main_dir', type=str, default=None, help='the main directory')
parser.add_argument('--k', type=int, default=None, help='network scales')
opt = parser.parse_args()

to_harmonize = opt.to_harmonize ## 0 or 1
FC_measure = opt.FC_measure ## 'correlation' or 'tangent'
feature_type = opt.feature_type ## 'between' or 'within'
tangent_method = opt.tangent_method ## 'geometric' or 'harmonic'
regression_method = opt.regression_method ## 'ridge' or 'grouplasso'
out_dir = opt.result_dir
k = opt.k
mean_index = 6
harmonize_state = ['nohar','har']



if feature_type == 'between':
    predict_between_scale(n_splits, n_cv_times, to_harmonize, FC_measure, tangent_method, regression_method, 6,
                          mean_index, out_dir, opt.main_dir)
elif feature_type == 'within':
    #k = 175
    if FC_measure == 'tangent':
        predict_within_scale_tangent(n_splits, k, n_cv_times, to_harmonize, regression_method, tangent_method,
                                     'covariance', out_dir, mean_index, opt.main_dir)
    elif FC_measure == 'correlation':
        predict_within_scale_fullcorr(n_splits, k, n_cv_times, to_harmonize, regression_method, out_dir, opt.main_dir)

elif feature_type == 'between_within':
    if FC_measure == 'tangent':
        predict_betweenNoWithin(n_splits, n_cv_times, regression_method, out_dir, opt.main_dir)

time.sleep(3)
elapsed_time = time.time() - st
print('Execution time:', time.strftime("%D:%H:%M:%S", time.gmtime(elapsed_time)))