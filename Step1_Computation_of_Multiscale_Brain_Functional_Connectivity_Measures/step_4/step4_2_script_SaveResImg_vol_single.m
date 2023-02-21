clc;
clear;

%addpath(genpath('C:\Users\LiHon\Google Drive\Code\Inhouse\ongoing\Code_mvNMF_l21_ard_v3_release\Release'));
%addpath(genpath('D:\Google_drive\Code\Inhouse\ongoing\Code_mvNMF_l21_ard_v3_release\Release'));
K=[17,25,50,75,100,125,150];
dataset='All'


for i=1:length(K)
    resFileName = ['/cbica/home/zhouz/projects/istaging/LiHM_NMF2/result/',dataset,'/',num2str(K(i)),'_network/res_cen.mat'];
    maskName = '/cbica/home/zhouz/projects/istaging/LiHM_NMF/atlas/mask_thr0p5_wmparc_2_cc.nii.gz';
    outDir = ['/cbica/home/zhouz/projects/istaging/LiHM_NMF2/result/',dataset,'/',num2str(K(i)),'_network/fig'];
    saveFig = 1;
    refNiiName = '/cbica/home/zhouz/projects/istaging/LiHM_NMF/atlas/mask_thr0p5_wmparc_2_cc.nii.gz';
    func_saveVolRes2Nii(resFileName,maskName,outDir,saveFig,refNiiName);
end