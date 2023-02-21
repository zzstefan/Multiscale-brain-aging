function step4_1_calMeanCentroid(k)

K=str2double(k);
dataset='All';

main_dir='/cbica/home/zhouz/projects/istaging/LiHM_NMF2';
%sub_ID=importdata([main_dir,filesep,'subList',filesep,'all_subID_',dataset,'.txt']);
load([main_dir,filesep,'Data',filesep,'new_CreatePrepData.mat']);


for j=1:length(K)
    resDir=['/cbica/home/zhouz/projects/istaging/LiHM_NMF2/result/',dataset,'/',num2str(K(j)),'_network'];
    V_centroid_all = zeros(length(gNb),K(j));

    res_cen=dir([main_dir,'/result/',dataset,'/',num2str(K(j)),'_network/res_single/*/*/res_cen.mat']);
    for i=1:length(res_cen)
        load([res_cen(i).folder '/res_cen.mat']);
        V_centroid_all=V_centroid_all + V_centroid;
    end
    
    V_centroid = V_centroid_all / length(res_cen);
    
    outName_cen = [resDir,filesep,'res_cen.mat'];
    save(outName_cen,'V_centroid','-v7.3');
end