function step3_single_new(sbjList_File,sbjTCFile,maskFile,prepDataFile,outDir,resId,initName,K,alphaS21,alphaL,vxI,spaR,ard,eta,iterNum,calcGrp,parforOn,subID)

if nargin~=18
    error('number of input should be 18 !');
end

% K=17;
% main_dir='/cbica/home/zhouz/projects/istaging/LiHM_NMF';
% result_dir=[main_dir, filesep, 'BLSA_test/',num2str(K),'_network/res_singleParcellation'];
% initName=[main_dir,filesep,'BLSA_test/',num2str(K),'_network/robust_init/init.mat'];
% maskFile=[main_dir,filesep,'atlas/mask_thr0p5_wmparc_2_cc.nii.gz'];
% prepDataFile=[main_dir,filesep,'new_CreatePrepData.mat'];
% 
% resId = 'IndividualParcellation';
% spaR=1;
% vxI=0;
% ard=1;
% eta=1;
% iterNum=30;
% alphaS21=2;
% alphaL=10;
% calcGrp=0;
% parforOn=0;
% 
% sub_ID=importdata([main_dir,filesep,'subList',filesep,'All_subID.txt']);
% sbj_AllList=importdata([main_dir,filesep,'subList',filesep,'All_subList.txt']);
addpath(genpath('/cbica/home/zhouz/projects/istaging/LiHM_NMF/Code_mvNMF_l21_ard_v3_debug'));

% K=str2double(K);
% alphaS21=str2double(alphaS21);
% alphaL=str2double(alphaL);
% vxI=str2double(vxI);
% spaR=str2double(spaR);
% ard=str2double(ard);
% eta=str2double(eta);
% iterNum=str2double(iterNum);
% calcGrp=str2double(calcGrp);
% parforOn=str2double(parforOn);

sub_ID=importdata(subID);
length(sub_ID);
sub_AllList=importdata(sbjList_File);
length(sub_AllList);

result_dir = outDir;

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

for i=1:length(sub_ID)
    temp=strsplit(sub_AllList{i}, '/');
    study=temp(4);
    if strcmp(study,'BLSA')
        ID=sub_ID{i};
    elseif strcmp(study,'cardia')
        ID=['CARDIA_',sub_ID{i}];
    elseif strcmp(study,'OASIS')
        ID=sub_ID{i};
    elseif strcmp(study,'chrisclark')
        ID=['PENN_',sub_ID{i}];
    else
        ID=['UKBB_',sub_ID{i}];
    end
    disp(ID);
    ResultantFolder=[result_dir,filesep,ID];
	sbjTCFolder=[sbjTCFile,filesep,ID];
    ResultFile_check = dir([ResultantFolder, '/**/final*.mat']);
    
    if isempty(ResultFile_check)
        if ~exist(ResultantFolder, 'dir')
            mkdir(ResultantFolder);
        end
        
        subIDFile=[ResultantFolder,filesep,'ID.mat'];
        save(subIDFile,'ID');
        
        sbjListFile=[ResultantFolder,filesep,'sbjListAllFile_',ID,'.txt'];
        if exist(sbjListFile, 'file')
	        system(['rm ', sbjListFile]);
        end
        
%         subfile=['/cbica/projects/BLSA/Pipelines/rsfMRI/rsfMRI_2020/Protocols/UKB_Pipeline/',ID, ...
%         '/fMRI_nosmooth/rfMRI.ica/reg_standard/filtered_func_data_clean.nii.gz'];
        subfile=sub_AllList{i};
        cmd=['echo ', subfile,' >> ',sbjListFile];
        system(cmd);
        
        %save([ResultantFolder,filesep,'Configuration.mat'],'sbjListFile', 'sbjTCFolder', 'maskFile', 'prepDataFile', 'ResultantFolder', 'resId', 'initName', 'K', 'alphaS21', 'alphaL', 'vxI', 'spaR', 'ard', 'eta', 'iterNum', 'calcGrp', 'parforOn');
        ScriptPath=[ResultantFolder,filesep,'tmp.sh'];
%         cmd=fprintf('./deployFuncMvnmfL21p1_func_vol_single.sh /cbica/software/external/matlab/R2018A "%s" "%s" "%s" "%s" "%s" "%s" "%s" %s %s %s %s %s %s %s %s %s %s "%s" > %s/ParcelFinal.log 2>&1' ...
%         ,sbjListFile,sbjTCFile,maskFile,prepDataFile,ResultantFolder,resId,initName,K,alphaS21, alphaL, vxI, spaR, ard, eta, iterNum, calcGrp, parforOn, sub_ID, ResultantFolder);
%     
        cmd = "/cbica/home/zhouz/projects/istaging/LiHM_NMF/Code_mvNMF_l21_ard_v3_debug/src/run_deployFuncMvnmfL21p1_func_vol_single.sh /cbica/software/external/matlab/R2018A '" + sbjListFile +"' '"+ sbjTCFolder +"' '"+ maskFile+"' '"+prepDataFile+"' '"+ResultantFolder+"' '"+resId+"' '"+initName+"' "+K+" "+alphaS21+" "+alphaL+" "+vxI+" "+spaR+" "+ard+" "+eta+" "+iterNum+" "+calcGrp+" "+parforOn+" > '"+ResultantFolder+"/ParcelFinal.log' "+ "2>&1";
        %cmd = "./deployFuncMvnmfL21p1_func_vol_single.sh /cbica/software/external/matlab/R2018A " + " nihao";
        cmd
        fid = fopen(ScriptPath, 'w');
        fprintf(fid, cmd);
        fclose(fid);
        
        system(['qsub -l h_vmem=20G ', ScriptPath]);
        pause(10);
    end
    
    
    
end