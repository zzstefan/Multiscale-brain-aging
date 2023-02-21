#!/bin/sh


echo -e "\nRunning commands on          : `hostname`"
echo -e "Start time                     : `date +%F-%H:%M:%S`\n"

dataset='All'
main_dir=/cbica/home/zhouz/projects/istaging/LiHM_NMF2

maskFile=/cbica/home/zhouz/projects/istaging/LiHM_NMF/atlas/mask_thr0p5_wmparc_2_cc.nii.gz
prepDataFile=/cbica/home/zhouz/projects/istaging/LiHM_NMF/new_CreatePrepData.mat
resId='IndividualParcellation'
sbjTCFile=/cbica/home/zhouz/projects/istaging/LiHM_NMF/result/All/AllsbjData

#sbjList_file=${main_dir}/subList/all_subList_${dataset}.txt
#sub_ID=${main_dir}/subList/all_subID_${dataset}.txt

#sub_ID=${main_dir}/subList/all_subList_All.txt
#sbjList_file=${main_dir}/subList/all_subList_All.txt

spaR=1
vxI=1
ard=0
eta=1
iterNum=30
alphaS21=2
alphaL=10
calcGrp=0
parforOn=0

#for K in 17 25 32 50 75 100
#for K in 175 200
for K in 150
do
    sbjList_file=${main_dir}/subList/all_subList_${dataset}.txt
    sub_ID=${main_dir}/subList/all_subID_${dataset}.txt
	init_file=${main_dir}/result/${dataset}/${K}_network/robust_init/init.mat
	outDir=${main_dir}/result/${dataset}/${K}_network/res_single
	jid=$(qsub \
			  -terse \
			  -b y \
			  -l h_vmem=20G \
			  -o ${main_dir}/scripts/sge_output/step_3/\$JOB_NAME-\$JOB_ID.stdout \
			  -e ${main_dir}/scripts/sge_output/step_3/\$JOB_NAME-\$JOB_ID.stderr \
			  ${main_dir}/scripts/NMF_steps/step_3/run_step3_single.sh -md ${main_dir} -s ${sbjList_file} -sTC ${sbjTCFile}\
			  -m ${maskFile} -p ${prepDataFile} \
			  -o ${outDir} -r ${resId} \
			  -init ${init_file} -k ${K} -a ${alphaS21}\
			  -b ${alphaL}  -v ${vxI} -sR ${spaR}\
			  -ad ${ard} -e ${eta} -iN ${iterNum}\
			  -cG ${calcGrp}  -pf ${parforOn} \
			  -s_id ${sub_ID}
			  )
done
