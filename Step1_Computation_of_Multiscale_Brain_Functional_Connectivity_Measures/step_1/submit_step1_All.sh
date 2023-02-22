#!/bin/sh

echo -e "\nRunning commands on          : `hostname`"
echo -e "Start time                     : `date +%F-%H:%M:%S`\n"

dataset='All'



main_dir=/cbica/home/zhouz/projects/istaging/LiHM_NMF2
sbjList_dir=${main_dir}/subList/${dataset}
maskFile=/cbica/home/zhouz/projects/istaging/LiHM_NMF/atlas/mask_thr0p5_wmparc_2_cc.nii.gz
prepDataFile=/cbica/home/zhouz/projects/istaging/LiHM_NMF/new_CreatePrepData.mat
spaR=1
vxI=1
ard=0
iterNum=1000
timeNum=250
alpha=2
beta=10
resId="${dataset}_vol"


for K in 17 25 50 75 100 125 150
do
	outDir=${main_dir}/result/${dataset}/${K}_network/init_r2
	for i in $(seq 1 50)
	do
		sbjList_file=${sbjList_dir}/subList_${i}.txt
		list_id=${i}
		jid=$(qsub \
			-terse \
			-b y \
			-j y \
			-l h_vmem=80G \
			-o ${main_dir}/scripts/sge_output/step_1/\$JOB_NAME-\$JOB_ID.log \
			${main_dir}/scripts/NMF_steps/step_1/run_step1.sh -md ${main_dir} -s ${sbjList_file} \
			-m ${maskFile} -p ${prepDataFile} \
			-o ${outDir} \
			-i_s ${list_id} \
			-sR ${spaR} \
			-v ${vxI} \
			-ad ${ard} \
			-iN ${iterNum} \
			-k ${K} \
			-tN ${timeNum} \
			-a ${alpha} \
			-b ${beta} \
			-r ${resId}  )
			
	done
done
