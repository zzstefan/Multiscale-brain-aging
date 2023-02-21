#!/bin/sh

echo -e "\nRunning commands on          : `hostname`"
echo -e "Start time                     : `date +%F-%H:%M:%S`\n"


main_dir=/cbica/home/zhouz/projects/istaging/LiHM_NMF2/scripts




for K in 17 25  50 75 100 125 150
do
 	jid=$(qsub \
 		-terse \
		-pe threaded 4 \
 		-l h_vmem=50G \
 		-o ${main_dir}/sge_output/step_4/\$JOB_NAME-\$JOB_ID.stdout \
 		-e ${main_dir}/sge_output/step_4/\$JOB_NAME-\$JOB_ID.stderr \
		${main_dir}/NMF_steps/step_4/cal_MeanCentroid.sh -k $K) 
done