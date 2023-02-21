#!/bin/sh

dataset='All'

#for i in $(seq 1 50)
#do
	#file=/cbica/home/zhouz/projects/istaging/LiHM_NMF/result/${dataset}/${K}_network/init_r2/${dataset}_vol_num100_comp${N}_Sub${i}_S_*/init.mat
	#echo ${file} >> /cbica/home/zhouz/projects/istaging/LiHM_NMF/result/${dataset}/${K}_network/init_file_network.txt
#done
for K in 150
do
	main_dir=/cbica/home/zhouz/projects/istaging/LiHM_NMF2/result/${dataset}/${K}_network/init_r2/

	dir=$(ls -l ${main_dir} |awk '/^d/ {print $NF}')
	for i in $dir
	do	
		file=${main_dir}$i/init.mat
		echo ${file} >> /cbica/home/zhouz/projects/istaging/LiHM_NMF2/result/${dataset}/${K}_network/init_file_network.txt
	done
done