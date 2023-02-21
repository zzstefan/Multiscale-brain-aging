#!/bin/sh

source `dirname $0`/settings.sh || { echo "Failed to source settings.sh!" 1>&2; exit 1; }
############################################################
help()
{
cat << HELP

This script does the following:
	
	1. Detecting subject-specific functional networks while establishing group level coorespondence
	
		Li, Hongming, Theodore D. Satterthwaite, and Yong Fan. "Large-scale sparse functional networks from resting state fMRI." Neuroimage 156 (2017): 1-13.
		Li, Hongming, Xiaofeng Zhu, and Yong Fan. "Identification of multi-scale hierarchical brain functional networks using deep matrix factorization." MICCAI, 2018.
		
#############################################################
Usage : $0 [OPTIONS]
OPTIONS:
Reqd:	
	-s	< file >	sbjListFile		--  text file contains paths to the fMRI image files (.nii or .nii.gz), each line refers to one image
	-m	< file > 	maskFile		--  mask image, for example grey matter mask (.nii or .nii.gz)
	-p 	< file > 	prepDataFile	--	auxiliary .mat file containing spatial neighborhood information, will be generated automatically using maskFile and saved for later use
	-d 	< path >	outDir			--	output directory
	-s	< int >	    spaR			-- 	radius of spatial neighborhood (1 voxel by default)
	-v  < int >     vxI				--  0 (spatial proximity only), 1 (image similarity and spatial proximity)
	-ad < int >		ard				--	0/1 (without/with parsimonious regularization)
	-iN	< int >     iterNum			--  # of iteration (use value larger than 1000)
	-K	< int >		K				-- 	# of ICNs
	-tN < int >		timeNum			--	# of time points of fMRI data
	-a	< int >		alpha			--	spatial sparsity regularization parameter (try 1 first), larger value leads to more spatial sparse ICNs
	-b	< int >		beta			--	spatial (graph) smoothness regularization parameter (try 10 first), larger value leads to more spatial smooth ICNs
	-r	< path >	resId			--	prefix string of the output result files.
	-o  < path >    outputDir		--	
output: ${outDir}/init.mat, file of initialization results, containing variable 'initU' and 'initV'(group ICNs).

ERROR: Not enough input arguments!!
HELP
exit 1
}

parse()
{
	while [ -n "$1" ];
	do
		case $1 in
			-h)
				help;
				shift 1;;
			-md)
				main_dir=$2;
				shift 2;;
			-s)
				sbjListFile=$2;
				shift 2;;
			-sTC)
				sbjTCFile=$2;
				shift 2;;
			-m)
				maskFile=$2;
				shift 2;;
			-p)
				prepDataFile=$2;
				shift 2;;
			-o)
				outDir=$2;
				shift 2;;
			-r)
				resId=$2;
				shift 2;;
			-init)
				init_file=$2;
				shift 2;;
			-k)
				K=$2;
				shift 2;;
			-a)
				alphaS21=$2;
				shift 2;;
			-b)
				alphaL=$2;
				shift 2;;
			-v)
				vxI=$2;
				shift 2;;
			-sR)
				spaR=$2;
				shift 2;;
			-ad)
				ard=$2;
				shift 2;;
			-e)
				eta=$2;
				shift 2;;
			-iN)
				iNum=$2;
				shift 2;;
			-cG)
				calcGrp=$2;
				shift 2;;
			-pf)
				parforOn=$2;
				shift 2;;
			-s_id)
				sub_ID=$2;
				shift 2;;
			-*)
				echo "ERROR:no such option $1"
				help;;
			*)
				break;;
		esac
	done

}

if [ $# -lt 1 ]
then
	help
fi

## Reading arguments
parse $*

# PID=$$
# tmp=${CBICA_TMPDIR}/step3_${PID}


matlab -nodisplay -nosplash -r "addpath(genpath('$main_dir/Code_mvNMF_l21_ard_v3_debug')); \
step3_single_new('${sbjListFile}','${sbjTCFile}', '${maskFile}','${prepDataFile}','${outDir}',\
'${resId}','${init_file}','${K}','${alphaS21}',\
'${alphaL}','${vxI}','${spaR}','${ard}','${eta}','${iNum}',\
'${calcGrp}','${parforOn}','${sub_ID}');exit"

# /cbica/home/zhouz/projects/istaging/LiHM_NMF/scripts/NMF_steps/step_3/run_step3_single_new.sh  /cbica/software/external/matlab/R2018A ${sbjListFile} \
# ${sbjTCFile} ${maskFile} ${prepDataFile} ${outDir} ${resId} ${init_file} ${K} ${alphaS21} ${alphaL} ${vxI} ${spaR} ${ard} ${eta} ${iNum} ${calcGrp} \
# ${parforOn} ${sub_ID}