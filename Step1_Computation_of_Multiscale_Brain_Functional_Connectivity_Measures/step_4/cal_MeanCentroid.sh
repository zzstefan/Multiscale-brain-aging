#!/bin/sh

############################################################
help()
{
cat << HELP


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
			-k)
				num_net=$2;
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


matlab -nodisplay -nosplash -r "addpath(genpath('/cbica/home/zhouz/projects/istaging/LiHM_NMF/Code_mvNMF_l21_ard_v3_debug/lib/NIfTI_20140122'));step4_1_calMeanCentroid('${num_net}');exit"