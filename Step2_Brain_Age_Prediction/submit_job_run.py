import os
harmonize_state = ['nohar', 'har']


####script_path is the final folder where all results will be saved
####fea_dir the folder where features are stored 
#### Need time need to look carefully when submitting new jobs
# FC = ['correlation', 'tangent']
# harmonized = [0, 1]
# fea = ['between', 'within']
# regression_method = 'RidgeRegression'
# tangent = ['geometric', 'harmonic']
main_dir = '/cbica/home/zhouz/projects/istaging/LiHM_NMF2'

FC = ['correlation']
harmonized = [0,1]
fea = ['within']
regression_method = 'ridge'
tangent = ['geometric']
K=[25,50,75,100,124,150,1]
#K=[17]

for k in K:
    for har in harmonized:
        for feature_type in fea:
            for fc in FC:
                if fc == 'correlation':
                    script_path = '%s/multiscale_result/%s/%s/%s/%s/scale_%s' % (main_dir, feature_type, fc, harmonize_state[har], regression_method, str(k))
                    fea_dir = '%s/multiscale_result/%s/%s/%s' % (main_dir, feature_type, fc, harmonize_state[har])
                    print(script_path)
                    print(fea_dir)
                    if not os.path.exists(script_path):
                        os.makedirs(script_path)
                    system_command = "source activate istaging; python %s/scripts/multiscale_scripts/generate_features.py --k %s --feature_type %s" \
                                     " --to_harmonize %s --FC_measure %s --regression_method %s --result_dir %s --main_dir %s" % (
                                         main_dir, str(k), feature_type, str(har), fc, regression_method, fea_dir, main_dir)
                    script = open(script_path + '/generate_fea.sh', 'w')
                    script.write(system_command)
                    script.close()
                    job_command = 'qsub -l h_vmem=100G,tmpfree=10M -j y -pe threaded 8 -N %s_%s_%s_%s -V -o %s/temp_para.o ' \
                                  '%s/generate_fea.sh' % (feature_type, str(har), fc, regression_method, script_path, script_path)
                    os.system(job_command)

                elif fc == 'tangent':
                    for tan in tangent:
                        script_path = '%s/multiscale_result/%s/%s/%s/%s/%s/scale_%s'%(main_dir, feature_type, fc, tan, harmonize_state[har], regression_method,str(k))
                        fea_dir = '%s/multiscale_result/%s/%s/%s/%s'%(main_dir, feature_type, fc, tan, harmonize_state[har])
                        print(fea_dir)
                        print(script_path)
                        if not os.path.exists(script_path):
                            os.makedirs(script_path)
                        system_command = "source activate istaging; python %s/scripts/multiscale_scripts/generate_features.py --k %s --feature_type %s" \
                                         " --to_harmonize %s --FC_measure %s --tangent_method %s --regression_method %s --result_dir %s --main_dir %s" %(
                        main_dir, str(k), feature_type, str(har), fc, tan, regression_method, fea_dir, main_dir)
                        script = open(script_path + '/generate_fea_%s.sh'%(str(k)), 'w')
                        script.write(system_command)
                        script.close()
                        job_command = 'qsub -l h_vmem=80G,tmpfree=10M -j y -pe threaded 16 -N %s_%s_%s_%s_%s -V -o %s/temp_para_%s.o ' \
                                      '%s/generate_fea_%s.sh' % (feature_type, str(har), fc, tan, regression_method, script_path, str(k), script_path, str(k))
                        os.system(job_command)


