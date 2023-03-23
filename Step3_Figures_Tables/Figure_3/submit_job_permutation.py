import os



main_dir = '/cbica/home/zhouz/projects/istaging_2022'
script_path = main_dir+'/Scripts_bash'
print(script_path)
if not os.path.exists(script_path):
    os.makedirs(script_path)

for i in range(10):
    system_command = "source activate istaging; python %s/repeat_cross_validation_cluster.py -seed %s"%(main_dir, str(i))
    script=open(script_path+'/repeat_%s.sh'%(str(i)),'w')
    script.write(system_command)
    script.close()
    job_command = 'qsub -l h_vmem=50G -j y -N repeat_%s -V -o %s/temp_%s.o ' \
                  '%s/repeat_%s.sh'%(str(i), script_path, str(i), script_path, str(i))
    os.system(job_command)