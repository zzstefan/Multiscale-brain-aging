# The computation of multiscale brain functional connectivity measures

1. We applied the personalized NMF brain decomposition method https://github.com/hmlicas/Collaborative_Brain_Decomposition and decomposed the brain into multiple scales. The basic functions can be referred to the brain decomposition repository above. It mainly consists of three steps:

- We first computed group-level FNs using a normalized-cuts based spectral clustering method (Step 2) to identify representative FNs from 50 sets of group-level FNs, each set being computed on a subset of 150 subjects (Step 1);
- The group-level FNs were then used as initializing FNs to compute personalized FNs based on each subject’s fMRI data (Step 3);
- We computed the FNs at seven scales (Step 3), yielding seven sets of  K (K=17, 25, 50, 75, 100, 125 and 150) FNs.


The brain decomposition was run on a cluster using Sun Grid Engine scheduler. We also have developed a Matlab-based toolbox (pNet, https://github.com/YuncongMa/pNet) which can provide a user-friendly interface to perform personalized functional network computation using the same method.