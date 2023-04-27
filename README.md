# Super-resolution-microscopy-project
General Guide for beginners:

a) Follow main program 1 2 3 step by step to creat your own dataset, complete training using SRR network and then make evaluation on a public test bench. You can of course create your own network(matters) and training base file(does not matter).

b) The simulation process is based on the modified version of OpenSIM. You can directly refer to openSIM or use simulation here.

c) If loss functions need to be tested, remember to make a switch in run.py. Jointloss can be imported from SSIM_L1_jointloss.py.

-------------------------------------------------------------------------------------
Remember to throw SRR.py to model_for_comparison.py and change the nework name in opt.
You can adjust SRR by choosing the most suitable hyperparameter on your dataset. 
The defult setting for I/O is 512x512, upscale = 1 

Hyperparmeter in SRR:
1. number of res group
2. number of res block
3. depth of network
4. number of encoder
5. use Channel Attention or not
6. use skip conenction or not
7. downsample layer: Maxpooling or conv
8. Input/Output size
9. Kernel size of conv

e.t.c

-------------------------------------------------------------------------------------
Microtubules.mrc only shows one cell folder, please go to BioSR dataset. You can also test on CCPs, ER, MTs and F-action in future work.



Credit:
1. OpenSIM (Matlab code for SIM illumiantion Simulation) https://github.com/LanMai/OpenSIM
2. Training(base file) https://github.com/charlesnchr/ML-SIM
3. BioSR(experimental SIM dataset) https://figshare.com/articles/dataset/BioSR/13264793?file=25714514
4. FairSIM https://github.com/fairSIM/fairSIM (remember to download ImageJ before using the plugin)



