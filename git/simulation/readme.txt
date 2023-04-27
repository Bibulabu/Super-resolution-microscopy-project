1. Follow main step 1 2 3 to creat your own dataset, complete training using SRR network and then make evaluation on a public test bench. You can of course create your own network(matters) and training base file(does not matter).

2. The simulation process is based on the modified version of OpenSIM. You can directly refer to openSIM or use simulation here.

3. If loss functions need to be tested, remember to make a switch in run.py. Jointloss can be imported from SSIM_L1_jointloss.py.


Credit:
OpenSIM (Matlab code) https://github.com/LanMai/OpenSIM
RCAN (Pytorch model) https://github.com/yulunzhang/RCAN
Training base file https://github.com/charlesnchr/ML-SIM

