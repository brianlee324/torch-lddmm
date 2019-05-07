import torch_lddmm
import numpy as np

#template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_atlas_forSTS.img'
#target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_target_forSTS.img'
target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/atlas_scrambled.img'
template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/atlas_images/ara_nissl_40_gauss_scale1.img'
#zeroimg = '/cis/home/leebc/Projects/Mouse_Histology/code/pytorch/zeroimg.img'

lddmm = torch_lddmm.LDDMM(template=template_file_name,target=target_file_name,costmask=None,outdir='/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/',gpu_number=0,a=8,p=2,niter=200,epsilon=2e-3,sigma=[1.0],sigmaR=3.0,nt=5,do_affine=1,checkaffinestep=0,epsilonL=1e-7,epsilonT=2e-5,optimizer='gdr',minbeta=1e-6,dtype='float',im_norm_ms = 0,slice_alignment = 1)
lddmm._checkParameters()
lddmm._load(lddmm.params['template'],lddmm.params['target'],lddmm.params['costmask'])
lddmm.initializeVariables()
lddmm._allocateGradientDivisors()
#a,b,theta,outtarget = lddmm.sa(lddmm.J[0],lddmm.I[0],dim=1,epsilonxy=2.5e-11, epsilontheta=5.5e-11)
a,b,theta,outtarget = lddmm.sa(lddmm.J[0],lddmm.I[0],dim=1,epsilonxy=2.5e-6, epsilontheta=5.5e-7,norm=1)
#lddmm.run()
#lddmm.setParams('a',4)
#lddmm.run(restart=False)
#lddmm.setParams('a',2)
#lddmm.run(restart=False)
