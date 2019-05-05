import torch_lddmm
import numpy as np

#template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_atlas_forSTS.img'
#target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_target_forSTS.img'
template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_affine.img'
target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_deformedatlas.img'
#template_file_name = '/cis/home/leebc/Projects/LDDMM/data/devin/processed_atlas.nii'
#target_file_name = '/cis/home/leebc/Projects/LDDMM/data/devin/processed_affine_deformed_target.nii'
zeroimg = '/cis/home/leebc/Projects/Mouse_Histology/code/pytorch/zeroimg.img'

lddmm = torch_lddmm.LDDMM(template=template_file_name,target=target_file_name,costmask=None,outdir='/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/',gpu_number=0,a=8,p=2,niter=200,epsilon=2e-3,sigma=[20.0],sigmaR=20.0,nt=5,doaffine=1,checkaffinestep=0,epsilonL=1e-7,epsilonT=2e-5,optimizer='gd',minbeta=1e-15,dtype='float',im_norm_ms = 0,cc=0,energy_fraction=0.0,we=3,sigmaW=[20.0,200.0,10.0],nMstep=5)
lddmm.run()
lddmm.setParams('a',4)
lddmm.run(restart=False)
lddmm.setParams('a',2)
lddmm.run(restart=False)
