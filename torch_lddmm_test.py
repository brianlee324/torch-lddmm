import torch_lddmm
import numpy as np

#template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_atlas_forSTS.img'
#target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_target_forSTS.img'
template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_affine.img'
target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_deformedatlas.img'

lddmm = torch_lddmm.LDDMM(template=template_file_name,target=target_file_name,outdir='/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/',gpu_number=0,a=8,p=2,niter=200,epsilon=2e-3,sigma=1.0,sigmaR=3.0,nt=5,doaffine=0,epsilonL=1e-5,epsilonT=2e-5,optimizer='gdr',minbeta=1e-6,dtype='float')
lddmm.run()
lddmm.setParams('a',4)
lddmm.run(restart=False)
lddmm.setParams('a',2)
lddmm.run(restart=False)