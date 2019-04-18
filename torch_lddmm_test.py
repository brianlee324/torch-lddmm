import torch_lddmm
import numpy as np
import sys
sys.path.insert(0,'/cis/home/leebc/Software/')
import nibabel as nib

template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_affine.img'
target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_deformedatlas.img'
zeroimg = '/cis/home/leebc/Projects/Mouse_Histology/code/pytorch/zeroimg.img'

template_image_struct = nib.load(template_file_name)
target_image_struct = nib.load(target_file_name)
dx = template_image_struct.header['pixdim'][1:4]

template_image = np.squeeze(template_image_struct.get_data()).astype(np.float32)
target_image = np.squeeze(target_image_struct.get_data()).astype(np.float32)

#lddmm = torch_lddmm.LDDMM(template=[template_file_name],target=[target_file_name],outdir='/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/',gpu_number=0,a=8,p=2,niter=200,epsilon=2e-3,sigma=[1.0],sigmaR=3.0,nt=5,doaffine=0,checkaffinestep=0,epsilonL=1e-7,epsilonT=2e-5,optimizer='gdr',minbeta=1e-6,dtype='float',im_norm_ms = 1)
lddmm = torch_lddmm.LDDMM(template=template_image,target=target_image,outdir='/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/',a=8,niter=200,epsilon=2e-3,sigma=[20.0],sigmaR=20.0,optimizer='gdr',dx=dx)
lddmm.run()
lddmm.setParams('a',4)
lddmm.run(restart=False)
lddmm.setParams('a',2)
lddmm.run(restart=False)
