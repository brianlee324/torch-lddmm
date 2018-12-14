import torch_lddmm

template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_affine.img'
target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_targetmasked.img'

lddmm = torch_lddmm.LDDMM(template=template_file_name,target=target_file_name,outdir='/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/',gpu_number=None,a=3.0,p=2,niter=100,epsilon=5e-3,sigma=2.0,sigmaR=1.0,nt=5)
lddmm.run()