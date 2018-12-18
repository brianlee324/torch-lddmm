import torch_lddmm

template_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_affine.img'
target_file_name = '/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_targetmasked.img'

lddmm = torch_lddmm.LDDMM(template=template_file_name,target=target_file_name,outdir='/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/',gpu_number=0,a=3.0,p=2,niter=100,epsilon=5e-3,sigma=2.0,sigmaR=3.0,nt=5,doaffine=0,epsilonL=1e-5,epsilonT=2e-5,optimizer='gdw')
lddmm.run()