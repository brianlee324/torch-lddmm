import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0,'/cis/home/leebc/Software/')
import nibabel as nib

start_time = time.time()

# replication-pad, artificial roll, subtract, single-sided difference on boundaries
def torch_gradient(arr, dx, dy, dz, grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu):
    arr = torch.squeeze(torch.nn.functional.pad(arr.unsqueeze(0).unsqueeze(0),(1,1,1,1,1,1),mode='replicate'))
    gradx = torch.cat((arr[1:,:,:],arr[0,:,:].unsqueeze(0)),dim=0) - torch.cat((arr[-1,:,:].unsqueeze(0),arr[:-1,:,:]),dim=0)
    grady = torch.cat((arr[:,1:,:],arr[:,0,:].unsqueeze(1)),dim=1) - torch.cat((arr[:,-1,:].unsqueeze(1),arr[:,:-1,:]),dim=1)
    gradz = torch.cat((arr[:,:,1:],arr[:,:,0].unsqueeze(2)),dim=2) - torch.cat((arr[:,:,-1].unsqueeze(2),arr[:,:,:-1]),dim=2)
    return gradx[1:-1,1:-1,1:-1]/dx/grad_divisor_x_gpu, grady[1:-1,1:-1,1:-1]/dy/grad_divisor_y_gpu, gradz[1:-1,1:-1,1:-1]/dz/grad_divisor_z_gpu


# set device
gpu_number = 0

# template
I = nib.load('/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_affine.img').get_data().astype(np.float32)
#I = nib.load('/cis/home/leebc/Projects/Mouse_Histology/data/registration/BNBoutput3/M918_forMRI/M918_defatlas_nosts.img').get_data().astype(np.float32)
#I = np.random.rand(340,340,340)

# target
J = nib.load('/cis/home/leebc/Projects/Mouse_Histology/data/registration/Hua121/Hua121_targetmasked.img').get_data().astype(np.float32)
#J = np.squeeze(nib.load('/cis/home/leebc/Projects/Mouse_Histology/data/registration/BNBoutput3/M918_forMRI/M918_80_full_cropped.img').get_data().astype(np.float32))
#J = np.random.rand(340,340,340)

I = (I - np.mean(I))/np.std(I)
J = (J - np.mean(J))/np.std(J)
#I = np.flip(I,axis=1).copy()
#J = np.flip(J,axis=1).copy()

# move
with torch.cuda.device(gpu_number):
    I_gpu = torch.tensor(I).float().cuda()
    J_gpu = torch.tensor(J).float().cuda()

dx = [0.04,0.04,0.04]
nx = I.shape

'''
# test interpolation
#ix,iy,iz = torch.meshgrid([torch.tensor(np.linspace(-1,1,I.shape[0])),torch.tensor(np.linspace(-1,1,I.shape[1])),torch.tensor(np.linspace(-1,1,I.shape[2]))])
ix,iy,iz = np.meshgrid(np.linspace(-1,1,I.shape[0]),np.linspace(-1,1,I.shape[1]),np.linspace(-1,1,I.shape[2]),indexing='ij')
with torch.cuda.device(gpu_number):
    ix_gpu = torch.tensor(ix).float().cuda()
    iy_gpu = torch.tensor(iy).float().cuda()
    iz_gpu = torch.tensor(iz).float().cuda()

out_gpu = torch.nn.functional.grid_sample(I_gpu.unsqueeze(0).unsqueeze(0),torch.stack((iz_gpu,iy_gpu,ix_gpu),dim=3).unsqueeze(0))
out = np.squeeze(out_gpu.to('cpu').numpy())
outfile = nib.AnalyzeImage(out,None)
outfile.header['dim'][1:4] = out.shape
outfile.header['pixdim'][1:4] = [0.04,0.04,0.04]
nib.save(outfile,'/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/out.img')
'''

# make smoothing kernel on CPU
f0 = np.arange(nx[0])/(dx[0]*nx[0])
f1 = np.arange(nx[1])/(dx[1]*nx[1])
f2 = np.arange(nx[2])/(dx[2]*nx[2])
F0,F1,F2 = np.meshgrid(f0,f1,f2,indexing='ij')
a = 3.0*dx[0] # a scale in mm
p = 2
Ahat = (1.0 - 2.0*a**2*((np.cos(2.0*np.pi*dx[0]*F0) - 1.0)/dx[0]**2 
                        + (np.cos(2.0*np.pi*dx[1]*F1) - 1.0)/dx[1]**2
                        + (np.cos(2.0*np.pi*dx[2]*F2) - 1.0)/dx[2]**2))**(2.0*p)
Khat = 1.0/Ahat
Khat = np.tile(np.reshape(Khat,(Khat.shape[0],Khat.shape[1],Khat.shape[2],1)),(1,1,1,2))

# move only one
with torch.cuda.device(gpu_number):
    Khat_gpu = torch.tensor(Khat).float().cuda()
    Ahat_gpu = torch.tensor(Ahat).float().cuda()

#K = np.fft.ifftn(Khat).real

# set parameters
niter = 100
epsilon = 5e-3
sigma = 2.0e0
sigmaR = 1.0e0
nt = 5
dt = 1.0/nt

# initialize
vt0, vt1, vt2 = np.zeros((nx[0],nx[1],nx[2],nt)), np.zeros((nx[0],nx[1],nx[2],nt)), np.zeros((nx[0],nx[1],nx[2],nt))
#It = np.tile(I[:,:,:,None],(1,1,1,nt+1))
EMAll = []
ERAll = []
EAll = []

# move
with torch.cuda.device(gpu_number):
    #vt0_gpu = torch.tensor(vt0).float().cuda()
    #vt1_gpu = torch.tensor(vt1).float().cuda()
    #vt2_gpu = torch.tensor(vt2).float().cuda()
    #It_gpu = torch.tensor(It).float().cuda()
    vt0_gpu = []
    vt1_gpu = []
    vt2_gpu = []
    It_gpu = []
    # pointers won't work
    It_gpu.append(torch.tensor(I[:,:,:]).float().cuda())
    for i in range(nt):
        vt0_gpu.append(torch.tensor(np.zeros((nx[0],nx[1],nx[2]))).float().cuda())
        vt1_gpu.append(torch.tensor(np.zeros((nx[0],nx[1],nx[2]))).float().cuda())
        vt2_gpu.append(torch.tensor(np.zeros((nx[0],nx[1],nx[2]))).float().cuda())
        It_gpu.append(torch.tensor(I[:,:,:]).float().cuda())
    #vt0_gpu = [torch.tensor(np.zeros((nx[0],nx[1],nx[2]))).float().cuda()]*nt
    #vt1_gpu = [torch.tensor(np.zeros((nx[0],nx[1],nx[2]))).float().cuda()]*nt
    #vt2_gpu = [torch.tensor(np.zeros((nx[0],nx[1],nx[2]))).float().cuda()]*nt
    #It_gpu = [torch.tensor(I[:,:,:]).float().cuda()]*(nt+1)
    

# image sampling domain
x0 = np.arange(nx[0])*dx[0]
x1 = np.arange(nx[1])*dx[1]
x2 = np.arange(nx[2])*dx[2]
X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')

# move
with torch.cuda.device(gpu_number):
    X0_gpu = torch.tensor(X0).float().cuda()
    X1_gpu = torch.tensor(X1).float().cuda()
    X2_gpu = torch.tensor(X2).float().cuda()

# allocate gradient divisor
grad_divisor_x = np.ones(I.shape)
grad_divisor_x[1:-1,:,:] = 2
grad_divisor_y = np.ones(I.shape)
grad_divisor_y[:,1:-1,:] = 2
grad_divisor_z = np.ones(I.shape)
grad_divisor_z[:,:,1:-1] = 2

# move
with torch.cuda.device(gpu_number):
    grad_divisor_x_gpu = torch.tensor(grad_divisor_x).float().cuda()
    grad_divisor_y_gpu = torch.tensor(grad_divisor_y).float().cuda()
    grad_divisor_z_gpu = torch.tensor(grad_divisor_z).float().cuda()


# main loop
for it in range(niter):
    #phiinv0,phiinv1,phiinv2 = X0,X1,X2
    phiinv0_gpu = X0_gpu.clone()
    phiinv1_gpu = X1_gpu.clone()
    phiinv2_gpu = X2_gpu.clone()
    ER = 0.0
    for t in range(nt):
        # update phiinv using method of characteristics
        #v0, v1, v2 = vt0[:,:,:,t], vt1[:,:,:,t], vt2[:,:,:,t]
        #X0s, X1s, X2s = X0_gpu-vt0_gpu[t]*dt, X1_gpu-v1_gpu[t]*dt, X2_gpu-v2_gpu[t]*dt
        
        # deal with boundary conditions: subtract identity, use flat boundary conditions, then add identity
        
        # change interpolation function
        #phiinv0 = spi.interpn([x0,x1,x2],phiinv0_gpu-X0_gpu,np.stack([X0_gpu-vt0_gpu[t]*dt,X1_gpu-v1_gpu[t]*dt,X2_gpu-v2_gpu[t]*dt],axis=-1),**interp_args)+(X0_gpu-vt0_gpu[t]*dt)
        #phiinv1 = spi.interpn([x0,x1,x2],phiinv1_gpu-X1_gpu,np.stack([X0_gpu-vt0_gpu[t]*dt,X1_gpu-v1_gpu[t]*dt,X2_gpu-v2_gpu[t]*dt],axis=-1),**interp_args)+(X1_gpu-v1_gpu[t]*dt)
        #phiinv2 = spi.interpn([x0,x1,x2],phiinv2_gpu-X2_gpu,np.stack([X0_gpu-vt0_gpu[t]*dt,X1_gpu-v1_gpu[t]*dt,X2_gpu-v2_gpu[t]*dt],axis=-1),**interp_args)+(X2_gpu-v2_gpu[t]*dt)
        
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-X0_gpu).unsqueeze(0).unsqueeze(0),torch.stack(((X2_gpu-vt2_gpu[t]*dt)/(nx[2]*dx[2]-dx[2])*2-1,(X1_gpu-vt1_gpu[t]*dt)/(nx[1]*dx[1]-dx[1])*2-1,(X0_gpu-vt0_gpu[t]*dt)/(nx[0]*dx[0]-dx[0])*2-1),dim=3).unsqueeze(0))) + (X0_gpu-vt0_gpu[t]*dt)
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-X1_gpu).unsqueeze(0).unsqueeze(0),torch.stack(((X2_gpu-vt2_gpu[t]*dt)/(nx[2]*dx[2]-dx[2])*2-1,(X1_gpu-vt1_gpu[t]*dt)/(nx[1]*dx[1]-dx[1])*2-1,(X0_gpu-vt0_gpu[t]*dt)/(nx[0]*dx[0]-dx[0])*2-1),dim=3).unsqueeze(0))) + (X1_gpu-vt1_gpu[t]*dt)
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-X2_gpu).unsqueeze(0).unsqueeze(0),torch.stack(((X2_gpu-vt2_gpu[t]*dt)/(nx[2]*dx[2]-dx[2])*2-1,(X1_gpu-vt1_gpu[t]*dt)/(nx[1]*dx[1]-dx[1])*2-1,(X0_gpu-vt0_gpu[t]*dt)/(nx[0]*dx[0]-dx[0])*2-1),dim=3).unsqueeze(0))) + (X2_gpu-vt2_gpu[t]*dt)
        
        # deform the image
        #It_gpu[t+1] = spi.interpn([x0,x1,x2],I,np.stack([phiinv0,phiinv1,phiinv2],axis=-1),**interp_args)
        It_gpu[t+1] = torch.squeeze(torch.nn.functional.grid_sample(It_gpu[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu/(nx[2]*dx[2]-dx[2])*2-1,phiinv1_gpu/(nx[1]*dx[1]-dx[1])*2-1,phiinv0_gpu/(nx[0]*dx[0]-dx[0])*2-1),dim=3).unsqueeze(0)))
        
        # calculate the energy of the flow
        #Av0 = np.fft.ifftn(np.fft.fftn(v0)*Ahat).real
        #Av1 = np.fft.ifftn(np.fft.fftn(v1)*Ahat).real
        #Av2 = np.fft.ifftn(np.fft.fftn(v2)*Ahat).real
        #ER += np.sum(v0*Av0 + v1*Av1 + v2*Av2)*0.5*dx[0]*dx[1]*dx[2]*dt
        
        # rfft produces a 2 channel matrix, torch does not support complex number multiplication yet
        ER += torch.sum(vt0_gpu[t]*torch.irfft(torch.rfft(vt0_gpu[t],3,onesided=False)*(1.0/Khat_gpu),3,onesided=False) + vt1_gpu[t]*torch.irfft(torch.rfft(vt1_gpu[t],3,onesided=False)*(1.0/Khat_gpu),3,onesided=False) + vt2_gpu[t]*torch.irfft(torch.rfft(vt2_gpu[t],3,onesided=False)*(1.0/Khat_gpu),3,onesided=False)) * 0.5 / sigmaR**2 * dx[0]*dx[1]*dx[2]*dt
        # this is slower. try vectorized rfft by stacking vt0_gpu
        #ER += torch.sum(torch.rfft(vt0_gpu[t],3,onesided=False)**2,dim=3) + torch.sum(torch.rfft(vt1_gpu[t],3,onesided=False)**2,dim=3) + torch.sum(torch.rfft(vt2_gpu[t],3,onesided=False)**2,dim=3)
    
    # compute loss
    #ER = torch.sum(Ahat_gpu * ER) * 0.5 / sigmaR**2 * dx[0]*dx[1]*dx[2]*dt / (nx[0]*nx[1]*nx[2])
    #err = It_gpu[-1] - J_gpu
    lambda1 = -(It_gpu[-1] - J_gpu)/sigma**2 # may not need to store this
    EM = torch.sum((It_gpu[-1] - J_gpu)**2/(2.0*sigma**2))*dx[0]*dx[1]*dx[2]
    E = ER+EM    
    EMAll.append(EM)        
    ERAll.append(ER)        
    EAll.append(E)
    end_time = time.time()
    print('iter: ' + str(it) + ', E = ' + str(E) + ', ER = ' + str(ER) + ', EM = ' + str(EM) + ', time = ' + str(end_time-start_time) + ' .\n')
    start_time = time.time()
    
    # backwards step
    phiinv0_gpu = X0_gpu.clone()
    phiinv1_gpu = X1_gpu.clone()
    phiinv2_gpu = X2_gpu.clone()
    
    for t in range(nt-1,-1,-1):
        # update phiinv using method of characteristics, note "+" because we are integrating backward
        #v0, v1, v2 = vt0[:,:,:,t], vt1[:,:,:,t], vt2[:,:,:,t]
        #X0s, X1s, X2s = X0+v0*dt, X1+v1*dt, X2+v2*dt
        #phiinv0 = spi.interpn([x0,x1,x2],phiinv0-X0,np.stack([X0s,X1s,X2s],axis=-1),**interp_args)+X0s
        #phiinv1 = spi.interpn([x0,x1,x2],phiinv1-X1,np.stack([X0s,X1s,X2s],axis=-1),**interp_args)+X1s
        #phiinv2 = spi.interpn([x0,x1,x2],phiinv2-X2,np.stack([X0s,X1s,X2s],axis=-1),**interp_args)+X2s
        
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-X0_gpu).unsqueeze(0).unsqueeze(0),torch.stack(((X2_gpu+vt2_gpu[t]*dt)/(nx[2]*dx[2]-dx[2])*2-1,(X1_gpu+vt1_gpu[t]*dt)/(nx[1]*dx[1]-dx[1])*2-1,(X0_gpu+vt0_gpu[t]*dt)/(nx[0]*dx[0]-dx[0])*2-1),dim=3).unsqueeze(0))) + (X0_gpu+vt0_gpu[t]*dt)
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-X1_gpu).unsqueeze(0).unsqueeze(0),torch.stack(((X2_gpu+vt2_gpu[t]*dt)/(nx[2]*dx[2]-dx[2])*2-1,(X1_gpu+vt1_gpu[t]*dt)/(nx[1]*dx[1]-dx[1])*2-1,(X0_gpu+vt0_gpu[t]*dt)/(nx[0]*dx[0]-dx[0])*2-1),dim=3).unsqueeze(0))) + (X1_gpu+vt1_gpu[t]*dt)
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-X2_gpu).unsqueeze(0).unsqueeze(0),torch.stack(((X2_gpu+vt2_gpu[t]*dt)/(nx[2]*dx[2]-dx[2])*2-1,(X1_gpu+vt1_gpu[t]*dt)/(nx[1]*dx[1]-dx[1])*2-1,(X0_gpu+vt0_gpu[t]*dt)/(nx[0]*dx[0]-dx[0])*2-1),dim=3).unsqueeze(0))) + (X2_gpu+vt2_gpu[t]*dt)
        
        
        # find the determinant of Jacobian
        #phiinv0_0,phiinv0_1,phiinv0_2 = np.gradient(phiinv0,dx[0],dx[1],dx[2])
        #phiinv1_0,phiinv1_1,phiinv1_2 = np.gradient(phiinv1,dx[0],dx[1],dx[2])
        #phiinv2_0,phiinv2_1,phiinv2_2 = np.gradient(phiinv2,dx[0],dx[1],dx[2])
        
        phiinv0_0,phiinv0_1,phiinv0_2 = torch_gradient(phiinv0_gpu,dx[0],dx[1],dx[2],grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu)
        phiinv1_0,phiinv1_1,phiinv1_2 = torch_gradient(phiinv1_gpu,dx[0],dx[1],dx[2],grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu)
        phiinv2_0,phiinv2_1,phiinv2_2 = torch_gradient(phiinv2_gpu,dx[0],dx[1],dx[2],grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu)
        detjac = phiinv0_0*(phiinv1_1*phiinv2_2 - phiinv1_2*phiinv2_1)\
            - phiinv0_1*(phiinv1_0*phiinv2_2 - phiinv1_2*phiinv2_0)\
            + phiinv0_2*(phiinv1_0*phiinv2_1 - phiinv1_1*phiinv2_0)
        
        # find lambda_t
        #lambdat = spi.interpn([x0,x1,x2],lambda1,np.stack([phiinv0,phiinv1,phiinv2],axis=-1),**interp_args)*detjac
        lambdat = torch.squeeze(torch.nn.functional.grid_sample(lambda1.unsqueeze(0).unsqueeze(0), torch.stack((phiinv2_gpu/(nx[2]*dx[2]-dx[2])*2-1,phiinv1_gpu/(nx[1]*dx[1]-dx[1])*2-1,phiinv0_gpu/(nx[0]*dx[0]-dx[0])*2-1),dim=3).unsqueeze(0)))*detjac
        
        # get the gradient of the image at this time
        #I_0, I_1, I_2 = np.gradient(It[:,:,:,t],dx[0],dx[1],dx[2])
        #I_0, I_1, I_2 = torch_gradient(It_gpu[t],dx[0],dx[1],dx[2],grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu)
        
        # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
        grad_list = [x*lambdat for x in torch_gradient(It_gpu[t],dx[0],dx[1],dx[2],grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu)]
        #grad_list[1], grad_list[0] = grad_list[0], grad_list[1]
        
        # 4. Compute the gradient and update
        # initialize the gradient with the matching term
        #grad0, grad1, grad2 = lambdat*I_0, lambdat*I_1, lambdat*I_2
        
        # smooth it
        #grad0 = np.fft.ifftn(np.fft.fftn(grad0)*Khat).real
        #grad1 = np.fft.ifftn(np.fft.fftn(grad1)*Khat).real
        #grad2 = np.fft.ifftn(np.fft.fftn(grad2)*Khat).real
        grad_list = [torch.irfft(torch.rfft(x,3,onesided=False)*Khat_gpu,3,onesided=False) for x in grad_list]
        
        # add the regularization term
        #grad0 += v0
        #grad1 += v1
        #grad2 += v2
        grad_list[0] += vt0_gpu[t]/sigmaR**2
        grad_list[1] += vt1_gpu[t]/sigmaR**2
        grad_list[2] += vt2_gpu[t]/sigmaR**2
        
        # update
        #vt0[:,:,:,t] -= epsilon*grad0
        #vt1[:,:,:,t] -= epsilon*grad1
        #vt2[:,:,:,t] -= epsilon*grad2
        vt0_gpu[t] -= epsilon*grad_list[0]
        vt1_gpu[t] -= epsilon*grad_list[1]
        vt2_gpu[t] -= epsilon*grad_list[2]


# save outputs
outvol = It_gpu[-1].to('cpu').numpy()
outimg = nib.AnalyzeImage(outvol,None)
outimg.header['pixdim'][1:4] = [0.04,0.04,0.04]
nib.save(outimg,'/cis/home/leebc/Projects/Mouse_Histology/data/registration/torch_test/Hua121_defatlas.img')
