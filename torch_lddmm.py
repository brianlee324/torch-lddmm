import torch
import numpy as np
import time
import sys
import time
import os
sys.path.insert(0,'/cis/home/leebc/Software/')
import nibabel as nib

class LDDMM:
    def __init__(self,template=None,target=None,outdir='./',gpu_number=0,a=5.0,p=2,niter=100,epsilon=5e-3,sigma=2.0,sigmaR=1.0,nt=5,doaffine=0):
        self.params = {}
        self.params['gpu_number'] = gpu_number
        self.params['a'] = float(a)
        self.params['p'] = float(p)
        self.params['niter'] = niter
        self.params['epsilon'] = float(epsilon)
        self.params['sigma'] = float(sigma)
        self.params['sigmaR'] = float(sigmaR)
        self.params['nt'] = nt
        self.params['template'] = template
        self.params['target'] = target
        self.params['outdir'] = outdir
        self.params['doaffine'] = doaffine
        print('\nCurrent parameters:')
        print('>    a            = ' + str(a) + ' (smoothing kernel, a*(pixel_size))')
        print('>    p            = ' + str(p) + ' (smoothing kernel power, p*2)')
        print('>    niter        = ' + str(niter) + ' (number of iterations)')
        print('>    epsilon      = ' + str(epsilon) + ' (gradient descent step size)')
        print('>    sigma        = ' + str(sigma) + ' (matching term coefficient (1/sigma**2))')
        print('>    sigmaR       = ' + str(sigmaR)+ ' (regularization term coefficient (1/sigmaR**2))')
        print('>    nt           = ' + str(nt) + ' (number of time steps in velocity field)')
        print('>    doaffine     = ' + str(doaffine) + ' (interleave affine registration: 0 = no, 1 = yes)')
        print('>    gpu_number   = ' + str(gpu_number) + ' (index of CUDA_VISIBLE_DEVICES to use)')
        print('>    outdir       = ' + str(outdir) + ' (output directory name)')
        print('\n')
        if template is None:
            print('WARNING: template file name is not set. Use LDDMM.setParams(\'template\',filename).\n')
        else:
            print('>    template     = ' + template + '\n')
        
        if target is None:
            print('WARNING: target file name is not set. Use LDDMM.setParams(\'target\',filename).\n')
        else:
            print('>    target       = ' + target + '\n')
    
    # manual edit parameter
    def setParams(self,parameter_name,parameter_value):
        if parameter_name in self.params:
            self.params[parameter_name] = parameter_value
        else:
            print('Parameter \'' + str(parameter_name) + '\' is not a valid parameter.\n')
        
        return
    
    # image loader
    def loadImage(self, filename):
        fname, fext = os.path.splitext(filename)
        if fext == '.img' or fext == '.hdr':
            img_struct = nib.load(fname + '.img')
            spacing = img_struct.header['pixdim'][1:4]
            size = img_struct.header['dim'][1:4]
            image = np.squeeze(img_struct.get_data().astype(np.float32))
            image = torch.tensor((image - np.mean(image)) / np.std(image)).float().to(device=self.params['cuda'])
            return (image, spacing, size)
        else:
            print('File format not supported.\n')
            return (-1,-1,-1)
    
    # helper function to check parameters before running registration
    def _checkParameters(self):
        flag = 1
        if self.params['gpu_number'] is not None and not isinstance(self.params['gpu_number'], (int, float)):
            flag = -1
            print('ERROR: gpu_number must be None or a number.')
        else:
            if self.params['gpu_number'] is None:
                self.params['cuda'] = 'cpu'
            else:
                self.params['cuda'] = 'cuda:' + str(self.params['gpu_number'])
        
        number_list = ['a','p','niter','epsilon','sigma','sigmaR','nt','doaffine']
        string_list = ['template','target','outdir']
        for i in range(len(number_list)):
            if not isinstance(self.params[number_list[i]], (int, float)):
                flag = -1
                print('ERROR: ' + number_list[i] + ' must be a number.')
        
        for i in range(len(string_list)):
            if not isinstance(self.params[string_list[i]], str):
                flag = -1
                print('ERROR: ' + string_list[i] + ' must be a string.')
        
        return flag
    
    # helper function to load images
    def _load(self, template, target):
        I,Ispacing,Isize = self.loadImage(template)
        J,Jspacing,Jsize = self.loadImage(target)
        #if I.shape[0] != J.shape[0] or I.shape[1] != J.shape[1] or I.shape[2] != J.shape[2]:
        if I.shape != J.shape:
            print('ERROR: the image sizes are not the same.\n')
            return -1
        #elif Ispacing[0] != Jspacing[0] or Ispacing[1] != Jspacing[1] or Ispacing[2] != Jspacing[2]
        elif np.sum(Ispacing==Jspacing) < len(I.shape):
            print('ERROR: the image pixel spacings are not the same.\n')
            return -1
        else:
            self.I = I
            self.J = J
            self.dx = list(Ispacing)
            self.dx = [float(x) for x in self.dx]
            self.nx = I.shape
            return 1
    
    # initialize lddmm kernels
    def initializeKernels(self):
        # make smoothing kernel on CPU
        f0 = np.arange(self.nx[0])/(self.dx[0]*self.nx[0])
        f1 = np.arange(self.nx[1])/(self.dx[1]*self.nx[1])
        f2 = np.arange(self.nx[2])/(self.dx[2]*self.nx[2])
        F0,F1,F2 = np.meshgrid(f0,f1,f2,indexing='ij')
        #a = 3.0*self.dx[0] # a scale in mm
        #p = 2
        self.Ahat = (1.0 - 2.0*(self.params['a']*self.dx[0])**2*((np.cos(2.0*np.pi*self.dx[0]*F0) - 1.0)/self.dx[0]**2 
                                + (np.cos(2.0*np.pi*self.dx[1]*F1) - 1.0)/self.dx[1]**2
                                + (np.cos(2.0*np.pi*self.dx[2]*F2) - 1.0)/self.dx[2]**2))**(2.0*self.params['p'])
        self.Khat = 1.0/self.Ahat
        # only move one kernel for now
        self.Khat = torch.tensor(np.tile(np.reshape(self.Khat,(self.Khat.shape[0],self.Khat.shape[1],self.Khat.shape[2],1)),(1,1,1,2))).float().to(device=self.params['cuda'])
    
    # initialize lddmm variables
    def initializeVariables(self):
        # helper variables
        self.dt = 1.0/self.params['nt']
        # loss values
        self.EMAll = []
        self.ERAll = []
        self.EAll = []
        # image sampling domain
        x0 = np.arange(self.nx[0])*self.dx[0]
        x1 = np.arange(self.nx[1])*self.dx[1]
        x2 = np.arange(self.nx[2])*self.dx[2]
        X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')
        self.X0 = torch.tensor(X0).float().to(device=self.params['cuda'])
        self.X1 = torch.tensor(X1).float().to(device=self.params['cuda'])
        self.X2 = torch.tensor(X2).float().to(device=self.params['cuda'])
        # v and I
        if self.params['gpu_number'] is not None:
            self.vt0 = []
            self.vt1 = []
            self.vt2 = []
            self.It = []
            # NOTE: you cannot use pointers / list multiplication for cuda tensors if you want actual copies
            self.It.append(torch.tensor(self.I[:,:,:]).float().cuda())
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).float().cuda())
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).float().cuda())
                self.vt2.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).float().cuda())
                self.It.append(torch.tensor(self.I[:,:,:]).float().cuda())
        else:
            self.vt0 = [torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).float()]*self.params['nt']
            self.vt1 = [torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).float()]*self.params['nt']
            self.vt2 = [torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).float()]*self.params['nt']
            self.It = [torch.tensor(self.I[:,:,:]).float()]*(self.params['nt']+1)
    
    # helper function for torch_gradient
    def _allocateGradientDivisors(self):
        # allocate gradient divisor for custom torch gradient function
        self.grad_divisor_x = np.ones(self.I.shape)
        self.grad_divisor_x[1:-1,:,:] = 2
        self.grad_divisor_x = torch.tensor(self.grad_divisor_x).float().to(device=self.params['cuda'])
        self.grad_divisor_y = np.ones(self.I.shape)
        self.grad_divisor_y[:,1:-1,:] = 2
        self.grad_divisor_y = torch.tensor(self.grad_divisor_y).float().to(device=self.params['cuda'])
        self.grad_divisor_z = np.ones(self.I.shape)
        self.grad_divisor_z[:,:,1:-1] = 2
        self.grad_divisor_z = torch.tensor(self.grad_divisor_z).float().to(device=self.params['cuda'])
        
    
    # replication-pad, artificial roll, subtract, single-sided difference on boundaries
    def torch_gradient(self,arr, dx, dy, dz, grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu):
        arr = torch.squeeze(torch.nn.functional.pad(arr.unsqueeze(0).unsqueeze(0),(1,1,1,1,1,1),mode='replicate'))
        gradx = torch.cat((arr[1:,:,:],arr[0,:,:].unsqueeze(0)),dim=0) - torch.cat((arr[-1,:,:].unsqueeze(0),arr[:-1,:,:]),dim=0)
        grady = torch.cat((arr[:,1:,:],arr[:,0,:].unsqueeze(1)),dim=1) - torch.cat((arr[:,-1,:].unsqueeze(1),arr[:,:-1,:]),dim=1)
        gradz = torch.cat((arr[:,:,1:],arr[:,:,0].unsqueeze(2)),dim=2) - torch.cat((arr[:,:,-1].unsqueeze(2),arr[:,:,:-1]),dim=2)
        return gradx[1:-1,1:-1,1:-1]/dx/grad_divisor_x_gpu, grady[1:-1,1:-1,1:-1]/dy/grad_divisor_y_gpu, gradz[1:-1,1:-1,1:-1]/dz/grad_divisor_z_gpu
    
    # deform template forward
    def forwardDeformation(self):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        phiinv2_gpu = self.X2.clone()
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X0-self.vt0[t]*self.dt)
            phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X1-self.vt1[t]*self.dt)
            phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X2-self.vt2[t]*self.dt)
            
            # deform the image
            self.It[t+1] = torch.squeeze(torch.nn.functional.grid_sample(self.It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0)))
        
        return self.It,phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # compute regularization energy for time varying velocity field in for loop to conserve memory
    def calculateRegularizationEnergyVt(self):
        ER = 0.0
        for t in range(self.params['nt']):
            # rfft produces a 2 channel matrix, torch does not support complex number multiplication yet
            ER += torch.sum(self.vt0[t]*torch.irfft(torch.rfft(self.vt0[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False) + self.vt1[t]*torch.irfft(torch.rfft(self.vt1[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False) + self.vt2[t]*torch.irfft(torch.rfft(self.vt2[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False)) * 0.5 / self.params['sigmaR']**2 * self.dx[0]*self.dx[1]*self.dx[2]*self.dt
        
        return ER
    
    # compute matching energy
    def calculateMatchingEnergyMSE(self):
        lambda1 = -(self.It[-1] - self.J)/self.params['sigma']**2 # may not need to store this
        EM = torch.sum((self.It[-1] - self.J)**2/(2.0*self.params['sigma']**2))*self.dx[0]*self.dx[1]*self.dx[2]
        return lambda1, EM
    
    # compute gradient per time step for time varying velocity field parameterization
    def calculateGradientVt(self,lambda1,t,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        # update phiinv using method of characteristics, note "+" because we are integrating backward
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X0+self.vt0[t]*self.dt)
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X1+self.vt1[t]*self.dt)
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X2+self.vt2[t]*self.dt)
        
        
        # find the determinant of Jacobian
        phiinv0_0,phiinv0_1,phiinv0_2 = self.torch_gradient(phiinv0_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
        phiinv1_0,phiinv1_1,phiinv1_2 = self.torch_gradient(phiinv1_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
        phiinv2_0,phiinv2_1,phiinv2_2 = self.torch_gradient(phiinv2_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
        detjac = phiinv0_0*(phiinv1_1*phiinv2_2 - phiinv1_2*phiinv2_1)\
            - phiinv0_1*(phiinv1_0*phiinv2_2 - phiinv1_2*phiinv2_0)\
            + phiinv0_2*(phiinv1_0*phiinv2_1 - phiinv1_1*phiinv2_0)
        
        # find lambda_t
        lambdat = torch.squeeze(torch.nn.functional.grid_sample(lambda1.unsqueeze(0).unsqueeze(0), torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0)))*detjac
        
        # get the gradient of the image at this time
        
        # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
        grad_list = [x*lambdat for x in self.torch_gradient(self.It[t],self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)]
        
        # smooth it
        grad_list = [torch.irfft(torch.rfft(x,3,onesided=False)*self.Khat,3,onesided=False) for x in grad_list]
        
        # add the regularization term
        grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
        grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
        grad_list[2] += self.vt2[t]/self.params['sigmaR']**2
        return grad_list
    
    # update gradient
    def updateGradientVt(self,t,grad_list):
        self.vt0[t] -= self.params['epsilon']*grad_list[0]
        self.vt1[t] -= self.params['epsilon']*grad_list[1]
        self.vt2[t] -= self.params['epsilon']*grad_list[2]
    
    # convenience function for calculating and updating gradients of Vt
    def calculateAndUpdateGradientsVt(self, lambda1):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        phiinv2_gpu = self.X2.clone()
        for t in range(self.params['nt']-1,-1,-1):
            grad_list = self.calculateGradientVt(lambda1,t,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
            self.updateGradientVt(t,grad_list)
    
    # main loop
    def registration(self):
        for it in range(self.params['niter']):
            # deform images forward
            _,_,_,_ = self.forwardDeformation()
            # get regularization energy
            ER = self.calculateRegularizationEnergyVt()
            # get matching energy and store the derivative of the matching term for gradient calculation
            lambda1,EM = self.calculateMatchingEnergyMSE()
            # save variables
            E = ER+EM
            self.EMAll.append(EM)        
            self.ERAll.append(ER)        
            self.EAll.append(E)
            # print function
            end_time = time.time()
            if it > 0:
                print('iter: ' + str(it) + ', E = ' + str(E.item()) + ', ER = ' + str(ER.item()) + ', EM = ' + str(EM.item()) + ', time = ' + str(end_time-start_time) + '.')
            else:
                print('iter: ' + str(it) + ', E = ' + str(E.item()) + ', ER = ' + str(ER.item()) + ', EM = ' + str(EM.item()) + '.')
            
            start_time = time.time()
            # calculate and update gradients
            self.calculateAndUpdateGradientsVt(lambda1)
    
    
    # main loop
    def registration_old(self):
        for it in range(self.params['niter']):
            # should these be global variables? does it matter?
            phiinv0_gpu = self.X0.clone()
            phiinv1_gpu = self.X1.clone()
            phiinv2_gpu = self.X2.clone()
            ER = 0.0
            for t in range(self.params['nt']):
                # update phiinv using method of characteristics
                phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X0-self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X1-self.vt1[t]*self.dt)
                phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X2-self.vt2[t]*self.dt)
                
                # deform the image
                self.It[t+1] = torch.squeeze(torch.nn.functional.grid_sample(self.It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0)))
                
                # rfft produces a 2 channel matrix, torch does not support complex number multiplication yet
                ER += torch.sum(self.vt0[t]*torch.irfft(torch.rfft(self.vt0[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False) + self.vt1[t]*torch.irfft(torch.rfft(self.vt1[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False) + self.vt2[t]*torch.irfft(torch.rfft(self.vt2[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False)) * 0.5 / self.params['sigmaR']**2 * self.dx[0]*self.dx[1]*self.dx[2]*self.dt
                # this is slower. try vectorized rfft by stacking self.vt0
                #ER += torch.sum(torch.rfft(self.vt0[t],3,onesided=False)**2,dim=3) + torch.sum(torch.rfft(self.vt1[t],3,onesided=False)**2,dim=3) + torch.sum(torch.rfft(self.vt2[t],3,onesided=False)**2,dim=3)
            
            # compute loss
            # this is faster but takes too much memory
            #ER = torch.sum( torch.stack(self.vt0,dim=0) * torch.irfft(torch.rfft( torch.stack(self.vt0,dim=0) ,3,onesided=False) * (1.0/self.Khat).unsqueeze(0).expand(5,-1,-1,-1,-1), 3,onesided=False) + torch.stack(self.vt1,dim=0) * torch.irfft(torch.rfft( torch.stack(self.vt1,dim=0) ,3,onesided=False) * (1.0/self.Khat).unsqueeze(0).expand(5,-1,-1,-1,-1), 3,onesided=False) + torch.stack(self.vt2,dim=0) * torch.irfft(torch.rfft( torch.stack(self.vt2,dim=0) ,3,onesided=False) * (1.0/self.Khat).unsqueeze(0).expand(5,-1,-1,-1,-1), 3,onesided=False) ) * 0.5 / self.params['sigmaR']**2 * self.dx[0]*self.dx[1]*self.dx[2]*self.dt
            
            #ER = torch.sum(Ahat_gpu * ER) * 0.5 / sigmaR**2 * dx[0]*dx[1]*dx[2]*dt / (nx[0]*nx[1]*nx[2])
            #err = self.It[-1] - self.J
            lambda1 = -(self.It[-1] - self.J)/self.params['sigma']**2 # may not need to store this
            EM = torch.sum((self.It[-1] - self.J)**2/(2.0*self.params['sigma']**2))*self.dx[0]*self.dx[1]*self.dx[2]
            E = ER+EM
            self.EMAll.append(EM)        
            self.ERAll.append(ER)        
            self.EAll.append(E)
            end_time = time.time()
            if it > 0:
                print('iter: ' + str(it) + ', E = ' + str(E.item()) + ', ER = ' + str(ER.item()) + ', EM = ' + str(EM.item()) + ', time = ' + str(end_time-start_time) + '.')
            else:
                print('iter: ' + str(it) + ', E = ' + str(E.item()) + ', ER = ' + str(ER.item()) + ', EM = ' + str(EM.item()) + '.')
            
            start_time = time.time()
            
            # backwards step
            phiinv0_gpu = self.X0.clone()
            phiinv1_gpu = self.X1.clone()
            phiinv2_gpu = self.X2.clone()
            
            for t in range(self.params['nt']-1,-1,-1):
                # update phiinv using method of characteristics, note "+" because we are integrating backward
                phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X0+self.vt0[t]*self.dt)
                phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X1+self.vt1[t]*self.dt)
                phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (self.X2+self.vt2[t]*self.dt)
                
                
                # find the determinant of Jacobian
                phiinv0_0,phiinv0_1,phiinv0_2 = self.torch_gradient(phiinv0_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
                phiinv1_0,phiinv1_1,phiinv1_2 = self.torch_gradient(phiinv1_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
                phiinv2_0,phiinv2_1,phiinv2_2 = self.torch_gradient(phiinv2_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
                detjac = phiinv0_0*(phiinv1_1*phiinv2_2 - phiinv1_2*phiinv2_1)\
                    - phiinv0_1*(phiinv1_0*phiinv2_2 - phiinv1_2*phiinv2_0)\
                    + phiinv0_2*(phiinv1_0*phiinv2_1 - phiinv1_1*phiinv2_0)
                
                # find lambda_t
                lambdat = torch.squeeze(torch.nn.functional.grid_sample(lambda1.unsqueeze(0).unsqueeze(0), torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0)))*detjac
                
                # get the gradient of the image at this time
                
                # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
                grad_list = [x*lambdat for x in self.torch_gradient(self.It[t],self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)]
                
                # smooth it
                grad_list = [torch.irfft(torch.rfft(x,3,onesided=False)*self.Khat,3,onesided=False) for x in grad_list]
                
                # add the regularization term
                grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
                grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
                grad_list[2] += self.vt2[t]/self.params['sigmaR']**2
                
                # update
                self.vt0[t] -= self.params['epsilon']*grad_list[0]
                self.vt1[t] -= self.params['epsilon']*grad_list[1]
                self.vt2[t] -= self.params['epsilon']*grad_list[2]
    
    # save files to disk
    def saveOutputs(self, save_template=False):
        if save_template:
            outimg = nib.AnalyzeImage(self.It[-1].to('cpu').numpy(),None)
            outimg.header['pixdim'][1:4] = self.dx
            nib.save(outimg,self.params['outdir'] + 'deformed_template.img')
        
    # convenience function
    def run(self):
        # check parameters
        flag = self._checkParameters()
        if flag==-1:
            print('ERROR: parameters did not check out.')
            return
        
        # load images
        flag = self._load(self.params['template'],self.params['target'])
        if flag==-1:
            print('ERROR: images did not load.')
            return
        
        # initialize initialize
        self.initializeVariables()
        # initialize kernels
        self.initializeKernels()
        # initialize stuff for gradient function
        self._allocateGradientDivisors()
        # main loop
        self.registration()
        # save outputs
        self.saveOutputs(save_template=True)
        
    