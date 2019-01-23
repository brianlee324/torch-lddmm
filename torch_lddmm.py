import torch
import numpy as np
import scipy.linalg
import time
import sys
import time
import os
sys.path.insert(0,'/cis/home/leebc/Software/')
import nibabel as nib

class LDDMM:
    def __init__(self,template=None,target=None,outdir='./',gpu_number=0,a=5.0,p=2,niter=100,epsilon=5e-3,epsilonL=1.0e-7,epsilonT=2.0e-5,sigma=2.0,sigmaR=1.0,nt=5,doaffine=0,checkaffinestep=1,optimizer='gd',maxclimbcount=3,savebestv=False,minenergychange = 0.000001,minbeta=1e-6,dtype='float',im_norm_ms=0):
        self.params = {}
        self.params['gpu_number'] = gpu_number
        self.params['a'] = float(a)
        self.params['p'] = float(p)
        self.params['niter'] = niter
        self.params['epsilon'] = float(epsilon)
        self.params['epsilonL'] = float(epsilonL)
        self.params['epsilonT'] = float(epsilonT)
        if isinstance(sigma,(int,float)):
            self.params['sigma'] = float(sigma)
        else:
            self.params['sigma'] = [float(x) for x in sigma]
        self.params['sigmaR'] = float(sigmaR)
        self.params['nt'] = nt
        self.params['template'] = template
        self.params['target'] = target
        self.params['outdir'] = outdir
        self.params['doaffine'] = doaffine
        self.params['checkaffinestep'] = checkaffinestep
        self.params['optimizer'] = optimizer
        self.params['maxclimbcount'] = maxclimbcount
        self.params['savebestv'] = savebestv
        self.params['minbeta'] = minbeta
        self.params['minenergychange'] = minenergychange
        self.params['im_norm_ms'] = im_norm_ms
        dtype_dict = {}
        dtype_dict['float'] = 'torch.FloatTensor'
        dtype_dict['double'] = 'torch.DoubleTensor'
        self.params['dtype'] = dtype_dict[dtype]
        optimizer_dict = {}
        optimizer_dict['gd'] = 'gradient descent'
        optimizer_dict['gdr'] = 'gradient descent with reducing epsilon'
        optimizer_dict['gdw'] = 'gradient descent with delayed reducing epsilon'
        print('\nCurrent parameters:')
        print('>    a               = ' + str(a) + ' (smoothing kernel, a*(pixel_size))')
        print('>    p               = ' + str(p) + ' (smoothing kernel power, p*2)')
        print('>    niter           = ' + str(niter) + ' (number of iterations)')
        print('>    epsilon         = ' + str(epsilon) + ' (gradient descent step size)')
        print('>    epsilonL        = ' + str(epsilonL) + ' (gradient descent step size, affine)')
        print('>    epsilonT        = ' + str(epsilonT) + ' (gradient descent step size, translation)')
        print('>    minbeta         = ' + str(minbeta) + ' (smallest multiple of epsilon)')
        print('>    sigma           = ' + str(sigma) + ' (matching term coefficient (1/sigma**2))')
        print('>    sigmaR          = ' + str(sigmaR)+ ' (regularization term coefficient (1/sigmaR**2))')
        print('>    nt              = ' + str(nt) + ' (number of time steps in velocity field)')
        print('>    doaffine        = ' + str(doaffine) + ' (interleave affine registration: 0 = no, 1 = yes)')
        print('>    checkaffinestep = ' + str(checkaffinestep) + ' (evaluate affine matching energy: 0 = no, 1 = yes)')
        print('>    im_norm_ms      = ' + str(im_norm_ms) + ' (normalize image by mean and std: 0 = no, 1 = yes)')
        print('>    gpu_number      = ' + str(gpu_number) + ' (index of CUDA_VISIBLE_DEVICES to use)')
        print('>    dtype           = ' + str(dtype) + ' (bit depth, \'float\' or \'double\')')
        print('>    outdir          = ' + str(outdir) + ' (output directory name)')
        if optimizer in optimizer_dict:
            print('>    optimizer       = ' + str(optimizer_dict[optimizer]) + ' (optimizer type)')
        else:
            print('WARNING: optimizer \'' + str(optimizer) + '\' not recognized. Setting to basic gradient descent with reducing step size.')
            self.params['optimizer'] = 'gdr'
        
        print('\n')
        if template is None:
            print('WARNING: template file name is not set. Use LDDMM.setParams(\'template\',filename).\n')
        else:
            print('>    template        = ' + str(template) + '\n')
        
        if target is None:
            print('WARNING: target file name is not set. Use LDDMM.setParams(\'target\',filename).\n')
        else:
            print('>    target          = ' + str(target) + '\n')
    
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
            if self.params['im_norm_ms'] == 1:
                if np.std(image) != 0:
                    image = torch.tensor((image - np.mean(image)) / np.std(image)).type(self.params['dtype']).to(device=self.params['cuda'])
                else:
                    image = torch.tensor((image - np.mean(image)) ).type(self.params['dtype']).to(device=self.params['cuda'])
                    print('WARNING: stdev of image is zero, not rescaling.')
            else:
                image = torch.tensor(image).type(self.params['dtype']).to(device=self.params['cuda'])
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
        
        number_list = ['a','p','niter','epsilon','sigmaR','nt','doaffine','epsilonL','epsilonT','im_norm_ms']
        string_list = ['outdir','optimizer']
        stringorlist_list = ['template','target']
        numberorlist_list = ['sigma']
        for i in range(len(number_list)):
            if not isinstance(self.params[number_list[i]], (int, float)):
                flag = -1
                print('ERROR: ' + number_list[i] + ' must be a number.')
        
        for i in range(len(string_list)):
            if not isinstance(self.params[string_list[i]], str):
                flag = -1
                print('ERROR: ' + string_list[i] + ' must be a string.')
        
        for i in range(len(stringorlist_list)):
            if not isinstance(self.params[stringorlist_list[i]], str) and not isinstance(self.params[stringorlist_list[i]], list):
                flag = -1
                print('ERROR: ' + stringorlist_list[i] + ' must be a string or a list of strings.')
            elif isinstance(self.params[stringorlist_list[i]], str):
                self.params[stringorlist_list[i]] = [self.params[stringorlist_list[i]]]
        
        for i in range(len(numberorlist_list)):
            if not isinstance(self.params[numberorlist_list[i]], (int,float)) and not isinstance(self.params[numberorlist_list[i]], list):
                flag = -1
                print('ERROR: ' + numberorlist_list[i] + ' must be a number or a list of numbers.')
            elif isinstance(self.params[numberorlist_list[i]], (int,float)):
                self.params[numberorlist_list[i]] = [self.params[numberorlist_list[i]]]
        
        # check channel length
        channel_check_list = ['sigma','template','target']
        channels = [len(self.params[x]) for x in channel_check_list]
        channel_set = list(set(channels))
        if len(channel_set) > 2 or (len(channel_set) == 2 and 1 not in channel_set):
            print('ERROR: number of channels is not the same between sigma, template, and target.')
            flag = -1
        elif (len(channel_set) == 2 and 1 in channel_set):
            channel_set.remove(1)
            for i in range(len(channel_check_list)):
                if channels[i] == 1:
                    self.params[channel_check_list[i]] = self.params[channel_check_list[i]]*channel_set[0]
            
            print('WARNING: one or more of sigma, template, and target has length 1 while another does not.')
        
        # optimizer flags
        if self.params['optimizer'] == 'gdw':
            self.params['savebestv'] = True
        
        return flag
    
    # helper function to load images
    def _load(self, template, target):
        if isinstance(template, str):
            I = [None]
            Ispacing = [None]
            Isize = [None]
            I[0],Ispacing[0],Isize[0] = self.loadImage(template)
        elif isinstance(template, list):
            I = [None]*len(template)
            Ispacing = [None]*len(template)
            Isize = [None]*len(template)
            for i in range(len(template)):
                I[i],Ispacing[i],Isize[i] = self.loadImage(template[i])
        
        if isinstance(target, str):
            J = [None]
            Jspacing = [None]
            Jsize = [None]
            J[0],Jspacing[0],Jsize[0] = self.loadImage(target)
        elif isinstance(target, list):
            J = [None]*len(target)
            Jspacing = [None]*len(target)
            Jsize = [None]*len(target)
            for i in range(len(target)):
                J[i],Jspacing[i],Jsize[i] = self.loadImage(target[i])
        
        if len(J) != len(I):
            print('ERROR: images must have the same number of channels.')
            return -1
            
        #if I.shape[0] != J.shape[0] or I.shape[1] != J.shape[1] or I.shape[2] != J.shape[2]:
        #if I.shape != J.shape:
        if not all([x.shape == I[0].shape for x in I+J]):
            print('ERROR: the image sizes are not the same.\n')
            return -1
        #elif Ispacing[0] != Jspacing[0] or Ispacing[1] != Jspacing[1] or Ispacing[2] != Jspacing[2]
        #elif np.sum(Ispacing==Jspacing) < len(I.shape):
        elif not all([list(x == Ispacing[0]) for x in Ispacing+Jspacing]):
            print('ERROR: the image pixel spacings are not the same.\n')
            return -1
        else:
            self.I = I
            self.J = J
            self.dx = list(Ispacing[0])
            self.dx = [float(x) for x in self.dx]
            self.nx = I[0].shape
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
        # TODO: try broadcasting this instead
        self.Khat = torch.tensor(np.tile(np.reshape(self.Khat,(self.Khat.shape[0],self.Khat.shape[1],self.Khat.shape[2],1)),(1,1,1,2))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # optimization multipliers (putting this in here because I want to reset this if I change the smoothing kernel)
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.GDBetaAffineR = float(1.0)
        self.GDBetaAffineT = float(1.0)
        self.climbcount = 0
        if self.params['savebestv']:
            self.best = {}
    
    # initialize lddmm kernels
    def initializeKernels2d(self):
        # make smoothing kernel on CPU
        f0 = np.arange(self.nx[0])/(self.dx[0]*self.nx[0])
        f1 = np.arange(self.nx[1])/(self.dx[1]*self.nx[1])
        F0,F1 = np.meshgrid(f0,f1,indexing='ij')
        #a = 3.0*self.dx[0] # a scale in mm
        #p = 2
        self.Ahat = (1.0 - 2.0*(self.params['a']*self.dx[0])**2*((np.cos(2.0*np.pi*self.dx[0]*F0) - 1.0)/self.dx[0]**2 
                                + (np.cos(2.0*np.pi*self.dx[1]*F1) - 1.0)/self.dx[1]**2))**(2.0*self.params['p'])
        self.Khat = 1.0/self.Ahat
        # only move one kernel for now
        # TODO: try broadcasting this instead
        self.Khat = torch.tensor(np.tile(np.reshape(self.Khat,(self.Khat.shape[0],self.Khat.shape[1],1)),(1,1,2))).type(self.params['dtype']).to(device=self.params['cuda'])
        
        # optimization multipliers (putting this in here because I want to reset this if I change the smoothing kernel)
        self.GDBeta = torch.tensor(1.0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.GDBetaAffineR = float(1.0)
        self.GDBetaAffineT = float(1.0)
        self.climbcount = 0
        if self.params['savebestv']:
            self.best = {}
    
    # initialize lddmm variables
    def initializeVariables(self):
        # TODO: handle 2D and 3D versions
        # helper variables
        self.dt = 1.0/self.params['nt']
        # loss values
        self.EMAll = []
        self.ERAll = []
        self.EAll = []
        if self.params['checkaffinestep'] == 1:
            self.EMAffineR = []
            self.EMAffineT = []
            self.EMDiffeo = []
        
        # image sampling domain
        x0 = np.arange(self.nx[0])*self.dx[0]
        x1 = np.arange(self.nx[1])*self.dx[1]
        x2 = np.arange(self.nx[2])*self.dx[2]
        X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')
        self.X0 = torch.tensor(X0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.X1 = torch.tensor(X1).type(self.params['dtype']).to(device=self.params['cuda'])
        self.X2 = torch.tensor(X2).type(self.params['dtype']).to(device=self.params['cuda'])
        # v and I
        if self.params['gpu_number'] is not None:
            self.vt0 = []
            self.vt1 = []
            self.vt2 = []
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).cuda())
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).cuda())
                self.vt2.append(torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype']).cuda())
            
            self.It = [[],[]]
            for ii in range(len(self.I)):
                # NOTE: you cannot use pointers / list multiplication for cuda tensors if you want actual copies
                #self.It.append(torch.tensor(self.I[:,:,:]).type(self.params['dtype']).cuda())
                self.It[ii].append(torch.tensor(self.I[ii][:,:,:]).type(self.params['dtype']).cuda())
                for i in range(self.params['nt']):
                    self.It[ii].append(torch.tensor(self.I[ii][:,:,:]).type(self.params['dtype']).cuda())
        else:
            self.vt0 = [torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype'])]*self.params['nt']
            self.vt1 = [torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype'])]*self.params['nt']
            self.vt2 = [torch.tensor(np.zeros((self.nx[0],self.nx[1],self.nx[2]))).type(self.params['dtype'])]*self.params['nt']
            self.It = [[None]]*len(self.I)
            for i in range(len(self.I)):
                self.It[i] = [torch.tensor(self.I[i][:,:,:]).type(self.params['dtype'])]*(self.params['nt']+1)
        
        # affine parameters
        self.affineA = torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.lastaffineA = torch.tensor(np.eye(4)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.gradA = torch.tensor(np.zeros((4,4))).type(self.params['dtype']).to(device=self.params['cuda'])
        
    # initialize lddmm variables
    def initializeVariables2d(self):
        # TODO: handle 2D and 3D versions
        # helper variables
        self.dt = 1.0/self.params['nt']
        # loss values
        self.EMAll = []
        self.ERAll = []
        self.EAll = []
        if self.params['checkaffinestep'] == 1:
            self.EMAffineR = []
            self.EMAffineT = []
            self.EMDiffeo = []
        
        # image sampling domain
        x0 = np.arange(self.nx[0])*self.dx[0]
        x1 = np.arange(self.nx[1])*self.dx[1]
        X0,X1 = np.meshgrid(x0,x1,indexing='ij')
        self.X0 = torch.tensor(X0).type(self.params['dtype']).to(device=self.params['cuda'])
        self.X1 = torch.tensor(X1).type(self.params['dtype']).to(device=self.params['cuda'])
        # v and I
        if self.params['gpu_number'] is not None:
            self.vt0 = []
            self.vt1 = []
            self.detjac = []
            self.It = [[],[]]
            for ii in range(len(self.I)):
                self.It[ii].append(torch.tensor(self.I[ii][:,:]).type(self.params['dtype']).cuda())
                for i in range(self.params['nt']):
                    self.It[ii].append(torch.tensor(self.I[ii][:,:]).type(self.params['dtype']).cuda())
            
            for i in range(self.params['nt']):
                self.vt0.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
                self.vt1.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
                self.detjac.append(torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype']).cuda())
        else:
            self.vt0 = [torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype'])]*self.params['nt']
            self.vt1 = [torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype'])]*self.params['nt']
            self.detjac = [torch.tensor(np.zeros((self.nx[0],self.nx[1]))).type(self.params['dtype'])]*self.params['nt']
            self.It = [[None]]*len(self.I)
            for i in range(len(self.I)):
                self.It[i] = [torch.tensor(self.I[i][:,:]).type(self.params['dtype'])]*(self.params['nt']+1)
        
        # affine parameters
        self.affineA = torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.lastaffineA = torch.tensor(np.eye(3)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.gradA = torch.tensor(np.zeros((3,3))).type(self.params['dtype']).to(device=self.params['cuda'])
    
    # helper function for torch_gradient
    def _allocateGradientDivisors(self):
        if self.J[0].dim() == 3:
            # allocate gradient divisor for custom torch gradient function
            self.grad_divisor_x = np.ones(self.I[0].shape)
            self.grad_divisor_x[1:-1,:,:] = 2
            self.grad_divisor_x = torch.tensor(self.grad_divisor_x).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_y = np.ones(self.I[0].shape)
            self.grad_divisor_y[:,1:-1,:] = 2
            self.grad_divisor_y = torch.tensor(self.grad_divisor_y).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_z = np.ones(self.I[0].shape)
            self.grad_divisor_z[:,:,1:-1] = 2
            self.grad_divisor_z = torch.tensor(self.grad_divisor_z).type(self.params['dtype']).to(device=self.params['cuda'])
        else:
            # allocate gradient divisor for custom torch gradient function
            self.grad_divisor_x = np.ones(self.I[0].shape)
            self.grad_divisor_x[1:-1,:] = 2
            self.grad_divisor_x = torch.tensor(self.grad_divisor_x).type(self.params['dtype']).to(device=self.params['cuda'])
            self.grad_divisor_y = np.ones(self.I[0].shape)
            self.grad_divisor_y[:,1:-1] = 2
            self.grad_divisor_y = torch.tensor(self.grad_divisor_y).type(self.params['dtype']).to(device=self.params['cuda'])
    
    # replication-pad, artificial roll, subtract, single-sided difference on boundaries
    def torch_gradient(self,arr, dx, dy, dz, grad_divisor_x_gpu,grad_divisor_y_gpu,grad_divisor_z_gpu):
        arr = torch.squeeze(torch.nn.functional.pad(arr.unsqueeze(0).unsqueeze(0),(1,1,1,1,1,1),mode='replicate'))
        gradx = torch.cat((arr[1:,:,:],arr[0,:,:].unsqueeze(0)),dim=0) - torch.cat((arr[-1,:,:].unsqueeze(0),arr[:-1,:,:]),dim=0)
        grady = torch.cat((arr[:,1:,:],arr[:,0,:].unsqueeze(1)),dim=1) - torch.cat((arr[:,-1,:].unsqueeze(1),arr[:,:-1,:]),dim=1)
        gradz = torch.cat((arr[:,:,1:],arr[:,:,0].unsqueeze(2)),dim=2) - torch.cat((arr[:,:,-1].unsqueeze(2),arr[:,:,:-1]),dim=2)
        return gradx[1:-1,1:-1,1:-1]/dx/grad_divisor_x_gpu, grady[1:-1,1:-1,1:-1]/dy/grad_divisor_y_gpu, gradz[1:-1,1:-1,1:-1]/dz/grad_divisor_z_gpu
    
    # 2D replication-pad, artificial roll, subtract, single-sided difference on boundaries
    def torch_gradient2d(self,arr, dx, dy, grad_divisor_x_gpu,grad_divisor_y_gpu):
        arr = torch.squeeze(torch.nn.functional.pad(arr.unsqueeze(0).unsqueeze(0),(1,1,1,1),mode='replicate'))
        gradx = torch.cat((arr[1:,:],arr[0,:].unsqueeze(0)),dim=0) - torch.cat((arr[-1,:].unsqueeze(0),arr[:-1,:]),dim=0)
        grady = torch.cat((arr[:,1:],arr[:,0].unsqueeze(1)),dim=1) - torch.cat((arr[:,-1].unsqueeze(1),arr[:,:-1]),dim=1)
        return gradx[1:-1,1:-1]/dx/grad_divisor_x_gpu, grady[1:-1,1:-1]/dy/grad_divisor_y_gpu
    
    # deform template forward
    def forwardDeformation(self):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        phiinv2_gpu = self.X2.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
            phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)
            phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2-self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (self.X2-self.vt2[t]*self.dt)
            
            if t == self.params['nt']-1 and self.params['doaffine'] == 1:
                if self.params['checkaffinestep'] == 1:
                    # new diffeo with old affine
                    phiinv0_temp,phiinv1_temp,phiinv2_temp = self.forwardDeformationAffineVectorized(self.lastaffineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    I = [None]*len(self.I)
                    for i in range(len(self.I)):
                        I[i] = torch.squeeze(torch.nn.functional.grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_temp/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border'))
                    
                    self.EMDiffeo.append( self.calculateMatchingEnergyMSEOnly(I) )
                    # new diffeo with new L and old T
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineR(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    phiinv0_temp,phiinv1_temp,phiinv2_temp = self.forwardDeformationAffineT(self.lastaffineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    I = [None]*len(self.I)
                    for i in range(len(self.I)):
                        I[i] = torch.squeeze(torch.nn.functional.grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_temp/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border'))
                    
                    self.EMAffineR.append( self.calculateMatchingEnergyMSEOnly(I) )
                    # new everything
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineT(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                else:
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineVectorized(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
            
            # deform the image
            for i in range(len(self.I)):
                self.It[i][t+1] = torch.squeeze(torch.nn.functional.grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border'))
        
        return self.It,phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # deform template forward
    def forwardDeformation2d(self):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        # TODO: evaluate memory vs speed for precomputing Xs, Ys, Zs
        for t in range(self.params['nt']):
            # update phiinv using method of characteristics
            phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0-self.vt0[t]*self.dt)
            phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1-self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0-self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1-self.vt1[t]*self.dt)            
            '''
            if t == self.params['nt']-1 and self.params['doaffine'] == 1:
                if self.params['checkaffinestep'] == 1:
                    # new diffeo with old affine
                    phiinv0_temp,phiinv1_temp,phiinv2_temp = self.forwardDeformationAffineVectorized(self.lastaffineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    I = torch.squeeze(torch.nn.functional.grid_sample(self.It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_temp/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border'))
                    self.EMDiffeo.append( self.calculateMatchingEnergyMSEOnly(I) )
                    # new diffeo with new L and old T
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineR(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    phiinv0_temp,phiinv1_temp,phiinv2_temp = self.forwardDeformationAffineT(self.lastaffineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                    I = torch.squeeze(torch.nn.functional.grid_sample(self.It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_temp/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_temp/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_temp/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border'))
                    self.EMAffineR.append( self.calculateMatchingEnergyMSEOnly(I) )
                    # new everything
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineT(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
                else:
                    phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.forwardDeformationAffineVectorized(self.affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
            '''
            # deform the image
            for i in range(len(self.I)):
                self.It[i][t+1] = torch.squeeze(torch.nn.functional.grid_sample(self.It[i][0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=2).unsqueeze(0),padding_mode='border'))
        
        return self.It,phiinv0_gpu, phiinv1_gpu
    
    # deform template forward using affine transform
    # TODO: can this be combined with forwardDeformation() for speed? then would I remove det(A) from the gradient?
    # TODO: this could be vectorized by stacking X0, X1, X2
    def forwardDeformationAffine(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        affineB = torch.inverse(affineA)
        
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3]
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3]
        #Zs = affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3]
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3])
        '''
        #Xs = affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3]
        #Ys = affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3]
        #Zs = affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3]
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3])
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3])
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3])/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(affineB[1,0]*self.X1 + affineB[1,1]*self.X0 + affineB[1,2]*self.X2 + affineB[1,3])/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(affineB[0,0]*self.X1 + affineB[0,1]*self.X0 + affineB[0,2]*self.X2 + affineB[0,3])/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0))) + (affineB[2,0]*self.X1 + affineB[2,1]*self.X0 + affineB[2,2]*self.X2 + affineB[2,3])
        '''
        # deform the last time point in the image list
        # actually, just compute phiinv0_gpu, phiinv1_gpu, etc, before the final time step's image deformation
        #self.It[-1] = torch.squeeze(torch.nn.functional.grid_sample(self.It[0].unsqueeze(0).unsqueeze(0),torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0)))
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # deform template forward using affine transform vectorized
    def forwardDeformationAffineVectorized(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        affineB = torch.inverse(affineA)
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2 + affineB[0,3]
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2 + affineB[1,3]
        #Zs = affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2 + affineB[2,3]
        s = torch.mm(affineB[0:3,0:3],torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,)),torch.reshape(self.X2,(-1,))), dim=0)) + torch.reshape(affineB[0:3,3],(3,1)).expand(-1,self.X0.numel())
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[2,:],(self.X2.shape)))
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # deform template forward using affine without translation
    def forwardDeformationAffineR(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        affineB = torch.inverse(affineA)
        #Xs = affineB[0,0]*self.X0 + affineB[0,1]*self.X1 + affineB[0,2]*self.X2
        #Ys = affineB[1,0]*self.X0 + affineB[1,1]*self.X1 + affineB[1,2]*self.X2
        #Zs = affineB[2,0]*self.X0 + affineB[2,1]*self.X1 + affineB[2,2]*self.X2
        s = torch.mm(affineB[0:3,0:3],torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,)),torch.reshape(self.X2,(-1,))), dim=0))
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[2,:],(self.X2.shape)))
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # deform template forward using affine translation
    def forwardDeformationAffineT(self,affineA,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        affineB = torch.inverse(affineA)
        s = torch.stack( (torch.reshape(self.X0,(-1,)),torch.reshape(self.X1,(-1,)),torch.reshape(self.X2,(-1,))), dim=0) + torch.reshape(affineB[0:3,3],(3,1)).expand(-1,self.X0.numel())
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[0,:],(self.X0.shape)))
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[1,:],(self.X1.shape)))
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((torch.reshape(s[2,:],(self.X2.shape)))/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(torch.reshape(s[1,:],(self.X1.shape)))/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(torch.reshape(s[0,:],(self.X0.shape)))/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (torch.reshape(s[2,:],(self.X2.shape)))
        return phiinv0_gpu, phiinv1_gpu, phiinv2_gpu
    
    # compute regularization energy for time varying velocity field in for loop to conserve memory
    def calculateRegularizationEnergyVt(self):
        ER = 0.0
        for t in range(self.params['nt']):
            # rfft produces a 2 channel matrix, torch does not support complex number multiplication yet
            ER += torch.sum(self.vt0[t]*torch.irfft(torch.rfft(self.vt0[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False) + self.vt1[t]*torch.irfft(torch.rfft(self.vt1[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False) + self.vt2[t]*torch.irfft(torch.rfft(self.vt2[t],3,onesided=False)*(1.0/self.Khat),3,onesided=False)) * 0.5 / self.params['sigmaR']**2 * self.dx[0]*self.dx[1]*self.dx[2]*self.dt
        
        return ER
    
    # compute regularization energy for time varying velocity field in for loop to conserve memory
    def calculateRegularizationEnergyVt2d(self):
        ER = 0.0
        for t in range(self.params['nt']):
            # rfft produces a 2 channel matrix, torch does not support complex number multiplication yet
            ER += torch.sum(self.vt0[t]*torch.irfft(torch.rfft(self.vt0[t],2,onesided=False)*(1.0/self.Khat),2,onesided=False) + self.vt1[t]*torch.irfft(torch.rfft(self.vt1[t],2,onesided=False)*(1.0/self.Khat),2,onesided=False)) * 0.5 / self.params['sigmaR']**2 * self.dx[0]*self.dx[1]*self.dt
        
        return ER
    
    # compute matching energy
    def calculateMatchingEnergyMSE(self):
        lambda1 = [None]*len(self.I)
        EM = 0
        for i in range(len(self.I)):
            lambda1[i] = -(self.It[i][-1] - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
            EM += torch.sum((self.It[i][-1] - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
        return lambda1, EM
    
    # compute matching energy
    def calculateMatchingEnergyMSE2d(self):
        lambda1 = [None]*len(self.I)
        EM = 0
        for i in range(len(self.I)):
            lambda1[i] = -(self.It[i][-1] - self.J[i])/self.params['sigma'][i]**2 # may not need to store this
            EM += torch.sum((self.It[i][-1] - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
        return lambda1, EM
    
    # compute matching energy without lambda1
    def calculateMatchingEnergyMSEOnly(self, I):
        EM = 0
        for i in range(len(self.I)):
            EM += torch.sum((I[i] - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]*self.dx[2]
        return EM
    
    # compute matching energy without lambda1
    def calculateMatchingEnergyMSEOnly2d(self, I):
        EM = 0
        for i in range(len(self.I)):
            EM += torch.sum((I[i] - self.J[i])**2/(2.0*self.params['sigma'][i]**2))*self.dx[0]*self.dx[1]
        return EM
    
    # update learning rate for gradient descent
    def updateGDLearningRate(self):
        flag = False
        if len(self.EAll) > 1:
            if self.params['optimizer'] == 'gdr':
                if self.params['checkaffinestep'] == 0:
                    # energy increased
                    if self.EAll[-1] > self.EAll[-2]:
                        self.GDBeta *= 0.7
                else:
                    # if diffeo energy increased
                    if self.ERAll[-1] + self.EMDiffeo[-1] > self.EAll[-2]:
                        self.GDBeta *= 0.8
                    
                    if self.EMAffineR[-1] > self.EMDiffeo[-1]:
                        self.GDBetaAffineR *= 0.95
                    
                    if self.EMAffineT[-1] > self.EMAffineR[-1]:
                        self.GDBetaAffineT *= 0.95
            
            elif self.params['optimizer'] == 'gdw':
                # energy increased
                if self.EAll[-1] > self.EAll[-2]:
                    self.climbcount += 1
                    if self.climbcount > self.params['maxclimbcount']:
                        flag = True
                        self.GDBeta *= 0.7
                        self.climbcount = 0
                        self.vt0 = [x.to(device=self.params['cuda']) for x in self.best['vt0']]
                        self.vt1 = [x.to(device=self.params['cuda']) for x in self.best['vt1']]
                        if self.J[0].dim() > 2:
                            self.vt2 = [x.to(device=self.params['cuda']) for x in self.best['vt2']]
                        print('Reducing epsilon to ' + str((self.GDBeta*self.params['epsilon']).item()) + ' and resetting to last best point.')
                # energy decreased
                elif self.EAll[-1] < self.bestE:
                    self.climbcount = 0
                    self.GDBeta *= 1.04
                elif self.EAll[-1] < self.EAll[-2]:
                    self.climbcount = 0
        
        if self.params['savebestv']:
            if self.EAll[-1] < self.bestE:
                self.bestE = self.EAll[-1]
                # TODO: this may be too slow to keep doing on cpu. possibly clone on gpu and eat memory
                self.best['vt0'] = [x.cpu() for x in self.vt0]
                self.best['vt1'] = [x.cpu() for x in self.vt1]
                if self.J[0].dim() > 2:
                    self.best['vt2'] = [x.cpu() for x in self.vt2]
        
        return flag
    
    
    # compute gradient of affine transformation
    # TODO: can i change the order of diffeo and A to remove one image gradient calculation?
    def calculateGradientA(self,affineA,lambda1):
        affineB = torch.inverse(affineA)
        gi_x = [None]*len(self.I)
        gi_y = [None]*len(self.I)
        gi_z = [None]*len(self.I)
        for i in range(len(self.I)):
            gi_x[i],gi_y[i],gi_z[i] = self.torch_gradient(self.It[i][-1],self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
            # TODO: can this be efficiently vectorized?
            for r in range(3):
                for c in range(4):
                    # allocating on the fly, not good
                    dA = torch.tensor(np.zeros((4,4))).type(self.params['dtype']).to(device=self.params['cuda'])
                    dA[r,c] = 1.0
                    AdAB = torch.mm(torch.mm(affineA,dA),affineB)
                    #AdABX = AdAB[0,0]*self.X0 + AdAB[0,1]*self.X1 + AdAB[0,2]*self.X2 + AdAB[0,3]
                    #AdABY = AdAB[1,0]*self.X0 + AdAB[1,1]*self.X1 + AdAB[1,2]*self.X2 + AdAB[1,3]
                    #AdABZ = AdAB[2,0]*self.X0 + AdAB[2,1]*self.X1 + AdAB[2,2]*self.X2 + AdAB[2,3]
                    # check if using lambda1 is faster
                    # TODO: this product has major bit depth issues, order of magnitude difference in gradient
                    if i == 0:
                        self.gradA[r,c] = torch.sum( lambda1[i] * ( gi_x[i]*(AdAB[0,0]*self.X0 + AdAB[0,1]*self.X1 + AdAB[0,2]*self.X2 + AdAB[0,3]) + gi_y[i]*(AdAB[1,0]*self.X0 + AdAB[1,1]*self.X1 + AdAB[1,2]*self.X2 + AdAB[1,3]) + gi_z[i]*(AdAB[2,0]*self.X0 + AdAB[2,1]*self.X1 + AdAB[2,2]*self.X2 + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
                    else:
                        self.gradA[r,c] += torch.sum( lambda1[i] * ( gi_x[i]*(AdAB[0,0]*self.X0 + AdAB[0,1]*self.X1 + AdAB[0,2]*self.X2 + AdAB[0,3]) + gi_y[i]*(AdAB[1,0]*self.X0 + AdAB[1,1]*self.X1 + AdAB[1,2]*self.X2 + AdAB[1,3]) + gi_z[i]*(AdAB[2,0]*self.X0 + AdAB[2,1]*self.X1 + AdAB[2,2]*self.X2 + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
                    #self.gradA[r,c] = torch.sum( lambda1 * ( gi_y*(AdAB[0,0]*self.X1 + AdAB[0,1]*self.X0 + AdAB[0,2]*self.X2 + AdAB[0,3]) + gi_x*(AdAB[1,0]*self.X1 + AdAB[1,1]*self.X0 + AdAB[1,2]*self.X2 + AdAB[1,3]) + gi_z*(AdAB[2,0]*self.X1 + AdAB[2,1]*self.X0 + AdAB[2,2]*self.X2 + AdAB[2,3]) ) ) * self.dx[0]*self.dx[1]*self.dx[2]
    
    # compute gradient per time step for time varying velocity field parameterization
    def calculateGradientVt(self,lambda1,t,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu):
        # update phiinv using method of characteristics, note "+" because we are integrating backward
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (self.X0+self.vt0[t]*self.dt)
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (self.X1+self.vt1[t]*self.dt)
        phiinv2_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv2_gpu-self.X2).unsqueeze(0).unsqueeze(0),torch.stack(((self.X2+self.vt2[t]*self.dt)/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,(self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border')) + (self.X2+self.vt2[t]*self.dt)
        
        
        # find the determinant of Jacobian
        phiinv0_0,phiinv0_1,phiinv0_2 = self.torch_gradient(phiinv0_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
        phiinv1_0,phiinv1_1,phiinv1_2 = self.torch_gradient(phiinv1_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
        phiinv2_0,phiinv2_1,phiinv2_2 = self.torch_gradient(phiinv2_gpu,self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)
        detjac = phiinv0_0*(phiinv1_1*phiinv2_2 - phiinv1_2*phiinv2_1)\
            - phiinv0_1*(phiinv1_0*phiinv2_2 - phiinv1_2*phiinv2_0)\
            + phiinv0_2*(phiinv1_0*phiinv2_1 - phiinv1_1*phiinv2_0)
        
        # deform phiinv back by affine transform if asked for
        if self.params['doaffine'] == 1:
            phiinv0_gpu = self.affineA[0,0]*phiinv0_gpu + self.affineA[0,1]*phiinv1_gpu + self.affineA[0,2]*phiinv2_gpu + self.affineA[0,3]
            phiinv1_gpu = self.affineA[1,0]*phiinv0_gpu + self.affineA[1,1]*phiinv1_gpu + self.affineA[1,2]*phiinv2_gpu + self.affineA[1,3]
            phiinv2_gpu = self.affineA[2,0]*phiinv0_gpu + self.affineA[2,1]*phiinv1_gpu + self.affineA[2,2]*phiinv2_gpu + self.affineA[2,3]
        
        for i in range(len(self.I)):
            # find lambda_t
            if self.params['doaffine'] == 0:
                lambdat = torch.squeeze(torch.nn.functional.grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border'))*detjac
            else:
                lambdat = torch.squeeze(torch.nn.functional.grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((phiinv2_gpu/(self.nx[2]*self.dx[2]-self.dx[2])*2-1,phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border'))*detjac*torch.det(self.affineA)
            
            # get the gradient of the image at this time
            # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
            if i == 0:
                grad_list = [x*lambdat for x in self.torch_gradient(self.It[i][t],self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)]
            else:
                grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient(self.It[i][t],self.dx[0],self.dx[1],self.dx[2],self.grad_divisor_x,self.grad_divisor_y,self.grad_divisor_z)])]
        
        # smooth it
        grad_list = [torch.irfft(torch.rfft(x,3,onesided=False)*self.Khat,3,onesided=False) for x in grad_list]
        
        # add the regularization term
        grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
        grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
        grad_list[2] += self.vt2[t]/self.params['sigmaR']**2
        return grad_list,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu
    
    # compute gradient per time step for time varying velocity field parameterization
    def calculateGradientVt2d(self,lambda1,t,phiinv0_gpu,phiinv1_gpu):
        # update phiinv using method of characteristics, note "+" because we are integrating backward
        phiinv0_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv0_gpu-self.X0).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=2).unsqueeze(0),padding_mode='border')) + (self.X0+self.vt0[t]*self.dt)
        phiinv1_gpu = torch.squeeze(torch.nn.functional.grid_sample((phiinv1_gpu-self.X1).unsqueeze(0).unsqueeze(0),torch.stack(((self.X1+self.vt1[t]*self.dt)/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,(self.X0+self.vt0[t]*self.dt)/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=2).unsqueeze(0),padding_mode='border')) + (self.X1+self.vt1[t]*self.dt)        
        
        # find the determinant of Jacobian
        phiinv0_0,phiinv0_1 = self.torch_gradient2d(phiinv0_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
        phiinv1_0,phiinv1_1 = self.torch_gradient2d(phiinv1_gpu,self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)
        detjac = phiinv0_0 * phiinv1_1 - phiinv0_1 * phiinv1_0
        self.detjac[t] = detjac.clone()
        '''
        # deform phiinv back by affine transform if asked for
        if self.params['doaffine'] == 1:
            phiinv0_gpu = self.affineA[0,0]*phiinv0_gpu + self.affineA[0,1]*phiinv1_gpu + self.affineA[0,2]
            phiinv1_gpu = self.affineA[1,0]*phiinv0_gpu + self.affineA[1,1]*phiinv1_gpu + self.affineA[1,2]
        '''
        for i in range(len(self.I)):
            # find lambda_t
            #if self.params['doaffine'] == 0:
            lambdat = torch.squeeze(torch.nn.functional.grid_sample(lambda1[i].unsqueeze(0).unsqueeze(0), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=2).unsqueeze(0),padding_mode='border'))*detjac
            #else:
            #    lambdat = torch.squeeze(torch.nn.functional.grid_sample(lambda1.unsqueeze(0).unsqueeze(0), torch.stack((phiinv1_gpu/(self.nx[1]*self.dx[1]-self.dx[1])*2-1,phiinv0_gpu/(self.nx[0]*self.dx[0]-self.dx[0])*2-1),dim=3).unsqueeze(0),padding_mode='border'))*detjac*torch.det(self.affineA)
            # get the gradient of the image at this time
            # is there a row column flip in matlab versus my torch_gradient function? yes, there is.
            if i == 0:
                grad_list = [x*lambdat for x in self.torch_gradient2d(self.It[i][t],self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)]
            else:
                grad_list = [y + z for (y,z) in zip(grad_list,[x*lambdat for x in self.torch_gradient2d(self.It[i][t],self.dx[0],self.dx[1],self.grad_divisor_x,self.grad_divisor_y)])]
        
        # smooth it
        grad_list = [torch.irfft(torch.rfft(x,2,onesided=False)*self.Khat,2,onesided=False) for x in grad_list]
        
        # add the regularization term
        grad_list[0] += self.vt0[t]/self.params['sigmaR']**2
        grad_list[1] += self.vt1[t]/self.params['sigmaR']**2
        return grad_list,phiinv0_gpu,phiinv1_gpu
    
    # update gradient
    def updateGradientVt(self,t,grad_list):
        self.vt0[t] -= self.params['epsilon']*self.GDBeta*grad_list[0]
        self.vt1[t] -= self.params['epsilon']*self.GDBeta*grad_list[1]
        if self.J[0].dim() > 2:
            self.vt2[t] -= self.params['epsilon']*self.GDBeta*grad_list[2]
    
    # convenience function for calculating and updating gradients of Vt
    def calculateAndUpdateGradientsVt(self, lambda1):
        phiinv0_gpu = self.X0.clone()
        phiinv1_gpu = self.X1.clone()
        if self.J[0].dim() > 2:
            phiinv2_gpu = self.X2.clone()
        
        for t in range(self.params['nt']-1,-1,-1):
            if self.J[0].dim() > 2:
                grad_list,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu = self.calculateGradientVt(lambda1,t,phiinv0_gpu,phiinv1_gpu,phiinv2_gpu)
            else:
                grad_list,phiinv0_gpu,phiinv1_gpu = self.calculateGradientVt2d(lambda1,t,phiinv0_gpu,phiinv1_gpu)
            self.updateGradientVt(t,grad_list)
    
    # update affine matrix
    def updateAffine(self):
        # transfer to cpu for matrix exponential, takes about 20ms round trip
        gradA_cpu_numpy = self.gradA.cpu().numpy()
        e = np.zeros((4,4))
        e[0:3,0:3] = self.params['epsilonL']*self.GDBetaAffineR
        e[0:3,3] = self.params['epsilonT']*self.GDBetaAffineT
        e = torch.tensor(scipy.linalg.expm(-e * gradA_cpu_numpy)).type(self.params['dtype']).to(device=self.params['cuda'])
        self.lastaffineA = self.affineA.clone()
        self.affineA = torch.mm(self.affineA,e)
    
    # main loop
    def registration(self):
        for it in range(self.params['niter']):
            # deform images forward
            if self.J[0].dim() == 2:
                _,_,_ = self.forwardDeformation2d()
                ER = self.calculateRegularizationEnergyVt2d()
                lambda1,EM = self.calculateMatchingEnergyMSE2d()
            else:
                _,_,_,_ = self.forwardDeformation()
                ER = self.calculateRegularizationEnergyVt()
                lambda1,EM = self.calculateMatchingEnergyMSE()
            
            # save variables
            E = ER+EM
            self.EMAll.append(EM)
            self.ERAll.append(ER)
            self.EAll.append(E)
            if self.params['checkaffinestep']:
                self.EMAffineT.append(EM)
            if it == 0 and self.params['savebestv']:
                self.bestE = E.clone()
            
            # print function
            if it > 0:
                start_time = end_time
            
            end_time = time.time()
            if it > 0:
                #print('iter: ' + str(it) + ', E = ' + str(E.item()) + ', ER = ' + str(ER.item()) + ', EM = ' + str(EM.item()) + ', ep = ' + str((self.GDBeta*self.params['epsilon']).item()) + ', time = ' + str(end_time-start_time) + '.')
                if self.params['checkaffinestep'] == 1:
                    print("iter: " + str(it) + ", E= {:.3f}, ER= {:.3f}, EM= {:.3f}, epd= {:.3f}, del_Ev= {:.4f}, del_El= {:.4f}, del_Et= {:.4f}, time= {:.2f}.".format(E.item(),ER.item(),EM.item(),(self.GDBeta*self.params['epsilon']).item(),self.ERAll[-1] + self.EMDiffeo[-1] - self.EAll[-2], self.EMAffineR[-1] - self.EMDiffeo[-1], self.EMAffineT[-1] - self.EMAffineR[-1],end_time-start_time))
                else:
                    print("iter: " + str(it) + ", E= {:.3f}, ER= {:.3f}, EM= {:.3f}, epd= {:.3f}, time= {:.2f}.".format(E.item(),ER.item(),EM.item(),(self.GDBeta*self.params['epsilon']).item(),end_time-start_time))
            else:
                #print('iter: ' + str(it) + ', E = ' + str(E.item()) + ', ER = ' + str(ER.item()) + ', EM = ' + str(EM.item()) + ', ep = ' + str((self.GDBeta*self.params['epsilon']).item()) + '.')
                print("iter: " + str(it) + ", E = {:.4f}, ER = {:.4f}, EM = {:.4f}, epd = {:.6f}.".format(E.item(),ER.item(),EM.item(),(self.GDBeta*self.params['epsilon']).item()))
            
            # or (self.EAll[-1]/self.EAll[-2] < 1-self.params['minenergychange'] and self.EAll[-2]/self.EAll[-3] < 1-self.params['minenergychange'] and self.EAll[-3]/self.EAll[-4] < 1-self.params['minenergychange'] and self.EAll[-4]/self.EAll[-5] < 1-self.params['minenergychange'])
            if it == self.params['niter']-1 or (self.GDBeta < self.params['minbeta'] and self.GDBetaAffineR < self.params['minbeta'] and self.GDBetaAffineT < self.params['minbeta']):
                break
            
            # update step sizes
            updateflag = self.updateGDLearningRate()
            # if asked for, recompute images
            if updateflag:
                if self.J[0].dim() == 2:
                    _,_,_ = self.forwardDeformation2d()
                    lambda1,EM = self.calculateMatchingEnergyMSE2d()
                else:
                    _,_,_,_ = self.forwardDeformation()
                    lambda1,EM = self.calculateMatchingEnergyMSE()
            
            # calculate affine gradient
            if self.params['doaffine'] == 1:
                self.calculateGradientA(self.affineA,lambda1)
            
            # calculate and update gradients
            self.calculateAndUpdateGradientsVt(lambda1)
            # update affine
            if self.params['doaffine'] == 1:
                self.updateAffine()
            
    
    
    
    # save files to disk
    def saveOutputs(self, save_template=False):
        if save_template:
            for i in range(len(self.I)):
                outimg = nib.AnalyzeImage(self.It[i][-1].to('cpu').numpy(),None)
                outimg.header['pixdim'][1:4] = self.dx
                nib.save(outimg,self.params['outdir'] + 'deformed_template_ch' + str(i) + '.img')
        
    # convenience function
    def run(self, restart=True):
        # check parameters
        flag = self._checkParameters()
        if flag==-1:
            print('ERROR: parameters did not check out.')
            return
        
        if restart:
            # load images
            flag = self._load(self.params['template'],self.params['target'])
            if flag==-1:
                print('ERROR: images did not load.')
                return
            
            # initialize initialize
            if self.J[0].dim() == 2:
                self.initializeVariables2d()
            else:
                self.initializeVariables()
            # initialize stuff for gradient function
            self._allocateGradientDivisors()
        
        # initialize kernels
        if self.J[0].dim() == 2:
            self.initializeKernels2d()
        else:
            self.initializeKernels()
        # main loop
        self.registration()
        # save outputs
        self.saveOutputs(save_template=True)
        
    
