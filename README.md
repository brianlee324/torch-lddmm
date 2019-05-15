#torch_lddmm - GPU/CPU implementation of modern dense image LDDMM registration algorithms in PyTorch

## Overview
This package performs optimization of LDDMM parameterized by a time-varying velocity field [1] on pairs of dense images (2D or 3D). This code is optimized to run on a GPU, but will also run on a CPU (original CPU version based on github.com/dtward/image_lddmm_tensorflow).

See ./examples/ directory for example Jupyter notebooks.

## Features
This package performs any combination of the following:
* Multi-channel images [2]
* Matching cost masking
* Tandem optimization of affine/rigid and diffeomorphism
* Image contrast correction [3]
* Voxelwise weight estimation [3]
* 2D to 3D rigid reference-guided image stack alignment [4]

## Quick-start guide
Dependencies: Python 3, PyTorch 1.0+, Numpy 1.15+
Download the package with: git clone github.com/brianlee324/torch-lddmm.git

In a Python session, "import torch_lddmm" and load image1 and image2 as numpy arrays. In our convention, the template image is warped towards the target image. The template and target images should be the same size and voxel spacing (pad or resample if not).

### Example: Basic LDDMM
lddmm = torch_lddmm.LDDMM(template=image1, target=image2, a=8, epsilon=1.0, sigma=10.0, sigmaR=10.0, dx=[0.1,0.1,0.1]) # create object
lddmm.run() # run registration with these settings
(vt0,vt1,vt2,A) = lddmm.outputTransforms() # output LDDMM and linear transforms
(phi0,phi1,phi2) = lddmm.computeThisDisplacement() # output resultant displacement field
deformed_template = lddmm.outputDeformedTemplate() # output deformed template as numpy array

### Example: Multichannel LDDMM
lddmm = torch_lddmm.LDDMM(template=[image1_channel1,image1_channel2],target=[image2_channel1,image2_channel2], a=8, epsilon=1.0, sigma=[10.0, 2.0], sigmaR=10.0, dx=[0.1,0.1,0.1])
lddmm.run()

## Example: Multichannel affine -> LDDMM+affine -> LDDMM
lddmm = torch_lddmm.LDDMM(template=[image1_channel1,image1_channel2],target=[image2_channel1,image2_channel2], a=8, epsilon=1.0, sigma=[10.0, 2.0], sigmaR=10.0, dx=[0.1,0.1,0.1], do_affine=1, do_lddmm=0, niter=50)
lddmm.run()
lddmm.setParams('niter',100) # increase iterations
lddmm.setParams('do_lddmm',1) # turn on lddmm, leave affine on
lddmm.run() # continue registration from current state
lddmm.setParams('do_affine',0) # turn off affine, leave lddmm on
lddmm.setParams('a',5) # shrink LDDMM kernel size
lddmm.run() # continue registration from current state

## Example: Multichannel LDDMM with contrast correction on channel 0 and weight estimation on channels 0 and 1
lddmm = torch_lddmm.LDDMM(template=[image1_channel1,image1_channel2],target=[image2_channel1,image2_channel2], a=8, epsilon=1.0, sigma=[10.0, 2.0], sigmaR=10.0, dx=[0.1,0.1,0.1], we=2, we_channels=[0,1], cc=1, cc_channels=[0])
lddmm.run()


### Parameter Guide
*a               = float (smoothing kernel, a*(pixel_size))
*p               = int (smoothing kernel power, p*2)
*niter           = int (number of iterations)
*epsilon         = float (gradient descent step size)
*epsilonL        = float (gradient descent step size, affine)
*epsilonT        = float (gradient descent step size, translation)
*minbeta         = float (smallest multiple of epsilon)
*sigma           = float (matching term coefficient (0.5/sigma**2))
*sigmaR          = float (regularization term coefficient (0.5/sigmaR**2))
*nt              = int (number of time steps in velocity field)
*do_lddmm        = 0/1 (perform LDDMM step, 0 = no, 1 = yes)
*do_affine       = 0/1/2 (interleave linear registration: 0 = no, 1 = affine, 2 = rigid)
*gpu_number      = int (index of CUDA_VISIBLE_DEVICES to use)
*dtype           = string (bit depth, 'float' or 'double')
*energy_fraction = float (fraction of initial energy at which to stop)
*cc              = 0/1 (contrast correction: 0 = no, 1 = yes)
*cc_channels     = list (image channels to run contrast correction (0-indexed))
*we              = 0/2/3/... (weight estimation: 0 = no, 2+ = yes)
*we_channels     = list (image channels to run weight estimation (0-indexed))
*sigmaW          = float (coefficient for each weight estimation class)
*nMstep          = int (update weight estimation every nMstep steps)
*costmask        = None or numpy.ndarray (costmask image)
*outdir          = string (output directory name)
*optimizer       = string (optimizer type 'gd' for gradient descent or 'gdr' for gradient descent with shrinking step size)
*template        = numpy.ndarray or list of numpy.ndarray
*target          = numpy.ndarray or list of numpy.ndarray

## References
1. Beg, Mirza Faisal & Miller, Michael & Trouvé, Alain & Younes, Laurent. (2005). Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms. International Journal of Computer Vision. 61. 139-157. 10.1023/B:VISI.0000043755.93987.aa. 
2. Ceritoglu, Can & Oishi, Kenichi & Mori, Susumu & Miller, Michael. (2009). Multi-contrast Large Deformation Diffeomorphic Metric Mapping and Diffusion Tensor Image Registration. NeuroImage. 47. S123. 10.1016/S1053-8119(09)71172-3. 
3. Tward, Daniel & Brown Timothy & Kageyama, Yusuke & Patel, Jaymin & Hou, Zhipeng & Mori, Susumu & Albert, Marilyn & Troncoso, Juan & Miller, Michael. (2018). Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer’s disease. bioRxiv 494005; doi: https://doi.org/10.1101/494005.
4. Lee, Brian & Tward, Daniel & Mitra, Partha & Miller, Michael. (2018). On variational solutions for whole brain serial-section histology using the computational anatomy random orbit model. PLOS Computational Biology. 14. 10.1371/journal.pcbi.1006610. 
