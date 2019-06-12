import numpy as np

class diffusion_filter():
    def __init__(self, niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1, gamma_efg = 1):
        self.niter = niter
        self.kappa = kappa
        self.gamma = gamma
        self.sigma = sigma
        self.option = option
        self.gamma_efg = gamma_efg
        
    def apply(self, img):
        return anisodiff(img,niter=self.niter,kappa=self.kappa,gamma=self.gamma,step=(1.,1.),sigma=self.sigma, option=self.option, gamma_efg = self.gamma_efg)
    
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1, gamma_efg = 1, gamma_tv_gauss = 1):
    img_min = img.min()
    img_max = img.max()
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.zeros_like(imgout) # Changed to zeros_like
    gE = gS.copy()

    for ii in np.arange(1,niter):
        
        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        if 0<sigma:
            deltaSf=flt.gaussian_filter(deltaS,sigma);
            deltaEf=flt.gaussian_filter(deltaE,sigma);
        else: 
            deltaSf=deltaS;
            deltaEf=deltaE;
        
        deltaSf = np.abs(deltaSf)
        deltaEf = np.abs(deltaEf)
        # conduction gradients (only need to compute one per dim!)
        if option == 1: # Perano malik
            gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
            gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
        elif option == 2: # Perano malik
            gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]
        elif option == 3: # BFB
            gS = 1/(deltaSf*(kappa + deltaSf) + 1e-7)/step[0]
            gE = 1/(deltaEf*(kappa + deltaEf) + 1e-7)/step[1]
        elif option == 4: # TV
            gS = 1/(deltaSf + kappa)/step[0]
            gE = 1/(deltaEf + kappa)/step[1]
        elif option == 5: # EFG

            qS = 1 - (deltaSf/gamma_efg)**2
            qE = 1 - (deltaEf/gamma_efg)**2

            gS[deltaSf <= gamma_efg] = (1 / (deltaSf + kappa) * np.exp(-qS**2) /step[0])[deltaSf <= gamma_efg]
            gS[deltaSf > gamma_efg] = (1 / (deltaSf + kappa) / step[0])[[deltaSf > gamma_efg]]

            gE[deltaEf <= gamma_efg] = (1 / (deltaEf + kappa) * np.exp(-qE**2) /step[0])[deltaEf <= gamma_efg]
            gE[deltaEf > gamma_efg] = (1 / (deltaEf + kappa) / step[0])[deltaEf > gamma_efg]
        
        elif option ==6:
            gS[deltaSf <= gamma_tv_gauss] = (np.ones_like(deltaSf)*(1/step[0]))[deltaSf <= gamma_tv_gauss]
            gS[deltaSf > gamma_tv_gauss] = (gamma_tv_gauss / (deltaSf + kappa) / step[0])[[deltaSf > gamma_efg]]

            gE[deltaEf <= gamma_tv_gauss] = (np.ones_like(deltaEf)*(1/step[0]))[deltaEf <= gamma_tv_gauss]
            gE[deltaEf > gamma_tv_gauss] = (gamma_tv_gauss / (deltaEf + kappa) / step[0])[[deltaEf > gamma_tv_gauss]]
        
        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        #plt.figure(figsize = (10,10))
        #plt.imshow((NS+EW))
        
        imgout += gamma*(NS+EW)
        imgout[imgout < img_min] = img_min
        imgout[imgout > img_max] = img_max
    return imgout
