import numpy as np

def dl_to_cl(el, cl_or_dl, inverse = 0):
    dl_fac = (el * (el+1)/2./np.pi)
    if inverse:
        return cl_or_dl*dl_fac
    else:
        return cl_or_dl/dl_fac
    
#take sets of astrophysical cls and ls
#combine into a covariance matrix with 1d cls
#ells and cls are both lists of arrays, len(ell[i]) must equal len(cl[i])
#for components that are the same across bands, cl[i] is one dimensional
#for components are different across bands, c[i] is multi dimensional
def create_ncov1d(ells,cls,ell_max,nbands):
    #ell_min=int(np.min(ells))
    #ell_astro=np.arange(int(ell_max))[ell_min:]
    #create a band by band matrix, to fill in each element with the cls for those bands
    ncov_1d=np.zeros([nbands,nbands,int(ell_max)])
    for c in range(0,len(ells)):
        #print('c number'+str(c))
        #print(cls[c])
        for b1 in range(0,nbands):
            #print(b1)
            for b2 in range(0,nbands):
                #print(str(b1)+'x'+str(b2))
                if cls[c].ndim==1:
                    ncov_1d[b1][b2][int(min(ells[c])):len(ells[c])+int(min(ells[c]))]+=cls[c]
                    #print(ncov_1d[b1][b2])
                else:
                    ncov_1d[b1][b2][int(min(ells[c])):len(ells[c])+int(min(ells[c]))]+=np.sqrt(cls[c][b1]*cls[c][b2])
    return ncov_1d

#take the band band covariance matrix of 1d cls, turn into the full NyXNxYbXb covariance matrix
def create_ncovbd(ncov_1d,beam_tf,psize,nbands,ny,nx,ell_max):
    ell_astro=np.arange(int(ell_max))
    print([ny,nx,nbands,nbands])
    noisemaps=np.reshape(np.array([[gridtomap(ell_grid(psize,ny,nx),ncov_1d[i][j],ell_astro)*beam_tf[i]*beam_tf[j] 
                         for i in range(0,nbands)] for j in range(0,nbands)]),(nbands**2,ny,nx))
    nmat=np.reshape(np.stack(noisemaps,axis=2),(ny,nx,nbands,nbands))
    return nmat
#create and pixel pixel band band covariance matrix
#takes an array of of noisemaps for each band
#THIS IS DOING IT WRONG for astro CLs! NEED TO CALCULATE CROSS CL'S SEPERATELY THEN ADD TOGETHER FOR DIAGONAL ELEMENTS
def create_N_d(noisemaps,diag=False):
    b=len(noisemaps)
    ny,nx=np.shape(noisemaps[0])
    stacked=np.stack(noisemaps,axis=2)
    stacked=np.reshape(stacked,(ny,nx,b,1))
    nmat=np.nan_to_num(np.sqrt(stacked*np.transpose(stacked,axes=(0,1,3,2))))
    if diag:
        nmat*=np.identity(b)
    return nmat    

#calculate the expected sigma of the filtered map, used to normalize filter
def sigma_faster(fsz,sf,nmat,ny,nx,b):
    s = sf*fsz
    s = np.reshape(s, (ny,nx,1,b))
    full_int=(s*np.linalg.inv(nmat)*np.transpose(s,axes=(0,1,3,2))).sum(axis=(2,3))
    sigma=np.real((np.sum(np.sum(full_int))/(ny*nx))**(-.5))
    return sigma

#create the optimal matched filter
#returns psi, sigma
def psi_faster(fsz,sf,nmat,ny,nx,b):
    s=sf*fsz
    s = np.reshape(s, (ny,nx,1,b))
    sigma=sigma_faster(fsz,sf,nmat,ny,nx,b)
    temp=(np.linalg.inv(nmat)*s).sum(axis=(3))
    return (sigma**2*temp), sigma

#apply the matched filter to the maps
#amaps=[[90][150][220]]
def multi_band_filter(amaps,psi,psize,b):
    resrad=psize*0.000290888
    kmaps=np.array([np.fft.fft2(np.fft.fftshift(m)*resrad,norm='ortho') for m in amaps])
    kmat=np.stack(kmaps,axis=2)
    filtered=np.sum(psi*kmat,axis=2)
    filtered = np.fft.fftshift(np.fft.ifft2(filtered,norm='ortho'))
    filtered = np.real(filtered)
    return filtered

#calculate sigma by fitting a gaussian and taking the width
#apod and mask should be 1s in unmasked pixels, 0 elsewhere
def ndh_sigma(amap,apod,mask):
    if apod is None:
        apod=np.ones(np.shape(amap))
    if mask is None:
        mask=np.ones(np.shape(amap))
    masked_pix=apod*mask
    tmap=amap[masked_pix==1].copy()
    mean=np.mean(amap)
    sig=np.std(amap)
    tmap=tmap[np.abs(tmap)<sig*5]
    #tmap=tmap[tmap!=0]
    #flat=tmap.flatten()
    #hist, bins = np.histogram(tmap, bins=200)
    #params,cov=op.curve_fit(gaus,bins[:len(bins)-1],hist)
    #this requires SPT-3G software, ommiting
    #atemp = gaussfit_hist(tmap,1000,-8.*sig,8.*sig,
    #                      do_plot = False)
    #signoise = atemp['params'][2]
    signoise=np.std(tmap)
    #print(signoise)
    if signoise>=1e-15:
        return np.abs(signoise)
    else:
        return 0

def calc_fsz(f):
    f*=10**9
    h=6.62607015*10**-34 
    Kb=1.380649*10**-23
    T=2.725
    x=(h*f)/(Kb*T)
    #print(x)
    return x*((np.e**x+1.)/(np.e**x-1.))-4

#from Matt
#are these axes switched? based on the numpy (y,x) convention, I think they are
def fft_grid(reso,nx,ny=None,dim=1):
    if dim == 2 and ny is None:
        ny = nx
    if ny is not None:
        dim = 2
    n = nx
    ind = np.arange(n,dtype=float)
    wh_lo = np.where(ind <= n/2.0)[0]
    n_lo = len(wh_lo)
    if np.mod(n,2) == 0:
        n_right = n_lo -1
    else:
        n_right = n_lo
    temp_ind = -ind[wh_lo[1:n_right]]
    ind = np.concatenate((ind[wh_lo],temp_ind[::-1]))
    grid = ind/reso/n
    if dim == 2:
        n = ny
        ind = np.arange(n,dtype=float)
        wh_lo = np.where(ind <= n/2.0)[0]
        n_lo = len(wh_lo)
        if np.mod(n,2) == 0:
            n_right = n_lo -1
        else:
            n_right = n_lo
        temp_ind = -ind[wh_lo[1:n_right]]
        ind = np.concatenate((ind[wh_lo],temp_ind[::-1]))
        gridy = ind/reso/n
        grid2d = np.zeros([nx,ny])
        for i in range(ny):
            grid2d[:,i] = np.sqrt(np.square(grid) + np.square(gridy[i]))
        # returns oscillations per radian on sky
        return grid2d
    else:
        return grid
    
#make an l space grid
def ell_grid(psize,ny,nx=None):
    if nx is None:
        nx=ny
    ellgrid=fft_grid(np.radians(psize/60),ny,nx)*2*np.pi
    ellgrid[np.where(ellgrid==0)] = 2
    return ellgrid.astype(int)

#fill in an l space grid with cl values
def gridtomap(ellgrid,cls,els):
    kmap=np.zeros(np.shape(ellgrid))
    ny,nx=np.shape(ellgrid)
    for i in range(0,nx):
        for j in range(0,ny):
            index=ellgrid[j,i]#-2 for some reason the code had these -2s in it? and i don't know why?
            maxi=len(cls)
            if index<maxi:
                kmap[j,i]=cls[ellgrid[j,i]]#-2]
    return np.asarray(kmap)