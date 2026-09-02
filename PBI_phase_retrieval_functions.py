import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import pyfftw
from scipy.ndimage import gaussian_filter


def calculate_resolution_modregger(image, detector_pixelsize, magnification, filterwidth=5, highfrq=2.0, nblfac=2.0):
    # Function from Rajmunds matlab script. 
    # Based on: Spatial resolution in Bragg-magnified X-ray images as determined by Fourier analysis, Peter Modregger et. al. (2007)
    """
    Calculate the resolution in both row and column directions based on the given image.
    
    Parameters:
    - image: 2D numpy array representing the image.
    - detector_pixelsize: The size of each pixel in meters.
    - magnification: The magnification factor.
    - filterwidth: The width of the convolution filter (default: 5).
    - highfrq: The frequency threshold for resolution estimation (default: 2.0).
    - nblfac: The factor used for resolution estimation (default: 2).
    - ax: The axis object for plotting (optional, default: None).

    Returns:
    - row_resolution: The resolution in the row direction.
    - row_uns: The uncertainty in the row resolution.
    - col_resolution: The resolution in the column direction.
    - col_uns: The uncertainty in the column resolution.
    """
    
    # Define the region of interest (ROI) based on user input
    def defineroi_func():
        # Placeholder function for defining ROI, adjust as needed
        return np.s_[:], np.s_[:]  # Select all for now, replace with actual ROI selection
    
    yroi, xroi = defineroi_func()
    data = image[yroi, xroi]

    # Rows processing (zeilen)
    N = data.shape[1]
    k = 2 * np.pi / N * np.arange(-N // 2, N // 2)  # Sample values in k-space
    PowerFdata = np.abs(fftshift(fft(data, axis=1), axes=1)) ** 2
    PowerFdata = PowerFdata[:, k > 0]
    k = k[k > 0]

    # Convolution and resolution estimation for rows
    ConvPowerFdline = np.zeros_like(PowerFdata)
    filter = np.ones(filterwidth)  # Length should be odd
    filter = filter / np.sum(filter)
    xres_row = np.zeros(PowerFdata.shape[0])
    uxres_row = np.zeros(PowerFdata.shape[0])

    for is_ in range(PowerFdata.shape[0]):
        zw = convolve(PowerFdata[is_, :], filter, mode='same')
        ConvPowerFdline[is_, :] = zw
        nbl = np.mean(ConvPowerFdline[is_, k > highfrq])
        mink = k[np.min(np.where(ConvPowerFdline[is_, :] <= nblfac * nbl))]
        maxk = k[np.max(np.where(ConvPowerFdline[is_, :] >= nblfac * nbl))]
        if np.isnan(mink) or np.isnan(maxk):
            mink = np.inf
            maxk = np.inf
        kres = np.mean([mink, maxk])
        ukres = 0.5 * (maxk - mink)

        xres_row[is_] = 2 * np.pi / kres
        uxres_row[is_] = 2 * np.pi / kres ** 2 * ukres

    # Columns processing (spalten)
    N = data.shape[0]
    k = 2 * np.pi / N * np.arange(-N // 2, N // 2)
    PowerFdata = np.abs(fftshift(fft(data, axis=0), axes=0)) ** 2
    PowerFdata = PowerFdata[k > 0, :]
    k = k[k > 0]

    # Convolution and resolution estimation for columns
    ConvPowerFdline = np.zeros_like(PowerFdata)
    filter = np.ones(filterwidth)  # Length should be odd
    filter = filter / np.sum(filter)
    xres_col = np.zeros(PowerFdata.shape[1])
    uxres_col = np.zeros(PowerFdata.shape[1])

    for is_ in range(PowerFdata.shape[1]):
        zw = convolve(PowerFdata[:, is_], filter, mode='same')
        ConvPowerFdline[:, is_] = zw
        nbl = np.mean(ConvPowerFdline[k > highfrq, is_])
        mink = k[np.min(np.where(ConvPowerFdline[:, is_] <= nblfac * nbl))]
        maxk = k[np.max(np.where(ConvPowerFdline[:, is_] >= nblfac * nbl))]
        kres = np.mean([mink, maxk])
        ukres = 0.5 * (maxk - mink)

        xres_col[is_] = 2 * np.pi / kres
        uxres_col[is_] = 2 * np.pi / kres ** 2 * ukres

    # Calculate final row and column resolutions
    row_resolution = (np.mean(xres_row) * detector_pixelsize) / magnification
    row_unc = (np.std(xres_row) * detector_pixelsize) / magnification
    col_resolution = (np.mean(xres_col) * detector_pixelsize) / magnification
    col_unc = (np.std(xres_col) * detector_pixelsize) / magnification

    # Return results as a dictionary
    output = {
        "row_resolution [m]": row_resolution,
        "row_uncertainty": row_unc,
        "col_resolution": col_resolution,
        "col_uncertainty": col_unc
    }

    return output,row_resolution,row_unc,col_resolution,col_unc


def apply_shift_rect(psi, p):
    psi = np.asarray(psi)
    p = np.asarray(p)

    ny, nx = psi.shape[1], psi.shape[2]

    tmp = np.pad(
        psi,
        ((0, 0), (ny//2, ny//2), (nx//2, nx//2)),
        mode="symmetric"
    )

    fy = np.fft.fftfreq(2 * ny)
    fx = np.fft.rfftfreq(2 * nx)
    x, y = np.meshgrid(fx, fy)

    phase = np.exp(
        -2j * np.pi * (
            x[None, :, :] * p[:, 1, None, None] +
            y[None, :, :] * p[:, 0, None, None]
        )
    )

    shifted = np.fft.irfft2(
        phase * np.fft.rfft2(tmp),
        s=tmp.shape[-2:]
    )

    return shifted[:, ny//2:ny//2 + ny, nx//2:nx//2 + nx]



def highpass(img, sigma=20):
    return img - gaussian_filter(img, sigma=sigma)

def center_crop_stack(stack, cy, cx, size):
    h = size // 2
    return stack[:, cy-h:cy+h, cx-h:cx+h]

def apply_shift(psi, p,n):
    """Apply shift for all projections."""
    psi = np.array(psi)
    p = np.array(p)
    tmp = np.pad(psi,((0,0),(n//2,n//2),(n//2,n//2)), 'symmetric')
    [x, y] = np.meshgrid(np.fft.rfftfreq(2*n),
                         np.fft.fftfreq(2*n))
    shift = np.exp(-2*np.pi*1j *    
                   (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res0 = np.fft.irfft2(shift*np.fft.rfft2(tmp))
    res = res0[:, n//2:3*n//2, n//2:3*n//2]#.get()
    return res

def _upsampled_dft(data, ups,
                   upsample_factor=1, axis_offsets=None):

    im2pi = 1j * 2 * np.pi
    tdata = data.copy()
    kernel = (np.tile(np.arange(ups), (data.shape[0], 1))-axis_offsets[:, 1:2])[
        :, :, None]*np.fft.fftfreq(data.shape[2], upsample_factor)
    kernel = np.exp(-im2pi * kernel)
    tdata = np.einsum('ijk,ipk->ijp', kernel, tdata)
    kernel = (np.tile(np.arange(ups), (data.shape[0], 1))-axis_offsets[:, 0:1])[
        :, :, None]*np.fft.fftfreq(data.shape[1], upsample_factor)
    kernel = np.exp(-im2pi * kernel)
    rec = np.einsum('ijk,ipk->ijp', kernel, tdata)

    return rec
    
def registration_shift(src_image, target_image, upsample_factor=1, space="real"):

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = np.fft.fft2(src_image)
        target_freq = np.fft.fft2(target_image)

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifft2(image_product)
    A = np.abs(cross_correlation)
    maxima = A.reshape(A.shape[0], -1).argmax(1)
    maxima = np.column_stack(np.unravel_index(maxima, A[0, :, :].shape))

    midpoints = np.array([np.fix(axis_size / 2)
                          for axis_size in shape[1:]])

    shifts = np.array(maxima, dtype=np.float64)
    ids = np.where(shifts[:, 0] > midpoints[0])
    shifts[ids[0], 0] -= shape[1]
    ids = np.where(shifts[:, 1] > midpoints[1])
    shifts[ids[0], 1] -= shape[2]
    
    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)

        normalization = (src_freq[0].size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate

        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                                upsampled_region_size,
                                                upsample_factor,
                                                sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        A = np.abs(cross_correlation)
        maxima = A.reshape(A.shape[0], -1).argmax(1)
        maxima = np.column_stack(
            np.unravel_index(maxima, A[0, :, :].shape))

        maxima = np.array(maxima, dtype=np.float64) - dftshift

        shifts = shifts + maxima / upsample_factor
           
    return shifts






def CTFPurePhase(rads, wlen, dists, fx, fy, alpha=1e-3):
    """
    Multi-distance pure-phase CTF retrieval.

    Parameters
    ----------
    rads : ndarray
        Stack of projections, shape (ndist, ny, nx).
    wlen : float
        X-ray wavelength in meters.
    dists : ndarray
        Effective propagation distances in meters.
    fx, fy : ndarray
        Spatial frequency grids in 1/m.
    alpha : float
        Regularization factor.

    Returns
    -------
    phase : ndarray
        Retrieved phase projection, shape (ny, nx).
    """
    rads = np.asarray(rads)
    dists = np.asarray(dists)

    numerator = np.zeros_like(np.fft.fft2(rads[0]), dtype=np.complex64)
    denominator = np.zeros_like(rads[0], dtype=np.float32)

    q2 = fx**2 + fy**2

    for j in range(len(dists)):
        rad_freq = np.fft.fft2(rads[j])
        ctf = np.sin(np.pi * wlen * dists[j] * q2)

        numerator += ctf * rad_freq
        denominator += 2 * ctf**2

    numerator /= len(dists)
    denominator = denominator /len(dists) + alpha

    phase = np.real(np.fft.ifft2(numerator / denominator))
    phase *= 0.5

    return phase.astype(np.float32)




def CTF(rads, wlen, dists, fx, fy, Rm, alpha): # Rm
    """
    Phase retrieval method based on Contrast Transfer Function.    This 
    method assumes weak absoprtion and slowly varying phase shift.
    Derived from Langer et al., 2008: Quantitative comparison of direct
    phase retrieval algorithms.

    Parameters
    ----------
    rads : list of 2D-array
        Elements of the list correspond to projections of the sample
        taken at different distance. One projection per element.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dists : list of float
        Object to detector distance (propagation distance) in mm. One 
        distance per element.
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space

    """

    A = np.zeros((rads[0].shape[0], rads[0].shape[1]))
    B = np.zeros((rads[0].shape[0], rads[0].shape[1]))
    C = np.zeros((rads[0].shape[0], rads[0].shape[1]))
    E = np.zeros((rads[0].shape[0], rads[0].shape[1]))
    F = np.zeros((rads[0].shape[0], rads[0].shape[1]))

    for j in range(0,len(dists)):
        sin = 2*np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) * Rm[:,:,j]
        cos = 2*np.cos(np.pi*wlen*dists[j]*(fx**2+fy**2)) * Rm[:,:,j]
        A = A + sin * cos
        B = B + sin * sin
        C = C + cos * cos
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])
        E = E + rad_freq * sin
        F = F + rad_freq * cos
    A = A / len(dists)
    B = B / len(dists)
    C = C / len(dists)    
    Delta = B * C - A**2
    
    phase = (C * E - A * F)    * (1 / (2*Delta+alpha)) 
    phase[0,0] = 0. + 0.j
    phase = pyfftw.interfaces.numpy_fft.ifft2(phase)
    phase = np.real(phase)

    return phase

def homoCTF(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):# 
    """



    Parameters
    ----------
    rad : 2D-array
        projection.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dist : float
        Object to detector distance (propagation distance) in mm.
    delta : float    
        refractive index decrement
    beta : float    
        absorption index
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space
    """    
    ny_, nx_ = rads[0].shape
    delta_dirac = np.bitwise_and(fx==0, fy==0).astype(np.double) * (ny_ * nx_)
    numerator = 0
    denominator = 0
    for j in range(0, len(dists)):    
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])
        cos = np.cos(np.pi*wlen*dists[j]*(fx**2+fy**2))
        sin = np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) 
        taylorExp = cos*Rm[:,:,j] + (delta/beta) * sin*Rm[:,:,j] # what is Rm????
        #taylorExp = cos + (delta/beta) * sin
        numerator = numerator + taylorExp * (rad_freq - delta_dirac)
        denominator = denominator + taylorExp**2

    numerator = numerator / len(dists)
    denominator = (denominator / len(dists)) + alpha
    
    phase = numerator / denominator    
    phase = np.real(  pyfftw.interfaces.numpy_fft.ifft2(phase) )
    phase = (delta/beta) * 0.5 * phase
    #phase = (delta/beta) * phase

    
    return phase

def CTFPurePhaseWithAbs(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
    argMin = np.argmin(dists)
    numerator = 0
    denominator = 0    

    for j in range(len(dists)):    
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j] / rads[argMin])
        taylorExp = np.sin(np.pi * wlen * dists[j] * (fx**2 + fy**2)) 
        numerator += taylorExp * rad_freq
        denominator += 2 * taylorExp**2 

    numerator /= len(dists)
    denominator = denominator / len(dists) + alpha

    tmp = np.real(pyfftw.interfaces.numpy_fft.ifft2(numerator / denominator))

    print("CTFPurePhaseWithAbs log input:")
    print("min:", np.nanmin(tmp), "max:", np.nanmax(tmp))
    print("negative:", np.sum(tmp < 0), "zeros:", np.sum(tmp == 0), "nan:", np.sum(np.isnan(tmp)))

    tmp = np.clip(tmp, 1e-6, None)

    phase = np.log(tmp)
    phase = (delta / beta) * 0.5 * phase

    return phase
def multiPaganin(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
    numerator = 0
    denominator = 0    

    for j in range(len(dists)):    
        rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])    
        
        # Ανάκτηση του Rm για τη συγκεκριμένη απόσταση (αν είναι 3D array)
        Rm_j = Rm[:, :, j] if Rm.ndim == 3 else Rm
        
        # Προσθήκη του Rm στο taylorExp (όπως ακριβώς στον single Paganin)
        taylorExp = 1 + wlen * dists[j] * np.pi * (delta / beta) * (fx**2 + fy**2) * Rm_j

        numerator += taylorExp * rad_freq
        denominator += taylorExp**2 

    numerator /= len(dists)
    denominator = denominator / len(dists) + alpha

    tmp = np.real(pyfftw.interfaces.numpy_fft.ifft2(numerator / denominator))
    
    # Ασφαλές clipping για αποφυγή αρνητικών τιμών/NaN στο log
    tmp = np.maximum(tmp, 1e-6)

    phase = (delta / beta) * 0.5 * np.log(tmp)

    return phase

def Paganin(rad, wlen, dist, delta, beta, fx, fy, Rm):            
    rad_freq = pyfftw.interfaces.numpy_fft.fft2(rad)

    '''from Paganin et al., 2002'''
    #~ mu = (4 * np.pi * beta) / wlen
    #~ phase = (rad_freq * mu) / (alpha+delta*dist*4*(np.pi**2)*(fx**2+fy**2)*Rm+mu) # 4 * pi^2 not explicit in manuscript
    #~ phase = np.real(pyfftw.interfaces.numpy_fft.ifft2(phase))
    #~ phase = (1/mu)*np.log(phase)
    #~ phase = (2*np.pi*delta/wlen)*phase

    '''from ANKA - Weitkamp et al., 2011'''
    filtre =  1 + (wlen*dist*delta*4*(np.pi**2)*(fx**2+fy**2) / (4*np.pi*beta)) # 4 * pi^2 not explicit in manuscript
    trans_func = np.log(np.real( pyfftw.interfaces.numpy_fft.ifft2( rad_freq / filtre)))
    phase = (delta/(2*beta)) * trans_func
        
    #~ phase = phase *(-wlen)/(2*np.pi*delta)
    return phase    

def sglDstCTF(rad, wlen, dist, delta, beta, fx, fy, Rm, alpha):
    """
    Phase retrieval method based on Contrast Transfer Function.    This 
    method relies on linearization of the direct problem, based  on  the
    first  order  Taylor expansion of the transmittance function.
    Found in Yu et al. 2018 and adapted from Cloetens et al. 1999


    Parameters
    ----------
    rad : 2D-array
        projection.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dist : float
        Object to detector distance (propagation distance) in mm.
    delta : float    
        refractive index decrement
    beta : float    
        absorption index
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space
    """    
    delta_dirac = np.bitwise_and(fx==0,fy==0).astype(np.double) #Aditya: Discretized Dirac Delta function
    rad_freq = pyfftw.interfaces.numpy_fft.fft2(rad)
    filtre = np.cos(np.pi*wlen*dist*(fx**2+fy**2)) + (delta/beta) * np.sin(np.pi*wlen*dist*(fx**2+fy**2))
    phase = (delta/beta) * 0.5 * ((rad_freq - delta_dirac) / filtre)
    phase = np.real(pyfftw.interfaces.numpy_fft.ifft2(phase))

    return phase
