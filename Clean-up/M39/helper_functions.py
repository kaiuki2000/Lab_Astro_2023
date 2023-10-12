import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from warnings import filterwarnings
import astroalign as aa
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia         
from astropy.wcs import WCS
from astropy import units as u
from astropy.visualization import simple_norm
from astropy.modeling import models, fitting
from astropy.modeling.models import Gaussian2D
from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm, SigmaClip
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MMMBackground # Can also use 'MedianBackground'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.psf import PSFPhotometry, IterativePSFPhotometry, BasicPSFPhotometry, IterativelySubtractedPSFPhotometry, IntegratedGaussianPRF, SourceGrouper, prepare_psf_model, DAOGroup
from astropy.table import Table, QTable
from shapely.geometry import LineString, Point

def im_plot(data, title = 'Default title', cbar_label = 'Counts', cbar_flag = True, stars_found = None, \
            cmap = 'RdPu', lo: float = 1.5, up: float = 98.5) -> None:
    """
    Plots an image with a colorbar and stars found if provided.
    
    Parameters
    ----------
    data : 2D array
        The image to be plotted.
    title : str, optional
        The title of the plot. The default is 'Default title'.
    cbar_label : str, optional
        The label of the colorbar. The default is 'Counts'.
    cbar_flag : bool, optional
        Whether to plot the colorbar. The default is True.
    stars_found : 2D array, optional
        The stars found by the algorithm. The default is None.
    cmap : str, optional
        The colormap to use. The default is 'RdPu'.
    lo : float, optional
        The lower percentile of the image to be plotted. The default is 1.5.
    up : float, optional
        The upper percentile of the image to be plotted. The default is 98.5.

    Returns
    -------
    None.
    """

    # Matplotlib style ;)
    plt.style.use('https://github.com/kaiuki2000/PitayaRemix/raw/main/PitayaRemix.mplstyle')

    fig = plt.figure()
    plt.title(title)
    l1 = np.percentile(data[np.isfinite(data)].flatten(), lo)
    l2 = np.percentile(data[np.isfinite(data)].flatten(), up)
    plt.imshow(data, cmap = cmap, origin = 'lower', clim = (l1, l2))
    if(cbar_flag == True):
        plt.colorbar(fraction = 0.030, pad = 0.035, label = cbar_label)
        plt.tight_layout()
    if(stars_found is not None): plt.plot(stars_found["x_fit"], stars_found["y_fit"], 'C1o', mfc = 'none', markersize = 2.5, alpha = 0.75)
    plt.xlabel(r'$x$ pixel')
    plt.ylabel(r'$y$ pixel')
    plt.grid(False)
    plt.show()



# Calibration functions
def masterBiasCreator(bias_dir: str = "./", plot_flag: bool = False, save_dir: str = "./", \
                      silent: bool = False) -> np.ndarray:
    """
    Description:
        Creates a master bias frame from a set of bias frames.
    Args:
        bias_dir (str): directory where the bias files are located.
        plot_flag (bool): if True, plots the resulting master bias frame.
        save_dir (str): directory where the master bias frame will be saved.
        silent (bool): if True, does not print anything.
    Outputs:
        masterBias (np.ndarray): master bias frame.
    """

    # Reading bias files
    biasFiles, biasAll, headersAll = [], [], []
    for file_n in os.listdir(bias_dir):
         if "bias" in file_n: biasFiles.append(file_n)
    if(not silent): print(f"Verification: biasFiles = {biasFiles}.")
    for file_n in biasFiles:
        biasAll.append(fits.open(bias_dir + file_n)[0].data)
        headersAll.append(fits.open(bias_dir + file_n)[0].header)
        
    biasAllNormalized = np.asarray([biasAll[i] for i in range(len(biasAll))])

    # Stacking bias files. Criteria: median.
    masterBias = np.median(biasAllNormalized, axis = 0)
    if(not silent): print(f'Shape tests: {np.shape(biasAllNormalized)}, {np.shape(masterBias)}')
    if(not silent): print(f'Normalized median (test) = {np.median(masterBias)}')

    # Plot the resulting master bias frame.
    if(plot_flag == True):
        im_plot(masterBias, title = 'Master bias frame', cbar_label = 'Counts', cbar_flag = True, stars_found = None, cmap = 'gray')

    # Save image, as a FITS file.
    fits.writeto(save_dir + 'MasterBias.fits', masterBias, headersAll[0], overwrite = True)
    if(not silent): print(f'Master bias frame saved as FITS file to {save_dir}/MasterBias.fits')
    return(masterBias)



def masterFlatCreator(masterBias: np.ndarray, flat_dir: str = './', plot_flag: bool = False, \
                      save_dir: str = "./", filter: str = 'Red', d: list = [1, 1, 3], \
                      silent: bool = False) -> np.ndarray:
    """
    Description:
        Creates a master flat frame from a set of flat frames.
    Args:
        flat_dir (str): directory where the flat files are located.
        plot_flag (bool): if True, plots the resulting master flat frame.
        save_dir (str): directory where the master flat frame will be saved.
        filter (str): filter used to take the flat frames.
        d (list): list of integers used to slice the flat files (Default values for M39).
        silent (bool): if True, does not print anything.
    Outputs:
        masterFlat (np.ndarray): master flat frame.
    """

    # Status message and initialization of 'i'.
    if(filter == "Red"):
        i = 0
        if(not silent): print(f"Creating master flat ({filter}) frame...")   
    elif(filter == "Green"):
        i = 1
        if(not silent): print(f"Creating master flat ({filter}) frame...")
    elif(filter == "Blue"):
        i = 2
        if(not silent): print(f"Creating master flat ({filter}) frame....")

    # Reading flat files, for filter X.
    flatX_Files, flatX_All, headersAll = [], [], []
    for file_n in os.listdir(flat_dir):
         if ("flat" in file_n and f"{filter}" in file_n): flatX_Files.append(file_n)
    flatX_Files = flatX_Files[d[i]:]
    if(not silent): print(f"Verification ({len(flatX_Files)} files read): flat{filter}Files = {flatX_Files}")
    for file_n in flatX_Files:
        flatX_All.append(fits.open(flat_dir + file_n)[0].data)
        headersAll.append(fits.open(flat_dir + file_n)[0].header)
    
    # Subtracting master bias from flatX files AND normalizing afterwards. Criteria: median.
    flatX_All_Normalized = np.asarray([(flatX_All[i] - masterBias)/np.median(flatX_All[i] - masterBias) for i in range(len(flatX_All))])
    
    # Stacking flatX files. Criteria: median.
    masterFlatX = np.median(flatX_All_Normalized, axis = 0)
    if(not silent): print(f'Shape tests: Before stacking = {np.shape(flatX_All_Normalized)}; After stacking = {np.shape(masterFlatX)}')
    if(not silent): print(f'Normalized median (test) = {np.median(masterFlatX)}')
    
    if(plot_flag == True):
        im_plot(masterFlatX, title = f'Master flat ({filter}) frame', cbar_label = 'Counts', cbar_flag = True, stars_found = None, cmap = 'gray')
    
    # Save image as a FITS file.
    fits.writeto(save_dir + f'MasterFlat{filter}.fits', masterFlatX, headersAll[0], overwrite = True) # Using fitsAll[0]'s header as header.
    if(not silent): print(f'Master flat ({filter}) frame saved as FITS file to {save_dir}/MasterFlat{filter}.fits')
    return(masterFlatX)



def generateCalibratedFrames(masterBias: np.ndarray, masterFlat: np.ndarray, light_dir: str = "./", \
                             save_dir: str = "./", filter: str = "Red", d: list = [10, 12, 13], \
                             object: str = "m39", silent: bool = False) -> None:
    """
    Description:
        Generates calibrated frames from a set of light frames.
    Args:
        masterBias (np.ndarray): master bias frame.
        masterFlat (np.ndarray): master flat frame.
        light_dir (str): directory where the light files are located.
        save_dir (str): directory where the calibrated frames will be saved.
        filter (str): filter used to take the light frames.
        d (list): list of integers used to slice the light files.
        object (str): name of the object ('m39' by default).
    Outputs:
        None.
    """

    # Status message and initialization of 'i'.
    if(filter == "Red"):
        i = 0
        if(not silent): print("Applying correction to \"Red\" files...")   
    elif(filter == "Green"):
        i = 1
        if(not silent): print("Applying correction to \"Green\" files...")
    elif(filter == "Blue"):
        i = 2
        if(not silent): print("Applying correction to \"Blue\" files...")

    # Reading light f'{filter}' files
    light_X_Files, light_X_All, header_X_All = [], [], []
    for file_n in os.listdir(light_dir):
         if (object in file_n and f"{filter}" in file_n): light_X_Files.append(file_n)
    light_X_Files = light_X_Files[d[i]:]
    if(not silent): print(f"Verification ({len(light_X_Files)} files read [Object: {object}]): light{filter}Files = {light_X_Files}")
    for file_n in light_X_Files:
        light_X_All.append(fits.open(light_dir + file_n)[0].data)
        header_X_All.append(fits.open(light_dir + file_n)[0].header)

    # Correction with bias and flat.
    light_X_All_corrected = (light_X_All - masterBias)/masterFlat
    
    for n, frame in enumerate(light_X_All_corrected):
        fits.writeto(save_dir + f'{light_X_Files[n]}', frame, header_X_All[n], overwrite = True)



def align_colour_frames(light_dir: str = "./", save_dir: str = "./", filter: str = 'Red', \
                        object: str = "m39", silent: bool = False) -> None:
    """
    Description:
        Aligns a set of light frames, from the specified filter colour.
    Args:
        light_dir (str): directory where the light files are located.
        save_dir (str): directory where the calibrated frames will be saved.
        filter (str): filter used to take the light frames.
        object (str): name of the object ('m39' by default).
    Outputs:
        None.
    """
    
    files = []
    for file_n in os.listdir(light_dir):
         if (object in file_n and f"{filter}" in file_n and "AlignedColour" not in file_n): files.append(file_n)
    if(not silent): print(f'Files to align: {files}.')

    reference_image = fits.open(light_dir + files[0])[0].data + 0
    for i in range(0, len(files)):
        image_data   = fits.open(light_dir + files[i])
        source_image = image_data[0].data + 0  # Here, the addition of zero (0) solves the the endian compiler issue.
        header       = image_data[0].header
        image_aligned, footprint = aa.register(source_image, reference_image, fill_value = np.nan)
        aligned_file = files[i].replace('.fits', '')
        fits.writeto(save_dir + aligned_file + '_AlignedColour' + '.fits', image_aligned, header, overwrite = True)
        if(not silent): print('No. %i alignment done.' %i)



def stack_colour_frames(light_dir: str = "./", save_dir: str = "./", filter: str = 'Red', \
                        object: str = "m39", silent: bool = False, plot_flag: bool = False) -> np.ndarray:
    """
    Description:
        Stacks a set of light frames, from the specified filter colour.
    Args:
        light_dir (str): directory where the light files are located.
        save_dir (str): directory where the calibrated frames will be saved.
        filter (str): filter used to take the light frames.
        object (str): name of the object ('m39' by default).
    Outputs:
        None.
    """

    # Reading light f'{filter}' files
    light_X_Files, light_X_All, header_X_All = [], [], []
    for file_n in os.listdir(light_dir):
         if (object in file_n and f"{filter}" in file_n and "AlignedColour" in file_n): light_X_Files.append(file_n)
    if(not silent): print(f"Verification ({len(light_X_Files)} files read): light{filter}Files = {light_X_Files}")
    for file_n in light_X_Files:
        light_X_All.append(fits.open(light_dir + file_n)[0].data)
        header_X_All.append(fits.open(light_dir + file_n)[0].header)

    light_X_Stack = np.median(light_X_All, axis = 0)

    # Plot the resulting master bias frame.
    if(plot_flag == True):
        im_plot(light_X_Stack, title = f'Stacked {filter} frame', cbar_label = 'Counts', cbar_flag = True, stars_found = None, cmap = 'gray')
    fits.writeto(save_dir + filter + '_Stacked' + '.fits', light_X_Stack, header_X_All[0], overwrite = True)
    if(not silent): print(f'Stacked \"{filter}\" frame saved as FITS file to {save_dir + filter}_Stacked.fits')
    return(light_X_Stack)



def align_3_stacked(light_dir: str = "./", save_dir: str = "./", silent: bool = False) -> None:
    """
    Description:
        Aligns a set of light frames, from the specified filter colour.
    Args:
        light_dir (str): directory where the light files are located.
        save_dir (str): directory where the calibrated frames will be saved.
        filter (str): filter used to take the light frames.
    Outputs:
        None.
    """
    
    files = []
    for file_n in os.listdir(light_dir):
         if ("Stacked" in file_n and "Aligned" not in file_n): files.append(file_n)
    if(not silent): print(f'Files to align: {files}.')

    reference_image = fits.open(light_dir + files[0])[0].data + 0
    for i in range(0, len(files)):
        image_data   = fits.open(light_dir + files[i])
        source_image = image_data[0].data + 0  # Here, the addition of zero (0) solves the the endian compiler issue.
        header       = image_data[0].header
        image_aligned, footprint = aa.register(source_image, reference_image, detection_sigma = 2, fill_value = np.nan)
        aligned_file = files[i].replace('.fits', '')
        fits.writeto(save_dir + aligned_file + '_Aligned' + '.fits', image_aligned, header, overwrite = True)
        if(not silent): print('No. %i alignment done.' %i)



# Star finding/photometry functions
def GetFWHM_2D(data: np.ndarray, fwhm_estimate: float = 10.0):
    """
    Description:
        Estimates the FWHM of stars in a given image.
    Args:
        data (np.ndarray): image to be used.
        fwhm_estimate (float): initial guess for the FWHM of the stars in the image.
        plot_flag (bool): if True, plots the brightest stars used to estimate the FWHM.
    Outputs:
        xfwhm, yfwhm, fwhm, sigxfwhm, sigyfwhm, sigfwhm, medtheta (floats): FWHM's, Theta and uncertainties.
    """

    # Get background
    _, median, std = sigma_clipped_stats(data, sigma = 3.0, maxiters = 5)

    # Find stars
    daofind = DAOStarFinder(fwhm = fwhm_estimate, threshold = 5.0 * std) 
    sources = daofind(data - median)

    # Take 'nbright' brightest stars
    nbright = 10
    brightest = np.argsort(sources['flux'])[::-1][0:nbright]
    brsources = sources[brightest]

    # Fit the Gaussian PSF to brightest stars
    rmax = 25
    (ny, nx)           = np.shape(data)
    fit_g              = fitting.LevMarLSQFitter()
    allxfwhm, allyfwhm = np.zeros(len(brsources)), np.zeros(len(brsources))
    allfwhm, alltheta  = np.zeros(len(brsources)), np.zeros(len(brsources))
    for i, src in enumerate(brsources):
      if int(src['ycentroid']) > rmax and int(src['ycentroid']) < ny - rmax and \
         int(src['xcentroid']) > rmax and int(src['xcentroid']) < nx - rmax:
        img = data[int(src['ycentroid']) - rmax:int(src['ycentroid']) + rmax,
                   int(src['xcentroid']) - rmax:int(src['xcentroid']) + rmax]
        subx,suby   = np.indices(img.shape) # instead of meshgrid
        p_init      = models.Gaussian2D(amplitude = np.max(img), x_mean = rmax, y_mean = rmax, x_stddev = 1.0, y_stddev = 1.0)
        fitgauss    = fit_g(p_init, subx, suby, img - np.min(img))
        allxfwhm[i] = np.abs(fitgauss.x_stddev.value)
        allyfwhm[i] = np.abs(fitgauss.y_stddev.value)
        allfwhm[i]  = 0.5 * (allxfwhm[i] + allyfwhm[i])
        alltheta[i] = fitgauss.theta.value
    xfwhm, yfwhm    = np.median(allxfwhm) * gaussian_sigma_to_fwhm,np.median(allyfwhm) * gaussian_sigma_to_fwhm
    fwhm            = np.median(allfwhm) * gaussian_sigma_to_fwhm
    sigfwhm, sigxfwhm, sigyfwhm = np.std(allfwhm), np.std(allxfwhm), np.std(allyfwhm)
    medtheta = np.median(alltheta);
    return(xfwhm, yfwhm, fwhm, sigxfwhm, sigyfwhm, sigfwhm, medtheta)



def CreateStarsTable(data: np.ndarray, iterative: bool = False, filter: str = 'Red', exp_time: float = 60,   \
                     plot_bkg_flag: bool = False, plot_stars_flag: bool = False, fit_shape: tuple = (5, 5),  \
                     threshold_sig: float = 10.0, aperture_radius: float = 12.0, grouper_dist: float = 20.0, \
                     model_2D: bool = False, silent: bool = False) -> Table:
    """
    Description:
        Creates a table with the stars found in a given image.
    Args:
        data (np.ndarray): image to be used.
        iterative (bool): if True, uses iterative sigma clipping to estimate the FWHM.
        filter (str): filter used to take the light frames.
        exp_time (float): exposure time of the image.
        plot_bkg_flag (bool): if True, plots the background estimation.
        plot_stars_flag (bool): if True, plots the stars found.
        fit_shape (tuple): shape of the PSF model.
        threshold_sig (float): threshold used to find stars.
        aperture_radius (float): radius of the aperture used to find stars.
        grouper_dist (float): distance used to group stars.
        model_2D (bool): if True, uses a 2D gaussian model to find stars.
        silent (bool): if True, does not print anything.
    Outputs:
        phot (Table): table with the stars found.
    """

    # For M71:
    sec_z  = 1.07                      # Airmass
    # m_zp = 0.0                       # Zero-point magnitude: introduce actual value later!

    # Extinction coefficient values
    if(filter   == 'Red'):   k = 0.09; m_zp = 20.427; m_zp_err = 0.015
    elif(filter == 'Green'): k = 0.15; m_zp = 20.405; m_zp_err = 0.019
    elif(filter == 'Blue'):  k = 0.25; m_zp = 19.802; m_zp_err = 0.016
        

    # Get background
    sigma_clip    = SigmaClip(sigma = 3.0)
    bkg_estimator = MMMBackground(sigma_clip = sigma_clip)
    bkg           = Background2D(data, (50, 50), filter_size = (3, 3), sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)
    
    # Plotting results
    if(plot_bkg_flag == True):
        filterwarnings("ignore") # Ignore 'np.nan masking' warnings.

        # Matplotlib style ;)
        plt.style.use('https://github.com/kaiuki2000/PitayaRemix/raw/main/PitayaRemix.mplstyle')

        lo, up  = np.percentile(data[~np.isnan(data)], 1.5), np.percentile(data[~np.isnan(data)], 98.5)
        fig, ax = plt.subplots(1, 3, figsize = (12, 6), sharey = True)

        # Axes' labels
        ax[0].set_xlabel(r'$x$ pixel'); ax[1].set_xlabel(r'$x$ pixel'); ax[2].set_xlabel(r'$x$ pixel')
        ax[0].set_ylabel(r'$y$ pixel')

        # Actual plotting
        im0     = ax[0].imshow(data, cmap = 'gray', origin = 'lower', clim = (lo, up))
        divider = make_axes_locatable(ax[0])
        cax     = divider.append_axes('right', size = '5%', pad = 0.05)
        fig.colorbar(im0, cax = cax, orientation = 'vertical')

        im1     = ax[1].imshow(bkg.background, cmap = 'gray', origin = 'lower', clim = (lo, up))
        divider = make_axes_locatable(ax[1])
        cax     = divider.append_axes('right', size = '5%', pad = 0.05)
        fig.colorbar(im1, cax = cax, orientation = 'vertical')

        lo, up  = np.percentile(data[~np.isnan(data)] - bkg.background[~np.isnan(data)], 1.5), np.percentile(data[~np.isnan(data)] - bkg.background[~np.isnan(data)], 98.5)
        im2     = ax[2].imshow(data - bkg.background, cmap = 'gray', origin = 'lower', clim = (lo, up))
        divider = make_axes_locatable(ax[2])
        cax     = divider.append_axes('right', size = '5%', pad = 0.05)
        fig.colorbar(im2, cax = cax, orientation = 'vertical')

        # Title
        fig.suptitle(f"2D MMMBackground estimation ({filter} filter)")
        fig.tight_layout()
        fig.subplots_adjust(top = 1.5)
        plt.show()

    data     -= bkg.background # Subtract background from image
    fit_shape = fit_shape      # Fit shape of the PSF model (9x9 pixel grid)

    xfwhm, yfwhm, fwhm, sigxfwhm, sigyfwhm, sigfwhm, medtheta   = GetFWHM_2D(data, fwhm_estimate = 10.0)
    if(not silent): print(f'Obtained FWHM values: xfwhm = {xfwhm}, yfwhm = {yfwhm}.')

    if(model_2D == False):
        psf_model = IntegratedGaussianPRF(flux = 1, sigma = fwhm / gaussian_sigma_to_fwhm)            # Sigma = FWHM / 2.355 (FWHM ~ 10 taken from ds9!)
        daofind   = DAOStarFinder(fwhm = fwhm, threshold = threshold_sig * bkg.background_rms_median) # If we assume bkg_rms = std, then threshold = 1.0 * std.
        psfphot   = PSFPhotometry(psf_model = psf_model, fit_shape = fit_shape, finder = daofind, \
                                  localbkg_estimator = None, aperture_radius = aperture_radius, \
                                  grouper = SourceGrouper(grouper_dist)) # This also assumes <bkg> = 0 (Average = 0).
        phot  = psfphot(data); phot = phot[phot['flags'] == 0]            # Create a mask to exclude flagged sources
        


        # Plotting residuals and model image.
        resid   = psfphot.make_residual_image(data, (35, 35))
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        norm    = simple_norm(data, 'sqrt', percent=98)
        ax[0].imshow(data, origin='lower', norm=norm)
        ax[1].imshow(data - resid, origin='lower', norm=norm)
        im      = ax[2].imshow(resid, origin='lower')
        ax[0].set_title('Data')
        ax[1].set_title('Model')
        ax[2].set_title('Residual Image')
        plt.tight_layout()



        if(iterative):
            # Take 'nbright' brightest stars and print list of their fluxes
            nbright   = 10
            brightest = np.argsort(phot['flux_fit'])[::-1][0:nbright]
            brsources = phot[brightest]
            pos       = (brsources['x_init'], brsources['y_init'])

            # Initial parameters for IterativePSFPhotometry
            init_params      = QTable()
            init_params['x'] = pos[0]
            init_params['y'] = pos[1]

            # Performing IterativePSFPhotometry
            psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder = daofind, localbkg_estimator = None, \
                                             aperture_radius = aperture_radius)
            phot    = psfphot(data); phot = phot[phot['flags'] == 0]
        
    else:
        filterwarnings("ignore") # Ignore deprecation warnings
        two2dgauss  = Gaussian2D(x_mean = 0.0, y_mean = 0.0, theta = medtheta, \
                                 x_stddev = xfwhm/gaussian_sigma_to_fwhm, y_stddev = yfwhm/gaussian_sigma_to_fwhm)
        psf_model   = prepare_psf_model(two2dgauss, xname = 'x_mean', yname = 'y_mean', fluxname = 'amplitude') 
        daofind     = DAOStarFinder(fwhm = fwhm, threshold = threshold_sig * bkg.background_rms_median)
        if(not iterative):
            psfphot = BasicPSFPhotometry(psf_model = psf_model, fitshape = fit_shape, finder = daofind, bkg_estimator = None,
                                         aperture_radius = aperture_radius, group_maker = DAOGroup(grouper_dist)) # DAOGroup() groups stars that are closer than 'grouper_dist' pixels.
            phot    = psfphot(data)

            # Masking stars that are outside the edges of the image.
            mask_lo_x = phot['x_fit'] > 0.0; mask_up_x = phot['x_fit'] < 1536.0
            mask_lo_y = phot['y_fit'] > 0.0; mask_up_y = phot['y_fit'] < 1024.0
            phot = phot[mask_lo_x & mask_up_x & mask_lo_y & mask_up_y]
        else:
            psfphot = IterativelySubtractedPSFPhotometry(psf_model = psf_model, fitshape = fit_shape, \
                                                         finder = daofind, bkg_estimator = None, \
                                                         aperture_radius = aperture_radius, \
                                                         group_maker = DAOGroup(grouper_dist))
            phot    = psfphot(data)

            # Masking stars that are outside the edges of the image.
            mask_lo_x = phot['x_fit'] > 0.0; mask_up_x = phot['x_fit'] < 1536.0
            mask_lo_y = phot['y_fit'] > 0.0; mask_up_y = phot['y_fit'] < 1024.0
            phot = phot[mask_lo_x & mask_up_x & mask_lo_y & mask_up_y]

    if(not silent): print(f'Found {len(phot)} stars in image ({filter} filter).')
    if(plot_stars_flag == True):
        im_plot(data, title = f'Image with {len(phot)} stars found ({filter} filter)', cbar_label = 'Counts', cbar_flag = True, stars_found = phot, cmap = 'viridis')

    # Adding magnitude column to Table.
    phot['mag']     = -2.5 * np.log10(phot['flux_fit']/exp_time) - k * sec_z + m_zp     # We're missing m_zp, from the standard star!
    if(model_2D == False): phot['mag_err'] = np.sqrt( (2.5 * 1/phot['flux_fit'] * 1/np.log(10) * phot['flux_err'])**2 + \
                                (-k * 0.01)**2 + (-sec_z * 0.01)**2 + m_zp_err**2 )

    # Print the table in ascending order of magnitudes, using sorted()
    # Different possible sorting criteria.
    def get_mag(Table):
       return Table['mag']
    def get_qfit(Table):
       return Table['qfit']
    def get_x_fit(Table):
       return Table['x_fit']

    return(Table(rows = sorted(phot, key = get_x_fit)[:], names = phot.colnames))



def GenerateHR(blue_table: Table, green_table: Table, match_dist: float = 1.0, \
               silent = False, plot_flag = True, object: str = 'M39', \
               filters: tuple = ('B', 'V'), error_bars_flag = True, par_flag = False) -> tuple:
    """
    Description:
        Generates a Herzsprung-Russell diagram from two tables of stars (from two different filters, B and V).
    Args:
        blue_table (Table): table of stars from the blue filter.
        green_table (Table): table of stars from the green filter.
        match_dist (float): maximum distance between stars to be considered a match.
        silent (bool): if True, does not print anything.
        plot_flag (bool): if True, plots the resulting Herzsprung-Russell diagram.
        object (str): name of the object ('M39' by default).
        filters (tuple): tuple of strings with the names of the filters used.~
        error_bars_flag (bool): if True, plots error bars.
        par_flag (bool): if True, outputs parallaxes as well.
    Outputs:
        BV_List (list): list of tuples (B-V, B) for each matched star. (Or other filters, e.g., V-R)
        par_List (list): list of parallaxes for each matched star.
        BV_err (list): list of errors in B-V for each matched star.
        V_err (list): list of errors in V for each matched star.
    """
    
    # Star matching (for B-V computations) algorithm:
    BV_List, par_List = [], []
    BV_err, V_err     = [], []

    for Row1 in green_table:
        for Row2 in blue_table:
            if(abs(Row2['x_fit'] - Row1['x_fit']) < match_dist and abs(Row2['y_fit'] - Row1['y_fit']) < match_dist):
                if(not silent): print(f"Match!: id_b = {Row2['id']:3.0f}, id_g = {Row1['id']:3.0f}, \
abs(delta_x) = {abs(Row2['y_fit'] - Row1['y_fit']):.4f}, abs(delta_y) = {abs(Row2['y_fit'] - Row1['y_fit']):.4f}, \
mag_B = {Row2['mag']:.4f}, mag_G = {Row1['mag']:.4f}, B-V = {Row2['mag'] - Row1['mag']:.4f}")
                BV_List.append((Row2['mag'] - Row1['mag'], Row2['mag'])) # (B-V, B)
                if(par_flag): par_List.append((Row1['parallax'] + Row2['parallax'])/2.)
                if(error_bars_flag == True): 
                    BV_err.append(np.sqrt( Row2['mag_err']**2 + Row1['mag_err']**2 ))
                    V_err.append(Row1['mag_err'])
                blue_table = blue_table[blue_table['id'] != Row2['id']]
                continue
    print(f'{len(BV_List)} matches found.')

    # Plotting the resulting Herzsprung-Russell diagram
    if(plot_flag == True):
        def colorFader(c1, c2, mix = 0):    # Fade (linear interpolate) from color c1 (at mix = 0) to c2 (mix = 1)
            c1 = np.array(mpl.colors.to_rgb(c1))
            c2 = np.array(mpl.colors.to_rgb(c2))
            return np.array([mpl.colors.to_hex((1 - mix[i]) * c1 + mix[i] * c2) for i in range(len(mix))])
        
        # Sigmoid scaling function
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        c1 = '#3266a8' # Blue
        c2 = '#a85f32' # Orange

        plt.scatter([x[0] for x in BV_List], [y[1] for y in BV_List], label = 'Data points', s = 5, \
                    color = colorFader(c1, c2, np.array([sigmoid(x*2.5) for x in [x[0] for x in BV_List]])))
        if(error_bars_flag == True):
            plt.errorbar([x[0] for x in BV_List], [y[1] for y in BV_List], xerr = BV_err, yerr = V_err, fmt = 'none', \
                          ecolor = colorFader(c1, c2, np.array([sigmoid(x*2.5) for x in [x[0] for x in BV_List]])), \
                          elinewidth = 0.5)

        plt.gca().invert_yaxis()
        plt.xlim(-2.0, 2.0)      # This is here for easier comparison with the literature.
        plt.ylim(20, 6)          # Same for this.
        plt.xlabel(f'{filters[0]} - {filters[1]}')
        plt.ylabel(f'{filters[0]}')
        plt.title(f'{object} Hertzsprung-Russell Diagram')
        plt.legend()
        plt.savefig("H-R_Diagram_M39.png", dpi = 600)
        plt.show()

    return(BV_List, par_List, BV_err, V_err)



def match_gaia(sources, header, ra, dec, width = 0.2, height = 0.2):
    """
    Description:
        Function that matches the stars in the image with the Gaia catalog.
    Parameters:
        sources: array of the sources' coordinates in the image
        header: header of the image
        ra: right ascension of the center of the image
        dec: declination of the center of the image
        width: width of the image
        height: height of the image
    Returns:
        star_ra: right ascension of the stars
        star_dec: declination of the stars
        star_par: parallax of the stars
        star_parer: parallax error of the stars
        star_dist: distance of the stars
        star_pmra: proper motion in right ascension of the stars
        star_pmdec: proper motion in declination of the stars
        matched_stars: Gaia catalog around the center of the image (matched stars).
    """

    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source" # edr3 or dr2
    Gaia.ROW_LIMIT       = -1
    
    # Get ra,dec from WCS astrometry header
    wcs_header = WCS(header)
    coords     = wcs_header.pixel_to_world(sources[0], sources[1])

    # Get Gaia catalog around center ra/dec values
    cencoord      = SkyCoord(ra = ra, dec = dec, unit = (u.deg, u.deg), frame = 'icrs')
    width, height = u.Quantity(width, u.deg), u.Quantity(height, u.deg)
    gaia_stars    = Gaia.query_object_async(coordinate = cencoord, width = width, height = height)
    gaia_coords   = SkyCoord(ra = gaia_stars['ra'], dec = gaia_stars['dec'])

    # Match catalogs
    gidx, gd2d, gd3d = coords.match_to_catalog_sky(gaia_coords)
    gbestidx         = (gd2d.deg < 0.0008) #<0.00015deg=0.54''

    # Output variables
    star_ra, star_dec           = np.zeros(len(sources[0]), dtype = float), np.zeros(len(sources[0]), dtype = float)
    star_ra[:], star_dec[:]     = np.nan, np.nan
    star_ra[gbestidx]           = gaia_stars['ra'][gidx[gbestidx]]
    star_dec[gbestidx]          = gaia_stars['dec'][gidx[gbestidx]]
    
    # Stars' distances
    star_dist                   = np.zeros(len(sources[0]), dtype = float)
    star_dist[:]                = np.nan
    star_dist[gbestidx]         = gaia_stars['dist'][gidx[gbestidx]]

    # Proper motions
    star_pmra, star_pmdec       = np.zeros(len(sources[0]), dtype = float), np.zeros(len(sources[0]), dtype = float)
    star_pmra[:], star_pmdec[:] = np.nan, np.nan
    star_pmra[gbestidx]         = gaia_stars['pmra'][gidx[gbestidx]]
    star_pmdec[gbestidx]        = gaia_stars['pmdec'][gidx[gbestidx]]

    # Stars' parallaxes
    star_par, star_parer        = np.zeros(len(sources[0]), dtype = float) * np.nan, np.zeros(len(sources[0]), dtype = float) * np.nan
    star_par[gbestidx]          = gaia_stars['parallax'][gidx[gbestidx]]
    star_parer[gbestidx]        = gaia_stars['parallax_error'][gidx[gbestidx]]

    matched_stars               = gaia_stars[gidx[gbestidx]]

    return star_ra, star_dec, star_par, star_parer, star_dist, star_pmra, star_pmdec, matched_stars



def GaiaEllipseMask_pm(astrometry_image, center: tuple, table: QTable, plot_flag = True, n_std = 1.0, \
                       silent = False, xlim = None, ylim = None, filter = 'Red'):
    """
    Description:
        Function that masks the stars that were not matched with the Gaia catalogue.
        Also uses a confidence ellipse to mask further stars that are (probably) not part of the cluster.
    Parameters:
        astrometry_image: image with the astrometry
        center: center of the image
        table: table with the stars' data
        plot_flag: if True, plots the results
        n_std: number of standard deviations
        silent: if True, does not print anything
        xlim: x axis limits
        ylim: y axis limits
    Returns:
        table_masked: table with the stars that were matched with the Gaia catalogue
        table_unmasked: table with the stars that were not matched with the Gaia catalogue
    """

    # Running the matching function:
    stars_found            = np.array((table['x_fit'], table['y_fit']))
    header                 = astrometry_image.header
    sources                = stars_found
    ra, dec, width, height = center[0], center[1], 0.2, 0.2

    star_ra, star_dec, star_par, _, _, star_pmra, star_pmdec, _ = match_gaia(sources, header, ra, dec, width, height)

    # Count number of matches
    if(not silent): print(f'Number of matches with Gaia catalogue = {np.count_nonzero(~(np.isnan(star_ra) | np.isnan(star_dec)))}.')

    # Masking the stars that were not matched with the Gaia catalogue.
    pm_ra  = star_pmra[~(np.isnan(star_pmra) | np.isnan(star_pmdec))]
    pm_dec = star_pmdec[~(np.isnan(star_pmra) | np.isnan(star_pmdec))]

    plt.figure()
    ax = plt.gca()

    # Define your 'nσ-confidence-ellipse' parameters:
    n_std      = n_std # Same as n_σ.
    cov        = np.cov(pm_ra, pm_dec)
    lambda_, v = np.linalg.eig(cov)
    lambda_    = np.sqrt(lambda_)

    ellipse_center_pmra  = np.mean(pm_ra)
    ellipse_center_pmdec = np.mean(pm_dec)
    major_axis           = lambda_[0]*2*n_std # 2a
    minor_axis           = lambda_[1]*2*n_std # 2b
    angle_deg            = np.rad2deg(np.arccos(v[0, 0]))
    angle                = np.radians(angle_deg)

    ellipse = Ellipse(xy = (ellipse_center_pmra, ellipse_center_pmdec), width = major_axis, height = minor_axis, 
                            edgecolor = 'C3', fc = 'None', lw = 0.5, angle = angle_deg, \
                            label = rf'Confidence ellipse (${n_std} \sigma$)')
    ax.add_patch(ellipse)

    # We only work with the stars that were matched with the Gaia catalogue.
    mask   = ~(np.isnan(star_pmra) | np.isnan(star_pmdec))
    pm_ra  = star_pmra [mask]
    pm_dec = star_pmdec[mask]
    par    = star_par[mask] # Parallaxes! Needed for distance modulus.

    # Calculate the proper motion ellipse
    ellipse = (( ( (pm_ra  - ellipse_center_pmra) * np.cos(angle) + (pm_dec - ellipse_center_pmdec)* np.sin(angle) )** 2  / (major_axis/2.)** 2 +
                 ( (pm_ra  - ellipse_center_pmra) * np.sin(angle) - (pm_dec - ellipse_center_pmdec)* np.cos(angle) )** 2  / (minor_axis/2.)** 2 ) <= 1)

    # Plotting the pmRA vs pmDEC graph:
    masked_ra            = pm_ra[ellipse]
    masked_dec           = pm_dec[ellipse]
    masked_par           = par[ellipse] # Parallaxes! Needed for distance modulus.

    # Plotting results.
    if(plot_flag == True):
        plt.plot(masked_ra, masked_dec, 'o', markersize = 1.0, label = 'Accepted data points')
        plt.plot(pm_ra[~ellipse], pm_dec[~ellipse], 'o', markersize = 1.0, label = 'Rejected data points')
        plt.title(rf'Gaia catalogue matching: Proper motions in RA ($\alpha$) vs DEC ($\delta$) [{filter} filter]')
        if(xlim != None and ylim != None):
            plt.xlim(xlim[0], xlim[1])
            plt.ylim(ylim[0], ylim[1])
        plt.xlabel(r'Proper motion $\alpha$ [mas/yr]')
        plt.ylabel(r'Proper motion $\delta$ [mas/yr]')
        plt.legend()
        plt.show()

    table_masked               = (table[mask])[ellipse]
    table_masked['parallax']   = masked_par

    table_unmasked             = (table[mask])[~ellipse]
    table_unmasked['parallax'] = par[~ellipse]
    return table_masked, table_unmasked



def isochrone_fitter(iso_file: str, data_x, data_y, data_x_err, data_y_err, \
                     dists: tuple = (600, 8000, 200), N: int = 150, \
                     mags: tuple = ('B', 'V'), object: str = 'M71', \
                     mag_lims: tuple = (21, 2), silent: bool = True, \
                     globular_method: bool = False, old_flag: bool = False) -> None:
    """
    Description
    -----------
    This function fits an isochrone to a given cluster, using the data from the HR diagram.
    It uses the data from the Gaia catalogue, and the isochrones from the PARSEC database.
    The isochrones are given in absolute magnitudes, while the data come in apparent.
    The function returns the distance modulus and the distance in parsecs.

    Parameters
    ----------
    iso_file : str
        The name of the file containing the isochrones.
    data_x : array-like
        The x-axis data, in this case the color of the stars.
    data_y : array-like
        The y-axis data, in this case the magnitude of the stars.
    data_x_err : array-like
        The error in the x-axis data.
    data_y_err : array-like
        The error in the y-axis data.
    dists : tuple, optional
        The minimum and maximum distance to be tested, and the step size.
        The default is (600, 8000, 200).
    N : int, optional
        The number of points to be used in the isochrone.
        The default is 150.
    mags : tuple, optional
        The magnitudes to be used in the isochrone.
        The default is ('B', 'V').
    object : str, optional
        The name of the cluster.
        The default is 'M71'.
    mag_lims : tuple, optional
        The limits of the y-axis.
        The default is (21, 2).
    silent : bool, optional
        If True, the function will not print anything.
        The default is True.
    globular_method : bool, optional
        If True, the function will use the "globular" cluster method (2D-histogram based).
        The default is False.
    old_flag : bool, optional
        If True, the function will only use 'old' ages (> 10Gyrs).
        The default is False.
    
    Returns
    -------
    None.
    """

    # Coloring scheme: Aesthetics only.
    # Sigmoid scaling function
    def colorFader(c1, c2, mix = 0):    # Fade (linear interpolate) from color c1 (at mix = 0) to c2 (mix = 1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return np.array([mpl.colors.to_hex((1 - mix[i]) * c1 + mix[i] * c2) for i in range(len(mix))])

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    c1 = '#3266a8' # Blue
    c2 = '#a85f32' # Orange
    
    # Read the models
    # Important: Open the model file (in any text editor) and remove the # at the beginning of the line 12 (names of the columns)
    # Otherwise python won't know the column names and will call them col1, col2 etc.
    isochrones = Table.read(iso_file, format='ascii', guess = False, fast_reader = False)
    if(not silent): print(isochrones.columns)

    # Let's see which ages we have in the model file
    logages = np.unique(isochrones['logAge'])         # Find the unique age entries
    ages    = np.unique(10**isochrones['logAge']/1e6) # in Myrs
    for logage, age in zip(logages,ages):
        if(not silent): print(f'Log_Age = {logage:.5f}, Age = {age:.5f} Myrs.') # print all of that

    # All the ages that we will test
    ages = np.unique(isochrones['logAge'])

    # Distance
    dmin, dmax, step = dists[0], dists[1], dists[2]
    distances        = np.arange(dmin, dmax, step) 

    if(globular_method == False):
        # Define an array to save the root-mean-square deviation values
        rmsd = np.zeros(shape = (len(ages), len(distances)))
        for i in range(len(ages)):
            age = ages[i]
            for j in range(len(distances)):

                # Model
                distance  = distances[j]
                DM        = 5 * np.log10(distance) - 5                   # Distance modulus
                isochrone = isochrones[isochrones['logAge'] == age][0:N]
                col_iso   = isochrone[f'{mags[0]}mag'] - isochrone[f'{mags[1]}mag']        # Color isochrone
                mag_iso   = isochrone[f'{mags[0]}mag'] + DM                       # Magnitude isochrone, shifted to the distance of the cluster
                line      = LineString(np.asarray([col_iso, mag_iso]).T) # Representation of the isochrone as a continuous line 

                # Data
                d = np.empty(len(data_x))
                for k in range(len(data_x)):
                    col_data = data_x[k]
                    mag_data = data_y[k]
                    point    = Point(col_data, mag_data)
                    d[k] = point.distance(line) # Shortest distance of the point to the line of the isochrone
                d = np.ma.masked_invalid(d)
                rmsd[i, j] = np.sqrt(np.nanmean(d**2)) 

        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 7))
        fig.suptitle(f'{object} - Isochrone fit', fontsize = 16)
        fig.subplots_adjust(top = 0.925)
        fig.subplots_adjust(wspace = 0.1)

        pos     = ax[0].imshow(rmsd, cmap = 'PiYG', norm = LogNorm(), origin = 'lower',
                               extent = [distances[0], distances[-1], 10**ages[0]/1e6, 10**ages[-1]/1e6], aspect = 'auto', \
                               interpolation = 'gaussian')
        fig.colorbar(pos, ax = ax[0]) # , format = "%d")

        # Find the grid position of the minimum rmsd
        minrmsd_pos = np.unravel_index(rmsd.argmin(), rmsd.shape)
        print(f'Best find indices: [i, j] = [{minrmsd_pos[0]}, {minrmsd_pos[1]}]. Rmsd = {np.nanmin(rmsd)}.')
        print("Best fit model:    age =", np.round(10**ages[minrmsd_pos[0]]/1e6),'Myr; distance =', distances[minrmsd_pos[1]], 'pc.')
        if(object == 'M71'): print("Literature values: age = 9-10 Gyr; distance = 4000 pc.")
        if(object == 'M39'): print("Literature values: age = 278.6 Myr; distance = 311 pc.")
        best_age  = ages[minrmsd_pos[0]]
        best_dist = distances[minrmsd_pos[1]]
        ax[0].set_xlabel('Distance in pc', fontsize = 15)
        ax[0].set_ylabel('Age in Myr', fontsize = 15)

        # Let's now plot the best-fit model on top of our data
        ax[1].scatter( data_x, data_y, s = 2.5, \
                       color = colorFader(c1, c2, np.array([sigmoid(x*2.5) for x in data_x])), label = 'Data points')
        ax[1].errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, fmt = 'none', \
                     ecolor = colorFader(c1, c2, np.array([sigmoid(x*2.5) for x in data_x])), \
                     elinewidth = 0.5)
        ax[1].set_ylim(mag_lims[0], mag_lims[1])
        ax[1].set_xlabel(f'{mags[0]} - {mags[1]}', fontsize = 14)
        ax[1].set_ylabel(f'{mags[0]}', fontsize = 14)

        # Important: isochrones are given in absolute magnitudes, while the data come in apparent
        # median_parallax = np.nanmedian(par_List)
        DM = 5 * np.log10(best_dist) - 5        # Distance modulus

        age_1 = isochrones['logAge'] == best_age
        ax[1].plot(isochrones[f'{mags[0]}mag'][age_1][0:N] - isochrones[f'{mags[1]}mag'][age_1][0:N], isochrones[f'{mags[0]}mag'][age_1][0:N] + DM, \
                 label = str(np.round(10**best_age/1e6)) + ' Myr', color = 'C4', linewidth = 1.0)
        ax[1].legend()
        plt.tight_layout()
        plt.show()

        print(f'Distance modulus: {DM:.5f}')
        print(f'Distance in pc:   {best_dist:.3f}') # Distance in parsecs
    else:
        bin_size              = 0.1
        min_edge_x,max_edge_x = -1.5 , 1.5
        min_edge_y,max_edge_y = 7.5 , 17.5
        bin_num_x             = int((max_edge_x-min_edge_x) / bin_size)
        bin_num_y             = int((max_edge_y-min_edge_y) / bin_size)
        bins_x                = np.linspace(min_edge_x, max_edge_x, bin_num_x + 1)
        bins_y                = np.linspace(min_edge_y, max_edge_y, bin_num_y + 1)

        col_data = data_x
        mag_data = data_y

        h2d, xedges, yedges   = np.histogram2d(mag_data, col_data, bins = (bins_y, bins_x))
        fig, ax               = plt.subplots(nrows = 1, ncols = 3, width_ratios = [1, 2.5, 2.5], figsize = (16.8, 7))
        fig.suptitle(f'{object} - Isochrone fit: Using \"Globular\" method', fontsize = 16)
        fig.subplots_adjust(top = 0.925)
        fig.subplots_adjust(wspace = 0.1)
        
        ax[0].imshow(h2d, norm   = LogNorm(vmin = 0.01, vmax = 100), extent = [min_edge_x, max_edge_x, min_edge_y, max_edge_y], \
                  origin = 'lower')
        ax[0].invert_yaxis()
        ax[0].set_xlabel(f'{mags[0]} - {mags[1]}'), ax[0].set_ylabel(f'{mags[0]}')
        ax[0].set_title('2D Histogram of the data')
        # 'bins_x' and 'bins_y' are the edegs of the histogram bins
        x = (bins_x + bin_size / 2)[0:-1]

        # Actual 'fitting' starts here.
        if(old_flag == True): ages = ages[ages >= 10] # Selecting only 'old' ages.
        
        # Values of the histogram for the fitting
        x      = (bins_x + bin_size/2)[0:-1] # colors
        y      = (bins_y + bin_size/2)[0:-1] # magnitudes
        weight = h2d

        # Define an array to save the root-mean-square deviation values
        rmsd = np.zeros(shape = (len(ages), len(distances)))
        for i in range(len(ages)):
            age = ages[i]
            for j in range(len(distances)):
                distance = distances[j]
                DM = 5 * np.log10(distance) - 5 # distance modulus
                isochrone = isochrones[isochrones['logAge'] == age][0:N]
                col_iso   = isochrone[f'{mags[0]}mag'] - isochrone[f'{mags[1]}mag'] # Color isochrone
                mag_iso   = isochrone[f'{mags[0]}mag'] + DM                         # Magnitude isochrone, shifted to the distance of the cluster
                line      = LineString(np.asarray([col_iso,mag_iso]).T) # Representation of the isochrone as a continuous line 

                d     = np.empty(len(x)*len(y))
                w     = np.empty(len(x)*len(y))
                count = 0
                for k_x in range(len(x)):
                    for k_y in range(len(y)):
                        col_data = x[k_x]
                        mag_data = y[k_y]

                        point    = Point(col_data, mag_data)
                        d[count] = point.distance(line) # Shortest distance of the point to the isochrone
                        w[count] = weight[k_y,k_x]
                        count    = count + 1

                rmsd[i, j] = np.sqrt(np.average(d**2, weights = w)) 

        pos = ax[1].imshow(rmsd, cmap = 'PiYG', norm = LogNorm(), origin = 'lower', \
                           extent = [distances[0], distances[-1], 10**ages[0]/1e6, 10**ages[-1]/1e6], aspect = 'auto')
        fig.colorbar(pos, ax = ax[1]) # , format= "%d")

        # Find the grid position of the minimum rmsd
        minrmsd_pos = np.unravel_index(rmsd.argmin(), rmsd.shape)
        print(np.nanmin(rmsd), minrmsd_pos)
        print(f'Best find indices: [i, j] = [{minrmsd_pos[0]}, {minrmsd_pos[1]}]. Rmsd = {np.nanmin(rmsd)}.')
        print("Best fit model:    age =", np.round(10**ages[minrmsd_pos[0]]/1e6),'Myr; distance =', distances[minrmsd_pos[1]], 'pc.')
        if(object == 'M71'): print("Literature values: age = 9-10 Gyr; distance = 4000 pc.")
        if(object == 'M39'): print("Literature values: age = 278.6 Myr; distance = 311 pc.")
        best_age  = ages[minrmsd_pos[0]]
        best_dist = distances[minrmsd_pos[1]]
        ax[1].set_xlabel('Distance in pc', fontsize = 15)
        ax[1].set_ylabel('Age in Myr', fontsize = 15)

        # Let's now plot the best-fit model on top of our data
        ax[2].scatter( data_x, data_y, s = 2.5, \
                       color = colorFader(c1, c2, np.array([sigmoid(x*2.5) for x in data_x])), label = 'Data points')
        ax[2].errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, fmt = 'none', \
                     ecolor = colorFader(c1, c2, np.array([sigmoid(x*2.5) for x in data_x])), \
                     elinewidth = 0.5)
        ax[2].set_ylim(mag_lims[0], mag_lims[1])
        ax[2].set_xlabel(f'{mags[0]} - {mags[1]}', fontsize = 14)
        ax[2].set_ylabel(f'{mags[0]}', fontsize = 14)

        # Important: isochrones are given in absolute magnitudes, while the data come in apparent
        # median_parallax = np.nanmedian(par_List)
        DM = 5 * np.log10(best_dist) - 5        # Distance modulus

        age_1 = isochrones['logAge'] == best_age
        ax[2].plot(isochrones[f'{mags[0]}mag'][age_1][0:N] - isochrones[f'{mags[1]}mag'][age_1][0:N], isochrones[f'{mags[0]}mag'][age_1][0:N] + DM, \
                 label = str(np.round(10**best_age/1e6)) + ' Myr', color = 'C4', linewidth = 1.0)
        ax[2].legend()
        plt.tight_layout()
        plt.show()

        print(f'Distance modulus: {DM:.5f}')
        print(f'Distance in pc:   {best_dist:.3f}') # Distance in parsecs