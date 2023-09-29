import os
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
import astroalign as aa
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.modeling.models import Gaussian2D
from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm, SigmaClip
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MMMBackground # Can also use 'MedianBackground'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.psf import PSFPhotometry, IterativePSFPhotometry, BasicPSFPhotometry, IterativelySubtractedPSFPhotometry, IntegratedGaussianPRF, SourceGrouper, prepare_psf_model, DAOGroup
from astropy.table import Table, QTable

def im_plot(data, title = 'Default title', cbar_label = 'Counts', cbar_flag = True, stars_found = None, cmap = 'RdPu', lo: float = 1.5, up: float = 98.5) -> None:
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
def masterBiasCreator(bias_dir: str = "./", plot_flag: bool = False, save_dir: str = "./", silent: bool = False) -> np.ndarray:
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



def masterFlatCreator(masterBias: np.ndarray, flat_dir: str = './', plot_flag: bool = False, save_dir: str = "./", filter: str = 'Red', silent: bool = False) -> np.ndarray:
    """
    Description:
        Creates a master flat frame from a set of flat frames.
    Args:
        flat_dir (str): directory where the flat files are located.
        plot_flag (bool): if True, plots the resulting master flat frame.
        save_dir (str): directory where the master flat frame will be saved.
        filter (str): filter used to take the flat frames.
        silent (bool): if True, does not print anything.
    Outputs:
        masterFlat (np.ndarray): master flat frame.
    """

    # Status message.
    if(not silent): print(f'Creating master flat ({filter}) frame...')

    # Reading flat files, for filter X.
    flatX_Files, flatX_All, headersAll = [], [], []
    for file_n in os.listdir(flat_dir):
         if ("flat" and f"{filter}" in file_n): flatX_Files.append(file_n)
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



def generateCalibratedFrames(masterBias: np.ndarray, masterFlat: np.ndarray, light_dir: str = "./", save_dir: str = "./", filter: str = "Red", d: list = [2, 4, 2], object: str = "M71", silent: bool = False) -> None:
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
        object (str): name of the object.
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



def align_colour_frames(light_dir: str = "./", save_dir: str = "./", filter: str = 'Red', object: str = "M71", silent: bool = False) -> None:
    """
    Description:
        Aligns a set of light frames, from the specified filter colour.
    Args:
        light_dir (str): directory where the light files are located.
        save_dir (str): directory where the calibrated frames will be saved.
        filter (str): filter used to take the light frames.
        object (str): name of the object.
    Outputs:
        None.
    """
    
    files = []
    for file_n in os.listdir(light_dir):
         if (object in file_n and f"{filter}" in file_n): files.append(file_n)
    if(not silent): print(f'Files to align: {files}.')

    reference_image = fits.open(light_dir + files[0])[0].data + 0
    for i in range(0, len(files)):
        image_data   = fits.open(light_dir + files[i])
        source_image = image_data[0].data + 0  # Here, the addition of zero (0) solves the the endian compiler issue.
        header       = image_data[0].header
        image_aligned, footprint = aa.register(source_image, reference_image)
        aligned_file = files[i].replace('.fits', '')
        fits.writeto(save_dir + aligned_file + '_AlignedColour' + '.fits', image_aligned, header, overwrite = True)
        if(not silent): print('No. %i alignment done.' %i)



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
        image_aligned, footprint = aa.register(source_image, reference_image)
        aligned_file = files[i].replace('.fits', '')
        fits.writeto(save_dir + aligned_file + '_Aligned' + '.fits', image_aligned, header, overwrite = True)
        if(not silent): print('No. %i alignment done.' %i)



def stack_colour_frames(light_dir: str = "./", save_dir: str = "./", filter: str = 'Red', object: str = "M71", silent: bool = False, plot_flag: bool = False) -> np.ndarray:
    """
    Description:
        Stacks a set of light frames, from the specified filter colour.
    Args:
        light_dir (str): directory where the light files are located.
        save_dir (str): directory where the calibrated frames will be saved.
        filter (str): filter used to take the light frames.
        object (str): name of the object.
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
    z    = 1.07                        # Airmass
    def sec(z): return 1.0 / np.cos(z) # sec(z) = 1/cos(z)
    # m_zp = 0.0                       # Zero-point magnitude: introduce actual value later!

    # Extinction coefficient values
    if(filter   == 'Red'):   k = 0.09; m_zp = 0.0 
    elif(filter == 'Green'): k = 0.15; m_zp = 0.0
    elif(filter == 'Blue'):  k = 0.25; m_zp = 0.0 
        

    # Get background
    sigma_clip = SigmaClip(sigma = 3.0)
    bkg_estimator = MMMBackground(sigma_clip = sigma_clip)
    bkg = Background2D(data, (50, 50), filter_size = (3, 3), sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)

    # Plotting results
    if(plot_bkg_flag == True):
        # Matplotlib style ;)
        plt.style.use('https://github.com/kaiuki2000/PitayaRemix/raw/main/PitayaRemix.mplstyle')

        lo, up = np.percentile(data, 1.5), np.percentile(data, 98.5)
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

        lo, up  = np.percentile(data - bkg.background, 1.5), np.percentile(data - bkg.background, 98.5)
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
        phot = psfphot(data); phot = phot[phot['flags'] == 0]            # Create a mask to exclude flagged sources
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
    phot['mag'] = -2.5 * np.log10(phot['flux_fit']/exp_time) - k * sec(z) + m_zp # We're missing m_zp, from the standard star!

    # Print the table in ascending order of magnitudes, using sorted()
    # Different possible sorting criteria.
    def get_mag(Table):
       return Table['mag']
    def get_qfit(Table):
       return Table['qfit']
    def get_x_fit(Table):
       return Table['x_fit']

    return(Table(rows = sorted(phot, key = get_x_fit)[:], names = phot.colnames))



def GenerateHR(blue_table: Table, green_table: Table, match_dist: float = 1.0, silent = False, plot_flag = True) -> tuple:
    """
    Description:
        Generates a Herzsprung-Russell diagram from two tables of stars (from two different filters, B and V).
    Args:
        blue_table (Table): table of stars from the blue filter.
        green_table (Table): table of stars from the green filter.
        match_dist (float): maximum distance between stars to be considered a match.
        silent (bool): if True, does not print anything.
        plot_flag (bool): if True, plots the resulting Herzsprung-Russell diagram.
    Outputs:
        BV_List (list): list of tuples (B-V, B) for each matched star.
    """
    
    # Star matching (for B-V computations) algorithm:
    BV_List = []
    for Row1 in green_table:
        for Row2 in blue_table:
            if(abs(Row2['x_fit'] - Row1['x_fit']) < match_dist and abs(Row2['y_fit'] - Row1['y_fit']) < match_dist):
                if(not silent): print(f"Match!: id_b = {Row2['id']:3.0f}, id_g = {Row1['id']:3.0f}, \
abs(delta_x) = {abs(Row2['y_fit'] - Row1['y_fit']):.4f}, abs(delta_y) = {abs(Row2['y_fit'] - Row1['y_fit']):.4f}, \
mag_B = {Row2['mag']:.4f}, mag_G = {Row1['mag']:.4f}, B-V = {Row2['mag'] - Row1['mag']:.4f}")
                BV_List.append((Row2['mag'] - Row1['mag'], Row2['mag'])) # (B-V, B)
                blue_table = blue_table[blue_table['id'] != Row2['id']]
                continue
    print(f'{len(BV_List)} matches found.')

    # Plotting the resulting Herzsprung-Russell diagram
    if(plot_flag == True):
        plt.scatter([x[0] for x in BV_List], [y[1] for y in BV_List], label = 'Data points')
        plt.gca().invert_yaxis()
        plt.ylabel('B')
        plt.xlabel('B-V')
        plt.title('Hertzsprung-Russell Diagram')
        plt.legend()
        plt.show()

    return(BV_List)