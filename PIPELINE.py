
#=======================================================================================================================================================#
#Lucky Imaging Data Reduction Pipeline for GUFI data 
#=======================================================================================================================================================#
'''
8/02/2019
Lucky Imaging Data Reduction Pipeline for GUFI
Takes in input data; bias files, flat files, and science files, and produces a final image that is sharper, lower fwhm, than an individual image
This is done by sorting the images based on the average fwhm. The average fhwm of each image is found by locating the stars in the image and then 
averaging the fwhm of each star. In order to get the visible stars in the field the images are first shifted and combined. The images are then sorted
using the average fhwm, taking the top 1%, lowest fwhm, of the images. Once the images have been sorted and selected, the remaining images are 
combined using the median values. '''

#======================================================================================================================================================#
#Used Libraries
#======================================================================================================================================================#

import time                                                                                            
import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits                                                                                   
import os                                                                                                      
import glob                                           
from astropy.stats import biweight_location

from astropy.nddata import Cutout2D
from scipy import optimize
from image_registration import chi2_shift
from image_registration.fft_tools import shift
from astropy import units as u
import astropy.nddata
from ccdproc import ccd_process#, cosmicray_lacosmic
from matplotlib.colors import LogNorm
from photutils.detection import DAOStarFinder





#=======================================================================================================================================================#
#Initialize Code
#=======================================================================================================================================================#

start = time.time()
def DATA_REDUCTION():   


#=======================================================================================================================================================#
#Detects whether or not a median, reduced, file exists for the bias and flat, for lucky imaging darks should not be needed as each individual exposure
#is very short. If a median file is not detected, the code will create the needed files
#=======================================================================================================================================================#
    
    def Median_Bias():
        if os.path.isfile('/Users/jamestaylor/Desktop/Test_Directory/bias/median_bias.fits') == False:
            file = glob.glob('/Users/jamestaylor/Desktop/Test_Directory/bias/*')
            j = 1
            for image in file:
                hdul = fits.open(image)
                hdr = hdul[0].header['naxis3']
                ccd_bias = fits.getdata(image)
                #Slices the data cube in such a way as to create a separate file from the designated slice
                for i in range(hdr):
                    print(i)
                    bias_slice = ccd_bias[i,:,:]
                    outfile = '/Users/jamestaylor/Desktop/Test_Directory/bias/bias_'+str(j)+'_'+str(i)+'.fits'
                    hdu = fits.PrimaryHDU(bias_slice)
                    hdu.writeto(outfile, overwrite = True)
                j+=1
          #Combines the newly created files into an array, gets the data from them, combines them into a median and then saves them as the master file
            files = glob.glob('/Users/jamestaylor/Desktop/Test_Directory/bias/bias_*')
            files_data = np.array([fits.getdata(image) for image in files])
            median = np.median(files_data, axis = 0)
            outfile = '/Users/jamestaylor/Desktop/Test_Directory/bias/median_bias.fits'
            hdu = fits.PrimaryHDU(median)
            hdu.writeto(outfile, overwrite = True)
            bias = astropy.nddata.CCDData.read('/Users/jamestaylor/Desktop/Test_Directory/bias/median_bias.fits', unit=u.electron)
        else:
            bias = astropy.nddata.CCDData.read('/Users/jamestaylor/Desktop/Test_Directory/bias/median_bias.fits', unit=u.electron)
        return bias

    def Median_Flat():
        if os.path.isfile('/Users/jamestaylor/Desktop/Test_Directory/flat/median_flat.fits') == False:
            file = glob.glob('/Users/jamestaylor/Desktop/Test_Directory/flat/flat.0*.fit')
            j = 1
            for image in file:
                hdul = fits.open(image)
                hdr = hdul[0].header['naxis3']
                ccd_flat = fits.getdata(image)
                for i in range(hdr):
                    print(i)
                    flat_slice = ccd_flat[i,:,:] / np.mean(ccd_flat)
                    outfile = '/Users/jamestaylor/Desktop/Test_Directory/flat/flat_'+str(j)+'_'+str(i)+'.fits'
                    hdu = fits.PrimaryHDU(flat_slice)
                    hdu.writeto(outfile, overwrite = True)
                j+=1
            files = glob.glob('/Users/jamestaylor/Desktop/Test_Directory/flat/flat_*')
            files_data = np.array([fits.getdata(image) for image in files])
            median = np.median(files_data, axis = 0)
            outfile = '/Users/jamestaylor/Desktop/Test_Directory/flat/median_flat.fits'
            hdu = fits.PrimaryHDU(median)
            hdu.writeto(outfile, overwrite = True)
            flat = astropy.nddata.CCDData.read('/Users/jamestaylor/Desktop/Test_Directory/flat/median_flat.fits', unit=u.electron)
        else:
            flat = astropy.nddata.CCDData.read('/Users/jamestaylor/Desktop/Test_Directory/flat/median_flat.fits', unit=u.electron)
        return flat
  #Splits science files from data cubes
    def SPLIT_DATA():
        if os.path.isfile('/Users/jamestaylor/Desktop/Test_Directory/SPLIT_2/split_1_0.fits') == False:
            file = glob.glob('/Users/jamestaylor/Desktop/Test_Directory/RAW/*')
            j = 1
            for image in file:
                #print('here')
                hdul = fits.open(image)
                hdr = hdul[0].header['naxis3']
                #print(hdr)
                ccd_raw = fits.getdata(image)
                #print('there')
                for i in range(hdr):
                    print(i)
                    raw_slice = ccd_raw[i,:,:]
                    outfile = '/Users/jamestaylor/Desktop/Test_Directory/SPLIT_2/split_'+str(j)+'_'+str(i)+'.fits'
                    hdu = fits.PrimaryHDU(raw_slice)
                    hdu.writeto(outfile, overwrite = True)
                j+=1
            science_split = glob.glob('/Users/jamestaylor/Desktop/Test_Directory/SPLIT_2/*')
            
        else:
            science_split = glob.glob('/Users/jamestaylor/Desktop/Test_Directory/SPLIT_2/*')
        return science_split
    SPLIT_DATA()
    
#=======================================================================================================================================================#
#Uses the ccdproc ccd_process in order to perform both bias subtraction and flat division in one step on each of the image files,cosmicray removal 
#is then done on the images(not sure if needed since taking the median), if the files already exist gather the files and store them in an array
#La_Cosmic Dokkum(2001), McCully(2014)
#=======================================================================================================================================================#
    
    def Reduced_Data():                                                                         
        if os.path.isfile('/Users/jamestaylor/Desktop/Test_Directory/science/reduced_data_1.fits') == False:
            science = glob.glob('/Users/jamestaylor/Desktop/Test_Directory/SPLIT_2/*')
            bias = Median_Bias()                
            flat = Median_Flat()
            i = 0
            flat = astropy.nddata.CCDData.read('/Users/jamestaylor/Desktop/Test_Directory/flat/median_flat.fits', unit=u.electron)
            bias = astropy.nddata.CCDData.read('/Users/jamestaylor/Desktop/Test_Directory/bias/median_bias.fits', unit=u.electron)
            for image in science:
                print('file: ',i)
                ccd = astropy.nddata.CCDData.read(image, unit=u.electron)
                #cr_subtract = cosmicray_lacosmic(ccd)
                reduction = ccd_process(ccd, master_flat = flat, master_bias = bias, add_keyword='Reduced')
                reduction_ = reduction / np.percentile(reduction, 99)
              
                outfile = '/Users/jamestaylor/Desktop/Test_Directory/science/reduced_data'+str(i)+'.fits'
                hdu = fits.PrimaryHDU(reduction_)
                hdu.writeto(outfile,overwrite = True)
                i += 1    
            reduced_data = np.array(glob.glob('/Users/jamestaylor/Desktop/Test_Directory/science/reduced_data*.fits'))
            
        else:
            reduced_data = np.array(glob.glob('/Users/jamestaylor/Desktop/Test_Directory/science/reduced_data*.fits'))
            print(reduced_data)
        return reduced_data
    reduced_data = Reduced_Data()
    return reduced_data
    
#=======================================================================================================================================================#
#Lucky Imaging: Sorts the images based on the fwhm and only takes the top 1%
#=======================================================================================================================================================#

#Shifts the images prior to detection in order to get location of stars for fwhm calculations
j = 0
median_array = []
def Pre_Shift():
    reduced_data = DATA_REDUCTION()
    
    if os.path.isfile('/Users/jamestaylor/Desktop/Test_Directory/Shifted_Images/shifted_sum.fits')==False:
        i = 0
        hdu = fits.open(reduced_data[0])
        data_ref = hdu[0].data
        for image in reduced_data:
            hdu = fits.open(image)
            img = hdu[0].data
            
            xoff, yoff, xerr, yerr = chi2_shift(data_ref,img, upsample_factor = 100)
            
            print(xoff, yoff)
            shifted_image = shift.shiftnd(img, (-xoff,-yoff))
            plt.imshow(shifted_image, cmap = 'gray', origin = 'lower')
            plt.show()
            outfile = '/Users/jamestaylor/Desktop/Test_Directory/Shifted_Images/chi2_shift_r1_'+str(i)+'.fits'
            hdu = fits.PrimaryHDU(shifted_image)
            hdu.writeto(outfile, overwrite = True)
            i+=1
        shifted_sum = 0
        shifts = np.array(glob.glob('/Users/jamestaylor/Desktop/Test_Directory/Shifted_Images/chi2_shift_r1*'))
        for image in shifts:
            print(image)
            hdu = fits.open(image)
            data = hdu[0].data
            shifted_sum += data
        outfile = '/Users/jamestaylor/Desktop/Test_Directory/Shifted_Images/shifted_sum.fits'
        hdu = fits.PrimaryHDU(shifted_sum)
        hdu.writeto(outfile, overwrite = True)
        hdu = fits.open('/Users/jamestaylor/Desktop/Test_Directory/Shifted_Images/shifted_sum.fits')
        shifted_sum = hdu[0].data
        
    else:
        shifts = np.array(glob.glob('/Users/jamestaylor/Desktop/Test_Directory/Shifted_Images/chi2_shift_r1*'))
        hdu = fits.open('/Users/jamestaylor/Desktop/Test_Directory/Shifted_Images/shifted_sum.fits')
        shifted_sum = hdu[0].data
    return shifts, shifted_sum, reduced_data
shifts, shifted_sum, reduced_data = Pre_Shift()
__all__ = ['find_stars']


def find_stars(data, n_sigma):
    """
    Find stars in an image using the DAOFIND (`Stetson 1987
    <http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_) algorithm.
    Parameters
    ----------
    data : 2D array_like
        The 2D image array.
    n_sigma : float
        The sigma threshold for detection.
    """
    # Determine background count central tendency
    mu = biweight_location(data)

    # Determine background count dispersion
    sigma = biweight_location(data)

    # Set the detection threshold to 5-sigma above background
    threshold = mu + n_sigma*sigma

    # Set the full-width half-maximum of the Gaussian kernel
    # applied to smooth the background
    # (Using the sqrt gives us roughly the right scale and
    # will scale for other image sizes)
    fwhm = np.sqrt(data.shape)[0]

    # Instantiate a finder from the DAOStarFinder class
    finder = DAOStarFinder(threshold, fwhm, sky=mu, exclude_border=True)

    # Apply the algorithm to the data
    table = finder.find_stars(data)

    return table


if __name__ == '__main__':
    # Test it out on two images
    

    file = '/Users/jamestaylor/Desktop/Test_Directory/Shifted_Images/shifted_sum.fits'
    hdu = fits.open(file)
    data = hdu[0].data
    # Load the data
    #data = pyfits.getdata(file)

# Find the stars
    stars = find_stars(data, 5)

# Print results
    #print(file)
    print(stars)
    i = 0
    guess_x, guess_y = stars['xcentroid'], stars['ycentroid']
    for line in stars:
        print(guess_x[i], guess_y[i])
        i+=1

    # Save images
    fig, ax = plt.subplots()
    im = ax.imshow(data, origin='lower',
                   interpolation='nearest', norm=LogNorm())
    ax.scatter(stars['xcentroid'], stars['ycentroid'],
               s=np.sqrt(data.shape)[0], color='none', edgecolor='white')
    plt.colorbar(im, label='Counts')
    ax.set_xlabel('x (pixel)')
    ax.set_ylabel('y (pixel)')
    plt.savefig('targets_median.pdf', bbox_inches='tight')
    plt.close()


for file in shifts:
    hdu = fits.open(file)
    data_guess = hdu[0].data
    
    
    __all__ = ['gaussian', 'fit_gaussian', 'get_fwhm', 'load_data', 'get_subimg']
        
    # Robust code to fit a 2D gaussian taken from#
    
    # https://scipy-cookbook.readthedocs.io/items/FittingData.html#
    
    def gaussian(height, center_x, center_y, width_x, width_y): #Produces a Gaussian function with the given parameters for the data#
        
        width_x = float(width_x)    
        width_y = float(width_y)    
        return lambda x,y: height*np.exp(
    
            -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
            
    def moments(data): # Returns the parameters (height, x, y, width_x, width_y), and calculates the moments#
           
        total = data.sum()    
        X, Y = np.indices(data.shape)    
        x = (X*data).sum()/total    
        y = (Y*data).sum()/total    
        col = data[:, int(y)]    
        width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())    
        row = data[int(x), :]    
        width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())    
        height = data.max()    
        return height, x, y, width_x, width_y    
        
    #Fits a Gaussian to the given data and optimizes using the error function to minimize the least square fit#
    def fit_gaussian(data): 
        
        params = moments(data)
        errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p
        
    #Calculates the FWHM for a chosen sub area of the image#
    def get_fwhm(data):
        
        params = fit_gaussian(data)
        #Convert the standard deviations of the 2D gaussian to FWHMs#
        sigmas = params[3:]    
        fwhms = 2*np.sqrt(2*np.log(2))*sigmas
        return fwhms.mean()
    
    #Read in a file#
    def load_data(file=file): 
        
        return fits.open(file)
    
    #Generates a sub-image of from the reference frame above#    
    def get_subimg(hdulist, center=None, size = 50):
  
        if center is None:    
            x_guess, y_guess = guess_x, guess_y    
            center = (x_guess, y_guess)
        subimg = Cutout2D(hdulist[0].data, (x_guess, y_guess), size=size)    
        return subimg
            
    #Runs a collection of the above functions in order to get the 2-D FWHM, returns the FWHM#
    def main(): 
  
        hdulist = load_data()    
        subimg = get_subimg(hdulist)    
        fwhm = get_fwhm(subimg.data)
        print(f'The FWHM is {fwhm} pixels')    
                
        # Plot the results#    
        # We need access to a few more parameters to illustrate the fit#
        
        params = fit_gaussian(subimg.data)    
        fit = gaussian(*params)  
        fig, ax = plt.subplots() 
        im = ax.imshow(subimg.data)  
        ax.contour(fit(*np.indices(subimg.shape)), cmap=plt.cm.magma, origin = 'lower')    
        ax.set_title('GUFI-'+str(j))    
        ax.set_xlabel('x (pix)')    
        ax.set_ylabel('y (pix)')
        fig.colorbar(im, label = 'Counts')
        # plt.savefig('example.png', bbox_inches='tight')    
        plt.show()    
        return fwhm
    j+=1
    i = 0
    fwhm_array = []
    for lines in stars:
        if __name__ == '__main__': 
            guess_x, guess_y = stars['xcentroid'][i], stars['ycentroid'][i] 
            fwhm = main()   
            fwhm_array.append(fwhm)
            fwhm_median = np.median(fwhm_array)
        i+=1
    median_array.append(fwhm_median)

#Sorts the FWHMs and get the indecies in order to sort and choose the top 1% of the science targets with the lowest FWHM#
def sort():
    print(median_array)
    idx = np.argsort(median_array)
    print(idx)
    files = shifts[idx]
    print(files)
    print(len(shifts))
    top = files[:int(0.01*len(shifts))]
    print(top)
    return top
top = sort()
    
#=======================================================================================================================================================#
#The reamining images are then combined using the median, the resultant image is then saved as a fits file. A subimage of each of the stars is also 
#saved as a pdf file
#=======================================================================================================================================================#

shifts = top
shifted = np.array([fits.getdata(image) for image in shifts])
shifted_median = np.median(shifted, axis = 0)
outfile = '/Users/jamestaylor/Desktop/Test_Directory/Results_Images/LI_reduced.fits'
hdu = fits.PrimaryHDU(shifted_median)
hdu.writeto(outfile, overwrite = True)

i = 0
hdu = fits.open(outfile)
data = hdu[0].data
stars = find_stars(data, 5)
for lines in stars:
    x_pos = stars['xcentroid'][i]
    y_pos = stars['ycentroid'][i]
    fig = plt.figure(figsize = (10,10))
    fig, ax = plt.subplots()
    
    ax.set_xlim([x_pos-20, x_pos+20])
    ax.set_ylim([y_pos-20, y_pos+20])
    img = ax.imshow(data, cmap = 'inferno', origin = 'lower', vmin = 0, vmax = data.max())
    ax.set_title('Star '+ str(i+1)+' Peak Position ('+str(x_pos)+','+str(y_pos)+')')
    ax.set_xlabel('x (pix)')    
    ax.set_ylabel('y (pix)')
    fig.colorbar(img, label = 'Counts')
    fig.savefig('/Users/jamestaylor/Desktop/Test_Directory/Results_Images/star'+str(i+1)+'.pdf', bbox_inches = 'tight')
    i+=1
end = time.time()
print('Total Run TIme: ', end - start)

#=======================================================================================================================================================#
#=======================================================================================================================================================#

