import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits                                                                                                                           
from astropy.stats import biweight_location
from astropy.nddata import Cutout2D
from scipy import optimize
from matplotlib.colors import LogNorm
from photutils.detection import DAOStarFinder
import glob


#=========================================================================================================================================================
#Calculates the FWHM for the resultant image and then saves a plot with the target and the fwhm in the title 
#=========================================================================================================================================================


file = '/Users/jamestaylor/Desktop/Test_Directory_LP/Sim_Science/simulated_median.fits'
#file = '/Users/jamestaylor/Desktop/Test_Directory_LP/Sim_Science/simulated_data_6.fits'

  
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

def FWHM(file):
    if __name__ == '__main__':
        
        
    
        # Load the data
        print(file)
        hdu = fits.open(file)
        data = hdu[0].data
        print(shape(data))
        
       
    
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
            center = (x_guess[i], y_guess[i])
        subimg = Cutout2D(hdulist[0].data, (x_guess[i], y_guess[i]), size=size)    
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
        sub_data = subimg.data
        im = ax.imshow(subimg.data, cmap = plt.cm.inferno, origin = 'lower', vmin = sub_data.min(), vmax = sub_data.max())  
        ax.contour(fit(*np.indices(subimg.shape)), cmap=plt.cm.viridis, origin = 'lower')    
        ax.set_title('Star '+str(i+1)+' Pos: ('+str(round(guess_x[i],3))+','+str(round(guess_y[i],3))+')\n' +'FWHM: '+str(round(fwhm*.08,3))+' arcsec\n')   
        ax.set_xlabel('x (pix)')    
        ax.set_ylabel('y (pix)')
        fig.colorbar(im, label = 'Counts')
        plt.savefig('/Users/jamestaylor/Desktop/Test_Directory/Results_Images/fwhm_norm_method_'+str(i+1)+'.pdf', bbox_inches='tight') 
        
        plt.show()    
        return fwhm, guess_x, guess_y
    if __name__ == '__main__': 
        fwhms = []
        file = open('fwhm_and_position.txt','w+')
        i = 0
        for line in stars:
            
            fwhm, guess_x, guess_y = main()
            file.write('Star '+str(i+1)+' Pos: ('+str(guess_x[i])+','+str(guess_y[i])+')\n' +'FWHM: '+str(fwhm)+'\n')
            i+=1
        file.close()
FWHM(file)