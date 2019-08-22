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
from matplotlib.colors import LogNorm
from photutils.detection import DAOStarFinder
import random
from scipy import ndimage


def star_sim():
    if os.path.isfile('/Users/jamestaylor/Desktop/Test_Directory_LP/Sim_Science/simulated_median.fits') == False:
        dim = 1000
        data = np.random.random((dim,dim))
        print(np.shape(data))
        print('')
        
        for k in range(5):
            i = random.randint(0+(dim*.10),dim-(.10*dim))
            j = random.randint(0+(dim*.10),dim-(.10*dim))
            l = random.randint(1,3)
            print(l)
            rx = random.randint(2,4)
            ry  = random.randint(2,4)
            for x in range(i-rx, i+rx):
                 
                for y in range(j-ry, j+ry):
                    print(x,y)
                    data[x][y] += 300*l
                    #print(data[x][y])
        
        for i in range(1000):
            
            data = ndimage.filters.gaussian_filter(data, sigma = 1.50)
            plt.imshow(data, cmap = 'inferno', origin = 'lower', vmin = 0, vmax = data.max())
            outfile = '/Users/jamestaylor/Desktop/Test_Directory_LP/Sim_Science/simulated_data_'+str(i)+'.fits'
            hdu = fits.PrimaryHDU(data)
            hdu.writeto(outfile, overwrite = True)
    else:
        pass
star_sim()
def Combine():
    
    
    sim = glob.glob('/Users/jamestaylor/Desktop/Test_Directory_LP/Sim_Science/simulated_data*.fits')
   
    
    data_ref = fits.getdata(sim[0])
    i = 0
    for im in sim:
        img = fits.getdata(im)
        xoff, yoff, xerr, yerr = chi2_shift(data_ref,img, return_error = True, upsample_factor = 100)           
        print(xoff)
        shifted_image = shift.shiftnd(img, (-xoff,-yoff))
        outfile = '/Users/jamestaylor/Desktop/Test_Directory_LP/Sim_Science/chi2_sim_shift_r1_'+str(i)+'.fits'
        hdu = fits.PrimaryHDU(shifted_image)
        hdu.writeto(outfile, overwrite = True)
        i+=1
    dt = glob.glob('/Users/jamestaylor/Desktop/Test_Directory_LP/Sim_Science/chi2_sim_shift_r1_*.fits')
    print(len(dt))
    dt_data = np.array([fits.getdata(image) for image in dt])
    print(shape(dt_data))
    med = np.median(dt_data, axis = 0)
    print(shape(med))
    plt.imshow(med, cmap = 'inferno', origin = 'lower')
    outfile = '/Users/jamestaylor/Desktop/Test_Directory_LP/Sim_Science/simulated_median.fits'
    hdu = fits.PrimaryHDU(med)
    hdu.writeto(outfile, overwrite = True)
Combine()