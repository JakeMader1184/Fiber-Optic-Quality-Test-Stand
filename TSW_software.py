
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages as PDF
#import pyfits
from datetime import datetime
import time
import math

from scipy.optimize import curve_fit
import scipy
import astropy

from photutils.aperture import CircularAperture
from scipy import ndimage
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAnnulus

import os
import imageio
from astropy.stats import sigma_clipped_stats, sigma_clip
import numpy.ma as ma

import warnings
warnings.filterwarnings('ignore')

import scipy.ndimage
import os
from matplotlib.ticker import FormatStrFormatter
import glob
from matplotlib.patches import Circle

def rebin( a, newshape ):
        '''Rebin an array to a new shape.
        '''
        assert len(a.shape) == len(newshape)

        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        return a[tuple(indices)]
    
def cart2pol(x, y):

    rho = np.sqrt((x)**2 + (y)**2)
    
    ## adding pi only because I want angles from 0 to 360
    phi = (np.arctan2(y,x))
    return(rho, phi)
    
def bg_sub(data,in_bg_an,out_bg_an,lim):
    center = centroid(data,lim)
    
    a= np.shape(data)[0]
    b= np.shape(data)[1]
    #print(a)
    #print(b)
    i_coords, j_coords = np.meshgrid(range(a),range(b), indexing='ij')
    
    
    rho,phi = cart2pol(i_coords-(center[0]),j_coords - (center[1]))
    
    maskk = np.where((rho > in_bg_an)&(rho < out_bg_an))
    
    check = np.ones(np.shape(data))
    for i in range(len(maskk[0])):
        check[maskk[0][i]][maskk[1][i]] = 0
    
    rho = np.ma.array(rho, mask = check)
    #rho = np.where(rho.mask, np.nan, rho)
    #plt.imshow(rho)
    
    data1 = np.copy(data)
    #print(data1)
    
    data_BG = np.array(data1)
    #data_BG = np.where(data_BG.mask, np.nan, data_BG)

    #print(data_BG)
    
    N = (len(check[np.where(check == 0)]))
    #print(N)
    print()
    flux = np.nansum(data_BG[np.where(check == 0)])
    print(f"flux: {flux}")
    #print((flux))
    #print(type(np.asarray(flux)))
    sb = (flux/N)
    print(f"BG: {sb}")
    print()
    BG_rms = np.nanstd(data_BG)
    #print(f"BG_rms {BG_rms}")

    
    ######maskk2 = np.where((data1 < 255))    
    ######check2 = np.ones(np.shape(data1))
    ######for i in range(len(maskk2[0])):
    ######    check2[maskk2[0][i]][maskk2[1][i]] = 0
    ######
    ######data2 = np.ma.array(data1,mask = check2)
    ######
    ######## such as find the rms in the BG 
    ######## Find the max counts in the pixels.
    ######
    ######img_dist = np.histogram(data2)
    ######max_counts = np.nanmax(data2) ## report instead the 95 or 99 percentile, just to show that there are
    ######## none near 255 really.
    ######print(f"MAX COUNTS {max_counts}")
    # for cold stuff, not tsw
    
    data2 = 0
    max_counts = 0
    
    #print(len(np.where(data1 == 255)[0]))
    
    #fig,ax = plt.subplots()
    #ax.hist(data_BG.flatten())
    #plt.savefig(f"{date}/{name}_BG_distribtion.png")
    #plt.close()

    
    #print('>LOOK: '+str(sb))
    #apertures = CircularAnnulus((center[1],center[0]), r_in = in_bg_an,r_out = out_bg_an)# TSW 200 -230
    #
    ### This bit does the analysis without any masking.
    #phot_table_list = aperture_photometry(data, apertures, method = 'exact')
    #flux = phot_table_list['aperture_sum']
    #
    #area = np.array([apertures.area])
    #print(flux)
    #print((area))
    #sb = flux/area
    out = data - sb
    
    
    BG_sb = sb

    #print(type(BG_sb))
    return(out,BG_sb,BG_rms,max_counts,data_BG,data2)

def med(date,Type,data_list,lamp_initial,lamp_list,null_list,newshape,in_bg_an,out_bg_an,lim):
##instead just use one of the original fits files and just replace the data with this median.
##Save it as a new fits file, with median in the name.
    print()
    print(Type)
    print(data_list[0])
    cwd = os.getcwd()    
    name = str.split(data_list[0],'.')[0]
    path = f"{cwd}/{date}/raw/{Type}"
    
    hdu_data_list = [fits.open(f"{path}/{data_list[i]}") for i in range(len(data_list))]

    A = [hdu_data_list[i][0].data for i in range(len(hdu_data_list))]

    #To ignore reshaping set newshape to empty list: newshape = []
    if len(newshape) > 0:
        A = [rebin(A[i],newshape) for i in range(len(A))]
        
    else:
        A = A
        
    
    M = np.median(np.array(A),axis=0)
    U = np.std(np.array(A),axis=0)
    
    A = M
    
    if lamp_initial == 0:
        
        A = A
        lamp_factor = 1
        
    else:
        init_lamp = fits.open(f"{path}/{lamp_initial}")[0].data
        lamp_data_list = fits.open(f"{path}/{lamp_list}")[0].data
        lamp_factor = np.nansum(lamp_data_list)/np.nansum(init_lamp) #Maybe do residuals between lamp frames
        ## instead
        
        A = A/lamp_factor # Correct for lamp factor

        print(f"Lamp Factor: {lamp_factor}")
    ################################################
    B = A[:,0:150]
    C = A[:,1129:1279]

    shapes_B = (np.shape(B))
    #print(shapes_B[0],shapes_B[1])
    
    shapes_C = (np.shape(C))
    #print(shapes_C[0],shapes_C[1])
    
    row_means = []
    for i in range(shapes_B[0]):
        row_avg = np.nanmean(B[i])
        
        row_avg2 = np.nanmean(C[i])
        #print(row_avg,row_avg2)
        #print()
        
        row_mean = (row_avg + row_avg2)/2 
        row_means.append(row_mean)
        #print(row_mean)
        #print(row_means)
        
    for i in range(len(row_means)):
        A[i] = A[i] - row_means[i]
    ##################################################
        

    if len(null_list) == 0:
        
        ## Image statistics are done after correcting for lamp factor.
        med_out = bg_sub(A,in_bg_an,out_bg_an,lim)
        
        A_null = med_out[1] * np.ones(np.shape(A))
        BG_rms = med_out[2]
        max_counts = med_out[3]
        print(med_out[1])
        
        print(f"Used BG annulus.")
        print(f"BG annulus SB: {np.mean(A_null)}") # this is really just the value of sb in the bg
        M_null = A_null
        #print(type(A_null))
        #print(np.shape(A_null))
        #print(A_null)
    else:
        hdu_null_list = [fits.open(f"{path}/{null_list[i]}") for i in range(len(null_list))]
        A_null = [hdu_null_list[i][0].data for i in range(len(hdu_null_list))]
        
        if len(newshape) > 0:
            A_null = [rebin(A_null[i],newshape)for i in range(len(A_null))]
        
        M_null = np.median(np.array(A_null),axis=0)
        print(f"Used null frame.")
    
    A = A - M_null
    #print(type(A))

    ########################
    
    data_bg = med_out[4]
    
    ##fig,ax = plt.subplots()
    ##
    ##binss = np.linspace(0,150,151)
    ##ax.hist(data_bg.flatten(),bins = binss,align = 'left')
    ##ax.set_xlabel('ADU')
    ##ax.set_ylabel('Number of Pixels in BG')
    ##ax.set_title('BG Distribution')
    ##plt.tight_layout()
    ##plt.savefig(f"{cwd}/{date}/{Type}_{name}_BG_distribution.png")
    ##plt.close()
    ##############################################   
    ##
    ##data2 = med_out[5]
    ##fig,ax2 = plt.subplots()
    ##
    ##binss = np.linspace(0,255,256)
    ##ax2.hist(data2.flatten(),bins = binss,align = 'left')
    ##ax2.set_xlabel('ADU')
    ##ax2.set_ylabel('Number of Pixels in Image')
    ##ax2.set_title('Full Image Distribution')
    ##ax2.set_ylim(0,175000)
    ##ax2.set_xlim(0,255)
##
    ##plt.tight_layout()
    ##plt.savefig(f"{cwd}/{date}/{Type}_{name}_full_distribution.png")
    ##plt.close()
    
    ###############################################
    ############################################
    
    U_null = np.std(np.array(A_null),axis=0)
    M=A
    D = A
    
    DU = np.sqrt((U**2)+(U_null)**2) 
    
    headerfits = f"{path}/{data_list[0]}"
    header = fits.getheader(headerfits)

    outfilename_med = f"{name}_{Type}_med"
    outpath_med = f"{cwd}/{date}/redux/{Type}/{outfilename_med}.fits"

    outfilename_null_med = f"{name}_null_{Type}_med"
    outpath_null_med = f"{cwd}/{date}/redux/{Type}/{outfilename_null_med}.fits"
    
    outfilename_nobg = f"{name}_{Type}_nobg"
    outpath_nobg = f"{cwd}/{date}/redux/{Type}/{outfilename_nobg}.fits"
    
    outfilename_U = f"{name}_{Type}_uncertainty"
    outpath_U = f"{cwd}/{date}/redux/{Type}/{outfilename_U}.fits"
    
    #fits.writeto(outpath_med,M,header,overwrite = True)
    #fits.writeto(outpath_null_med,M_null,header,overwrite = True)
    fits.writeto(outpath_nobg,D,header,overwrite = True)
    #fits.writeto(outpath_U,DU,header,overwrite = True)
    
    return(np.mean(A_null),lamp_factor,BG_rms,max_counts)

def med_for_TSW(cwd,date,Type,data_list,null_list,newshape,lim,query,query_d,bgsub):
##instead just use one of the original fits files and just replace the data with this median.
##Save it as a new fits file, with median in the name.
    #cwd = os.getcwd()
    #print(cwd)
    
        
    if os.path.isdir(f"{cwd}/{date}/raw/fiber") == False:
        os.mkdir(f"{cwd}/{date}/raw/fiber") 
        
    if os.path.isdir(f"{cwd}/{date}/redux") == False:
        os.mkdir(f"{cwd}/{date}/redux") 
        
    if os.path.isdir(f"{cwd}/{date}/analysis") == False:
        os.mkdir(f"{cwd}/{date}/analysis") 
        
    if os.path.isdir(f"{cwd}/{date}/redux/fiber") == False:
        os.mkdir(f"{cwd}/{date}/redux/fiber") 
    
    if os.path.isdir(f"{cwd}/{date}/redux/direct") == False:
        os.mkdir(f"{cwd}/{date}/redux/direct") 
        
    
    name = str.split(data_list[0],'.')[0]

    path = f"{cwd}/{date}/raw/{Type}"
    
    hdu_data_list = [fits.open(f"{path}/{data_list[i]}") for i in range(len(data_list))]

    A = [hdu_data_list[i][0].data for i in range(len(hdu_data_list))]

    #To ignore reshaping set newshape to empty list: newshape = []
    if len(newshape) > 0:
        A = [rebin(A[i],newshape) for i in range(len(A))]

    M = np.median(np.array(A),axis=0)
    U = np.std(np.array(A),axis=0)
    
    if len(null_list) == 0:
        
        A_null = bg_sub(M,in_bg_an,out_bg_an)[1] * np.ones(np.shape(M))
        print(f"Mean_annulus_counts: {np.mean(A_null)}")
        M_null = A_null
        print(f"Used BG annulus.")

    else:
        hdu_null_list = [fits.open(f"{path}/{null_list[i]}") for i in range(len(null_list))]
        A_null = [hdu_null_list[i][0].data for i in range(len(hdu_null_list))]
        
        if len(newshape) > 0:
            A_null = [rebin(A_null[i],newshape)for i in range(len(A_null))]
        
        M_null = np.median(np.array(A_null),axis=0)
        print(f"Used null frame.")
    
    ##################################################################### I may want to feed bg_subtraction 
    ## the same center that is computed
    
    
    U_null = np.std(np.array(A_null),axis=0)
    
    D = M - M_null
    
    ## I dont want to do this in the presence of a non-flat BG

    #print(type(D))
    # D is null subtracted
    
    if bgsub == True:
        med_out = bg_sub(D,230,350,lim)
        
        A_null = np.array(med_out[1] * np.ones(np.shape(M)))
        BG_rms = med_out[2]
        max_counts = med_out[3]
        
        
        D = D-A_null
    else:
        D=D
    ### now its also BG subtracted.
    ##print(type(A_null))
    ##print(type(D))
    ##################################################################################
    #DU = np.sqrt((U**2)+(U_null)**2) 
    
    headerfits = f"{path}/{data_list[0]}"
    header = fits.getheader(headerfits)

    outfilename_med = f"{name}_{Type}_med"
    outpath_med = f"{cwd}/{date}/redux/{Type}/{outfilename_med}.fits"

    outfilename_null_med = f"{name}_null_{Type}_med"
    outpath_null_med = f"{cwd}/{date}/redux/{Type}/{outfilename_null_med}.fits"
    
    outfilename_nobg = f"{name}_{Type}_nobg"
    outpath_nobg = f"{cwd}/{date}/redux/{Type}/{outfilename_nobg}.fits"
    
    #outfilename_U = f"{name}_{Type}_uncertainty"
    #outpath_U = f"{cwd}/{date}/redux/{Type}/{outfilename_U}.fits"
        
    fits.writeto(outpath_med,M,header,overwrite = True) 
    fits.writeto(outpath_null_med,M_null,header,overwrite = True) 
    fits.writeto(outpath_nobg,D,header,overwrite = True) 
    #fits.writeto(outpath_U,DU,header,overwrite = True)
    
def multi_med(cwd,date,direct_data_list,direct_null_list,fiber_data_list,fiber_null_list,\
              newshape,lim,query,query_d,bgsub):
    
    med_for_TSW(cwd,date,'direct',direct_data_list,direct_null_list,newshape,lim,query,query_d,bgsub)
    med_for_TSW(cwd,date,'fiber',fiber_data_list,fiber_null_list,newshape,lim,query,query_d,bgsub)

def neo_annulize(data,cen,lim):
    radius = range(350)
    radius = [(i+1) for i in radius]
    center = centroid(data,lim)
    
    ############################################
    if cen == (0,0):
        center = centroid(data,lim)
    else:
        center = cen
    ############################################
    
    
    print(f"x y center from neo_annulize: {center[0]},{center[1]}")
    apertures = np.array([CircularAnnulus((center[1],center[0]), r_in=i,r_out = i+(1)) for i in radius])
    
    phot_table_list = [aperture_photometry(data, ap, method = 'exact') for ap in apertures]
    #,error = uncertainty
    flux =[float(i['aperture_sum']) for i in phot_table_list]
    #flux_error = [float(i['aperture_sum_err']) for i in phot_table_list]
    flux_error=0
    area = np.array([i.area for i in apertures]).flatten()
    sb = flux/area
    #sb_error = flux_error/area
    sb_error = 0
    
    ###
    #num_an = len(radius)
    #fluxes = np.zeros(num_an, dtype=np.float32)
    #r_vec = np.zeros(num_an, dtype=np.float32)
    #dims = data.shape
    #vecvec = np.indices(dims,dtype=np.float32)
    #distances = ((center[0] - vecvec[0,])**2 + (center[1] - vecvec[1,])**2)**0.5
    #rlimit = distances.max()
    #rstep = rlimit/num_an
    #r1 = 0.0
    #r2 = rstep
    #for i in range(num_an):
    #    idx = np.where((distances <= r2) & (distances > r1))

    #    fluxes[i] = np.sum(data[idx])
    #    r_mid = (r1 + r2)*0.5
    #    r_vec[i] = r_mid
    #    r1 = r2
    #    r2 = r1 + rstep
    ### another method: ,r_vec,fluxes
    
    
    out = np.array([radius,flux,flux_error,sb,sb_error,center]) 
    return(out)

def neo_FReD(direct_img_data, fiber_img_data,d_cen,f_cen,\
             lim,FL3=55.616,FR=4.2):
    
    print(FR)
    
    d_rvec1, d_fluxes, d_err,d_sb,d_sberrors,d_cenn = \
    neo_annulize(direct_img_data,d_cen,lim)  
    
    f_rvec1, f_fluxes, f_err,f_sb,f_sberrors,f_cenn = \
    neo_annulize(fiber_img_data,f_cen,lim)  
    
    '''Turn pixels into an physical length. The SBIG STL-1001E has 24 micron
    square pixels'''
    d_rvec = np.array([i*0.024*2 for i in d_rvec1]) # ---> mm  *2 because the data is binned 2x2
    f_rvec = np.array([i*0.024*2 for i in f_rvec1]) # TSW is 0.024*2 cold test is 0.0052*2
    

    d_N = FL3/(2*d_rvec)
    f_N = FL3/(2*f_rvec) # These would be exact if we had the right FL3. We are assumin that we know the
    ## iris and FL2 perfectly, such that it forms an f/4.2 beam. I will therefore scale FL3 around
    ## until the EE95 of the f/4.2 beam is at f/4.2...
    
    d_cumflux = np.nancumsum(d_fluxes)
    f_cumflux = np.nancumsum(f_fluxes)
    d_cumerr = (np.cumsum(d_err**2))**0.5
    f_cumerr = (np.cumsum(f_err**2))**0.5
        
    edgerad = 230 # goes out to the corner
    t = (np.abs(np.array(d_rvec1) - edgerad)).argmin()
    d_max = d_cumflux[t]
    print(f"Direct max flux defined within f/{d_N[t]} or {edgerad} pixels")
    #d_max = np.max(d_cumflux)

    # I will make this different for the fiber potentially
    
    t = (np.abs(np.array(f_rvec1) - edgerad)).argmin()
    f_max = f_cumflux[t] 
    print(f"Fiber max flux defined within f/{f_N[t]} or {edgerad} pixels")

    #f_max = np.max(f_cumflux)
 
    f_EE = f_cumflux/f_max
    f_EEerr = f_cumerr/f_max

    d_EE = d_cumflux/d_max
    d_EEerr = d_cumerr/d_max
    
    ######## Fit a parabola to the direct beam I DONT ACTUALLY NEED TO FIT THE PARABOLA.
    ######t = (np.abs(d_EE - 0.95)).argmin()
    ######EEfit = np.poly1d(np.polyfit(d_rvec[:t],d_EE[:t],2))
    ######fitr = np.linspace(d_rvec.min(),d_rvec.max(),500)
    ######fitEE = EEfit(fitr)
    ######
    ######## Combined with assuming an FL3 and knowing the diameter of the input aperture (iris, ~12.27mm)
    ######## 
    ######r1 = np.interp(0.95,fitEE,fitr) ## because I am just assuming that F2 and the iris forms an f/4.2 beam,
    ######## this, EE95 radius the f/4.2
    ######
######
    ######fit_N = FL3/(2*r1)
        
    ## I need to write a bit that finds the EE95 of my spot. This will be f/4.2.... Or as my
    ## back-distance analysis shows, f/4.3
    ## This radius will be r1
    t = (np.abs(d_EE - 0.95)).argmin()
    r1 = d_rvec[t]

    #fit_N = FR
    #actual_FR = fit_N 

    #rf25 = r1 * fit_N/2.5 ##Replace FIT_N with 4.2 this sets the platescale based on us knowing we have
    #rf38 = r1 * fit_N/3.8 ## an f/4.2 input beam with absolute certainty
    #rf42 = r1 * fit_N/FR ##
    
    #########

    d_N = r1*FR/d_rvec
    f_N = r1*FR/f_rvec

    r38 = np.abs((d_N - 3.8)).argmin()
    r42 = np.abs((d_N - 4.2)).argmin()

    print(f"f/4.2 at this radius {r42}")

    print(f"f/3.8 at this radius{r38}")
    
    #########
    
    #d25_func = np.abs(d_rvec - rf25)
    #d25_flux =  d_cumflux[d25_func.argmin()]
    #
    #d38_func = np.abs(d_rvec - rf38)
    #d38_flux =  d_cumflux[d38_func.argmin()]
    #
    #dap_func = np.abs(d_rvec - r1) # could also be d_N - actual_FR
    #dap_flux =  d_cumflux[dap_func.argmin()]
    #
    #
    #f25_func = np.abs(f_rvec - rf25)
    #f25_flux =  f_cumflux[f25_func.argmin()]
    #
    #f38_func = np.abs(f_rvec - rf38)
    #f38_flux =  f_cumflux[f38_func.argmin()]
    #
    #fap_func = np.abs(f_rvec - r1)
    #fap_flux =  f_cumflux[fap_func.argmin()]
        
    ### TT metrics
    #f25_d25 = (f25_flux/d25_flux)*100 ##This is chosen as the absolute total throughput
    #f25_dap = (f25_flux/dap_flux)*100 ## This is assuming only the flux within the direct ap is injected
#
    ### RT metrics of direct beam
    #d38_d25 = (d38_flux/d25_flux)*100
    #dap_d25 = (dap_flux/d25_flux)*100
    #
    ### RT metrics of fiber beam
    #f38_f25 = (f38_flux/f25_flux)*100
    #fap_f25 = (fap_flux/f25_flux)*100
    #
    #fit_fN_func = np.abs(f_EE-0.95)  
    #EE95 = f_N[fit_fN_func.argmin()]   
    
    ####### ONLY RELEVANT FOR f/5 INPUT BEAM MEASUREMENTS LIKE I MADE ON ma_053
    ####### I WONT BOTHER WITH THE CORRECTED VERSIONS OF THIS
    #rf5 = r1 * fit_N/5.0
    #rf4 = r1 * fit_N/4.0
    #rf32 = r1 * fit_N/3.2
    #rf3 = r1 * fit_N/3
    
    #f3_func = np.abs(f_rvec - rf3)
    #f3_flux =  f_cumflux[f3_func.argmin()]
    #
    #f32_func = np.abs(f_rvec - rf32)
    #f32_flux =  f_cumflux[f32_func.argmin()]
    #
    #f4_func = np.abs(f_rvec - rf4)
    #f4_flux =  f_cumflux[f4_func.argmin()]
    #
    #fap_func = np.abs(f_rvec - r1)
    #fap_flux =  f_cumflux[fap_func.argmin()]
    #
    #
    #d3_func = np.abs(d_rvec - rf3)
    #d3_flux =  d_cumflux[d3_func.argmin()]
    #
    #d32_func = np.abs(d_rvec - rf32)
    #d32_flux =  d_cumflux[d32_func.argmin()]
    #
    #d4_func = np.abs(d_rvec - rf4)
    #d4_flux =  d_cumflux[d4_func.argmin()]
    #
    #dap_func = np.abs(d_rvec - r1)
    #dap_flux =  d_cumflux[dap_func.argmin()]
    
    ### TT metrics
    #fap_dap = fap_flux/dap_flux*100
    #f4_d4 = f4_flux/d4_flux*100
    #f32_d32 = f32_flux/d32_flux*100
    #f3_d3 = f3_flux/d3_flux*100
    #f3_dap = f3_flux/dap_flux*100
    #
    ### RT metrics of direct beam
    #dap_d3 = dap_flux/d3_flux*100
    #d4_d3 = d4_flux/d3_flux*100
    #
    ### RT metrics of fiber beam
    #fap_f3 = fap_flux/f3_flux*100
    #f4_f3 = f4_flux/f3_flux*100
    #######
    
    '''Now we need to use the difference between an ideal beam and the direct
    beam (which should be ideal) to create a correction that we can apply
    to both the direct and fiber data'''
    
    # _correction will correspond to corrections that will be applied to
    # get corrected values and _c will correspond to values that have 
    # been corrected
    
    #f_r_correction = np.zeros(f_EE.size,dtype=float)
    #d_r_c = np.zeros(d_EE.size,dtype=float)
#
    #j=0
    #for k in range(f_r_correction.size):
    #    # Naming conventions here match what is in my notebook on pages
    #    # 47 through 51
    #    
    #    '''First the direct beam'''
    #    d_r_c[k] = (d_EE[k]*(FL3/(2*actual_FR))**2)**0.5
    #    
    #    '''Now the fiber beam'''
    #    f_r_i = (f_EE[k]*(FL3/(2*actual_FR))**2)**0.5
#
    #    '''find the closest d_EE value that is less than f_EE[k]'''
    #    while d_EE[j] < f_EE[k]: j += 1
 #
    #    '''interpolate'''
    #    m = (d_EE[j] - d_EE[j-1])/(d_rvec[j] - d_rvec[j-1])
    #    r_d = (f_EE[k] - d_EE[j-1])/m + d_rvec[j-1]
#
    #    '''f_dr2 is f_dr**2'''
    #    f_dr2 = r_d**2 - f_r_i**2
    #    f_r_correction[k] = f_dr2
#
    #    '''We do this to fix some weirdness that might happen at really large
    #    radii. It's more of a visual appeal thing than anything else'''
    #    if (np.abs(f_rvec[k]**2 - f_r_correction[k]))**0.5 <\
    #            (np.abs(f_rvec[k-1]**2 - f_r_correction[k-1]))**0.5 \
    #            or (f_rvec[k]**2 - f_r_correction[k]) < 0:
#
    #        f_r_correction[k] = f_r_correction[k-1]
#
    #'''Actually perform the correction on the fiber data'''
    #f_r_c = (f_rvec**2 - f_r_correction)**0.5
#
    #d_rerr = (np.abs(d_r_c - d_rvec))**0.5
    #f_rerr = (np.abs(f_r_c - f_rvec))**0.5
    #
#
    #d_N_c = FL3/(2*d_r_c)
    #f_N_c = FL3/(2*f_r_c)
#
    #
    ###### Compute all the metrics from before, but using the corrected radius vector
    #d25_func_c = np.abs(d_r_c - rf25)
    #d25_flux_c =  d_cumflux[d25_func_c.argmin()]
    #
    #d38_func_c = np.abs(d_r_c - rf38)
    #d38_flux_c =  d_cumflux[d38_func_c.argmin()]
    #
    #dap_func_c = np.abs(d_r_c - r1) # could be d_N - actual_FR
    #dap_flux_c =  d_cumflux[dap_func_c.argmin()]
    #
    #
    #f25_func_c = np.abs(f_r_c - rf25)
    #f25_flux_c =  f_cumflux[f25_func_c.argmin()]
    #
    #f38_func_c = np.abs(f_r_c - rf38)
    #f38_flux_c =  f_cumflux[f38_func_c.argmin()]
    #
    #fap_func_c = np.abs(f_r_c - r1)
    #fap_flux_c =  f_cumflux[fap_func_c.argmin()]
#
    #
    ### TT metrics
    #f25_d25_c = (f25_flux_c/d25_flux_c)*100 ##This is chosen as the absolute total throughput
    #f25_dap_c = (f25_flux_c/dap_flux_c)*100 ## This is assuming only the flux within the direct ap is injected
    #
    ### RT metrics of fiber beam
    #f38_f25_c = (f38_flux_c/f25_flux_c)*100
    #fap_f25_c = (fap_flux_c/f25_flux_c)*100
    #
    ### RT metrics of direct beam
    #d38_d25_c = (d38_flux_c/d25_flux_c)*100
    #dap_d25_c = (dap_flux_c/d25_flux_c)*100
    #
    #fit_fN_func_c = np.abs(f_EE-0.95)  
    #EE95_c = f_N_c[fit_fN_func_c.argmin()]    
    #######
    d_r_c =[]
    f_r_c= []
    d_N_c = []
    f_N_c = []
    
    direct = np.array([d_rvec, d_r_c, d_N, d_N_c, d_cumflux,d_max, d_EE,d_sb,d_EEerr,\
                       d_sberrors])
    fiber = np.array([f_rvec, f_r_c, f_N, f_N_c, f_cumflux,f_max, f_EE,f_sb,f_EEerr,\
                      f_sberrors])

    #if FR == 4.2:
    #
    #    out0 = direct,fiber,direct_img_data,fiber_img_data
    #    metrics = f25_d25,f25_dap,f38_f25,fap_f25,d38_d25,dap_d25,EE95,actual_FR
    #    metrics_c = f25_d25_c,f25_dap_c,f38_f25_c,fap_f25_c,d38_d25_c,dap_d25_c,EE95_c
    #    
    #    return(out0,metrics,metrics_c)

    #else:
    out0 = direct,fiber,direct_img_data,fiber_img_data,d_cenn,f_cenn
    metrics = []
    #metrics = fap_dap,f4_d4,f32_d32,f3_d3,f3_dap,dap_d3,d4_d3,fap_f3,f4_f3,actual_FR
    #metrics_c = f25_d25_c,f25_dap_c,f38_f25_c,fap_f25_c,d38_d25_c,dap_d25_c,EE95_c
        
    return(out0,metrics)

def EE_from_TSW(cwd,date,dir_img_data,fib_img_data,query,lim):

    ### This needs to take the center that was already produced in the 
    
    #For the science cable tests I will use specific direct spot centers.
    d_cen = (255,242)
    #242,255
    f_cen = (0,0)
    
    
    FL3 = 55.616 ##########
    FR = 4.2  ######This was 4.3 on accident...
    
    out = neo_FReD(dir_img_data,fib_img_data,d_cen,f_cen,lim,FL3,FR)
        
    direct= (out[0][0])
    fiber = (out[0][1])
    
    func = np.abs(direct[2] - 3.8)
    relative_throughput = fiber[6][func.argmin()]
    
    total_tput = fiber[5]/direct[5]
    locs = np.linspace(0.2,0.8,6)
    
    fig, ax = plt.subplots(figsize = (8,5))
    
    ax.plot(direct[2],direct[6],color = 'k', label = 'Injection Beam',linewidth = 2)
    ax.plot(fiber[2],fiber[6],color = 'r', label = f"Fiber Output",linewidth = 2)
    
    ax.grid(which = 'minor',linestyle='--')
    ax.grid(which = 'major',linestyle='--')
    
    ax.minorticks_on()
    ax.set_xlim(1,20)
    ax.set_xscale('log')
    ax.set_ylim(0,1.1)
    
    ax.set_title(query,fontsize = 16)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.vlines(ymin = 0,ymax = 1.1, x =4.2,linestyle = '--',label = 'f/4.2',alpha = 0.7,color='m')
    ax.vlines(ymin = 0,ymax = 1.1, x =3.8,linestyle = '--',label = 'f/3.8',alpha = 0.7,color='b')
    ax.vlines(ymin = 0,ymax = 1.1, x =2.5,linestyle = '--',label = 'f/2.5 - Max Aperture',alpha = 0.7,color='r')

    ax.set_xlabel('f/#',fontsize = 16)
    ax.set_ylabel('EE',fontsize =16)
    ax.legend(loc = 'upper right',fontsize = 14)
    
    ## example code: g = float("{0:.2f}".format(x))
    
    ax.text(0.50, locs[1], 'Transmission Throughput: '+str(float("{0:.2f}".format(total_tput*100))) + '%',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        fontsize=12,color = 'g')
    
    ax.text(0.50, locs[0], 'Relative Throughput @ f/3.8: '+str(float("{0:.2f}".format(relative_throughput*100))) + '%',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        fontsize=12,color = 'g')
    
   # plt.xticks(fontsize = 16)
    #plt.yticks(fontsize = 16)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{cwd}/{date}/analysis/{query}_EE_COG.png")
    #return(fib_img_data)
    return(total_tput*100,relative_throughput*100,out[0][4],out[0][5])

def plot_spots(out_from_neo_FReD,out_path,name):
    
    direct,fiber,direct_img_data,fiber_img_data = out_from_neo_FReD[0]

    f, ax = plt.subplots(1,3,sharey=False, sharex=False, figsize = (10,3))
    ax[0].plot(direct[0],direct[7],color='r',label='Direct')
    
    #ax[0].plot(direct[0],direct[7]+(3*direct[9]),color='r',label='Direct + 3sigma',alpha = 0.3)
    #ax[0].plot(direct[0],direct[7]-(3*direct[9]),color='r',label='Direct - 3sigma',alpha = 0.3)

    ax[0].set_title('Input Spot Surface Brightness (counts)')
    ax[0].set_xlabel('Radius (mm)')
    ax[0].set_xlim(0,20)
    ax[0].set_ylim(0,np.max(direct[7])+0.15*np.max(direct[7]))
    ax[0].plot(fiber[0],fiber[7], color = 'k', label='Fiber')
    #ax[0].plot(fiber[0],fiber[7]+(3*fiber[9]),color='k',label='Fiber + 3sigma',alpha = 0.3)
    #ax[0].plot(fiber[0],fiber[7]-(3*fiber[9]),color='k',label='Fiber - 3sigma',alpha = 0.3)

    
    ax[0].legend(loc=0,numpoints=1,frameon=False,fontsize=8)
    ax[0].grid(which = 'major')
    
    ax[1].imshow(direct_img_data, origin = 'lower',vmin=np.min(direct[7]),vmax = np.max(direct[7])*1.5)
    ax[1].set_title('Input Spot Image')
    
    ax[2].imshow(fiber_img_data, origin = 'lower',vmin =np.min(direct[7]),vmax = np.max(direct[7])*1.5)
    ax[2].set_title('Output Spot Image')
    f.suptitle(name, fontsize=12, y = 1.035)
    plt.tight_layout()
    
    #if os.path.isdir(cwd + '/TSW_Investigation/' + 'analysis/' + name) == False:
    #    os.mkdir(cwd + '/TSW_Investigation/' + 'analysis/' + name)
    
    #plt.savefig(cwd + '/TSW_Investigation/analysis/'+ name + '/' + name + '_SB_Spots')
    plt.savefig(f"{out_path}/{name}_SB_spots.pdf")

def produce(cwd,date,query,query_d,query_null_f,lim,bgsub):

    dir_img_list,dir_null_list,fib_img_list,fib_null_list = get_lists(cwd,date,query,query_d,query_null_f)
    multi_med(cwd,date,dir_img_list,dir_null_list,fib_img_list,fib_null_list,(),lim,query,query_d,bgsub)
    
    #date = '20201106'
    path = f"{cwd}/{date}/"
    data_list = [query]
    dir_img_data = fits.open(f"{cwd}/{date}/redux/direct/{query_d}_direct_nobg.fits")[0].data 
    fib_img_data = fits.open(f"{cwd}/{date}/redux/fiber/{query}_fiber_nobg.fits")[0].data


    total_tput,relative_throughput,d_cenn,f_cenn = EE_from_TSW(cwd,date,dir_img_data,fib_img_data,query,lim)

    # draw circles
    
    figa,ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(np.log10(dir_img_data),origin='lower',cmap='Greys_r',vmax=np.log10(900))
    circ1 = Circle((d_cenn[1],d_cenn[0]),radius=230,color='r',linestyle='--',linewidth=2,fill=False)
    circ2 = Circle((d_cenn[1],d_cenn[0]),radius=128,color='m',linestyle='--',linewidth=2,fill=False)
    circ3 = Circle((d_cenn[1],d_cenn[0]),radius=142,color='b',linestyle='--',linewidth=2,fill=False)

    ax[0].add_patch(circ1)
    ax[0].add_patch(circ2)
    ax[0].add_patch(circ3)

    ax[0].legend([circ1,circ2,circ3], ['f/2.5','f/4.2','f/3.8'],fontsize = 16)
    
    ax[1].imshow(np.log10(dir_img_data),origin='lower',cmap='Greys_r',vmin=np.log10(900))
    circ1 = Circle((d_cenn[1],d_cenn[0]),radius=230,color='r',linestyle='--',linewidth=2,fill=False)
    circ2 = Circle((d_cenn[1],d_cenn[0]),radius=128,color='m',linestyle='--',linewidth=2,fill=False)
    circ3 = Circle((d_cenn[1],d_cenn[0]),radius=142,color='b',linestyle='--',linewidth=2,fill=False)

    ax[1].add_patch(circ1)
    ax[1].add_patch(circ2)
    ax[1].add_patch(circ3)

    #ax[1].legend([circ1,circ2,circ3], ['f/2.5','f/4.2','f/3.8'],fontsize = 14)
    figa.suptitle('Direct Image',y=0.9,fontsize = 16)
    
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    plt.show()

    plt.tight_layout()
    figa.savefig(f"{cwd}/{date}/analysis/{query}__dir_img.png")
    plt.close()
    
    #dir_img_uncertainty = fits.open(f"{path}redux/direct/{name}_direct_uncertainty.fits")[0].data
    
    figb,bx = plt.subplots(1,2,figsize=(12,6))
    bx[0].imshow(np.log10(fib_img_data),origin='lower',cmap='Greys_r',vmax=np.log10(1100))
    
    circ1 = Circle((f_cenn[1],f_cenn[0]),radius=230,color='r',linestyle='--',linewidth=2,fill=False)
    circ2 = Circle((f_cenn[1],f_cenn[0]),radius=128,color='m',linestyle='--',linewidth=2,fill=False)
    circ3 = Circle((f_cenn[1],f_cenn[0]),radius=142,color='b',linestyle='--',linewidth=2,fill=False)

    bx[0].add_patch(circ1)
    bx[0].add_patch(circ2)
    bx[0].add_patch(circ3)

    #bx[0].legend([circ1,circ2,circ3], ['f/2.5','f/4.2','f/3.8'],fontsize = 14)
    
    bx[1].imshow(np.log10(fib_img_data),origin='lower',cmap='Greys_r',vmin=np.log10(5500))
    
    circ1 = Circle((f_cenn[1],f_cenn[0]),radius=230,color='r',linestyle='--',linewidth=2,fill=False)
    circ2 = Circle((f_cenn[1],f_cenn[0]),radius=128,color='m',linestyle='--',linewidth=2,fill=False)
    circ3 = Circle((f_cenn[1],f_cenn[0]),radius=142,color='b',linestyle='--',linewidth=2,fill=False)

    bx[1].add_patch(circ1)
    bx[1].add_patch(circ2)
    bx[1].add_patch(circ3)

    #bx[1].legend([circ1,circ2,circ3], ['f/2.5','f/4.2','f/3.8'],fontsize = 16)
    figb.suptitle('Fiber Image',y=0.9,fontsize = 16)

    bx[0].set_xticklabels([])
    bx[0].set_yticklabels([])
    bx[1].set_xticklabels([])
    bx[1].set_yticklabels([])
    plt.show()
    

    plt.tight_layout()
    figb.savefig(f"{cwd}/{date}/analysis/{query}_fib_img.png")
    plt.close()
    
    #fib_img_uncertainty = fits.open(f"{path}redux/fiber/{name}_fiber_uncertainty.fits")[0].data
    
    return(fib_img_data,total_tput,relative_throughput)

def bmp_to_fits(date,Type,filename):
    cwd = os.getcwd()

    if (Type == 'direct') or (Type == 'fiber'):
        outpath = f"{cwd}/{date}/raw/{Type}"
        image = imageio.imread(os.path.join(outpath +'/',filename))

    else:
        outpath = f"{cwd}/{date}/{Type}"
        image = imageio.imread(os.path.join(outpath +'/',filename),pilmode='L')

    #outpath = inpath + '/' + filename + '.fits'
    
    print(np.shape(image))
    #image_2x2 = np.round(rebin(image,(np.shape(image)[0]//2,np.shape(image)[1]//2)),0)
    
    fits.writeto(outpath +'/'+ str.split(filename,'.')[0] + '.fits',image,overwrite=True)
    
def centroid(image,lim):
    data = np.copy(image)
    size = data.shape
    
    ###will need to adjust for TSW use: 900 for the 12 bit Manta, 7 or 8 or 15 ish for 8bit ueye
    # 15 for cold ueye, 20 for TSW ueye
    ig_x = (np.where(data < lim)[0])## 20 for cold testing latest, TSW should be using 900.
    ig_y = (np.where(data < lim)[1])
    
    kp_x = (np.where(data > lim)[0])
    kp_y = (np.where(data > lim)[1])
    
    
    for i in range(len(ig_x)):
        data[ig_x[i]][ig_y[i]] = 0

    for i in range(len(kp_x)):
        data[kp_x[i]][kp_y[i]] = 1
    
    totalMass = np.sum(data)
    xcom = np.sum(np.sum(data,1) * np.arange(size[0]))/totalMass
    ycom = np.sum(np.sum(data,0) * np.arange(size[1]))/totalMass

    print(f"x y center from centroid: {ycom},{xcom}")
    return (xcom,ycom)

def get_filenames(date,Type,query,cwd):

    path_list = glob.glob(f"{cwd}/{date}/raw/{Type}/{query}")
          
    data_list = []
    
    
    for i in range(len(path_list)):
        
        #print(path_list[i])
        if path_list[i].split('/')[8].split('.')[1] == 'bmp' or \
        path_list[i].split('/')[8].split('.')[1] == 'tiff':
    
            data_list.append(path_list[i].split('/')[8])
        
    return(sorted(data_list))

def get_rad(name,FL3,Type,ITERATE,lim,date,cwd):###################

    cwd = os.getcwd()
    path = f"{cwd}/{date}/"
    
    img_data = fits.open(f"{path}redux/{Type}/{name}_{Type}_nobg.fits")[0].data 
    #img_uncertainty = fits.open(f"{path}redux/{Type}/{name}_{Type}_uncertainty.fits")[0].data
    #plt.imshow(img_data)
    d_rvec1, d_fluxes, d_err,d_sb,d_sberrors = \
    mod_annulize(img_data,lim)  ###################
            
    d_rvec = np.array([i*0.0052 for i in d_rvec1]) # ---> mm  *2 because the data is binned 2x2
    d_cumflux = np.nancumsum(d_fluxes)
    #d_cumerr = (np.cumsum(d_err**2))**0.5
    
    t = (np.abs(np.array(d_rvec1) - 500)).argmin()
    d_max = d_cumflux[t]
        
    d_EE = d_cumflux/d_max
    
    if ITERATE == 0:
        t = (np.abs(d_EE - 0.95)).argmin()
        r1 = d_rvec[t]
    else:
        t = (np.abs(d_EE - ITERATE)).argmin()
        r1 = d_rvec[t]

    return(r1)

def mod_annulize(data,lim):
    radius0 = range(500)
    radius = [(i+1) for i in radius0]
    
    #if center_in[0] == 0:
    #    center = FS.centroid(data,lim)
    #else:
    #    center = center_in
    #print(radius[0])
    center = centroid(data,lim)

    #print(f"x y center_unweighted: {np.round(center_unweighted[1],2)},{np.round(center_unweighted[0],2)}")
    
    print(f"x y center used in FS.mod_annulize: {np.round(center[1],2)},{np.round(center[0],2)}")
    print()
    #980/2,1000/2
    apertures = np.array([CircularAnnulus((center[1],center[0]), r_in=i,r_out = i+(1)) for i in radius])
    
    ## This bit does the analysis without any masking.
    phot_table_list = [aperture_photometry(data, ap, method = 'exact') for ap in apertures]
    flux =[float(i['aperture_sum']) for i in phot_table_list]
    #flux_error = [(i['aperture_sum_err']) for i in phot_table_list]
    flux_error = 1
    area = np.array([i.area for i in apertures])
    sb = flux/area
    sb_error = flux_error/area
    ##
    out = np.array([radius,flux,flux_error,sb,sb_error]) 
    return(out)
    
def calibrate(date,data_set,data_set_d,lim,cwd,injection_fn):

    
    if os.path.isdir(f"{cwd}/{date}/raw/fiber") == False:
        os.mkdir(f"{cwd}/{date}/raw/fiber") 
        
    if os.path.isdir(f"{cwd}/{date}/redux") == False:
        os.mkdir(f"{cwd}/{date}/redux") 
        
    if os.path.isdir(f"{cwd}/{date}/analysis") == False:
        os.mkdir(f"{cwd}/{date}/analysis") 
        
    if os.path.isdir(f"{cwd}/{date}/redux/fiber") == False:
        os.mkdir(f"{cwd}/{date}/redux/fiber") 
    
    if os.path.isdir(f"{cwd}/{date}/redux/direct") == False:
        os.mkdir(f"{cwd}/{date}/redux/direct") 
        
    ######Convert all direct and fiber bmp to fits
    ######
    
    Type = 'direct'
    
    data_list = get_filenames(date,Type,'*',cwd)
    
    [bmp_to_fits(date,Type, data_list[i]) for i in range(len(data_list))]
    print(data_list)

    path = f"{cwd}/{date}/raw/{Type}"

    
    data_list0 = get_filenames(date,Type,f"*_{data_set_d}.*",cwd)
    data_list = [data_list0[i].split('.')[0] + '.fits' for i in range(len(data_list0))]
    
    print(f'Relevant Images {data_list}')
    print()

    null_list = []
    
    direct_med_out = [med(date,Type,[data_list[i]],0,0,null_list,[],980/2,1000/2,lim) \
     for i in range(len(data_list))]

    data_list0 = get_filenames(date,Type,f"*_{data_set_d}.*",cwd)
    data_name_list = [data_list0[i].split('.')[0]  for i in range(len(data_list0))]
    
    F = [1*np.int(data_list0[i].split('_')[0]) for i in range(len(data_list0))]
    #print(f'Backdistances {F}')
    EE_fit = np.linspace(0.15,0.95,5)

    print(data_name_list[0])
    rs0 = [get_rad(data_name_list[i],F[i],Type,EE_fit[0],lim,date,cwd) for i in range(len(data_name_list))]
    rs1 = [get_rad(data_name_list[i],F[i],Type,EE_fit[1],lim,date,cwd) for i in range(len(data_name_list))]
    rs2 = [get_rad(data_name_list[i],F[i],Type,EE_fit[2],lim,date,cwd) for i in range(len(data_name_list))]
    rs3 = [get_rad(data_name_list[i],F[i],Type,EE_fit[3],lim,date,cwd) for i in range(len(data_name_list))]
    rs4 = [get_rad(data_name_list[i],F[i],Type,EE_fit[4],lim,date,cwd) for i in range(len(data_name_list))]
    
    diams0 = [rs0[i] * 2 for i in range(len(rs0))]
    diams1 = [rs1[i] * 2 for i in range(len(rs1))]
    diams2 = [rs2[i] * 2 for i in range(len(rs2))]
    diams3 = [rs3[i] * 2 for i in range(len(rs3))]
    diams4 = [rs4[i] * 2 for i in range(len(rs4))]
    
    
    diams = np.vstack([diams0,diams1,diams2,diams3,diams4])
    locs = np.linspace(0.2,0.8,6)
    
    
    fig,ax = plt.subplots(figsize =(8,6))
    
    colors = ['r','darkorange','y','c','b']
    
    [ax.scatter(F,diams[i],c = colors[i],s = 15) for i in range(len(colors))]
    
    slopes = []
    y_int = []
    
    for i in range(len(colors)):
        line = np.poly1d(np.polyfit(F,diams[i],1))
        slopes.append(line[1])
        y_int.append(line[0])
        
        fitr = np.linspace(0,25,500)
        fitEE = line(fitr)
        ax.plot(fitr,fitEE,color = colors[i],linestyle = '--',linewidth = 1,\
                label = 'EE'+str(np.int(np.round(EE_fit[i],2)*100)) +' as D')
    
        ax.text(0.98, locs[i],  'f = ' + str(np.round(-(line[0])/line[1],3))+'mm',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color=colors[i], fontsize=14)
    d_slopes = slopes
    plt.legend()
    ax.set_xlim(-5,25)
    ax.set_ylim(-1,6)
    ax.set_ylabel('D (mm)')
    ax.set_xlabel('f (mm)')
    
    
    ax.set_title('Direct Beam Platescale as a Function of EE')
    ax.grid()
    plt.tight_layout()
    
    plt.savefig(f"{cwd}/{date}/analysis/{data_set_d}_direct_beam_calibration_{injection_fn}.png")
    
    #### Lets grab the common x intercept of all the EE slopes: 
    
    x_int = [-(y_int[i])/slopes[i] for i in range(len(slopes))]
    
    direct_f = np.mean(np.array([x_int[0],x_int[1],x_int[2],x_int[3],x_int[4]]))
    print()
    print(f'Direct image calibration {direct_f}')
    print()
    [print(f"Injection f/# = {1/(slopes[i])}") for i in range(len(slopes))]
    #[print((F[i] - direct_f)/(diams4[i])) for i in range(len(diams4))]
    print()

    return(direct_f,direct_med_out)

def get_lists(cwd,date,query,query_d,query_null_f):

    #######################################################
    Type = 'direct'
    
    ## That last period gets the science data
    path_list = glob.glob(f"{cwd}/{date}/raw/{Type}/{query_d}.*")
    
    #print(cwd)
    #print(path_list)
    print(path_list[0].split('/')[7])
    
    
    
    data_list = []
    for i in range(len(path_list)):
        if path_list[i].split('/')[8].split('.')[2] == 'FIT':
            data_list.append(path_list[i].split('/')[8])
            
    data_list = sorted(data_list)
    print(data_list[0])
    print()
    ########################################################
    dir_img_list = data_list
    
    
    #######################################################
    Type = 'direct'
    
    ## That last period gets the science data
    path_list = glob.glob(f"{cwd}/{date}/raw/{Type}/{query_d}_*") ###   _
    
    data_list = []
    for i in range(len(path_list)):
        if path_list[i].split('/')[8].split('.')[2] == 'FIT':
            data_list.append(path_list[i].split('/')[8])
            
    data_list = sorted(data_list)
    print(data_list[0])
    print()

    ########################################################
    dir_null_list = data_list
    
    
    
    #######################################################
    Type = 'fiber'
    
    ## That last period gets the science data
    path_list = glob.glob(f"{cwd}/{date}/raw/{Type}/{query}.*")
    
    data_list = []
    for i in range(len(path_list)):
        if path_list[i].split('/')[8].split('.')[2] == 'FIT':
            data_list.append(path_list[i].split('/')[8])
            
    data_list = sorted(data_list)
    print(data_list[0])
    print()

    ########################################################
    fib_img_list = data_list
    
    #######################################################
    ## That last period gets the science data
    path_list = glob.glob(f"{cwd}/{date}/raw/{Type}/{query_null_f}_*")
    
    data_list = []
    for i in range(len(path_list)):
        if path_list[i].split('/')[8].split('.')[2] == 'FIT':
            data_list.append(path_list[i].split('/')[8])
            
    data_list = sorted(data_list)
    print(data_list[0])
    print()

    ########################################################
    fib_null_list = data_list
    
    return(dir_img_list,dir_null_list,fib_img_list,fib_null_list)

def main():
    # Set your working directory and parameters
    cwd = r"C:\\Users\\jakes\\Documents\\Madison Research Code\\Test Stand Code"
    date = "20230113"  # <-- Change to your dataset date
    query = "600FBP_test.00003082.FIT"  # <-- Change to your fiber FITS filename (without path)
    query_d = "your_direct_file_name"  # <-- Change to your direct FITS filename (without path)
    query_null_f = "600FBP_test_null.00003082"  # <-- Change if you have a null fiber file
    lim = 900  # <-- Adjust threshold as needed
    bgsub = True  # <-- Set to True if you want background subtraction

    produce(cwd, date, query, query_d, query_null_f, lim, bgsub)

if __name__ == "__main__":
    main()