from astropy.io import fits
import astropy.units as u
import hermite as h
from scipy import trapz
from lmfit import Model
import numpy as np
import pylab as pl


ATCA_cube = fits.open('data/ngc1512.line.4km.na.icln.fits') #loading fits data cube

data = ATCA_cube[0].data

rms_noise = np.std(data[:,0, :, :])    #rms noise in line-free channel

x = np.arange(125)  #channels
velocity = 700 + 4.*x    #1st cannel + chan_width*x

file = open('files/h_3_cons_chan.txt', 'w')
file.write('%4s %4s %4s %4s %4s %4s %4s\n' %('i','j','fit_amp', 'fit_mean', 'fit_width', 'h3', 'her3_area'))

gmodel = Model(h.Hermite)    #modelling hermite polynomial of 3rd order

for i in np.arange(778):
	for j in np.arange(779):

		mean = velocity[np.where(data[0,:, i, j] == np.max(data[0,:, i, j]))]  #mean
		guess_params = [np.max(data[0,:, i, j]), mean[0], 3.5, 0.2]   #initial parameters for fitting

		if np.max(data[0,:, i, j]) > 3.*rms_noise:     #applying 3*sigma flux cut
			#fitting a 3rd Gauss-Hermite polynomial to parameterise the cube
			"""
           
            checking for 3 consecutive
            channels with max(flux) > 3sig
            
            """
			fl_cut = np.where(data[0,:, i, j] > 3.*rms_noise)
			cons_chann = np.where(np.diff(fl_cut[0]) != 1.)
			fl_cut = np.delete(fl_cut, cons_chann, None)


			if len(fl_cut) >= 3.0: #and sum(np.diff(fl_cut[0][cons_chann])) == len(np.diff(fl_cut[0][cons_chann])):

				fit = gmodel.fit(data[0,:, i, j], x = velocity, amp = guess_params[0], mu = guess_params[1], sig = guess_params[2], h3 = guess_params[3])
				fit_amp, fit_mu, fit_sig, fit_h3 = fit.best_values['amp'], fit.best_values['mu'], fit.best_values['sig'], fit.best_values['h3']
				her3_area = trapz(fit.best_fit, velocity)
				file.write('%4d %4d %6.5f %6.5f %6.5f %6.3f %6.3f\n'%(i, j, fit_amp, fit_mu, fit_sig, fit_h3, her3_area))
				
				if i in range(500, 510) and j in range(500, 510):
					pl.plot(velocity, data[0,:, i, j]*1000, 'k.',linewidth = 1.25, label = 'HI_line emission')
					pl.plot(velocity, fit.best_fit*1000, 'g-',linewidth = 1.25, label = 'Fitted HER3')
					pl.axhline(3000*rms_noise, color = 'grey', ls = '-.', linewidth = 1.25, label = '3$\sigma$')
					pl.legend()
					pl.xlabel('Velocity [km/s]', fontsize = 14)
					pl.ylabel('Flux [kJy beam$^{-1}$]', fontsize = 14)
					pl.savefig('images/profile_her_cons_chan'+str(i)+'_'+str(j)+'.png', format = 'png', dpi = 150)
					pl.close()
file.close()
         
