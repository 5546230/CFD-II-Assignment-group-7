"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import src.mesh as mesh
import src.mimeticSEM as mimeticSEM
import matplotlib.pyplot as plt
import numpy as np
from src.geometryMaps import squareGeom
from stokesFlow2D import stokesSolver
from mpltools import annotation
from scipy.stats import linregress
## ============================================== ##
## =================== Inputs =================== ##
## ============================================== ##
p_arr = np.arange(1, 5, dtype = int);
N_arr = np.arange(5, 20, 2, dtype = int);
# Geometry definition
geomMap = squareGeom();
## ============================================================== ##
## +++++++++++++++++++ Fill in Source term of PDE +++++++++++++++ ##
## +++++++++++ corresponding to the manufactured solution +++++++ ##
## ++++++++++++++++++++++ in the lines below ++++++++++++++++++++ ##
## ============================================================== ##
fx = lambda x, y: -8*np.pi**2*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)-2*np.pi*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)# Add x−component of the forcing term as a function of x and y
fy = lambda x, y: -4*np.pi**2*np.sin(2*np.pi*x)*(1 - np.cos(2*np.pi*y)) + 4*np.pi**2*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + 2*np.pi*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)# Add y−component of the forcing term as a function of x and y
## ============================================================== ## # (Remove later); -4*np.pi**2*np.sin(2*np.pi*x) + 8*np.pi**2*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + 2*np.pi*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
## ============================================================== ##

## ============================================================== ##
## =================== Manufactured solution ==================== ##
## ============================================================== ##
uExact = lambda x, y: y + np.cos(2*np.pi*x)*np.sin(2*np.pi*y);
vExact = lambda x, y: np.sin(2*np.pi*x)*(1 - np.cos(2*np.pi*y));
pExact = lambda x, y: np.sin(2*np.pi*x)*np.cos(2*np.pi*y);
omegaExact = lambda x, y: -2*np.pi*np.cos(2*np.pi*x)*(1 - np.cos(2*np.pi*y)) + 1 + 2*np.pi*np.cos(2*np.pi*x)*np.cos(2*np.pi*y);
curl_x_omegaExact = lambda x, y: -8*np.pi**2*np.cos(2*np.pi*x)*np.sin(2*np.pi*y);
curl_y_omegaExact = lambda x, y: -4*np.pi**2*np.sin(2*np.pi*x)*(1 - np.cos(2*np.pi*y)) + 4*np.pi**2*np.sin(2*np.pi*x)*np.cos(2*np.pi*y);
DIVuExact = lambda x, y: np.zeros_like(x);
## ============================================================== ##
## ============================================================== ##

## ============================================================== ##
## =================== Error norms definition =================== ##
## ============================================================== ##
def computeL2_error(phi_Reconstruct, phi_exact, w):
	L2_error = np.sqrt((phi_Reconstruct - phi_exact)**2@w);
	return L2_error;

def computeHdiv_error(u_Reconstruct, v_Reconstruct, DIVu_Reconstruct, x, y, w, uExact = uExact, vExact = vExact, DIVuExact = DIVuExact):
	error_u_L2_pow2 = computeL2_error(u_Reconstruct, uExact(x, y), w)**2 + computeL2_error(v_Reconstruct, vExact(x, y), w)**2;
	error_DIVu_L2_pow2 = computeL2_error(DIVu_Reconstruct, DIVuExact(x, y), w)**2;
	error_Hdiv = np.sqrt(error_u_L2_pow2 + error_DIVu_L2_pow2);
	return np.sqrt(error_u_L2_pow2), error_Hdiv;

def computeHcurl_error(omega_Reconstruct, curl_x_omega_Reconstruct, curl_y_omega_Reconstruct, x, y, w, omegaExact = omegaExact, curl_x_omegaExact = curl_x_omegaExact, curl_y_omegaExact = curl_y_omegaExact):
	error_omega_L2_pow2 = computeL2_error(omega_Reconstruct, omegaExact(x, y), w)**2;
	error_curl_omega_L2_pow2 = computeL2_error(curl_x_omega_Reconstruct, curl_x_omegaExact(x, y), w)**2 + computeL2_error(curl_y_omega_Reconstruct, curl_y_omegaExact(x, y), w)**2;
	error_Hcurl = np.sqrt(error_omega_L2_pow2 + error_curl_omega_L2_pow2);
	return np.sqrt(error_omega_L2_pow2), error_Hcurl;
## ============================================================== ##
## ============================================================== ##

## ============================================================== ##
## ======= Compute errors for different mesh refinements ======== ##
## ============================================================== ##
error_lst = np.zeros((5, p_arr.shape[0], N_arr.shape[0]));
elem_h = np.zeros((p_arr.shape[0], N_arr.shape[0]));
for i, p in enumerate(p_arr):
	for j, N in enumerate(N_arr):
		X0, X1 = geomMap.X0, geomMap.X1;
		mapping_parms = geomMap.mapping_parms;
		Phi, dPhi = geomMap.Phi, geomMap.dPhi;
		msh = mesh.mesh2D(X0, X1, N, N, p, Phi, dPhi, mapping_parms, p + 4);
		sem = mimeticSEM.SEM2D(msh);
		msh.buildMesh();

		omega, u, pressure, streamFunc, DIVu, CURLomega = stokesSolver(msh, sem, fx, fy);

		psi0_plot = sem.k_formBasis(0, msh.xi_hiDOP, msh.eta_hiDOP);
		psi1x_plot, psi1y_plot = sem.k_formBasis(1, msh.xi_hiDOP, msh.eta_hiDOP);
		psi2_plot = sem.k_formBasis(2, msh.xi_hiDOP, msh.eta_hiDOP);

		error_lst[0, i, j], error_lst[1, i, j] = computeHdiv_error(psi1x_plot@u, psi1y_plot@u, psi2_plot@DIVu, msh.x_hiDOP.flatten(), msh.y_hiDOP.flatten(), msh.W_hiDOP);
		error_lst[2, i, j], error_lst[3, i, j] = computeHcurl_error(psi0_plot@omega, psi1x_plot@CURLomega, psi1y_plot@CURLomega, msh.x_hiDOP.flatten(), msh.y_hiDOP.flatten(), msh.W_hiDOP);
		error_lst[4, i, j] = computeL2_error(psi2_plot@pressure, pExact(msh.x_hiDOP.flatten(), msh.y_hiDOP.flatten()), msh.W_hiDOP);
		print('p = %i, N = %i,	error_u_L2 = %0.5e,	error_u_Hdiv = %0.5e,	error_omega_L2 = %0.5e,\n		error_omega_Hcurl = %0.5e,	error_pressure_L2 = %0.5e\n'%(p, N, *error_lst[:, i, j]));
		elem_h[i, j] = 1/np.sqrt(np.mean(msh.h_x.flatten()*msh.h_y.flatten()));
## ============================================================== ##
## ============================================================== ##


## ============================================================== ##
## ======================= Error plots ========================== ##
## ============================================================== ##
# Save data
save_data = True;

fig0 = plt.figure();
ax0 = fig0.add_subplot(111);

fig1 = plt.figure();
ax1 = fig1.add_subplot(111);

fig2 = plt.figure();
ax2 = fig2.add_subplot(111);

fig3 = plt.figure();
ax3 = fig3.add_subplot(111);

fig4 = plt.figure();
ax4 = fig4.add_subplot(111);

for i, p in enumerate(p_arr):
	# Error u in L2
	ax0.loglog(elem_h[i, :], error_lst[0, i, :], label = r'$p = %i$'%p);
	slope = linregress(np.log10(elem_h[i, -2:]), np.log10(error_lst[0, i, -2:]))[0];
	annotation.slope_marker((elem_h[i, elem_h.shape[1]*5//9], error_lst[0, i, elem_h.shape[1]*5//9]*1.1), (round(slope, 0), 1), ax = ax0, invert = False);
	ax0.grid(True);
	ax0.minorticks_on();
	ax0.grid(visible = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax0.tick_params(axis = 'both', which = 'minor', labelsize = 10);
	
	ax0.set_xlabel(r'$\dfrac{1}{\sqrt{h_x h_y}}$');
	ax0.set_ylabel(r'$|| u - u_{ex} ||_{L^2}$');
	ax0.legend(bbox_to_anchor = (0., 1.01, 1., .102), loc = 'lower left',
					ncol = 4, mode = 'expand', borderaxespad = 0.);
	
	# Error u in Hdiv
	ax1.loglog(elem_h[i, :], error_lst[1, i, :], label = r'$p = %i$'%p);
	slope = linregress(np.log10(elem_h[i, -2:]), np.log10(error_lst[1, i, -2:]))[0];
	annotation.slope_marker((elem_h[i, elem_h.shape[1]*5//9], error_lst[1, i, elem_h.shape[1]*5//9]*1.1), (round(slope, 0), 1), ax = ax1, invert = False);
	ax1.grid(True);
	ax1.minorticks_on();
	ax1.grid(visible = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax1.tick_params(axis = 'both', which = 'minor', labelsize = 10);
	
	ax1.set_xlabel(r'$\dfrac{1}{\sqrt{h_x h_y}}$');
	ax1.set_ylabel(r'$\left(|| u - u_{ex} ||_{L^2}^2 + || \nabla \cdot u - \nabla \cdot u_{ex} ||_{L^2}^2 \right)^{\dfrac{1}{2}}$');
	ax1.legend(bbox_to_anchor = (0., 1.01, 1., .102), loc = 'lower left',
					ncol = 4, mode = 'expand', borderaxespad = 0.);
	
	# Error omega in L2
	ax2.loglog(elem_h[i, :], error_lst[2, i, :], label = r'$p = %i$'%p);
	slope = linregress(np.log10(elem_h[i, -2:]), np.log10(error_lst[2, i, -2:]))[0];
	annotation.slope_marker((elem_h[i, elem_h.shape[1]*5//9], error_lst[2, i, elem_h.shape[1]*5//9]*1.1), (round(slope, 0), 1), ax = ax2, invert = False);
	ax2.grid(True);
	ax2.minorticks_on();
	ax2.grid(visible = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax2.tick_params(axis = 'both', which = 'minor', labelsize = 10);
	
	ax2.set_xlabel(r'$\dfrac{1}{\sqrt{h_x h_y}}$');
	ax2.set_ylabel(r'$|| \bar{\omega} - \omega_{ex} ||_{L^2}$');
	ax2.legend(bbox_to_anchor = (0., 1.01, 1., .102), loc = 'lower left',
					ncol = 4, mode = 'expand', borderaxespad = 0.);
	
	# Error omega in Hcurl
	ax3.loglog(elem_h[i, :], error_lst[3, i, :], label = r'$p = %i$'%p);
	slope = linregress(np.log10(elem_h[i, -2:]), np.log10(error_lst[3, i, -2:]))[0];
	annotation.slope_marker((elem_h[i, elem_h.shape[1]*5//9], error_lst[3, i, elem_h.shape[1]*5//9]*1.1), (round(slope, 0), 1), ax = ax3, invert = False);
	ax3.grid(True);
	ax3.minorticks_on();
	ax3.grid(visible = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax3.tick_params(axis = 'both', which = 'minor', labelsize = 10);
	
	ax3.set_xlabel(r'$\dfrac{1}{\sqrt{h_x h_y}}$');
	ax3.set_ylabel(r'$\left(|| \bar{\omega} - \omega_{ex} ||_{L^2}^2 + || \nabla \times \bar{\omega} - \nabla \times \omega_{ex} ||_{L^2}^2\right)^{\dfrac{1}{2}}$');
	ax3.legend(bbox_to_anchor = (0., 1.01, 1., .102), loc = 'lower left',
					ncol = 4, mode = 'expand', borderaxespad = 0.);
	# Error pressure in L2
	ax4.loglog(elem_h[i, :], error_lst[4, i, :], label = r'$p = %i$'%p);
	slope = linregress(np.log10(elem_h[i, -2:]), np.log10(error_lst[3, i, -2:]))[0];
	annotation.slope_marker((elem_h[i, elem_h.shape[1]*5//9], error_lst[4, i, elem_h.shape[1]*5//9]*1.1), (round(slope, 0), 1), ax = ax4, invert = False);
	ax4.grid(True);
	ax4.minorticks_on();
	ax4.grid(visible = True, which = 'minor', color = '#999999', linestyle = '-', alpha = 0.2);
	ax4.tick_params(axis = 'both', which = 'minor', labelsize = 10);
	
	ax4.set_xlabel(r'$\dfrac{1}{\sqrt{h_x h_y}}$');
	ax4.set_ylabel(r'$|| \bar{p} - p_{ex} ||_{L^2}$');
	ax4.legend(bbox_to_anchor = (0., 1.01, 1., .102), loc = 'lower left',
					ncol = 4, mode = 'expand', borderaxespad = 0.);
	
	fig0.tight_layout()
	fig1.tight_layout()
	fig2.tight_layout()
	fig3.tight_layout()
	fig4.tight_layout()

if save_data:
	fig0.savefig('error_figures/Error_u_in_L2.pdf');
	fig1.savefig('error_figures/Error_u_in_Hdiv.pdf');
	fig2.savefig('error_figures/Error_omega_in_L2.pdf');
	fig3.savefig('error_figures/Error_omega_in_Hcurl.pdf');
	fig4.savefig('error_figures/Error_pressure_in_L2.pdf');
else:
	plt.show();
## ============================================================== ##
## ============================================================== ##
