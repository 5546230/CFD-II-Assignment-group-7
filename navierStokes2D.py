"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import src.mesh as mesh
import src.mimeticSEM as mimeticSEM
from src.mimeticMachinery import sparseLinAlgTools as splaTools
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import matplotlib.colors as colors
from src.geometryMaps import geomdef
import sys
import os
import matplotlib.animation as pltani
import pickle
import warnings
warnings.filterwarnings('ignore')
## ============================================== ##
## ============ Main Solver function ============ ##
## ============================================== ##
def navierStokesSolver(msh: mesh.mesh2D, sem: mimeticSEM.SEM2D, dt, Ndt, Re, fx, fy, saveData = False):
	# Call global Incidence and Mass matrices
	print('-----------Assembling global matrices');
	t_arr = np.arange(0, dt*(Ndt + 1), dt);
	E10 = sem.E10();
	E21 = sem.E21();
	M0 = sem.massMatrix(0);
	M1 = sem.massMatrix(1);
	M1inv = sem.massMatrix(1, inv = True);
	M2inv = sem.massMatrix(2, inv = True);
	# Call trace matrices
	Elambda1, _ = sem.Elambda1();
	Egamma0, weak_bc_mat, _ = sem.Egamma0();
	Ethetagamma, _ = sem.Ethetagamma();
	psi0 = sem.k_formBasis(0, msh.xi, msh.eta);
	psi1x, psi1y = sem.k_formBasis(1, msh.xi, msh.eta);
	shp0, shp1 = psi1x.shape;
	zeros_12 = np.zeros((msh.nNodes, msh.nSurfs));
	# Assemble global system
	print('-----------Initialising global system and rhs');
	NSsys = sp.bmat([[M0, -E10.T@M1, None, None, Egamma0.T, None], \
				[1/(2*Re)*M1@E10, 1/dt*M1, E21.T, Elambda1.T, None, None], \
					[None, E21, None, None, None, None], \
					[None, Elambda1, None, None, None, None], \
					[Egamma0, None, None, None, None, Ethetagamma.T], \
					[None, None, None, None, Ethetagamma, None]], format = 'csc');
	# Build RHS
	top_wall_velocity = 1.0;
	bottom_wall_velocity = 0.0;
	weak_wall_tangentail_velocity = weak_bc_mat@np.concatenate((bottom_wall_velocity*(msh.wx*msh.J[:msh.N, 0, 0, :]).flatten(), top_wall_velocity*(msh.wx*msh.J[-msh.N:, 0, -1, :]).flatten()));
	# Invert sytem using Shur Complement approach
	dinv00, dinv01, dinv10, dinv11 = splaTools.shurComplementInv(dt*M1inv, E21.T, E21, np.zeros((sem.mesh.nSurfs, sem.mesh.nSurfs)));
	dinv = sp.bmat([[dinv00, dinv01], [dinv10, dinv11]]);
	Ainv11, Ainv10, Ainv01, Ainv00 = splaTools.shurComplementInv(dinv, sp.bmat([[1/(2*Re)*M1@E10], [zeros_12.T]], format = 'csc'), sp.bmat([[-E10.T@M1, zeros_12]], format = 'csc'), M0);
	Ainv = sp.bmat([[Ainv00, Ainv01], [Ainv10, Ainv11]], format = 'csc');
	B = NSsys[:sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs, sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs:];
	C = NSsys[sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs:, :sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs];
	D = NSsys[sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs:, sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs:];

	inv_D_C_Ainv_B = la.splu(D - C@Ainv@B);

	RHSx1 = np.vstack((np.zeros((sem.mesh.nNodes, shp0)), psi1x.T.toarray(), np.zeros((sem.mesh.nSurfs, shp0))));
	RHSy1 = np.vstack((np.zeros((sem.mesh.nNodes, shp0)), psi1y.T.toarray(), np.zeros((sem.mesh.nSurfs, shp0))));
	RHS2 = np.zeros((Elambda1.shape[0] + Egamma0.shape[0] + Ethetagamma.shape[0], shp0));

	Ux_lm = inv_D_C_Ainv_B.solve(RHS2 - C@Ainv@RHSx1);
	Uy_lm = inv_D_C_Ainv_B.solve(RHS2 - C@Ainv@RHSy1);

	Ux = Ainv@RHSx1 - Ainv@B@Ux_lm;
	Uy = Ainv@RHSy1 - Ainv@B@Uy_lm;

	RHS_bc = np.concatenate((weak_wall_tangentail_velocity, np.zeros(msh.nEdges + msh.nSurfs)));
	RHS_bc2 = np.zeros(Elambda1.shape[0] + Egamma0.shape[0] + Ethetagamma.shape[0]);
	weak_bc_lm = inv_D_C_Ainv_B.solve(RHS_bc2 - C@Ainv@RHS_bc);
	Ubc = np.concatenate((Ainv@RHS_bc - Ainv@B@weak_bc_lm, weak_bc_lm));

	t_psi_H_curlx = Ux[:sem.mesh.nNodes, :]@msh.w_mat.T;
	t_psi_H_divx = Ux[sem.mesh.nNodes:sem.mesh.nNodes + shp1, :]@msh.w_mat.T;
	t_psi_H1x = Ux[sem.mesh.nNodes + shp1:sem.mesh.nNodes + shp1 + sem.mesh.nSurfs, :]@msh.w_mat.T;
	# t_lambdax = Ux[:Elambda1.shape[0], :]@msh.w_mat.T;
	# t_gammax = Ux[Elambda1.shape[0]:Elambda1.shape[0] + Egamma0.shape[0], :].T;
	# t_thetax = Ux[Elambda1.shape[0] + Egamma0.shape[0]:, :].T;

	t_psi_H_curly = Uy[:sem.mesh.nNodes, :]@msh.w_mat.T;
	t_psi_H_divy = Uy[sem.mesh.nNodes:sem.mesh.nNodes + shp1, :]@msh.w_mat.T;
	t_psi_H1y = Uy[sem.mesh.nNodes + shp1:sem.mesh.nNodes + shp1 + sem.mesh.nSurfs, :]@msh.w_mat.T;
	# t_lambday = Uy[:Elambda1.shape[0], :]@msh.w_mat.T;
	# t_gammay = Uy[Elambda1.shape[0]:Elambda1.shape[0] + Egamma0.shape[0], :].T;
	# t_thetay = Uy[Elambda1.shape[0] + Egamma0.shape[0]:, :].T;

	weak_bc_H_curl = Ubc[:sem.mesh.nNodes];
	weak_bc_H_div = Ubc[sem.mesh.nNodes:sem.mesh.nNodes + shp1];
	weak_bc_H1 = Ubc[sem.mesh.nNodes + shp1:sem.mesh.nNodes + shp1 + sem.mesh.nSurfs];
	# weak_bc_lambda = Ubc[sem.mesh.nNodes + shp1 + sem.mesh.nSurfs:sem.mesh.nNodes + shp1 + sem.mesh.nSurfs + Elambda1.shape[0]];

	# Initialise solution vectors
	omega = np.zeros((t_psi_H_curlx.shape[0], Ndt));
	u = np.zeros((t_psi_H_divx.shape[0], Ndt));
	pressure = np.zeros((t_psi_H1x.shape[0], Ndt));
	# lambdas = np.zeros((t_lambdax.shape[0], Ndt));

	# Solve system
	print('-----------Solving system');
	n_correctors = 2;
	for t_idx in range(Ndt - 1):
		for i in range(n_correctors):
			rhsx = fx(msh.x.flatten(), msh.y.flatten(), t_idx*dt + dt/2) + psi1x@(1/dt*u[:, t_idx] - 1/(2*Re)*E10@omega[:, t_idx]) - (0.5**2)**(i > 0)*((psi0@(omega[:, t_idx] + omega[:, t_idx + 1]))*(-psi1y@(u[:, t_idx] + u[:, t_idx + 1])));
			rhsy = fy(msh.x.flatten(), msh.y.flatten(), t_idx*dt + dt/2) + psi1y@(1/dt*u[:, t_idx] - 1/(2*Re)*E10@omega[:, t_idx]) - (0.5**2)**(i > 0)*((psi0@(omega[:, t_idx] + omega[:, t_idx + 1]))*(psi1x@(u[:, t_idx] + u[:, t_idx + 1])));
			omega[:, t_idx + 1] = weak_bc_H_curl + t_psi_H_curlx@rhsx + t_psi_H_curly@rhsy;
			u[:, t_idx + 1] = weak_bc_H_div + t_psi_H_divx@rhsx + t_psi_H_divy@rhsy;
			pressure[:, t_idx + 1] = weak_bc_H1 + t_psi_H1x@rhsx + t_psi_H1y@rhsy;
			# lambdas[:, t_idx + 1] = weak_bc_lambda + t_lambdax@rhsx + t_lambday@rhsy;
		sys.stdout.write('\rt = %0.4f s,	max(DIV_u^{n + 1}) = %0.3e,	max((u^{n + 1} - u^{n})/dt) = %0.3e'%(t_arr[t_idx + 1], np.abs(E21@(u[:, t_idx + 1])).max(), np.abs(u[:, t_idx + 1] - u[:, t_idx]).max()/dt));
		# sys.stdout.flush();
	sys.stdout.write('\n');

	# Retrieve degrees of freedom
	DIVu = E21@u;
	streamFunc = la.spsolve(sp.bmat([[E10.T@E10, Egamma0.T, None], [Egamma0, None, Ethetagamma.T], [None, Ethetagamma, None]], format = 'csr'), np.vstack((E10.T@u, np.zeros((Egamma0.shape[0], Ndt)), np.zeros((Ethetagamma.shape[0], Ndt)))))[:E10.shape[1], :];
	pressure = M2inv@pressure;
	if (saveData):
		print('-----------Saving data');
		with open('./savedData/solution_p_%i_NxM_%ix%i_Re_%0.1f_dt_%0.2e.pkl'%(p, N, M, Re, dt), 'wb') as file:
			pickle.dump({'t_arr': t_arr, 'omega': omega, 'u': u, 'pressure': pressure, 'streamFunc': streamFunc, 'DIVu': DIVu}, file);
			file.close();
	print('-----------Solve completed');
	return t_arr, omega, u, pressure, streamFunc, DIVu;
## ============================================== ##
## ============================================== ##

if __name__ == '__main__':
	## ============================================== ##
	## =================== Inputs =================== ##
	## ============================================== ##
	# Polynomial degree
	p = 4;
	# Number of elements in x-direction
	N = 16;
	# Number of elements in y-direction
	M = 8;
	# Number of nodes to plot in
	pRefined = p + 2;
	# Time step size
	dt = 1e-3;
	# Number of time steps
	Ndt = 20001;
	# Reynolds number
	Re = 40;
	# Geometry definition
	geomMap = geomdef();
	# Source term of PDE
	fx = lambda x, y, t: 0.25 + 0.25*np.tanh(-15*(y - 0.5));
	fy = lambda x, y, t: np.zeros_like(x);
	# Boolian flag to save data
	saveData = True;
	# Boolian flag whether or not to force the simulation to run
	forceRun = False;
	## ============================================== ##
	## ============================================== ##

	## ============================================== ##
	## ============== Assemble objects ============== ##
	## ============================================== ##
	X0, X1 = geomMap.X0, geomMap.X1;
	mapping_parms = geomMap.mapping_parms;
	Phi, dPhi = geomMap.Phi, geomMap.dPhi;
	msh = mesh.mesh2D(X0, X1, N, M, p, Phi, dPhi, mapping_parms, pRefined);
	sem = mimeticSEM.SEM2D(msh);
	msh.buildMesh();
	## ============================================== ##
	## ============================================== ##

	## ============================================== ##
	## ========= Solve Navier-Stokes system ========= ##
	## ============================================== ##
	if not os.path.isdir('savedData'):
		os.mkdir('savedData')
	if forceRun:
		t_arr, omega, u, pressure, streamFunc, DIVu = navierStokesSolver(msh, sem, dt, Ndt, Re, fx, fy, saveData = saveData);
	else:
		try:
			with open('./savedData/solution_p_%i_NxM_%ix%i_Re_%0.1f_dt_%0.2e.pkl'%(p, N, M, Re, dt), 'rb') as file:
					data = pickle.load(file);
					print('-----------Found saved data')
					saveData = False
					file.close();
			# print(data.keys())
			t_arr, omega, u, pressure, streamFunc, DIVu = data['t_arr'], data['omega'], data['u'], data['pressure'], data['streamFunc'], data['DIVu'];
		except FileNotFoundError:
			t_arr, omega, u, pressure, streamFunc, DIVu = navierStokesSolver(msh, sem, dt, Ndt, Re, fx, fy, saveData = saveData);

	## ============================================== ##
	## ============================================== ##

	print(np.abs(u[:, -1] - u[:, -2]).max()/dt)

	## ============================================== ##
	## ============ Post process solution =========== ##
	## ============================================== ##
	# Call basis functions
	psi0_plot = sem.k_formBasis(0, msh.xi_hiDOP, msh.eta_hiDOP);
	psi1x_plot, psi1y_plot = sem.k_formBasis(1, msh.xi_hiDOP, msh.eta_hiDOP);
	psi2_plot = sem.k_formBasis(2, msh.xi_hiDOP, msh.eta_hiDOP);

	# Compute integral of vorticity over the domain
	integral_omega = msh.W@omega[:, 1:];

	# Reconstruct solution (Note, this reconstruction can be very memory intensive for fine meshes with many time steps.
	# To save cost, you can slice the array so as to reconstruct only a select few time steps instead of all the time steps)
	omega_Reconstruct = msh.gridit((psi0_plot@omega).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	streamFunc_Reconstruct = msh.gridit((psi0_plot@streamFunc).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	DIVu_Reconstruct = msh.gridit((psi2_plot@DIVu).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	p_Reconstruct = msh.gridit((psi2_plot@pressure).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	u_Reconstruct = msh.gridit((psi1x_plot@u).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	v_Reconstruct = msh.gridit((psi1y_plot@u).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));

	msh.plotMesh();

	skip_frames = 20;
	omega_Reconstruct_plot = omega_Reconstruct[..., ::skip_frames];
	t_arr_plot = t_arr[::skip_frames];
	levels = np.arange(-30, 31, 2);
	fig = plt.figure(figsize = (19.2, 6.4));
	ax = fig.add_subplot(111);
	ax.set_xlabel(r'$x$');
	ax.set_ylabel(r'$y$');
	ax.set_aspect('equal');
	cax = ax.contourf(msh.xPlot, msh.yPlot, omega_Reconstruct_plot[..., 1], levels = levels, cmap = plt.cm.twilight_shifted, extend = 'both');
	cax1 = ax.contourf(msh.xPlot + 2, msh.yPlot, omega_Reconstruct_plot[..., 1], levels = levels, cmap = plt.cm.twilight_shifted, extend = 'both');
	cax2 = ax.contourf(msh.xPlot + 4, msh.yPlot, omega_Reconstruct_plot[..., 1], levels = levels, cmap = plt.cm.twilight_shifted, extend = 'both');
	text = ax.text(0.05, 1.05, '', transform = ax.transAxes, usetex = False);
	fig.colorbar(cax, orientation = 'horizontal');
	
	def init():
		return cax, cax1, cax2, text;

	def step(i):
		for c in ax.collections: c.remove();
		cax = ax.contourf(msh.xPlot, msh.yPlot, omega_Reconstruct_plot[..., i + 1], levels = levels, cmap = plt.cm.twilight_shifted, extend = 'both');		
		cax1 = ax.contourf(msh.xPlot + 2, msh.yPlot, omega_Reconstruct_plot[..., i + 1], levels = levels, cmap = plt.cm.twilight_shifted, extend = 'both');		
		cax2 = ax.contourf(msh.xPlot + 4, msh.yPlot, omega_Reconstruct_plot[..., i + 1], levels = levels, cmap = plt.cm.twilight_shifted, extend = 'both');		
		text.set_text('time = %.3fs'%(t_arr_plot[i]));
		return cax, cax1, cax2, text;

	ani = pltani.FuncAnimation(fig, step, omega_Reconstruct_plot.shape[-1] - 1, init_func = init, interval = 30);
	if saveData:
		ani.save('%s.gif'%('./figures/geom4_solution_anim_p_%i_NxM_%ix%i_Re_%0.1f_dt_%0.2e.pkl'%(p, N, M, Re, dt)), writer = 'ffmpeg');
	# plt.show();

	static_timeStamps = (np.linspace(0, 1, 5)*(Ndt - 1)).astype(int);
	static_timeStamps = static_timeStamps[-2:]
	
	for i in static_timeStamps[1:]:
		# Plot of vorticity
		fig0 = plt.figure('t = %0.1f'%t_arr[i]);
		ax0 = fig0.add_subplot(111);
		ax0.set_xlabel(r'$x$');
		ax0.set_ylabel(r'$y$');
		ax0.set_aspect('equal');
		cax0 = ax0.contourf(msh.xPlot, msh.yPlot, omega_Reconstruct[..., i], levels = levels, cmap = 'viridis', extend = 'both');
		fig0.colorbar(cax0, orientation = 'horizontal');
		fig0.tight_layout()

		fig1 = plt.figure();
		ax1 = fig1.add_subplot(111)
		ax1.set_xlabel(r'$x$');
		ax1.set_ylabel(r'$y$');
		ax1.set_aspect('equal');
		cax1 = ax1.pcolormesh(msh.xPlot, msh.yPlot, DIVu_Reconstruct[..., i], norm=colors.SymLogNorm(1e-14), cmap = 'viridis');
		fig1.colorbar(cax1, orientation = 'horizontal')
		fig1.tight_layout()

		y = np.linspace(0, 1, 20, endpoint = True);
		# find the streamfunc values at (0, y)
		streamfuncIsoValues = np.zeros_like(y);
		for j in range(len(y)):
			idx = np.argmin(np.abs(msh.yPlot[:, 0] - y[j]));
			streamfuncIsoValues[j] = streamFunc_Reconstruct[..., i][idx, 0];
		uMag = np.sqrt(u_Reconstruct**2 + v_Reconstruct**2);
		fig2 = plt.figure();
		ax2 = fig2.add_subplot(111);
		lev = np.linspace(0, uMag.max(), 20)
		cax2 = ax2.contourf(msh.xPlot, msh.yPlot, uMag[..., i], cmap = 'viridis', extend = 'max', levels = lev);
		ax2.contour(msh.xPlot, msh.yPlot, streamFunc_Reconstruct[..., i], levels = sorted(streamfuncIsoValues), colors = 'black', linestyles = 'solid', );
		ax2.set_aspect('equal');
		ax2.set_xlabel(r'$x$');
		ax2.set_ylabel(r'$y$');
		fig2.colorbar(cax2, orientation = 'horizontal');
		fig2.tight_layout();

		fig3, ax3 = plt.subplots(1, 3, figsize = (12, 4), sharey = True)
		ax3[0].set_ylabel(r'$u_x$')
		for x, axs in zip([0.5, 1.0, 1.5], ax3):
			ndx = np.isclose(msh.xPlot, x)
			axs.plot(msh.yPlot[ndx], u_Reconstruct[..., i][ndx])
			axs.set_xlabel(r'$y$')
			axs.set_title(f'x = {x}')
			# axs.set_aspect('equal')
			axs.grid()
		fig3.tight_layout()	
		# ax3.plot(x_locs, interp, 'o', label = 'Numerical')

		fig4 = plt.figure()
		ax4 = fig4.add_subplot(111)
		cax4 = ax4.contourf(msh.xPlot, msh.yPlot, p_Reconstruct[..., i], cmap='viridis', levels = 20, extend='both');
		ax4.set_aspect('equal');
		ax4.set_xlabel(r'$x$');
		ax4.set_ylabel(r'$y$');
		fig4.colorbar(cax4, orientation = 'horizontal');
		fig4.tight_layout();

		fig5 = plt.figure()
		ax5 = fig5.add_subplot(111)
		ax5.plot(t_arr[2:], (integral_omega + 2))
		ax5.set_xlabel(r'$t$')
		ax5.set_ylabel(r'$\int \omega dA -\int \omega_{exact} dA $')
		ax5.grid()
		# log scale
		# ax5.set_yscale('symlog')
		# add ticks for the log scale
		# ax5.yaxis.set_minor_locator(plt.MultipleLocator(1))
		# ax5.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}' if x > 0 else ''))
		# ax5.yaxis.set_major_locator(plt.MultipleLocator(.1))


		fig5.tight_layout()



		if i == 20000:
			fig0.savefig(f'figures/navier_stokes_vorticity_Re{Re}_dt{dt}_t{i}.png')
			fig1.savefig(f'figures/navier_stokes_divergence_Re{Re}_dt{dt}_t{i}.png')
			fig2.savefig(f'figures/navier_stokes_streamlines_Re{Re}_dt{dt}_t{i}.png')
			fig3.savefig(f'figures/navier_stokes_profile_Re{Re}_dt{dt}_t{i}.pdf')
			fig4.savefig(f'figures/navier_stokes_pressure_Re{Re}_dt{dt}_t{i}.png')
			fig5.savefig(f'figures/navier_stokes_integral_Re{Re}_dt{dt}.pdf')
		

		# Now implement your own plotting routine for the streamlines, the divergence plots, etc

	# plt.show();
	
	## ============================================== ##
	## ============================================== ##
