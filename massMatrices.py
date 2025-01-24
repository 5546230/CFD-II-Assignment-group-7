import numpy as np

def assemble_element_M0(psi0_i, psi0_j, w_xi, w_eta, det_F, dx_dxi, dx_deta, dy_dxi, dy_deta):
	""" 
		This function is called internally to generate the mass matrix
		M0 for a single element with polynomial degree 'p'.
		The function must return a numpy array of size 
		((p + 1)*(p + 1)) x ((p + 1)*(p + 1)).
		The input arguments 'psi0_i', 'psi0_j' are the ((p + 1)*(p + 1)) 
		basis functions evaluated at the Gauss-Lobatto quadrature nodes.
		'w_xi', 'w_eta' are the Gauss-Lobatto weights, and 'det_F, dx_dxi, 
		dx_deta, dy_dxi, dy_deta' are the Jacobian of the mapping along 
		with its components. 

		psi0_i, psi0_j are 3D numpy arrays of shape (((p + 1)*(p + 1)), (p + 1), (p + 1))
		det_F, dx_dxi, dx_deta, dy_dxi, dy_deta are 2D numpy arrays of shape ((p + 1), (p + 1))
		w_xi, w_eta are 1D numpy arrays of shape (p + 1)
	"""
	## ============================================================== ##
	## ++++++++++++++ Implement the zero-form mass matrix +++++++++++ ##
	## ++++++++++++++++++++++++ assembly below ++++++++++++++++++++++ ##
	## ============================================================== ##
	w = np.outer(w_xi, w_eta)

	# vectorized version
	M0 = np.einsum('ikl,kl,jkl->ij', psi0_i, w*det_F, psi0_j, optimize=True)	

	return M0

def assemble_element_M1(psi1_xi_i, psi1_eta_i, psi1_xi_j, psi1_eta_j, w_xi, w_eta, det_F, dx_dxi, dx_deta, dy_dxi, dy_deta):
	""" 
		This function is called internally to generate the mass matrix
		M1 for a single element with polynomial degree 'p'.
		The function must return a numpy array of size 
		(2*p*(p + 1)) x (2*p*(p + 1)).
		The input arguments 'psi1_xi_i, psi1_eta_i, psi1_xi_j, psi1_eta_j' are the 
		basis functions evaluated at the Gauss-Lobatto quadrature nodes. 'w_xi', 'w_eta' 
		are the Gauss-Lobatto weights, and 'det_F, dx_dxi, dx_deta, dy_dxi, dy_deta' are the Jacobian
		of the mapping along with its components. 

		psi_xi_i, psi_eta_i, psi_xi_j, psi_eta_j are 3D numpy arrays of shape (p*(p + 1), (p + 1), (p + 1)) 
		det_F, dx_dxi, dx_deta, dy_dxi, dy_deta are 2D numpy arrays of shape ((p + 1), (p + 1))
		w_xi, w_eta are 1D numpy arrays of shape (p + 1)
	"""
	## ============================================================== ##
	## ++++++++++++++ Implement the one-form mass matrix ++++++++++++ ##
	## ++++++++++++++++++++++++ assembly below ++++++++++++++++++++++ ##
	## ============================================================== ##
	w = np.outer(w_xi, w_eta)

	F = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])

	FTF = np.einsum('ij...,ik...->kj...', F, F) # F^T*F 

	# vectorized version
	M_xi_xi = np.einsum('ikl,kl,jkl->ij', psi1_xi_i, FTF[0, 0] * w / det_F, psi1_xi_j, optimize=True)
	M_xi_eta = np.einsum('ikl,kl,jkl->ij', psi1_xi_i, FTF[0, 1] * w / det_F, psi1_eta_j, optimize=True)
	M_eta_xi = np.einsum('ikl,kl,jkl->ij', psi1_eta_i, FTF[1, 0] * w / det_F, psi1_xi_j, optimize=True)
	M_eta_eta = np.einsum('ikl,kl,jkl->ij', psi1_eta_i, FTF[1, 1] * w / det_F, psi1_eta_j, optimize=True)

	M1 = np.block([[M_xi_xi, M_xi_eta], [M_eta_xi, M_eta_eta]])
	
	return M1


def assemble_element_M2(psi2_i, psi2_j, w_xi, w_eta, det_F, dx_dxi, dx_deta, dy_dxi, dy_deta):
	""" 
		This function is called internally to generate the mass matrix
		M2 for a single element with polynomial degree 'p'.
		The function must return a numpy array of size 
		(p*p) x (p*p).
		The input arguments 'psi2_i', 'psi2_j' are the basis functions evaluated 
		at the Gauss-Lobatto quadrature nodes. 'w_xi', 'w_eta' are the Gauss-Lobatto
		weights, and 'det_F, dx_dxi, dx_deta, dy_dxi, dy_deta' are the Jacobian
		of the mapping along with its components. 

		psi_i, psi_j are 3D numpy arrays of shape (p*p, (p + 1), (p + 1))
		det_F, dx_dxi, dx_deta, dy_dxi, dy_deta are 2D numpy arrays of shape ((p + 1), (p + 1))
		w_xi, w_eta are 1D numpy arrays of shape (p + 1)
	"""
	## ============================================================== ##
	## ++++++++++++++ Implement the two-form mass matrix ++++++++++++ ##
	## ++++++++++++++++++++++++ assembly below ++++++++++++++++++++++ ##
	## ============================================================== ##
	w = np.outer(w_xi, w_eta)

	M2 = np.einsum('ikl,kl,jkl->ij', psi2_i, w/det_F, psi2_j, optimize=True)
	
	return M2