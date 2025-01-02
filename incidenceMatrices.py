import numpy as np
import scipy.sparse as sp

def assemble_element_E10(p):
	""" 
		This function is called internally to 
		generate the incidence matrix E10 for a 
		single element with polynomial degree 'p'.
		The function must return a numpy array 
		or scipy sparse array of size 
		(2*p*(p + 1)) x ((p + 1)*(p + 1))
	"""
	## ============================================================== ##
	## ++++++++++++++ Implement the incidence matrix E10 ++++++++++++ ##
	## +++++++++++++++++++++++++++ below ++++++++++++++++++++++++++++ ##
	## ============================================================== ##
	E10_1d = sp.diags([1, -1], [0, 1], shape=(p, p+1))
	I = sp.eye(p+1)
	E10 = sp.block_array(
		[
			[sp.kron(-E10_1d, I)], 
			[sp.kron(I, E10_1d)]
		]
	)
	return E10

def assemble_element_E21(p):
	""" 
		This function is called internally to 
		generate the incidence matrix E21 for a 
		single element with polynomial degree 'p'.
		The function must return a numpy array 
		or scipy sparse array of size 
		(p*p) x (2*p*(p + 1))
	"""
	## ============================================================== ##
	## ++++++++++++++ Implement the incidence matrix E21 ++++++++++++ ##
	## +++++++++++++++++++++++++++ below ++++++++++++++++++++++++++++ ##
	## ============================================================== ##
	E21_1d = -sp.diags([1, -1], [0, 1], shape=(p, p+1))
	E21 = sp.block_array(
		[[
			sp.kron(sp.eye(p), E21_1d),
			sp.kron(E21_1d, sp.eye(p))
		]]
	)
	return E21