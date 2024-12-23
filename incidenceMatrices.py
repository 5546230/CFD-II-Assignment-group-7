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
	raise NotImplementedError

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
	raise NotImplementedError