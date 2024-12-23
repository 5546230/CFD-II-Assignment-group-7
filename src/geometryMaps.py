"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import numpy as np

class geomdef():
	def __init__(self):
		self.X0 = [0.0, 2.0];
		self.X1 = [0.0, 1.0];
		self.mapping_parms = (0.3, (self.X0[1] - self.X0[0])/4, 3*(self.X0[1] - self.X0[0])/4, 15);

	def Phi(self, xi, eta, X0, X1, Kx, Ky, a, b, c, d, i, j):
		r = X0[0] + ((X0[1] - X0[0])/Kx)*(i + (xi + 1)/2);
		s = X1[0] + ((X1[1] - X1[0])/Ky)*(j + (eta + 1)/2);
		x = r;
		y = s - 0.5*(1 - s)*s + a*(np.exp(-(d*(r - b)**2)) + np.exp(-(d*(r - c)**2)))*(1 - (s - (1 - s)*s)**2);
		return x, y;

	def dPhi(self, xi, eta, X0, X1, Kx, Ky, a, b, c, d, i, j):
		r = X0[0] + ((X0[1] - X0[0])/Kx)*(i + (xi + 1)/2);
		s = X1[0] + ((X1[1] - X1[0])/Ky)*(j + (eta + 1)/2);
		dx_xi = ((X0[1] - X0[0])/Kx)/2*np.ones_like(r);
		dx_eta = np.zeros_like(r);
		dy_xi = ((X0[1] - X0[0])/Kx)/2*(a*d*((-2*r + 2*b)*np.exp(-d*(r - b)**2) + (-2*r + 2*c)*np.exp(-d*(r - c)**2))*(1 - (s - (1 - s)*s)**2));
		dy_eta = ((X1[1] - X1[0])/Ky)/2*(0.5 + s - 4*a*(np.exp(-d*(r - b)**2) + np.exp(-d*(r - c)**2))*(s - (1 - s)*s)*s);
		return dx_xi, dx_eta, dy_xi, dy_eta;

class squareGeom():
	def __init__(self):
		self.X0 = [0.0, 1.0];
		self.X1 = [0.0, 1.0];
		self.mapping_parms = (0.2,);

	def Phi(self, xi, eta, X0, X1, Kx, Ky, c, i, j):
		r = X0[0] + ((X0[1] - X0[0])/Kx)*(i + (xi + 1)/2);
		s = X1[0] + ((X1[1] - X1[0])/Ky)*(j + (eta + 1)/2);
		x = r + 0.5*c*np.sin(2*np.pi*r)*np.sin(2*np.pi*s);
		y = s - 0.5*c*np.sin(2*np.pi*r)*np.sin(2*np.pi*s);
		return x, y;

	def dPhi(self, xi, eta, X0, X1, Kx, Ky, c, i, j):
		r = X0[0] + ((X0[1] - X0[0])/Kx)*(i + (xi + 1)/2);
		s = X1[0] + ((X1[1] - X1[0])/Ky)*(j + (eta + 1)/2);
		dx_xi = ((X0[1] - X0[0])/Kx)/2*(1 + 0.5*c*2*np.pi*np.cos(2*np.pi*r)*np.sin(2*np.pi*s));
		dx_eta = ((X1[1] - X1[0])/Ky)/2*(0.5*c*2*np.pi*np.sin(2*np.pi*r)*np.cos(2*np.pi*s));
		dy_xi = ((X0[1] - X0[0])/Kx)/2*(-0.5*c*2*np.pi*np.cos(2*np.pi*r)*np.sin(2*np.pi*s));
		dy_eta = ((X1[1] - X1[0])/Ky)/2*(1 - 0.5*c*2*np.pi*np.sin(2*np.pi*r)*np.cos(2*np.pi*s));
		return dx_xi, dx_eta, dy_xi, dy_eta;
