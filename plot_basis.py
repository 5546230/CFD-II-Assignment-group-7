import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.mimeticMachinery import mimeticBasis


p = 8;
xi = np.linspace(-1, 1, 100);
eta = np.linspace(-1, 1, 100);
c = np.zeros(p+1);
c[-1] = 1;
roots = np.polynomial.legendre.legroots(np.polynomial.legendre.legder(c));
roots = np.concatenate(([-1], roots, [1]));
print(roots);
basis = mimeticBasis();
print(eta.shape)
i_0, j_0 = 2, 3;
i_1, j_1 = 2, 2;
i_2, j_2 = 2, 2;
psi0 = np.outer(basis.evalBasis(0, roots, eta)[i_0,:],  basis.evalBasis(0, roots, xi)[j_0,:])
psi1 = np.outer(basis.evalBasis(1, roots, eta)[i_1,:],  basis.evalBasis(0, roots, xi)[j_1,:])
psi2 = np.outer(basis.evalBasis(101, roots, eta)[i_2,:],  basis.evalBasis(101, roots, xi)[j_2,:])
xi, eta = np.meshgrid(xi, eta);

fig = plt.figure(figsize=(9, 3));
ax = fig.add_subplot(131, projection='3d');
ax.plot_surface(xi, eta, psi0, cmap='viridis');
# plot the roots
for i, root in enumerate(roots):
    for j, root_ in enumerate(roots):
        ax.scatter(root_, root, 1 , c='r', marker='o');

ax.set_xlabel(r'$\xi$');
ax.set_ylabel(r'$\eta$');
# ax.set_zlabel(r'$\psi^{(0)}_0$');
ax.set_title(rf'$ψ^{{(0)}}_{{{i_0*(p+1)+j_0}}}$');

ax = fig.add_subplot(132, projection='3d');
ax.plot_surface(xi, eta, psi1, cmap='viridis');
# plot the roots
for i, root in enumerate(roots):
    for j, root_ in enumerate(roots):
        ax.scatter(root_, root, 2 , c='r', marker='o');

ax.set_xlabel(r'$\xi$');
ax.set_ylabel(r'$\eta$');
ax.set_title(rf'$ψ^{{(1)}}_{{{i_1*(p+1)+j_1}}}$');
# ax.set_zlabel(r'$\psi^{(1)}_0$');

ax = fig.add_subplot(133, projection='3d');
ax.plot_surface(xi, eta, psi2, cmap='viridis');
# plot the roots
for i, root in enumerate(roots):
    for j, root_ in enumerate(roots):
        ax.scatter(root_, root, 2, c='r', marker='o');

ax.set_xlabel(r'$\xi$');
ax.set_ylabel(r'$\eta$');
ax.set_title(rf'$ψ^{{(2)}}_{{{i_2*(p)+j_2}}}$');
# ax.set_zlabel(r'$\psi^{(2)}_0$');

fig.tight_layout();
fig.savefig('basisFunctions.pdf', bbox_inches='tight', pad_inches=0.1);
