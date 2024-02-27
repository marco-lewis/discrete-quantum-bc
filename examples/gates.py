import numpy as np

xgate = np.array([[0,1],[1,0]])
zgate = np.array([[1,0],[0,-1]])
tgate = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
sxgate = 1/2 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]])

hadamard = np.dot(np.array([[1,1],[1,-1]]), 1/np.sqrt(2))
hadamard_n = lambda n: hadamard if n == 1 else np.kron(hadamard, hadamard_n(n-1))

Rx = lambda phi: np.array([[np.cos(phi/2), -1j*np.sin(phi/2)],[-1j*np.sin(phi/2), np.cos(phi/2)]])
Ry = lambda phi: np.array([[np.cos(phi/2), -1*np.sin(phi/2)],[-1*np.sin(phi/2), np.cos(phi/2)]])
Rz = lambda phi: np.array([[np.exp(-1j * phi/2), 0],[0, np.exp(1j * phi/2)]])