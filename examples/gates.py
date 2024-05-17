import numpy as np

Xgate = np.array([[0,1],[1,0]])
Ygate = np.array([[0, 1j],[-1j,0]])
Zgate = np.array([[1,0],[0,-1]])
Tgate = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
SQRTXgate = 1/2 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]])
CNOTgate = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
CZgate = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]])

Hgate = np.dot(np.array([[1,1],[1,-1]]), 1/np.sqrt(2))

Rx = lambda phi: np.array([[np.cos(phi/2), -1j*np.sin(phi/2)],[-1j*np.sin(phi/2), np.cos(phi/2)]])
Ry = lambda phi: np.array([[np.cos(phi/2), -1*np.sin(phi/2)],[-1*np.sin(phi/2), np.cos(phi/2)]])
Rz = lambda phi: np.array([[np.exp(-1j * phi/2), 0],[0, np.exp(1j * phi/2)]])

n_gate = lambda gate, n: gate if n == 1 else np.kron(gate, n_gate(gate, n-1))