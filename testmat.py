import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import save
import os

os.remove('theta.txt')
os.remove('delta.txt')

K = 10 # The number of trajectories
N = 100  # Number of sample points in each trajectory
theta =[0.1,0.13,0.2,0.34, 1.0, 0.64, 2.0, 1.24, 0.5, 0.7]
delta = [0.01, 0.012, 0.008, 0.005, 0.004, 0.006, 0.003, 0.004, 0.005, 0.0015]



data_theta = np.empty([K,N])
for x in range (K):

    t_0 = 0 # Start time
    t_end = delta[x]*N # End time
    t = np.linspace(t_0,t_end,num = N)

    g = 9.82 # Gravitational acceleration (m/s^2)
    l = 0.3 # Length of the pendelum (m)
    theta_0 = theta[x] # The start angle (rad)
    w = g/l
    F = 20 # The starting force which acts on the mass (N)
    m = 20 # The mass at the end of the pendelum (kg)

    xtheta = theta_0 * np.cos(w*t) + F/(w**2 * m)

    #plt.plot(t,xtheta)
    #plt.show()

    with open('theta.txt','a') as datafil:
        datafil.write('$$')
        np.savetxt(datafil,xtheta,delimiter= ',')
    

np.savetxt('delta.txt',delta)

    


    
  