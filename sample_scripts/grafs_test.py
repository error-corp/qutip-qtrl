#!/usr/bin/env python
# coding: utf-8

# This notebook is a tutorial of the qutip-cntrl quantum optimal control library, found [here](https://github.com/qutip/qutip-qtrl). I used [this](https://nbviewer.org/github/qutip/qutip-notebooks/blob/master/examples/control-pulseoptim-Hadamard.ipynb) tutorial via [QuTiP](https://qutip.readthedocs.io/en/master/index.html) to explore the package. The goal here is to calculate control amplitudes needed to implement a single-qubit Hadamard gate using the well-known [GRAPE](https://www.sciencedirect.com/science/article/abs/pii/S1090780704003696) algorithm, which uses gradient ascent to optimize constant control fields across discrete time interals.

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys
import shutil


# In[2]:


from qutip import Qobj, identity, sigmax, sigmaz, core
#QuTiP control modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qutip_qtrl import pulseoptim as cpo


# In[3]:


# Defining physics notation
H_d = sigmaz() # drift Hamiltonian
H_c = [sigmax(), sigmaz()] # control Hamiltonian
U_0 = identity(2) # initial Gate
U_tg = core.gates.hadamard_transform(1) # target Gate (Hadamard)


# In[4]:


# Defining Parameters
M = 10 # number of time intervals
T = 10 # total evolution time


# In[5]:


# Defining fidelity threshold, iterations/time limit, and gradient minimum
threshold = 1e-10 # fidelity theshold
max_iter = 300 # max iterations
max_time = 120 # max time (seconds)
min_grad = 1e-20 # min of gradient should approach 0 - stop when we're close to 0


# In[6]:


# Define pulse type (qutip-control options are: RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW)
pulse_type = 'RND'


# In[7]:


# Create an output file extension using our parameters
output_file_extension = f"M_{M}_T_{T}_Pulse_{pulse_type}.txt"


# In[8]:

"""
# Run GRAPE optimization algorithm (https://qutip.org/docs/4.0.2/modules/qutip/control/pulseoptim.html)
result = cpo.optimize_pulse_unitary(
    H_d, H_c, U_0, U_tg, M, T, fid_err_targ=threshold, min_grad=min_grad, max_iter=max_iter, 
    max_wall_time=max_time, init_pulse_type=pulse_type, gen_stats=True, out_file_ext=output_file_extension
)
"""
result = cpo.opt_pulse_grafs(
    H_d, H_c, U_0, U_tg, M, T, fid_err_targ=threshold, min_grad=min_grad, max_iter=max_iter, 
    max_wall_time=max_time, init_pulse_type=pulse_type, gen_stats=True, out_file_ext=output_file_extension
)
"""
result = cpo.optimize_pulse(
    H_d, H_c, U_0, U_tg, M, T, fid_err_targ=threshold, min_grad=min_grad, max_iter=max_iter, 
    max_wall_time=max_time, init_pulse_type=pulse_type, gen_stats=True, out_file_ext=output_file_extension
)
"""

# In[9]:


result.stats.report()


# In[10]:


fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
#ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
ax1.step(result.time,
         np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
         where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
ax2.step(result.time,
         np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
         where='post')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




