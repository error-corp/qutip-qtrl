# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
Created on Tue Mar 18 12:18:14 2014
@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

Statistics for the optimisation
Note that some of the stats here are redundant copies from the optimiser
used here for calculations
"""
#import numpy as np
import datetime

class Stats:
    """
    Base class for all optimisation statistics
    Used for configurations where all timeslots are updated each iteration
    e.g. exact gradients
    Note that all times are generated using timeit.default_timer() and are
    in seconds
    
    Attributes
    ----------
    wall_time_optim_start : float
        Start time for the optimisation

    wall_time_optim_end : float
        End time for the optimisation
        
    wall_time_optim : float
        Time elasped during the optimisation
    
    num_iter : integer
        Number of iterations of the optimisation algorithm (e.g. BFGS)
    
    wall_time_dyn_gen_compute : float
        Total wall (elasped) time computing combined dynamics generator
        (for example combining drift and control Hamiltonians)
        
    wall_time_prop_compute : float
        Total wall (elasped) time computing propagators, that is the 
        time evolution from one timeslot to the next
        Includes calculating the propagator gradient for exact gradients
        
    wall_time_fwd_prop_compute : float
        Total wall (elasped) time computing combined forward propagation, 
        that is the time evolution from the start to a specific timeslot.
        Excludes calculating the propagators themselves
        
    wall_time_onwd_prop_compute : float
        Total wall (elasped) time computing combined onward propagation, 
        that is the time evolution from a specific timeslot to the end time.
        Excludes calculating the propagators themselves
        
    wall_time_gradient_compute : float
        Total wall (elasped) time computing the fidelity error gradient.
        Excludes calculating the propagator gradients (in exact gradient
        methods)
        
    num_fidelity_func_calls : integer
        Number of calls to fidelity function by the optimisation algorithm
    
    num_grad_func_calls : integer
        Number of calls to gradient function by the optimisation algorithm

    num_fidelity_computes : integer
        Number of time the fidelity is computed
        (It is only computed if any amplitudes changed since the last call)
        
    num_grad_computes : integer
        Number of time the gradient is computed
        (It is only computed if any amplitudes changed since the last call)

    num_ctrl_amp_updates : integer
        Number of times the control amplitudes are updated
        
    mean_num_ctrl_amp_updates_per_iter : float
        Mean number of control amplitude updates per iteration

    num_timeslot_changes : integer
        Number of times the amplitudes of a any control in a timeslot changes
        
    mean_num_timeslot_changes_per_update : float
        Mean average number of timeslot amplitudes that are changed per update
        
    num_ctrl_amp_changes : integer
        Number of times individual control amplitudes that are changed
    
    mean_num_ctrl_amp_changes_per_update : float
        Mean average number of control amplitudes that are changed per update
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.wall_time_optim_start = 0.0
        self.wall_time_optim_end = 0.0
        self.wall_time_optim = 0.0
        self.num_iter = 0
        # Duration attributes
        self.wall_time_dyn_gen_compute = 0.0        
        self.wall_time_prop_compute = 0.0
        self.wall_time_fwd_prop_compute = 0.0
        self.wall_time_onwd_prop_compute = 0.0
        self.wall_time_gradient_compute = 0.0
        # Fidelity and gradient function calls and computes
        self.num_fidelity_func_calls = 0
        self.num_grad_func_calls = 0
        self.num_fidelity_computes = 0
        self.num_grad_computes = 0
        # Control amplitudes
        self.num_ctrl_amp_updates = 0
        self.mean_num_ctrl_amp_updates_per_iter = 0.0
        self.num_timeslot_changes = 0
        self.mean_num_timeslot_changes_per_update = 0.0
        self.num_ctrl_amp_changes = 0
        self.mean_num_ctrl_amp_changes_per_update = 0.0
        
    def calculate(self):
        """
        Perform the calculations (e.g. averages) that are required on the stats
        Should be called before calling report
        """
        # If the optimation is still running then the optimisation 
        # time is the time so far
        if self.wall_time_optim_end > 0.0:
            self.wall_time_optim = \
                    self.wall_time_optim_end - self.wall_time_optim_start
                    
        self.mean_num_ctrl_amp_updates_per_iter = \
                self.num_ctrl_amp_updates / float(self.num_iter)
        
        self.mean_num_timeslot_changes_per_update = \
                self.num_timeslot_changes / float(self.num_ctrl_amp_updates)
                
        self.mean_num_ctrl_amp_changes_per_update = \
                self.num_ctrl_amp_changes / float(self.num_ctrl_amp_updates)               

    def _format_datetime(self, t, tot=0.0):
        dtStr = str(datetime.timedelta(seconds=t))
        if tot > 0:
            percent = 100*t/tot
            dtStr += " ({:03.2f}%)".format(percent)
        return dtStr
        
    def report(self):
        """
        Print a report of the stats to the console
        """
        print ""
        print "------------------------------------"
        print "---- Control optimisation stats ----"
        self.report_timings()
        self.report_func_calls()
        self.report_amp_updates()
        print "---------------------------"
        
    def report_timings(self):
        print "**** Timings (HH:MM:SS.US) ****"
        tot = self.wall_time_optim
        print "Total wall time elapsed during optimisation: " + \
                self._format_datetime(tot)
        print "Wall time computing Hamiltonians: " + \
                self._format_datetime(self.wall_time_dyn_gen_compute, tot)
        print "Wall time computing propagators: " + \
                self._format_datetime(self.wall_time_prop_compute, tot)
        print "Wall time computing forward propagation: " + \
                self._format_datetime(self.wall_time_fwd_prop_compute, tot)
        print "Wall time computing onward propagation: " + \
                self._format_datetime(self.wall_time_onwd_prop_compute, tot)
        print "Wall time computing gradient: " + \
                self._format_datetime(self.wall_time_gradient_compute, tot)
        print ""
        
    def report_func_calls(self):
        print "**** Iterations and function calls ****"
        print "Number of iterations: " + str(self.num_iter)
        print "Number of fidelity function calls: " + \
                str(self.num_fidelity_func_calls)
        print "Number of times fidelity is computed: " + \
                str(self.num_fidelity_computes)
        print "Number of gradient function calls: " + \
                str(self.num_grad_func_calls)
        print "Number of times gradients are computed: " + \
                str(self.num_grad_computes)
        print ""
        
    def report_amp_updates(self):
        print "**** Control amplitudes ****"
        print "Number of control amplitude updates: " + \
                str(self.num_ctrl_amp_updates)
        print "Mean number of updates per iteration: " + \
                str(self.mean_num_ctrl_amp_updates_per_iter)                
        print "Number of timeslot values changed: " + \
                str(self.num_timeslot_changes)
        print "Mean number of timeslot changes per update: " + \
                str(self.mean_num_timeslot_changes_per_update)
        print "Number of amplitude values changed: " + \
                str(self.num_ctrl_amp_changes)
        print "Mean number of amplitude changes per update: " + \
                str(self.mean_num_ctrl_amp_changes_per_update)
                

class StatsDynTsUpdate(Stats):
    """
    Optimisation stats class for configurations where all timeslots are not  
    necessarily updated at each iteration. In this case it may be interesting
    to know how many Hamiltions etc are computed each ctrl amplitude update
        
    Attributes
    ----------
    num_dyn_gen_computes : integer
        Total number of dynamics generator (Hamiltonian) computations,
        that is combining drift and control dynamics to calculate the 
        combined dynamics generator for the timeslot

    mean_num_dyn_gen_computes_per_update : float
        # Mean average number of dynamics generator computations per update

    mean_wall_time_dyn_gen_compute : float
        # Mean average time to compute a timeslot dynamics generator
    
    num_prop_computes : integer
        Total number of propagator (and propagator gradient for exact 
        gradient types) computations

    mean_num_prop_computes_per_update : float
        Mean average number of propagator computations per update
        
    mean_wall_time_prop_compute : float
        Mean average time to compute a propagator (and its gradient)
    
    num_fwd_prop_step_computes : integer
        Total number of steps (matrix product) computing forward propagation
    
    mean_num_fwd_prop_step_computes_per_update : float
        Mean average number of steps computing forward propagation

    mean_wall_time_fwd_prop_compute : float
        Mean average time to compute forward propagation
        
    num_onwd_prop_step_computes : integer
        Total number of steps (matrix product) computing onward propagation

    mean_num_onwd_prop_step_computes_per_update : float
        Mean average number of steps computing onward propagation

    mean_wall_time_onwd_prop_compute
        Mean average time to compute onward propagation
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        Stats.reset(self)
        # Dynamics generators (Hamiltonians)        
        self.num_dyn_gen_computes = 0
        self.mean_num_dyn_gen_computes_per_update = 0.0
        self.mean_wall_time_dyn_gen_compute = 0.0
        # **** Propagators *****
        self.num_prop_computes = 0
        self.mean_num_prop_computes_per_update = 0.0
        self.mean_wall_time_prop_compute = 0.0
        # **** Forward propagation ****
        self.num_fwd_prop_step_computes = 0
        self.mean_num_fwd_prop_step_computes_per_update = 0.0
        self.mean_wall_time_fwd_prop_compute = 0.0
        # **** onward propagation ****
        self.num_onwd_prop_step_computes = 0
        self.mean_num_onwd_prop_step_computes_per_update = 0.0
        self.mean_wall_time_onwd_prop_compute = 0.0
        
    def calculate(self):
        Stats.calculate(self)
        self.mean_num_dyn_gen_computes_per_update = \
                self.num_dyn_gen_computes / float(self.num_ctrl_amp_updates)
                
        self.mean_wall_time_dyn_gen_compute = \
                self.wall_time_dyn_gen_compute / \
                        float(self.num_dyn_gen_computes)

        self.mean_num_prop_computes_per_update = \
                self.num_prop_computes / float(self.num_ctrl_amp_updates)
                
        self.mean_wall_time_prop_compute = \
                self.wall_time_prop_compute / float(self.num_prop_computes)
                
        self.mean_num_fwd_prop_step_computes_per_update = \
                self.num_fwd_prop_step_computes / \
                        float(self.num_ctrl_amp_updates)
                
        self.mean_wall_time_fwd_prop_compute = \
                self.wall_time_fwd_prop_compute / \
                        float(self.num_fwd_prop_step_computes)
    
        self.mean_num_onwd_prop_step_computes_per_update = \
                self.num_onwd_prop_step_computes / \
                        float(self.num_ctrl_amp_updates)
                
        self.mean_wall_time_onwd_prop_compute = \
                self.wall_time_onwd_prop_compute / \
                        float(self.num_onwd_prop_step_computes)
        
    def report(self):
        """
        Print a report of the stats to the console
        """
        print ""
        print "------------------------------------"
        print "---- Control optimisation stats ----"
        self.report_timings()
        self.report_func_calls()
        self.report_amp_updates()
        self.report_dyn_gen_comps()
        self.report_fwd_prop()
        self.report_onwd_prop()
        print "---------------------------"
        
    def report_dyn_gen_comps(self):
        print "**** Hamiltonian Computations ****"
        print "Total number of Hamiltonian computations: " + \
                str(self.num_dyn_gen_computes)        
        print "Mean number of Hamiltonian computations per update: " + \
                str(self.mean_num_dyn_gen_computes_per_update)
        print "Mean wall time to compute Hamiltonian {} s".\
                format(self.mean_wall_time_dyn_gen_compute)               
        print "**** Propagator Computations ****"
        print "Total number of propagator computations: " + \
                str(self.num_prop_computes)        
        print "Mean number of propagator computations per update: " + \
                str(self.mean_num_prop_computes_per_update)
        print "Mean wall time to compute propagator {} s".\
                format(self.mean_wall_time_prop_compute)
                
    def report_fwd_prop(self):
        print "**** Forward Propagation ****"
        print "Total number of forward propagation step computations: " + \
                str(self.num_fwd_prop_step_computes)        
        print "Mean number of forward propagation step computations" + \
                " per update: " + \
                str(self.mean_num_fwd_prop_step_computes_per_update)
        print "Mean wall time to compute forward propagation {} s".\
                format(self.mean_wall_time_fwd_prop_compute)
                
    def report_onwd_prop(self):
        print "**** Onward Propagation ****"
        print "Total number of onward propagation step computations: " + \
                str(self.num_onwd_prop_step_computes)        
        print "Mean number of onward propagation step computations" + \
                " per update: " + \
                str(self.mean_num_onwd_prop_step_computes_per_update)
        print "Mean wall time to compute onward propagation {} s".\
                format(self.mean_wall_time_onwd_prop_compute)    