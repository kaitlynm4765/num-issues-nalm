import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

def compute_r_and_K(t_i, rb, rs, Kb, Ks):
    ''' time-varying growth rate and carrying capacity '''
    p = 365/(2*np.pi) # period of oscillation
    r = rb - rs*np.cos((t_i)/p) # growth rate
    K = Kb - Ks*np.cos((t_i)/p) # carrying capacity
    return (r,K)

def logistic(t, x, rb, rs, Kb, Ks):
    rv, Kv = compute_r_and_K(t, rb, rs, Kb, Ks)
    dxdt = (1 - (x/Kv))*rv*x
    return dxdt

def true_sol(x0, rb, rs, K, t):
    ''' true solution for time-varying growth rate 
    (constant carrying capacity) '''
    p = 365/(2*np.pi)
    ft = rb*t - p*rs*np.sin(t/p)
    num = K*x0
    den = x0 + np.exp(-ft)*(K-x0)
    true_x = num/den
    return true_x
 
def manual_RK4(x0, delta_t, params, t, threshold = True):
    ''' manually-programmed RK4 method '''
    n = int((1/delta_t)*len(t)) # total number of steps = 1/(step size)

    P = np.zeros(int(n))
  
    rb, rs, Kb, Ks = params
    
    P[0] = x0   # initial condition
        
    for i in np.arange(0,n-1):
        t_i = i*delta_t
        k1 = delta_t*logistic(t_i, P[i], rb, rs, Kb, Ks)
        k2 = delta_t*logistic(t_i + delta_t/2, P[i] + k1/2, rb, rs, Kb, Ks)
        k3 = delta_t*logistic(t_i + delta_t/2, P[i] + k2/2, rb, rs, Kb, Ks)
        k4 = delta_t*logistic(t_i + delta_t, P[i] + k3, rb, rs, Kb, Ks)
        
        next_val = P[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        
        # implement threshold for maximum carrying capacity
        if (threshold == False) or (next_val <= Kb + abs(Ks)):
            P[i+1] = next_val
        else:
            P[i+1] = P[i]
    return P

def test_numerical(x0, t, solver, PS, delta_t = 1, threshold = True):
    rb, rs, Kb, Ks = PS # parameter set
     
    if solver == 'manual RK4':
        start_time = time.time()
        P = manual_RK4(x0, delta_t, PS, t, threshold)
        end_time = time.time()
        nfev = 'null'
    else:
        start_time = time.time()
        P = solve_ivp(logistic, (t.min(), t.max()), [x0],
                      method = solver, t_eval = t,
                      args = (rb, rs, Kb, Ks))
        end_time = time.time()
        nfev = P.nfev
        P = P.y[0]
    
    ellapsed_time = end_time - start_time
    
    if Ks == 0:
        t_range = np.arange(0, t.max()+1, delta_t)
        true_solution = true_sol(x0, rb, rs, Kb, t_range)
        
        if len(true_solution) > len(P):
            error = f'error occured at t = {len(P)}'
        else:
            error = np.linalg.norm(P - true_solution)*delta_t
    else:
         error = 'null'   
    
    return P, error, ellapsed_time, nfev


#%% time-varying growth rate and constant carrying capacity

x0 = 1  # initial condition
t = np.arange(0, 5000) # time span

# Define parameter sets
PS1 = [0.05, 0.15, 2e5, 0]
PS2 = [-0.002, 0.25, 2e5, 0]
PS3 = [0.01, 0.25, 2e5, 0]
PS4 = [0.005, 0.05, 2e5, 0]
PS5 = [0.01, 0.005, 2e5, 0]
PS6 = [0.01, 0, 2e5, 0]

# true solutions
true_solution_PS1 = true_sol(x0, PS1[0], PS1[1], PS1[2], t)
r_PS1, K_PS1 = compute_r_and_K(t, PS1[0], PS1[1], PS1[2], PS1[3])

true_solution_PS2 = true_sol(x0, PS2[0], PS2[1], PS2[2], t)
r_PS2, K_PS2 = compute_r_and_K(t, PS2[0], PS2[1], PS2[2], PS2[3])

true_solution_PS3 = true_sol(x0, PS3[0], PS3[1], PS3[2], t)
r_PS3, K_PS3 = compute_r_and_K(t, PS3[0], PS3[1], PS3[2], PS3[3])

true_solution_PS4 = true_sol(x0, PS4[0], PS4[1], PS4[2], t)
r_PS4, K_PS4 = compute_r_and_K(t, PS4[0], PS4[1], PS4[2], PS4[3])

true_solution_PS5 = true_sol(x0, PS5[0], PS5[1], PS5[2], t)
r_PS5, K_PS5 = compute_r_and_K(t, PS5[0], PS5[1], PS5[2], PS5[3])

true_solution_PS6 = true_sol(x0, PS6[0], PS6[1], PS6[2], t)
r_PS6, K_PS6 = compute_r_and_K(t, PS6[0], PS6[1], PS6[2], PS6[3])

# numerical approximations
LSODA_PS1 = test_numerical(x0, t, 'LSODA', PS1)
BDF_PS1 = test_numerical(x0, t, 'BDF', PS1)
RK45_PS1 = test_numerical(x0, t, 'RK45', PS1)
RK23_PS1 = test_numerical(x0, t, 'RK23', PS1)
Radau_PS1 = test_numerical(x0, t, 'Radau', PS1)
DOP853_PS1 = test_numerical(x0, t, 'DOP853', PS1)
manual_RK4_PS1 = test_numerical(x0, t, 'manual RK4', PS1)

LSODA_PS2 = test_numerical(x0, t, 'LSODA', PS2)
BDF_PS2 = test_numerical(x0, t, 'BDF', PS2)
RK45_PS2 = test_numerical(x0, t, 'RK45', PS2)
RK23_PS2 = test_numerical(x0, t, 'RK23', PS2)
Radau_PS2 = test_numerical(x0, t, 'Radau', PS2)
DOP853_PS2 = test_numerical(x0, t, 'DOP853', PS2)
manual_RK4_PS2 = test_numerical(x0, t,'manual RK4', PS2)

LSODA_PS3 = test_numerical(x0, t, 'LSODA', PS3)
BDF_PS3 = test_numerical(x0, t, 'BDF', PS3)
RK45_PS3 = test_numerical(x0, t, 'RK45', PS3)
RK23_PS3 = test_numerical(x0, t, 'RK23', PS3)
Radau_PS3 = test_numerical(x0, t, 'Radau', PS3)
DOP853_PS3 = test_numerical(x0, t, 'DOP853', PS3)
manual_RK4_PS3 = test_numerical(x0, t, 'manual RK4', PS3)

LSODA_PS4 = test_numerical(x0, t, 'LSODA', PS4)
BDF_PS4 = test_numerical(x0, t, 'BDF', PS4)
RK45_PS4 = test_numerical(x0, t, 'RK45', PS4)
RK23_PS4 = test_numerical(x0, t, 'RK23', PS4)
Radau_PS4 = test_numerical(x0, t, 'Radau', PS4)
DOP853_PS4 = test_numerical(x0, t, 'DOP853', PS4)
manual_RK4_PS4 = test_numerical(x0, t, 'manual RK4', PS4)

LSODA_PS5 = test_numerical(x0, t, 'LSODA', PS5)
BDF_PS5 = test_numerical(x0, t, 'BDF', PS5)
RK45_PS5 = test_numerical(x0, t, 'RK45', PS5)
RK23_PS5 = test_numerical(x0, t, 'RK23', PS5)
Radau_PS5 = test_numerical(x0, t, 'Radau', PS5)
DOP853_PS5 = test_numerical(x0, t, 'DOP853', PS5)
manual_RK4_PS5 = test_numerical(x0, t, 'manual RK4', PS5)

LSODA_PS6 = test_numerical(x0, t, 'LSODA', PS6)
BDF_PS6 = test_numerical(x0, t, 'BDF', PS6)
RK45_PS6 = test_numerical(x0, t, 'RK45', PS6)
RK23_PS6 = test_numerical(x0, t, 'RK23', PS6)
Radau_PS6 = test_numerical(x0, t, 'Radau', PS6)
DOP853_PS6 = test_numerical(x0, t, 'DOP853', PS6)
manual_RK4_PS6 = test_numerical(x0, t, 'manual RK4', PS6)

# compare errors
error_PS1 = (LSODA_PS1[1], BDF_PS1[1], RK45_PS1[1], RK23_PS1[1], 
              Radau_PS1[1], DOP853_PS1[1], manual_RK4_PS1[1])
error_PS2 = (LSODA_PS2[1], BDF_PS2[1], RK45_PS2[1], RK23_PS2[1], 
              Radau_PS2[1], DOP853_PS2[1], manual_RK4_PS2[1])
error_PS3 = (LSODA_PS3[1], BDF_PS3[1], RK45_PS3[1], RK23_PS3[1], 
              Radau_PS3[1], DOP853_PS3[1], manual_RK4_PS3[1])
error_PS4 = (LSODA_PS4[1], BDF_PS4[1], RK45_PS4[1], RK23_PS4[1], 
              Radau_PS4[1], DOP853_PS4[1], manual_RK4_PS4[1])
error_PS5 = (LSODA_PS5[1], BDF_PS5[1], RK45_PS5[1], RK23_PS5[1], 
              Radau_PS5[1], DOP853_PS5[1], manual_RK4_PS5[1])
error_PS6 = (LSODA_PS6[1], BDF_PS6[1], RK45_PS6[1], RK23_PS6[1], 
              Radau_PS6[1], DOP853_PS6[1], manual_RK4_PS6[1])

error_df = pd.DataFrame({'PS1' : error_PS1, 'PS2' : error_PS2, 'PS3' : error_PS3,
              'PS4' : error_PS4, 'PS5' : error_PS5, 'PS6' : error_PS6}, 
              index = ('LSODA', 'BDF', 'RK45', 'RK23', 'Radau', 'DOP853', 
              'Manual RK4'))

def PS1_plot():
    ftsz = 12
    fig = plt.figure(figsize = (12,8), dpi = 300)
  
    ax = fig.add_subplot(2, 3, 1)
    plt.sca(ax)
    plt.title('LSODA', fontsize = ftsz)
    plt.plot(true_solution_PS1, 'b', linewidth = 3)
    plt.plot(LSODA_PS1[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])

    ax = fig.add_subplot(2, 3, 2)
    plt.sca(ax)
    plt.title('BDF', fontsize = ftsz)
    plt.plot(true_solution_PS1, 'b', linewidth = 3)
    plt.plot(BDF_PS1[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 3)
    plt.sca(ax)
    plt.title('RK45', fontsize = ftsz)
    plt.plot(true_solution_PS1, 'b', linewidth = 3)
    plt.plot(RK45_PS1[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 4)
    plt.sca(ax)
    plt.title('RK23', fontsize = ftsz)
    plt.plot(true_solution_PS1, 'b', linewidth = 3)
    plt.plot(RK23_PS1[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 5)
    plt.sca(ax)
    plt.title('Manual RK4', fontsize = ftsz)
    plt.plot(true_solution_PS1, 'b', linewidth = 3)
    plt.plot(manual_RK4_PS1[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    plt.legend(('True Solution', 'Approximation'), fontsize = ftsz)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1,
                    top=1,
                    wspace=0.5,
                    hspace=0.25)
    plt.show()
                                       
def PS2_plot():
    ftsz = 12
    fig = plt.figure(figsize = (12,8), dpi = 300)
  
    ax = fig.add_subplot(2, 3, 1)
    plt.sca(ax)
    plt.title('LSODA', fontsize = ftsz)
    plt.plot(true_solution_PS2, 'b', linewidth = 3)
    plt.plot(LSODA_PS2[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2e5+10])
    plt.xlim([0, 4000])

    ax = fig.add_subplot(2, 3, 2)
    plt.sca(ax)
    plt.title('BDF', fontsize = ftsz)
    plt.plot(true_solution_PS2, 'b', linewidth = 3)
    plt.plot(BDF_PS2[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 3)
    plt.sca(ax)
    plt.title('RK45', fontsize = ftsz)
    plt.plot(true_solution_PS2, 'b', linewidth = 3)
    plt.plot(RK45_PS2[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 4)
    plt.sca(ax)
    plt.title('RK23', fontsize = ftsz)
    plt.plot(true_solution_PS2, 'b', linewidth = 3)
    plt.plot(RK23_PS2[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 5)
    plt.sca(ax)
    plt.title('Manual RK4', fontsize = ftsz)
    plt.plot(true_solution_PS2, 'b', linewidth = 3)
    plt.plot(manual_RK4_PS2[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2e5+10])
    plt.xlim([0, 4000])
    plt.legend(('True Solution', 'Approximation'), fontsize = ftsz)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1,
                    top=1,
                    wspace=0.5,
                    hspace=0.25)
    plt.show()

def PS3_plot():
    ftsz = 12
    fig = plt.figure(figsize = (12,8), dpi = 300)
  
    ax = fig.add_subplot(2, 3, 1)
    plt.sca(ax)
    plt.title('LSODA', fontsize = ftsz)
    plt.plot(true_solution_PS3, 'b', linewidth = 3)
    plt.plot(LSODA_PS3[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5])
    plt.xlim([0, 4000])

    ax = fig.add_subplot(2, 3, 2)
    plt.sca(ax)
    plt.title('BDF', fontsize = ftsz)
    plt.plot(true_solution_PS3, 'b', linewidth = 3)
    plt.plot(BDF_PS3[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 3)
    plt.sca(ax)
    plt.title('RK45', fontsize = ftsz)
    plt.plot(true_solution_PS3, 'b', linewidth = 3)
    plt.plot(RK45_PS3[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 4)
    plt.sca(ax)
    plt.title('RK23', fontsize = ftsz)
    plt.plot(true_solution_PS3, 'b', linewidth = 3)
    plt.plot(RK23_PS3[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 5)
    plt.sca(ax)
    plt.title('Manual RK4', fontsize = ftsz)
    plt.plot(true_solution_PS3, 'b', linewidth = 3)
    plt.plot(manual_RK4_PS3[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5])
    plt.xlim([0, 4000])
    plt.legend(('Solution', 'Approx'), fontsize = ftsz-2)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1,
                    top=1,
                    wspace=0.5,
                    hspace=0.25)
    plt.show()

def PS4_plot():
    ftsz = 12
    fig = plt.figure(figsize = (12,8), dpi = 300)
  
    ax = fig.add_subplot(2, 3, 1)
    plt.sca(ax)
    plt.title('LSODA', fontsize = ftsz)
    plt.plot(true_solution_PS4, 'b', linewidth = 3)
    plt.plot(LSODA_PS4[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])

    ax = fig.add_subplot(2, 3, 2)
    plt.sca(ax)
    plt.title('BDF', fontsize = ftsz)
    plt.plot(true_solution_PS4, 'b', linewidth = 3)
    plt.plot(BDF_PS4[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 3)
    plt.sca(ax)
    plt.title('RK45', fontsize = ftsz)
    plt.plot(true_solution_PS4, 'b', linewidth = 3)
    plt.plot(RK45_PS4[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 4)
    plt.sca(ax)
    plt.title('RK23', fontsize = ftsz)
    plt.plot(true_solution_PS4, 'b', linewidth = 3)
    plt.plot(RK23_PS4[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    
    ax = fig.add_subplot(2, 3, 5)
    plt.sca(ax)
    plt.title('Manual RK4', fontsize = ftsz)
    plt.plot(true_solution_PS4, 'b', linewidth = 3)
    plt.plot(manual_RK4_PS4[0], 'r--', linewidth = 3)
    plt.xticks(fontsize = ftsz)
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000), fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    plt.legend(('Solution', 'Approx'), fontsize = ftsz-2)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1,
                    top=1,
                    wspace=0.5,
                    hspace=0.25)
    plt.show()
    
def PS5_plot():
    ftsz = 12
    fig = plt.figure(figsize = (12,8), dpi = 300)
  
    ax = fig.add_subplot(1, 1, 1)
    plt.sca(ax)
    plt.plot(true_solution_PS5, 'b', linewidth = 3, label = 'True Solution')
    plt.plot(LSODA_PS5[0], linewidth = 3, label = 'LSODA')
    plt.plot(BDF_PS5[0], linewidth = 3, label = 'BDF')
    plt.plot(RK45_PS5[0], linewidth = 3, label = 'RK45')
    plt.plot(RK23_PS5[0], linewidth = 3, label = 'RK23')
    plt.plot(manual_RK4_PS5[0], linewidth = 3, label = 'Manual RK4')
    plt.yticks(np.arange(0, 200010, 50000), 
               labels = np.arange(0, 200010, 50000))
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.ylim([-25000, 2.25e5+10])
    plt.xlim([0, 4000])
    plt.legend()
    
    plt.show()
    
PS1_plot()
PS2_plot()
PS3_plot()
PS4_plot()
PS5_plot()
    
#%% time-varying growth rate and carrying capacity

x0 = 2e5
t = np.arange(6000)

PS7 = (0.1, 0.05, 2e5, 500)
PS8 = (0.001, 0.002, 2e5, 500)
max_K = 200500

LSODA_PS7 = test_numerical(x0, t, 'LSODA', PS7)
BDF_PS7 = test_numerical(x0, t, 'BDF', PS7)
RK45_PS7 = test_numerical(x0, t, 'RK45', PS7)
manual_RK4_PS7 = test_numerical(x0, t, 'manual RK4', PS7)

LSODA_PS8 = test_numerical(x0, t, 'LSODA', PS8)
BDF_PS8 = test_numerical(x0, t, 'BDF', PS8)
RK45_PS8 = test_numerical(x0, t, 'RK45', PS8)
manual_RK4_PS8 = test_numerical(x0, t, 'manual RK4', PS8)

def plot_PS7_PS8():
    ftsz = 14
    fig = plt.figure(figsize = (12,12), dpi = 300)
  
    ax = fig.add_subplot(2, 2, 1)
    plt.sca(ax)
    plt.plot(LSODA_PS7[0], color = 'b', linewidth = 3)
    plt.plot(LSODA_PS8[0], color = 'r', linewidth = 3)
    plt.hlines(y = max_K, xmin = 2000, xmax = 6000, color = 'k', linewidth = 2)
    plt.title('LSODA', fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.xticks(ticks = np.arange(2000, 6500, 1000), 
               labels = np.arange(2000, 6500, 1000), fontsize = ftsz)
    plt.yticks(ticks = np.arange(199000, 202000, 500), 
               labels = np.arange(199000, 202000, 500), fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.xlim([2000, 6000])
    plt.ylim([199000, 202000])

    ax = fig.add_subplot(2, 2, 2)
    plt.sca(ax)
    plt.plot(BDF_PS7[0], color = 'b', linewidth = 3)
    plt.plot(BDF_PS8[0], color = 'r', linewidth = 3)
    plt.hlines(y = max_K, xmin = 0, xmax = 6000, color = 'k', linewidth = 2)
    plt.title('BDF', fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.xticks(ticks = np.arange(2000, 6500, 1000), 
               labels = np.arange(2000, 6500, 1000), fontsize = ftsz)
    plt.yticks(ticks = np.arange(199000, 202000, 500), 
               labels = np.arange(199000, 202000, 500), fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.xlim([2000, 6000])
    plt.ylim([199000, 202000])

    ax = fig.add_subplot(2, 2, 3)
    plt.sca(ax)
    plt.plot(RK45_PS7[0], color = 'b', linewidth = 3)
    plt.plot(RK45_PS8[0], color = 'r', linewidth = 3)
    plt.hlines(y = max_K, xmin = 0, xmax = 6000, color = 'k', linewidth = 2)
    plt.title('RK45', fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.xticks(ticks = np.arange(2000, 6500, 1000), 
               labels = np.arange(2000, 6500, 1000), fontsize = ftsz)
    plt.yticks(ticks = np.arange(199000, 202000, 500), 
               labels = np.arange(199000, 202000, 500), fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.xlim([2000, 6000])
    plt.ylim([199000, 202000])

    ax = fig.add_subplot(2,2,4)
    plt.sca(ax)
    plt.plot(manual_RK4_PS7[0], color = 'b', linewidth = 3)
    plt.plot(manual_RK4_PS8[0], color = 'r', linewidth = 3)
    plt.hlines(y = max_K, xmin = 0, xmax = 6000, color = 'k', linewidth = 2)
    plt.title('Manual RK4', fontsize = ftsz)
    plt.xlabel('Time $t$', fontsize = ftsz)
    plt.ylabel('$P(t)$', fontsize = ftsz)
    plt.xticks(ticks = np.arange(2000, 6500, 1000), 
               labels = np.arange(2000, 6500, 1000), fontsize = ftsz)
    plt.yticks(ticks = np.arange(199000, 202000, 500), 
               labels = np.arange(199000, 202000, 500), fontsize = ftsz)
    plt.grid(axis = 'both')
    plt.xlim([2000, 6000])
    plt.ylim([199000, 202000])
    plt.legend(['$r_b$ = 0.1, $r_s$ = 0.05', '$r_b$ = 0.001, $r_s$ = 0.002'], 
               fontsize = ftsz)

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=1,
                    top=1,
                    wspace=0.25,
                    hspace=0.25)
    plt.show()

plot_PS7_PS8()

