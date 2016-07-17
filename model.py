import numpy as np
from scipy.special import erf

def filter_trajectory(times, heights, mass, L):
    stop = len(heights)
    for i in range(2,len(heights)):
        if heights[i] > 24.0:
            stop = i
            break
        if i + 1 < stop and all([heights[i] > heights[j] for j in range(i + 1, stop)]) :
            stop = i
            break

    return times[:stop], heights[:stop], mass[:stop]

guess_dyn = np.array([0.64, 113.0, 1.0, 1.0])
guess_mix = [4.0,]

fix = [
        1.0,  #C_dyn[0]
        1./(2.*np.pi), #C_dyn[4]
        1./(np.pi), #C_dyn[6]
        1.0, #C_mix[1]
      ]

bounds_dyn = [[ 0.00,   0.0, 1.00,  1.0],
              [10.00, 400.0, 8.0,   8.002] ]
scaling_dyn = [1.0,  50.0, 3.501, 3.501]
reg_param = 0.0

bounds_dyn_t = []
for i in range(len(guess_dyn)):
    bounds_dyn_t.append((bounds_dyn[0][i], bounds_dyn[1][i]))

bounds_mix = [(1., 5.),]


def exp_dyn(x):
    return [fix[0], x[0], x[1], x[2], fix[1], x[3], fix[2]]

def exp_mix(x):
    return [x[0], fix[3]]

guess_dyn_full = np.array(exp_dyn(guess_dyn))

def get_scaling(bounds):
    scaling_of_vars = np.ones(len(bounds[0]))
    for i in range(scaling_of_vars.shape[0]):
        scaling_of_vars[i] = (bounds[1][i] - bounds[0][i]) / 2.
    return scaling_of_vars

def merge_coef(var, fix):
    return [fix[0], var[0], var[1], var[2], fix[1], var[3], fix[2], var[4], fix[3]]

def f_dyn(t, y, C, Atwood, visc, L, mixed_fluid):
    x = (2 * np.pi) ** (3./2) * visc / np.sqrt(Atwood * L**3) 
    foo = 2 * x * (np.sqrt(1.0 + x*x) + x)
    M  = (C[4]*L*(1.0+foo) + C[3] * abs(y[0])) * L * L
    V  = (C[6]*L + C[5] * abs(y[0])) * L * L
   
    if isinstance(mixed_fluid, float):
        mix = 1. - mixed_fluid / V
    else:
        mix = 1. - mixed_fluid(t) / V
         
    F = C[0] * Atwood * L * L * y[0] * mix
    F += - C[1] * L * L * y[1] * abs(y[1])
    F += - C[2] * visc * abs(y[0]) * y[1] 

    return [y[1], F / M]

def f_full(t, y, C_dyn, C_mix, Atwood, visc, L, diff, delta_i):
    mixed_fluid = mix_direct(C_mix, L, diff, delta_i, t, y[0])

    return f_dyn(t, y, C_dyn, Atwood, visc, L, mixed_fluid)

def integrate(r, times):
    Ny = len(r.y)
    soln = []
    for i in range(Ny):
        soln.append([r.y[i],])
    
    for t in times[1:]:
        r.integrate(t)
        if not r.successful():
            raise RuntimeError("integration failed")
        for i in range(Ny):
            soln[i].append(r.y[i])
    soln_np = [times,] + [np.array(x) for x in soln]
    return soln_np

def dyn_model(C, Atwood, visc, L, mix, y0, times):
    r = ode(f_dyn).set_integrator('vode', method='bdf', atol=1.0e-6, with_jacobian=False)
    r.set_initial_value(y0, times[0]).set_f_params(C, Atwood, visc, L, mix)

    [T, H, v] = integrate(r, times)
    return T, H, v

from scipy.integrate import ode
def full_model(C_dyn, C_mix, Atwood, visc, L, diff, delta_i, y0, times):

    r = ode(f_full).set_integrator('vode', method='bdf', atol=1.0e-6, with_jacobian=False)
    r.set_initial_value(y0, times[0]).set_f_params(C_dyn, C_mix, Atwood, visc, L, diff, delta_i)
   
    [T, H, v] = integrate(r, times)
    MM = mix_direct(C_mix, L, diff, delta_i, T, H)

    return T, H, v, MM

def mix_direct(C, L, diff, delta_i, time, height):
    offset = delta_i**2. / (4. * diff)
    delta = 2. * np.sqrt((time+offset) * diff)
    diam = C[0] * L/(2.*4)
    SA =  (C[1] * L + C[0]*height) * L
    integral = (
                 2*delta/np.sqrt(np.pi)*(1-np.exp(-(diam**2./np.square(delta)))) 
               + 2*diam*(1-erf(diam / delta))
               )
    DM = integral * SA
    return DM

def mix_model(C, L, diff, delta_i, times, h_func):
    return times, mix_direct(C, L, diff, delta_i, times, h_func(times))

from numpy.linalg import norm
def error(C_dyn, C_mix, Atwood, v, L, c, delta_i, y0, times, heights, mix, reg_param = 0.0):
    derr, merr = both_error(C_dyn, C_mix, Atwood, v, L, c, delta_i, y0, times, heights, mix)
    se = derr * derr 
    return np.sqrt(se) + reg_param * np.sqrt(np.sum(np.square((C_dyn - guess_dyn_full)/guess_dyn_full)))


def both_error(C_dyn, C_mix, Atwood, v, L, c, delta_i, y0, times, heights, mix):
    try:
        T, H, V, MM = full_model(C_dyn, C_mix, Atwood, v, L, c, delta_i, y0, times)
    except RuntimeError:
        return 1000.0, 1000.0
    dyn_error = np.sqrt(np.sum(np.square(H - heights))/heights.size)
    mix_error = np.sqrt(np.sum(np.square(mix - MM)) / (L**4.) / heights.size)
    return dyn_error, mix_error

def dyn_error(C_dyn, Atwood, v, L, y0, times, height, mix):
    T, H, V = dyn_model(C_dyn, Atwood, v, L, mix, y0, times)

    se = np.sum(np.square(H - height))
    re = np.sqrt(np.sum(np.square((np.array(C_dyn) - guess_dyn_full)/guess_dyn_full)))
    return np.sqrt(se / times.size) + reg_param * re

def mix_error(C_mix, L, c, delta_i, times, h_func, mix):
    T, DM = mix_model(C_mix, L, c, delta_i, times, h_func)

    se = np.sum(np.square(mix - DM)) / (L**4.)
    return np.sqrt(se / times.size)

