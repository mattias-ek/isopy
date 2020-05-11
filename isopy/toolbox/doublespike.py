import numpy as np
import scipy.optimize as spo
import isopy.core as _dtypes
import itertools as _itertools
#import matplotlib.pyplot as plt
from matplotlib import cm, colors

__all__ = []

"""
Functions for doublespike data reduction
"""

class DoubleSpikeInversionOutput:
    def __init__(self, method, lambda_, alpha, beta, Fnat, Fins, spike_faction, sample_fraction, Q):
        self.method = method

        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta

        self.Fnat = Fnat
        self.Fins = Fins

        self.spike_fraction = spike_faction
        self.sample_fraction = sample_fraction
        self.Q = Q

    def delta(self, mass_ratio):
        #use Fnat to calcualte a offset for the mass ratio given
        pass

    def ppm(self, mass_ratio):
        pass


def ds_inversion_rudge(m, n, T, P, xtol = 1e-10):
    """
    Double spike inversion using the method of Rudge et al. 2009, Chemical Geology 265 420-431.

    Parameters
    ----------
    self
    m : ndarray
        Measured isotope ratios. Must be an 1-dim array of size 3 or a 2-dim array with first dimension of 3.
    n : ndarray
        Reference isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `m`.
    T : ndarray
        Spike isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `m`.
    P : ndarray
        Log of mass isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `m`.
    xtol : float
        Required tolerance for inversion

    Returns
    -------
    DoubleSpikeInversionOutput
    """
    mndim = m.ndim
    if m.size == 3: m = m.reshape(3,1)
    elif m.ndim == 1: raise ValueError()
    elif m.ndim == 2:
        if m.shape[0] != 3: raise ValueError()
    else: raise ValueError('m has to many dimensions ({})'.format(m.ndim))

    if n.size == 3: n = np.ones(m.shape) * n.reshape(3,1)
    elif n.shape != m.shape: raise ValueError('Shape of n {} does not match shape of m {}'.format(n.shape, m.shape))

    if T.size == 3: T = np.ones(m.shape) * T.reshape(3,1)
    elif T.shape != m.shape: ValueError('Shape of T {} does not match shape of m {}'.format(T.shape, m.shape))

    if P.size == 3: P = np.ones(m.shape) * P.reshape(3,1)
    elif P.shape != m.shape: ValueError('Shape of P {} does not match shape of m {}'.format(P.shape, m.shape))

    A = np.transpose([T-n, -n*P, m*P])
    b = np.transpose(m-n)
    x0 = np.transpose(np.linalg.solve(A,b))

    x, infodict, ier, mesg =  spo.fsolve(_inversion_rudge_function, x0, (P, n, T, m), None, True, xtol = xtol)

    if ier != 1:
        x = np.ones(m.shape) * np.nan
    error_message = mesg #Shoudl this raises an errror. I guess it primaraly happens when there are o many htings to solve

    if mndim == 2: x = x.reshape(3,-1)
    lambda_ = x[0]
    alpha = x[1]/(1-lambda_)
    beta = x[2]

    Fnat = alpha * -1
    Fins = beta

    spike_fraction = (1+((1-lambda_)/lambda_)*((1+np.sum((n/np.exp(alpha*P)), axis = 0))/(1+np.sum(T, axis = 0))))**-1
    sample_fraction = (1-spike_fraction)
    Q = sample_fraction / spike_fraction

    return DoubleSpikeInversionOutput('Rudge', lambda_, alpha, beta, Fnat, Fins, spike_fraction, sample_fraction, Q)

def _inversion_rudge_function( x, P, n, T, m):
    x  = x.reshape(3,-1)
    lambda_ = x[0]
    alpha = x[1]/(1-lambda_)
    beta = x[2]

    return ((lambda_ * T) + ((1-lambda_) * n * np.exp(-alpha * P)) - m * np.exp(-beta * P)).flatten()

def _inversion_rudge_jacobian( x, P, n, T, m):
    #follwing code only works for size = 3
    x = x.reshape(3,-1)
    lambda_ = x[0]
    alpha = x[1]/(1-lambda_)
    beta = x[2]
    output = np.transpose([T - (n * np.exp(-alpha * P) * (1 + alpha * P)), -n * P * np.exp(-alpha * P), m * P * np.exp(-beta * P)])
    return output

def ds_inversion_siebert(MS, ST, SP, Mass, Fnat_guess = 2, Fins_guess = -0.00001, loop_one_iterations = 3, loop_two_iterations=6):
    """
    Double spike inversion using the method of Siebert et al. 2001, Geochemistry, Geophysics, Geosystems 2,

    Parameters
    ----------
    self
    MS : ndarray
        Measured isotope ratios. Must be an 1-dim array of size 3 or a 2-dim array with first dimension of 3.
    ST : ndarray
        Reference isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `MS`.
    SP : ndarray
        Spike isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `MS`.
    Mass : ndarray
        Log of mass isotope ratios. Must be a 1-dim array of size 3 or a 2-dim array with the same shape as `MS`.

    Fnat_guess : float
        Starting guess for Fnat
    Fins_guess : float
        Starting guess for Fins
    loop_one_iterations : int
        Number of iterations of the outer loop
    loop_two_iterations
        Number of iterations of the inner loop

    Returns
    -------
    DoubleSpikeInversionOutput
    """
    output_index = 0
    mndim = MS.ndim
    if MS.size == 3:
        MS = MS.reshape(3, 1)
    elif MS.ndim == 1: raise ValueError()
    elif MS.ndim == 2:
        if MS.shape[0] != 3: raise ValueError()
    else: raise ValueError('MS has to many dimensions ({})'.format(MS.ndim))

    if ST.size == 3: ST = ST.reshape(3)
    elif ST.shape != MS.shape: raise ValueError('Shape of ST {} does not match shape of MS {}'.format(ST.shape, MS.shape))

    if ST.size == 3: ST = ST.reshape(3)
    elif ST.shape != MS.shape: ValueError('Shape of ST {} does not match shape of MS {}'.format(ST.shape, MS.shape))

    if Mass.size == 3: Mass = Mass.reshape(3)
    elif Mass.shape != MS.shape: ValueError('Shape of Mass {} does not match shape of MS {}'.format(Mass.shape, MS.shape))

    dlen = MS.shape[1]
    SA = np.zeros((loop_one_iterations, 3, dlen), dtype = np.float64)
    Fins = np.zeros((loop_one_iterations, 3, dlen), dtype = np.float64)
    Fnat = np.zeros((loop_one_iterations, 3, dlen), dtype = np.float64)
    MT = np.zeros((3, dlen), dtype = np.float64)

    x = 0
    y = 1
    z = 2
    Ri = output_index

    Fnat_Ri = Fnat_guess
    Fins_Ri = Fins_guess

    Q = np.zeros(dlen, dtype = np.float64)

    for i1 in range(loop_one_iterations):
        SA[i1, x,:] = ST[x] * Mass[x] ** Fnat_Ri
        SA[i1, y,:] = ST[y] * Mass[y] ** Fnat_Ri
        SA[i1, z,:] = ST[z] * Mass[z] ** Fnat_Ri

        a = (ST[y] * (SA[i1, z] - SP[z]) + SA[i1, y] * (SP[z] - ST[z]) + SP[y] * (ST[z] - SA[i1, z])) / (ST[y] * (SA[i1, x] - SP[x]) + SA[i1, y] * (SP[x] - ST[x]) + SP[y] * (ST[x] - SA[i1, x]))
        b = (ST[x] * (SA[i1, z] - SP[z]) + SA[i1, x] * (SP[z] - ST[z]) + SP[x] * (ST[z] - SA[i1, z])) / (ST[x] * (SA[i1, y] - SP[y]) + SA[i1, x] * (SP[y] - ST[y]) + SP[x] * (ST[y] - SA[i1, y]))
        c = ST[z] - a * ST[x] - b * ST[y]

        for i2 in range(loop_two_iterations):
            MT[x] = MS[x] * Mass[x] ** -Fins_Ri
            MT[y] = MS[y] * Mass[y] ** -Fins_Ri
            MT[z] = MS[z] * Mass[z] ** -Fins_Ri

            d = (MS[z] - MT[z]) / (MS[x] - MT[x])
            e = MS[z] - d * MS[x]
            f = (MS[z] - MT[z]) / (MS[y] - MT[y])
            g = MS[z] - f * MS[y]

            MT[x] = (b * g - b * e + e * f - c * f) / (a * f + b * d - d * f)
            MT[y] = (a * e - a * g + d * g - c * d) / (a * f + b * d - d * f)
            MT[z] = a * MT[x] + b * MT[y] + c

            #gives positive Fins
            Fins[i1, 0] = np.log(MS[x] / MT[x]) / np.log(Mass[x])
            Fins[i1, 1] = np.log(MS[y] / MT[y]) / np.log(Mass[y])
            Fins[i1, 2] = np.log(MS[z] / MT[z]) / np.log(Mass[z])
            Fins_Ri = Fins[i1, Ri]

        a = (MS[y] * (MT[z] - SP[z]) + MT[y] * (SP[z] - MS[z]) + SP[y] * (MS[z] - MT[z])) / (MS[y] * (MT[x] - SP[x]) + MT[y] * (SP[x] - MS[x]) + SP[y] * (MS[x] - MT[x]))
        b = (MS[x] * (MT[z] - SP[z]) + MT[x] * (SP[z] - MS[z]) + SP[x] * (MS[z] - MT[z])) / (MS[x] * (MT[y] - SP[y]) + MT[x] * (SP[y] - MS[y]) + SP[x] * (MS[y] - MT[y]))
        c = MS[z] - a * MS[x] - b * MS[y]

        d = (ST[z] - SA[i1, z]) / (ST[x] - SA[i1, x])
        e = ST[z] - d * ST[x]
        f = (ST[z] - SA[i1, z]) / (ST[y] - SA[i1, y])
        g = ST[z] - f * ST[y]

        SA[i1, x] = (b * g - b * e + e * f - c * f) / (a * f + b * d - d * f)
        SA[i1, y] = (a * e - a * g + d * g - c * d) / (a * f + b * d - d * f)
        SA[i1, z] = a * SA[i1, x] + b * SA[i1, y] + c

        Fnat[i1, x] = np.log(SA[i1, x] / ST[x]) / np.log(Mass[x])
        Fnat[i1, y] = np.log(SA[i1, y] / ST[y]) / np.log(Mass[y])
        Fnat[i1, z] = np.log(SA[i1, z] / ST[z]) / np.log(Mass[z])
        Fnat_Ri = Fnat[i1, Ri]


    sample_fraction = ((1 / (MT[x] + MT[y] + MT[z] + 1)) - (1 / (SP[x] + SP[y] + SP[z] + 1))) / ((1 / (SA[loop_one_iterations - 1, x] + SA[loop_one_iterations - 1, y] + SA[loop_one_iterations - 1, z] + 1)) - (1 / (SP[x] + SP[y] + SP[z] + 1)))
    Fnat_f = Fnat[-1, Ri]
    Fins_f = Fins[-1, Ri]

    lambda_ = ((1-sample_fraction) ** (-1) - 1) / ((1 + np.sum(SA[-1], axis=0)) / (1 + np.sum(SP, axis=0)))
    lambda_ = 1 - (lambda_ / (lambda_ + 1))

    if mndim == 1:
        sample_fraction =  sample_fraction[0]
        Fnat_f= Fnat_f[0]
        Fins_f = Fins_f[0]
        lambda_ = lambda_[0]

    spike_fraction = (1-sample_fraction)
    Q = sample_fraction/spike_fraction

    alpha = Fnat_f * -1
    beta = Fins_f

    return DoubleSpikeInversionOutput(
        'Siebert', lambda_, alpha, beta, Fnat_f, Fins_f, spike_fraction, sample_fraction, Q)

def ds_inversion(measured, standard, spike, mass, inversion_ratios = None, method ='rudge', **method_kwargs):
    """
    Double spike inversion

    Examples
    --------
    lots of usage examples here

    Parameters
    ----------
    measured : RatioArray
        Measured isotope ratios
    standard : RatioArray, IsopyDict
        References isotope ratios or a dict or references values
    spike : RatioArray, IsopyDict
        Spike isotope ratios or a dict or references values
    mass : RatioArray, IsopyDict
        Mass isotope ratios or a dict or references values
    inversion_ratios : RatioList, None
        List of keys in measured that will be used for the inversion. Only necessary if measured contains more than 3 keys.
    method : str
        Inversion method to be used. Options are 'Rudge' and 'Siebert'.
    method_kwargs
        Keyword arguments for inversion method. See `inversion_rudge` and `inversion_siebert` for list of possible arguments.

    Returns
    -------
    DoubleSpikeInversionOutput
    """
    #Measured must be RatioArray
    #Standard, spike, and mass can be either RatioArray or dict
    if inversion_ratios is None:
        if len(measured.keys()) > 3: raise ValueError('If "measured" has more than 3 keys then "inversion_ratios" must be given')
        inversion_ratios = measured.keys()
    m = np.array([measured[x] for x in inversion_ratios])
    n = np.array([standard[x] for x in inversion_ratios])
    T = np.array([spike[x] for x in inversion_ratios])
    P = np.array([mass[x] for x in inversion_ratios])

    if method == 'rudge': return ds_inversion_rudge(m, n, T, np.log(P), **method_kwargs)
    elif method == 'siebert': return ds_inversion_siebert(m, n, T, P)
    else: raise ValueError('method "{}" not recognized'.format(method), **method_kwargs)


def johnson_nyquist_noise(voltages, resistors = None, integration_time = None, include_counting_statistics = True,
                          k = None, T = None, R = None, cpv = None):

    #Follows the equations from Liu & Pearson, Chemical Geology, 2014, 10, p301-311
    if integration_time is None: integration_time = np.float64(8.389)
    if k is None: k = np.float64(1.3806488E-023)
    #e = np.float64(1.602176565E-019)
    if T is None: T = np.float64(309)
    if R is None: R = np.float64(1E11)
    if cpv is None: cpv = np.float64(62500 * 1000)

    voltages = np.asarray(voltages)
    if resistors is None: resistors = [R] * voltages.shape[-1]
    resistors = np.asarray(resistors)

    t_noise = np.sqrt((4 * k * T * resistors) / integration_time) / (voltages * (resistors / R))

    if include_counting_statistics:
        c_stat = np.sqrt(1 / (voltages * cpv * integration_time))
        return np.sqrt(c_stat**2 + t_noise **2)*voltages
    else:
        return t_noise

def ds_uncertainty_grid(standard, spike1, spike2, mass, *, isotopes = None, max_voltage = 10, resistors = None,
                        integration_time = 4, cycles = 100, resolution = 19, Fnat = 0.1, Fins = 1.6,
                        ds_method ='Siebert', fast_mode = True, output_attr = 'alpha'):

    if isinstance(standard, _dtypes.IsotopeArray):
        if isotopes is None: isotopes = standard.keys()
        std = np.array([standard[i] for i in isotopes]).flatten()
    elif isinstance(standard, _dtypes.IsopyDict):
        std = np.array([standard[i] for i in isotopes]).flatten()
    else:
        std = np.asarray(standard).flatten()

    if std.size != 4: raise ValueError('standard must contain exactly four values')

    if isinstance(spike1, str):
        if isotopes is None: raise ValueError('spike1: "isotopes" not set')
        sp1 = np.zeros(4, dtype='f8')
        sp1[isotopes.index(spike1)] = 1
    elif isinstance(spike1, _dtypes.IsotopeArray):
        if isotopes is None: raise ValueError('spike1: "isotopes" not set')
        std = np.array([spike1[i] for i in isotopes]).flatten()
    else:
        sp1 = np.asarray(spike1).flatten()
    if sp1.size != 4: raise ValueError('spike1 must contain exactly four values')

    if isinstance(spike2, str):
        if isotopes is None: raise ValueError('spike2: "isotopes" not set')
        sp2 = np.zeros(4, dtype='f8')
        sp2[isotopes.index(spike2)] = 1
    elif isinstance(spike2, _dtypes.IsotopeArray):
        if isotopes is None: raise ValueError('spike2: "isotopes" not set')
        std = np.array([spike2[i] for i in isotopes]).flatten()
    else:
        sp2 = np.asarray(spike2).flatten()
    if sp2.size != 4: raise ValueError('spike2 must contain exactly four values')

    if isinstance(mass, (_dtypes.IsotopeArray, _dtypes.IsopyDict)):
        if isotopes is None: raise ValueError('mass: "isotopes" not set')
        mas = np.array([mass[i] for i in isotopes]).flatten()
    else:
        mas = np.asarray(mass).flatten()
    if mas.size != 4: raise ValueError('mass must contain exactly four values')

    #At this point standard, spike1 and spike2 are arrays of 4.
    standard = std / np.sum(std)
    spike1 = sp1 / np.sum(sp1)
    spike2 = sp2 / np.sum(sp2)
    mass = mas

    prop = np.linspace(0.0,1, resolution+2)[1:-1]

    denom = np.argmax(spike2)
    rat_index = [True] * 4
    rat_index[denom] = False

    mass_ratio = mass[rat_index] / mass[denom]

    if ds_method == 'Rudge':
        method = Rudge_inversion
        mass_ratio2 = np.log(mass_ratio)

    elif ds_method == 'Siebert':
        method = Siebert_inversion
        mass_ratio2 = mass_ratio

    std_ratio = (standard[rat_index] / standard[denom]) * mass_ratio ** Fnat

    if fast_mode is False:
        z = np.zeros((resolution, resolution, cycles), dtype=np.float64) * np.nan
        for y in range(resolution):
            spike = prop[y] * spike1 + (1-prop[y]) * spike2
            sp_rat = spike[rat_index] / spike[denom]
            for x in range(resolution):
                sample = prop[x] * spike + (1 - prop[x]) * standard
                sample = sample * max_voltage/np.max(sample)
                noise = johnson_nyquist_noise(sample, resistors, integration_time)
                noisy_sample = np.random.normal(sample, noise, (cycles,4)).transpose()
                smp_rat = noisy_sample[rat_index] / noisy_sample[denom]
                smp_rat = np.transpose(smp_rat.transpose() * mass_ratio ** Fins)

                solution = method(smp_rat, std_ratio, sp_rat, mass_ratio2)
                z[y,x, :] = solution.__getattribute__(output_attr)

    if fast_mode is True:
        big = np.ones((resolution, resolution, cycles, 4))
        small = np.ones((resolution, resolution, 4))

        spike = big * spike1 * prop.reshape(resolution, 1, 1, 1) + big * spike2 * (1 - prop).reshape(resolution, 1, 1,                                                                                          1)
        smp = spike[:, :, 1, :] * prop.reshape(1, resolution, 1) + small * standard * (1 - prop).reshape(1, resolution,
                                                                                                         1)
        smp = smp.reshape(-1, 4)
        smp = smp * (max_voltage / np.max(smp, axis=1).reshape(-1, 1))
        noise = johnson_nyquist_noise(smp, resistors, integration_time)

        noisy_sample = np.random.normal(smp, noise, (cycles, resolution ** 2, 4))
        noisy_sample = noisy_sample.transpose().reshape(4, -1)
        smp_ratio = noisy_sample[rat_index] / noisy_sample[denom] * (mass_ratio ** Fins).reshape(3, 1)

        spike = spike.flatten().reshape(-1, 4).transpose()
        sp_ratio = spike[rat_index] / spike[denom]

        #TODO check error that comes from rudge and do a specieal error
        solution = method(smp_ratio, std_ratio, sp_ratio, mass_ratio2)
        z = solution.__getattribute__(output_attr).transpose().flatten().reshape(resolution, resolution, -1)

    std = np.nanstd(z, axis=2) * 2
    count = np.count_nonzero(~np.isnan(z), axis=2)

    std[count < cycles * 0.9] = np.nan

    if output_attr is 'alpha':
        mean = np.nanmean(z, axis=2)
        std[np.abs(mean - Fnat) > std] = np.nan

    argmin_z = np.nanargmin(std)
    min_x_pos = argmin_z % std.shape[0]
    min_y_pos = int(argmin_z / std.shape[0])

    #x = sp in sp+smp, y = sp1 in sp1+sp2
    return prop, prop, std / np.sqrt(count), (min_y_pos, min_x_pos)

def plot_ds_uncertainty_grid(*args, **kwargs):
    x, y, z, best = ds_uncertainty_grid(*args, **kwargs)

    levels = [0] + [x * 0.0001 for x in range(1, 10)] + [x * 0.001 for x in range(1, 10)] + [x * 0.01 for x in
                                                                                             range(1, 10)] + [0.1]
    min_z = z[best]
    min_x = x[best[1]]
    min_y = y[best[0]]

    title = 'Smallest uncertianty: {:.5f} ($2\\sigma/\\sqrt{}n{}$) ' \
            'at x: {:.2f}, y: {:.2f}'.format(min_z, '{', '}', min_x, min_y)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.3)
    CS = ax.contourf(x, y, z, levels=levels, cmap=cm.get_cmap(cm.jet), norm=colors.PowerNorm(0.23))
    fig.colorbar(CS, ax=ax, ticks=[0, 0.001, 0.01, 0.1, 1])
    plt.xlabel('Spike fraction in Spike/Sample mix')
    plt.ylabel('Spike1 fraction in Spike1/Spike2 mix')
    plt.title(title)
    plt.show()


def ds_cocktail_list(standard, mass, spikes = None, isotopes = None, output_mass_ratio = None, output_multiple = 1000, **kwargs):
    if isinstance(standard, _dtypes.IsotopeArray):
        if isotopes is None: isotopes = standard.keys()
    elif isinstance(standard, _dtypes.IsopyDict):
        if isotopes is None: raise ValueError('"isotopes" not given')
    else:
        raise ValueError('"standard" must be either a IsotopeArrray or a IsopyDict')

    if isotopes is None: raise ValueError('"isotopes" not given')
    if len(isotopes) < 4: raise ValueError('At least 4 isotopes are needed')

    if spikes is None: spikes = {}
    if not isinstance(spikes, dict): raise ValueError('"spikes" must be a dict')
    for s in spikes:
        if not isinstance(s, _dtypes.IsotopeArray, _dtypes.IsopyDict):
            raise ValueError('Each entry in "spikes" must be either a IsotopeArrray or a IsopyDict')

    if not isinstance(mass, (_dtypes.IsotopeArray, _dtypes.IsopyDict)):
        raise ValueError('"mass" must be either a IsotopeArray or a IsopyDict')

    output = []
    for iso in _itertools.combinations(isotopes, 4):
        for spike in _itertools.combinations(iso, 2):
            x, y, z, best = uncertianty_grid(standard, spikes.get(spike[0], spike[0]), spikes.get(spike[1], spike[1]), mass, isotopes = iso, **kwargs)

            x_min = x[best[1]]
            y_min = y[best[0]]

            z_min = z[best]

            if output_mass_ratio is not None:
                z_min = (output_mass_ratio ** z[best] -1) * output_multiple

            output.append([tuple(iso), tuple(spike), (x_min, y_min, z_min), x, y, z, best])
    return output







