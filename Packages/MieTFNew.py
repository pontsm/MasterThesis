import tensorflow as tf
import tensorflow_probability as tfp
import math as math
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys


def get_Dn():
    
    radius = tf.range(50, 15000, 10)
    radius = tf.cast(radius, tf.float64)
    
    pi = tf.cast(math.pi, tf.float64)
    wavelength = tf.cast(532, dtype = tf.float64)
    
    x = 2*pi*radius/wavelength
    
    x = tf.cast(x, tf.complex128)
    
    intsteps = 1495
       
    @tf.function(experimental_relax_shapes=True)
    def TFDn(m):

        m = tf.cast(m, tf.complex128)

        ta = tf.TensorArray(tf.complex128, size=0, dynamic_size=True)
        ta = ta.unstack(tf.zeros((61, intsteps), dtype=tf.complex128))

        for i in range(59, 0,-1):
            ta = ta.write(i, (i+1.)/(m*x) - 1./((ta.read(i + 1)) + (i+1.)/(m*x)))

        output = ta.stack()[1:51]

        return output
    
    return TFDn


@tf.function(experimental_relax_shapes=True)
def TFsizedistribution1(r, Vtot, sigma, rmean):

    sigma = tf.math.log(sigma)
    pi = tf.cast(math.pi, tf.float64)
    mu = tf.math.log(rmean) - 3*sigma**2
    N0 = 3. / 4. / pi * Vtot * tf.exp(-(3.*mu + 4.5*sigma**2))

    return N0 /(tf.sqrt(2*pi)) * 1/sigma * 1/r * tf.exp(-(tf.math.log(r)-mu)**2/(2*sigma**2))


def get_TFEnsemble():
    
    n1 = tf.range(1.,51.)
    n1 = tf.cast(n1, tf.complex128)
    n1 = n1[tf.newaxis]

    inumber = tf.cast(tf.complex(0.,1.), tf.complex128)
    
    RiccatiSn = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/RiccatiSn.npy'))
    RiccatiSn0 = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/RiccatiSn0.npy'))
    RiccatiXin = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/RiccatiXin.npy')) 
    RiccatiXin0 = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/RiccatiXin0.npy'))
    
    pi_n = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/finalpi.npy'))
    tau_n = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/finaltau.npy'))
    
    n2 = tf.range(1.,51.)
    n2 = tf.cast(n2, tf.complex128)

    Prefactor = (2.*n2+1.)/(n2*(n2+1.))
    Prefactor = Prefactor[tf.newaxis,tf.newaxis]
    Prefactor = tf.transpose(Prefactor)
    
    radius = tf.range(50, 15000, 10)
    radius = tf.cast(radius, tf.float64)
    radiusrepeat = tf.transpose(tf.repeat(radius[:,tf.newaxis], 180, axis=1))

    intsteps = 1495
    
    zero = tf.constant(0, dtype=tf.float64)
    
    epsilon = tf.constant(1e-100, dtype=tf.float64)
    
    pi = tf.cast(math.pi, tf.float64)
    wavelength = tf.cast(532, dtype = tf.float64)
    
    x = 2*pi*radius/wavelength
    
    x = tf.cast(x, tf.complex128)
    
    TFDn = get_Dn()
    
    @tf.function(experimental_relax_shapes=True)
    def TFEnsemble(m, Vtot, sigma, rmean):
        
        m = tf.cast(m, tf.complex128)
        Vtot = tf.cast(Vtot, dtype = tf.float64)
        sigma = tf.cast(sigma, dtype = tf.float64)
        rmean = tf.cast(rmean, dtype = tf.float64)
        
        m = tf.cast(m, tf.complex128)
        
        D_n = TFDn(m)

        FactorFora_n = D_n/m + tf.transpose(n1)/x
        FactorForb_n = m*D_n + tf.transpose(n1)/x
        
        a = tf.math.multiply(FactorFora_n,RiccatiSn)
        b = tf.math.multiply(FactorForb_n,RiccatiSn)
        c = tf.math.multiply(FactorFora_n,RiccatiXin)
        d = tf.math.multiply(FactorForb_n,RiccatiXin)
        e = tf.math.subtract(a, RiccatiSn0)
        f = tf.math.subtract(c,RiccatiXin0)
        g = tf.math.subtract(b, RiccatiSn0)
        h = tf.math.subtract(d,RiccatiXin0)

        a_n = e / f
        b_n = g / h
        
        a_n = a_n[:,tf.newaxis,:]
        b_n = b_n[:,tf.newaxis,:]
        
        S_1 = tf.reduce_sum(Prefactor * ( pi_n * a_n + tau_n * b_n),0)
        S_2 = tf.reduce_sum(Prefactor * ( tau_n * a_n + pi_n * b_n),0)

        S_11 = 0.5*(tf.abs(S_2)**2+tf.abs(S_1)**2)
        S_12 = 0.5*(tf.abs(S_2)**2-tf.abs(S_1)**2)

        S11Integrand = S_11*TFsizedistribution1(radius, Vtot, sigma, rmean)
        S12Integrand = S_12*TFsizedistribution1(radius, Vtot, sigma, rmean)

        S11Result = tfp.math.trapz(S11Integrand, x = radiusrepeat)
        S12Result = tfp.math.trapz(S12Integrand, x = radiusrepeat)
        
        S11Result = tf.math.add(S11Result, epsilon)
        
        ppf = tf.math.divide(S12Result, S11Result)
        ppf = tf.subtract(zero, ppf)
        
        return S11Result, ppf
    
    return TFEnsemble    


def get_TFEnsembleP11():
    
    n1 = tf.range(1.,51.)
    n1 = tf.cast(n1, tf.complex128)
    n1 = n1[tf.newaxis]

    inumber = tf.cast(tf.complex(0.,1.), tf.complex128)
    
    RiccatiSn = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/RiccatiSn.npy'))
    RiccatiSn0 = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/RiccatiSn0.npy'))
    RiccatiXin = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/RiccatiXin.npy')) 
    RiccatiXin0 = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/RiccatiXin0.npy'))
    
    pi_n = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/finalpi.npy'))
    tau_n = tf.constant(np.load('/data/user/ponts_m/aerosol/aerosol_code1/helper_functions/MIE_NN/finaltau.npy'))
    
    n2 = tf.range(1.,51.)
    n2 = tf.cast(n2, tf.complex128)

    Prefactor = (2.*n2+1.)/(n2*(n2+1.))
    Prefactor = Prefactor[tf.newaxis,tf.newaxis]
    Prefactor = tf.transpose(Prefactor)
    
    radius = tf.range(50, 15000, 10)
    radius = tf.cast(radius, tf.float64)
    radiusrepeat = tf.transpose(tf.repeat(radius[:,tf.newaxis], 180, axis=1))

    intsteps = 1495
    
    zero = tf.constant(0, dtype=tf.float64)
    
    epsilon = tf.constant(1e-100, dtype=tf.float64)
    
    pi = tf.cast(math.pi, tf.float64)
    wavelength = tf.cast(532, dtype = tf.float64)
    
    x = 2*pi*radius/wavelength
    
    x = tf.cast(x, tf.complex128)
    
    TFDn = get_Dn()
    
    @tf.function(experimental_relax_shapes=True)
    def TFEnsemble(m, Vtot, sigma, rmean):
        
        m = tf.cast(m, tf.complex128)
        Vtot = tf.cast(Vtot, dtype = tf.float64)
        sigma = tf.cast(sigma, dtype = tf.float64)
        rmean = tf.cast(rmean, dtype = tf.float64)
        
        m = tf.cast(m, tf.complex128)
        
        D_n = TFDn(m)

        FactorFora_n = D_n/m + tf.transpose(n1)/x
        FactorForb_n = m*D_n + tf.transpose(n1)/x
        
        a = tf.math.multiply(FactorFora_n,RiccatiSn)
        b = tf.math.multiply(FactorForb_n,RiccatiSn)
        c = tf.math.multiply(FactorFora_n,RiccatiXin)
        d = tf.math.multiply(FactorForb_n,RiccatiXin)
        e = tf.math.subtract(a, RiccatiSn0)
        f = tf.math.subtract(c,RiccatiXin0)
        g = tf.math.subtract(b, RiccatiSn0)
        h = tf.math.subtract(d,RiccatiXin0)

        a_n = e / f
        b_n = g / h
        
        a_n = a_n[:,tf.newaxis,:]
        b_n = b_n[:,tf.newaxis,:]
        
        S_1 = tf.reduce_sum(Prefactor * ( pi_n * a_n + tau_n * b_n),0)
        S_2 = tf.reduce_sum(Prefactor * ( tau_n * a_n + pi_n * b_n),0)

        S_11 = 0.5*(tf.abs(S_2)**2+tf.abs(S_1)**2)

        S11Integrand = S_11*TFsizedistribution1(radius, Vtot, sigma, rmean)

        S11Result = tfp.math.trapz(S11Integrand, x = radiusrepeat)
        
        return S11Result
    
    return TFEnsemble    


def get_PINNLossFunctionP11():

    TFE = get_TFEnsembleP11()
    
    @tf.function(experimental_relax_shapes=True)
    def PINNLossFunction(output, y_true, batchsize):

        phasefunctions = output[:,:-5]

        n = output[:,-5]
        k = output[:,-4]

        n = tf.cast(n, dtype=tf.complex128)
        k = tf.cast(k, dtype=tf.complex128)
        k = k*1j

        m = n+k

        Vtot = output[:,-1]
        Vtot = tf.cast(Vtot, dtype=tf.float64)

        sigma = tf.math.abs(output[:,-3])
        sigma = tf.cast(sigma, dtype=tf.float64)

        rmean = tf.math.abs(output[:,-2])
        rmean = tf.cast(rmean, dtype=tf.float64)

        loss = tf.TensorArray(dtype = tf.float64, size=batchsize, dynamic_size=True, clear_after_read=False)
        loss = loss.unstack(tf.zeros((batchsize, 180), dtype=tf.float64))
        

        for i in range(batchsize):

            S11= TFE(m[i], Vtot[i], sigma[i], rmean[i])

            loss = loss.write(i, S11)

        mse = tf.reduce_mean(tf.square(output[:,-5:]-y_true))
        rmse = tf.math.sqrt(mse)
        
        intermediate = tf.subtract(loss.stack(), phasefunctions)
        intermediate = tf.math.square(intermediate)
        intermediate = tf.reduce_mean(intermediate)

        rmsresult = tf.math.sqrt(intermediate)

        return rmsresult + rmse
    
    return PINNLossFunction

PINNLossFunctionP11 = get_PINNLossFunctionP11()




