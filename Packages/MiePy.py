# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jv, yv
from scipy.stats import qmc
import warnings
from scipy.integrate import quad_vec
import functools
import pandas as pd

# This function is used to calculate the Mie coefficients a_n and b_n.
# For reference, see the book Bohren & Huffman. The basic solution is given
# in equation (4.53). The version implemented here is from equations
# (4.56) and (4.57), which assume the permeability of particle and
# surrounding medium to be the same!! This is an ASSUMPTION.

# To compute the coefficients we use the equations given in section
# 4.8 of the book.

# +
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jv, yv
from scipy.stats import qmc
import warnings
from scipy.integrate import quad_vec
import functools

#This function is used to calculate the Mie coefficients a_n and b_n.
#For reference, see the book Bohren & Huffman. The basic solution is given
#in equation (4.53). The version implemented here is from equations
#(4.56) and (4.57), which assume the permeability of particle and
#surrounding medium to be the same!! This is an ASSUMPTION.

#To compute the coefficients we use the equations given in section
#4.8 of the book.

#Inputs: m is the (complex) refractive index of the particle, x is the
#size parameter calculated as 2 * pi * n * r / lambda, and n is the
#refractive index of the medium (assumed to be 1 by default here), r
#is the particle size (radius) and lambda the wavelength of the incoming light
def MieCoefficients_ab(m,x,c=1,mediumindex=1):

  #for x correctly
  x = x* mediumindex.real
  m = m/ mediumindex.real
  mx = m*x

  #A good rule of thumb is that we need about x (size parameter) terms for
  #the sum to give good reslts (start of section 4.8 in Bohren and Huffman)
  #so we set the max sum index to the expression below.
  #c is 1 per default but can be changed in the function input increase
  #the upper bound of the summation.
  nmax = np.amax(np.round(c*(2+x+4*(x**(1/3)))))
  mxmax = np.amax(np.round(np.abs(mx)))

  #this will be used as the max index for the calculation of the
  #D_n below, given in eq. (4.89). The +16 terms are used for initialization
  #and will be dropped later. (see calculation of Dn below)

  nmax_Dn = np.round(max(nmax,mxmax)+16)

  #this will give us the index for the bessel/hankel/etc. functions
  n = np.arange(1,nmax+1) #
  nu = n + 0.5 #
  nu = nu[np.newaxis]
  n = n[np.newaxis]
  #this is the prefactor used to convert the normal bessel functions into
  #the so called Riccati Bessel functions. These are defined in Bohren and
  #Huffman right above equation (4.56)
  prefactor = np.sqrt(0.5*np.pi*x)

  #Here we define the Ricatti functions. The second line is needed because
  #we start at n=1 but we need to add the zeroth function, which is sin(x) for
  #the Ricatti function dubbed Sn and cos(x) for Cn. This effectively shifts
  #the index by 1, which is exactly what we need for the formula (4.88) in
  #the book
  RicattiSn = prefactor*jv(nu.T,x) #
  RicattiSn0 = np.insert(RicattiSn, 0, np.sin(x), 0)
  RicattiSn0 = np.delete(RicattiSn0, -1, 0) #

  RicattiCn = -prefactor*yv(nu.T,x) #
  RicattiCn0 = np.insert(RicattiCn, 0, np.cos(x), 0)
  #RicattiCn0 = np.append(np.cos(x), RicattiCn[0:int(nmax)-1]) #
  RicattiCn0 = np.delete(RicattiCn0, -1, 0)

  RicattiXin = RicattiSn-(0+1j)*RicattiCn #
  RicattiXin0 = RicattiSn0-(0+1j)*RicattiCn0 #

  #Here we add an axis to be able to do the .T command later, where we want to
  #subtract for example the first element from all the columns in the first row.
  #this is done by subtracting a row-vector from the matrix

  #here we define Dn, which is a quantity also needed to calculate the
  #coefficients in (4.88), and the definition of Dn is given in (4.89).
  #each term in the recurrence relation reduces the error signigicantly,
  #which is why here I initiate with zero and calculate more terms then
  #necessary and in the end throw away the first few terms.
  D_n = np.zeros((int(nmax_Dn),len(m)),dtype=complex)
  for i in range(int(nmax_Dn)-1,1,-1):
    D_n[i-1,:] = (i/mx)-(1/(D_n[i,:]+i/mx))


  #Here we drop the terms beyond nmax that we needed to initialize and we
  #start at index 1 because there is no a0 but we start with n=1.
  D_n = D_n[1:int(nmax)+1,:]

  #Here I just define the two factors in brackets from eq. (4.88)
  #Python is amazing here because the array-matrix multiplication does
  #exactly what we want here. multiplying an array with a matrix,
  #the rows are multiplied elementwise with the array and this happens for
  #every row, which is exactly what we want.
  FactorFora_n = D_n/m + n.T/x
  FactorForb_n = m*D_n + n.T/x

  #Now, finally, calculating the coefficients according to eq. (4.88) :D
  a_n = (FactorFora_n*RicattiSn - RicattiSn0) / (FactorFora_n*RicattiXin-RicattiXin0)
  b_n = (FactorForb_n*RicattiSn - RicattiSn0) / (FactorForb_n*RicattiXin-RicattiXin0)

  return a_n, b_n
    
#Here the angle-dependent functions pi_n and tau_n are calculated
#via a recursion relation as shown in equation (4.47)
def PiTau(angle,nmax):

  mu = np.cos(angle/180*np.pi)

  #This creates a 2D array with zeroes where the columns are the angles
  #(which is why there are 180 columns) and the rows are the pi_n and tau_n
  #for every angle
  pi = np.zeros((int(nmax),len(angle)))
  tau = np.zeros((int(nmax),len(angle)))

  #Here I calculate the first two elements according to relation (4.47) in
  #the book. Caveat!: the zeroth index is actually the element pi_1, tau_1.
  #this is because in the sum we start with n=1 so we don't consider n=0,
  #but it is nevertheless used to calculate n=1 via the relation (4.47)
  pi[0,:] = 1
  pi[1,:] = 3*mu

  tau[0,:] = mu
  #tau[1,:] = 3*(2*mu**2-1)
  tau[1,:] = 3.0*np.cos(2*np.arccos(mu))

  #After initializing the first few elements, now the rest is calculated
  #with the aforementioned recurrence relation
  #Caveat!: the element at index n=0 is actually the term for n=1. so
  #when calculating pi_n, we need to replace n in (4.47) by n+1, because
  #our index lags one behind (index 0 is actually for n=1, etc.)
  for n in range(2,int(nmax)):
    pi[n,:] = (2*n+1)/n * (mu*pi[n-1,:]) - (n+1)/n * pi[n-2,:]
    tau[n,:] = (n+1)*mu * pi[n,:] - (n+2) * pi[n-1,:]
  return pi, tau


#Here the Matrix Elements of the Scattering (2x2) matrix are calculated.
#The formulas for these matrix elements are given in equation (4.74)
#from these formulas we can then calculate the phase functions in the
#mueller matrix
def ScatteringMatrixElements(m,x,angle,c=1,n=1,limit=0.5,auto=True):

  xmax = np.amax(x)
  nmax = np.round(c*(2+xmax+4*np.power(xmax,1/3)))

  #determine the coefficients used to calculate the matrix elements S1 and S2
  #see function description for MieCoefficients_ab above
  a_n, b_n = MieCoefficients_ab(m,x,c,n)
  a_n = a_n[:,np.newaxis,:]
  b_n = b_n[:,np.newaxis,:]

  #Calculate the functions pi and tau needed for the calculation.
  #see function description for PieTau above
  pi_n, tau_n = PiTau(angle,nmax)

  #This gives the matrix a new (3rd) axis. This is needed because we multiply
  #with the a_n and b_n later. why? --> we have two matrices: the pi/tau_n
  #which are all the n's for every angle, and the an/bn, which are all the
  #n's for every refractive index. we now need to multiply the first entry
  #of the pi_n matrix (=pi_1 for angle theta=1) with the entire first row
  #of the index vs. an matrix in order to get pi_1*a_1 at angle theta = 1
  #for every refractive index in the matrix. So every angle gives an entirely
  #new matrix and in the end we have a rank 3 tensor.
  #when we then sum over one axis of this rank 3 tensor, we are left with
  #a rank 2 tensor (=matrix) whose elements then are the ScatteringMatrixElements
  #for each refractive index and each angle. The rows will be the angles
  #and the columns will be the different refractive indices
  pi_n = pi_n[:,:,np.newaxis]
  tau_n = tau_n[:,:,np.newaxis]

  #Here I create an array for the summation in formula (4.74)
  #remember that for all our quantities, the zero-index value is actually
  #the n=1 value. So here an array is created with numeric value 1 in its
  #first entry (= zeroth index) in order to sum properly according to formula (4.74)
  n = np.arange(1,int(nmax)+1)

  #this is the prefactor in the sum (4.74) and needs to be transposed with
  #the nexaxis command in order to multiply the correct entries in the rank 3
  #tensor below
  Prefactor = (2*n+1)/(n*(n+1))
  Prefactor = Prefactor[np.newaxis,np.newaxis]
  Prefactor = Prefactor.T

  #a_n is a matrix with the columns being different refractive indices and the
  #rows being the a_n for these indices, so I need the number or rows to know
  #how many a_n there are, to know until what n to sum.
  UpperSumIndex_a = np.shape(a_n)[0]
  UpperSumIndex_b = np.shape(b_n)[0]

  #finally, actual calculation of the sum (4.74) to get the matrix elements S1 and S2 ;)
  S_1 = np.sum(Prefactor[0:UpperSumIndex_a] * ( pi_n[0:UpperSumIndex_a,:] * a_n + tau_n[0:UpperSumIndex_b,:] * b_n),0)
  S_2 = np.sum(Prefactor[0:UpperSumIndex_a] * (tau_n[0:UpperSumIndex_a,:] * a_n + pi_n[0:UpperSumIndex_b,:] * b_n),0)

  #The returned matrices will have the S-values as entries. Since S depends on the angle and the
  #refractive index, we have "two dimensions". The columns will correspond to different refractive
  #indices and the rows will correspond to different angles. so the entry M_57 in the matrix
  #will be the S-element for the angle of 5 degrees of the 7th refractive index in the data set
  return S_1, S_2


#Here we use equation(s) (4.77) in the book to get the two elements we want
#we want S11 and -S12/S11, which is the polarized phase function
#the radius and the wavelength have to be input with the same units. so if one
#used nanometers for the wavelength, the particle radius needs to be nanometers as well.
def PhaseFunctions(m,wavelength,radius,angle,mediumindex=1.0,limit=0.5,auto=True,c=1):

  #n = n.real
  wavelength = wavelength/mediumindex.real
  x = 2*np.pi*radius/wavelength

  S_1, S_2 = ScatteringMatrixElements(m,x,angle,c,mediumindex,limit,auto)

  a_n, b_n = MieCoefficients_ab(m,x,c,mediumindex)

  S_11 = 0.5*(np.abs(S_2)**2+np.abs(S_1)**2)
  S_12 = 0.5*(np.abs(S_2)**2-np.abs(S_1)**2)

  #PPF: polarized phase function
  PPF = -S_12/S_11

  return S_11, PPF

def SinglePhaseFunctionsFromParameterSet(ParameterSet, wavelength, mediumindex=1.0, limit=0.5, auto=True, c=1):

    RefractiveIndex = ParameterSet['RealRefractiveIndex'].to_numpy() + (0+1j)*ParameterSet['ImaginaryRefractiveIndex'].to_numpy()
    RelativeIndex = RefractiveIndex/mediumindex

    Radius = ParameterSet['MeanRadius'].to_numpy()

    Angles = np.arange(0,180)

    S_11, PPF = PhaseFunctions(RelativeIndex,wavelength,Radius,Angles,mediumindex=mediumindex,limit=limit,auto=auto,c=c)

    return S_11, PPF


def sizedistribution(r, Vtot, sigma, rmean):
    
    sigma = np.log(sigma)
    mu = np.log(rmean) - 3*sigma**2
    N0 = 3 / 4 / np.pi * Vtot * np.exp(-(3*mu + 4.5*sigma**2))
    
    return N0 /(np.sqrt(2*np.pi)) * 1/sigma * 1/r * np.exp(-(np.log(r)-mu)**2/(2*sigma**2))

    
def integrandS11(radius, m, wavelength, mediumindex, angle, Vtot, sigma, rmean, c, limit, auto):

            wavelength = wavelength/mediumindex.real
            x = 2*np.pi*radius/wavelength

            S_1, S_2 = ScatteringMatrixElements(m,x,angle,c,mediumindex,limit,auto)
            
            a_n, b_n = MieCoefficients_ab(m,x,c,mediumindex)

            S_11 = 0.5*(np.abs(S_2)**2+np.abs(S_1)**2)

            return (S_11*sizedistribution(radius, Vtot, sigma, rmean))
        
def integrandPPF(radius, m, wavelength, mediumindex, angle, Vtot, sigma, rmean, c, limit, auto):

            wavelength = wavelength/mediumindex.real
            x = 2*np.pi*radius/wavelength

            S_1, S_2 = ScatteringMatrixElements(m,x,angle,c,mediumindex,limit,auto)
            
            a_n, b_n = MieCoefficients_ab(m,x,c,mediumindex)

            S_12 = 0.5*(np.abs(S_2)**2-np.abs(S_1)**2)

            return (S_12*sizedistribution(radius, Vtot, sigma, rmean))
    
def EnsemblePhaseFunctions(m, wavelength, Vtot, sigma, rmean, radiusbounds = np.array([50,15000]),angle=np.arange(0,180,1), mediumindex=1.0, limit=0.5, auto=True, c=1):

    PartialIntegrandS11 = functools.partial(integrandS11, m=m, wavelength=wavelength, mediumindex=mediumindex, angle=angle, Vtot=Vtot, sigma=sigma, rmean=rmean, c=c, limit=limit, auto=auto)

    IntegratedMatrixS11 = quad_vec(PartialIntegrandS11,radiusbounds[0],radiusbounds[1],workers=-1)[0]
    
    PartialIntegrandPPF = functools.partial(integrandPPF, m=m, wavelength=wavelength, mediumindex=mediumindex, angle=angle, Vtot=Vtot, sigma=sigma, rmean=rmean, c=c, limit=limit, auto=auto)

    IntegratedMatrixPPF = quad_vec(PartialIntegrandPPF,radiusbounds[0],radiusbounds[1],workers=-1)[0]

    return IntegratedMatrixS11, -IntegratedMatrixPPF/(IntegratedMatrixS11+1e-100)

def IntegratedEnsembleMatrixFromParameterSet(ParameterSet, wavelength, radiusbounds = np.array([50,15000]), angles = np.arange(0,180), mediumindex=1.0, limit=0.5, auto=True, c=1):

        RefractiveIndex = ParameterSet['RealRefractiveIndex'].to_numpy() + (0+1j)*ParameterSet['ImaginaryRefractiveIndex'].to_numpy()
        RelativeIndex = RefractiveIndex/mediumindex
        Vtot = ParameterSet['VolumeConcentration'].to_numpy()
        sigma = ParameterSet['Sigma'].to_numpy()
        rmean = ParameterSet['MeanRadius'].to_numpy()

        IntegratedMatrixS11, IntegratedMatrixPPF = EnsemblePhaseFunctions(RelativeIndex, wavelength, Vtot, sigma, rmean, radiusbounds=radiusbounds, angle=angles, mediumindex=mediumindex, limit=limit, auto=auto, c=c)

        return IntegratedMatrixS11, IntegratedMatrixPPF
    
    

