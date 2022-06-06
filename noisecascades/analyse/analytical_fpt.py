

import numpy as np
from scipy.special import gamma



def meanFPT_imkeller_pavlyukevich(alpha, sigma, barrier_distance = 1):

    if alpha == 1.0:
        meanFPT = barrier_distance*np.pi/sigma**alpha
    else:
        meanFPT = (barrier_distance**alpha)*2*gamma(1-alpha)*np.cos(np.pi*alpha/2)/sigma**alpha

    return meanFPT

def meanFPT_kramer(sigma, barrier_height = 0.25):

    return np.pi*np.exp(barrier_height/(sigma**2))/np.sqrt(2)

def estimate_gauss_sigma(alpha, levy_sigma, barrier_height = 0.25, barrier_distance = 1):

    if alpha == 1.0:
        gauss_sigma = np.sqrt(1/((1/barrier_height)*np.log(barrier_distance*np.sqrt(2)/levy_sigma**alpha)))
    else:
        gauss_sigma = np.sqrt(1/((1/barrier_height)*np.log((barrier_distance**alpha)*2*np.sqrt(2)*gamma(1-alpha)*np.cos(np.pi*alpha/2)/(np.pi*levy_sigma**alpha))))

    return gauss_sigma

def estimate_levy_sigma(alpha, gauss_sigma, barrier_height = 0.25, barrier_distance = 1):

    if alpha == 1.0:
        levy_sigma = (barrier_distance*np.sqrt(2) / np.exp(1 / ((1/barrier_height) * gauss_sigma**2)))**(1/alpha)
    else:
        levy_sigma = ((barrier_distance**alpha)*2*np.sqrt(2)*gamma(1-alpha)*np.cos(np.pi*alpha/2) / (np.pi*np.exp(1 / ((1/barrier_height) * gauss_sigma**2))))**(1/alpha)
    return levy_sigma



