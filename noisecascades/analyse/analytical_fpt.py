

import numpy as np
from scipy.special import gamma



def meanFPT_imkeller_pavlyukevich(alpha, sigma):

    if alpha == 1.0:
        meanFPT = np.pi/sigma**alpha
    else:
        meanFPT = 2*gamma(1-alpha)*np.cos(np.pi*alpha/2)/sigma**alpha

    return meanFPT

def meanFPT_kramer(sigma):

    return np.pi*np.exp(1/(4*(sigma**2)))/np.sqrt(2)

def estimate_gauss_sigma(alpha, levy_sigma):

    if alpha == 1.0:
        gauss_sigma = np.sqrt(1/(4*np.log(np.sqrt(2)/levy_sigma**alpha)))
    else:
        gauss_sigma = np.sqrt(1/(4*np.log(2*np.sqrt(2)*gamma(1-alpha)*np.cos(np.pi*alpha/2)/(np.pi*levy_sigma**alpha))))

    return gauss_sigma

def estimate_levy_sigma(alpha, gauss_sigma):

    if alpha == 1.0:
        levy_sigma = (np.sqrt(2) / np.exp(1 / (4 * gauss_sigma**2)))**(1/alpha)
    else:
        levy_sigma = (2*np.sqrt(2)*gamma(1-alpha)*np.cos(np.pi*alpha/2) / (np.pi*np.exp(1 / (4 * gauss_sigma**2))))**(1/alpha)
    return levy_sigma