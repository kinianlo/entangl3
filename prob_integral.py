# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:43:52 2018

@author: Kin Ian Lo
"""
from scipy.special import binom
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from utilities import get_VN_entropy as S_VN
from numpy.linalg import norm


def anti_traces(A):
    """
    Compute the sum of entry(s) along every anti-diagonal line of the matrix:
    
    The result is an array of 2N+1 elements. 
    The i-th element of the result is the sum of the entries labelled i which
    is examplifed below: 
    [[0, 1, 2, 3],
     [1, 2, 3, 4],
     [2, 3, 4, 5],
     [3, 4, 5, 6]]
    """
    T = np.zeros(sum(A.shape)-1)
    Aflip = np.flip(A, 0)
    for i in range(len(T)):
        T[i] = np.trace(Aflip, i-A.shape[0]+1)
    return T

def n_row_pascal(n, alternating=False):
    """
    Gives the n-th row in the pascal triganle.
    If alternating is True, the result is elementwisely multiplied by
    [1, -1, 1, -1, 1, ...]
    """
    a = binom(n, np.arange(n+1))
    if alternating:
        a = a * alternate(n+1)
    return a


def alternate(n):
    """
    Gives [1, -1, 1, -1, 1, ...] with n elements. 
    """
    a = np.ones(n, dtype=np.int)
    a[1::2] = -1
    return a


def integrate_2pi(A):
    """
    
    """
    n = max(A.shape)
    T = np.zeros((n, n))
    T[0, 0] = 2*np.pi
    for i in range(int((n-1)/2)):
        p = 2*i
        q = 0
        T[p+2, 0] = T[p, 0] * (p+1)/(p+q+2)
    for i in range(int((n-1)/2)):
        p = np.arange(n)
        q = 2*i
        T[:, q+2] = T[:, q] * (q+1)/(p+q+2)

    return A * T[:A.shape[0], :A.shape[1]]


def integrate_pi(A):
    n = max(A.shape)
    T = np.zeros((n, n))
    T[0, 0] = np.pi
    T[1, 0] = 2  # this may be irrelavent to the result
    for i in range(int((n-1)/2)):
        p = 2*i
        q = 0
        T[p+2, 0] = T[p, 0] * (p+1)/(p+q+2)
    for i in range(int(n/2)-1):
        p = 2*i+1
        q = 0
        T[p+2, 0] = T[p, 0] * (p+1)/(p+q+2)
    for i in range(int((n-1)/2)):
        p = np.arange(n)
        q = 2*i
        T[:, q+2] = T[:, q] * (q+1)/(p+q+2)

    return A * T[:A.shape[0], :A.shape[1]]


def promote(a):
    """
    Shift an array to the right by entry. 
    e.g. 
    [1,2,3] becomes [0, 1, 2, 3]
    """
    return np.insert(a, 0, 0)


def bloch_length_polynomial(N, N_pos):
    """
    Simulate a set of experimental outcomes. 
    First nb_sample of Bloch vectors are sampled according to the given
    distribution. For each sampled Bloch vector, nb_measurement spin-1/2 
    measurements are done for each of the x, y and directions. 
    
    Args:
        nb_sample: number of samples of Bloch vector
        nb_measurement: number of measurements made on each x, y and z axis 
        distribution: the distribtuion of Bloch vectors sampled
        usage: can be 'train' or 'test' (for bookkepping only)
        
    Returns:
        A panda dataframe with the following columns:
            nb_measurement: (see above)
            distribution: (see above)
            usage: (see above)
            bloch_vector_1: x-component of the Bloch vector 
            bloch_vector_2: y-component of the Bloch vector 
            bloch_vector_3: z-component of the Bloch vector 
            nb_positive_outcome_1: no. of 'up-state' sigma_x measurement outcomes
            nb_positive_outcome_2: no. of 'up-state' sigma_y measurement outcomes
            nb_positive_outcome_3: no. of 'up-state' sigma_z measurement outcomes      
        There are nb_sample rows and each row represent an experimental trial
    """
    N1, N2, N3 = N_pos

    X = np.convolve(n_row_pascal(N1), n_row_pascal(N-N1, True))
    Y = np.convolve(n_row_pascal(N2), n_row_pascal(N-N2, True))
    Z = np.convolve(n_row_pascal(N3), n_row_pascal(N-N3, True))

    A = np.outer(Y, X)
    A = integrate_2pi(A)  # integrate phi
    B = promote(anti_traces(A))

    C = np.outer(B, Z)
    C = integrate_pi(C)  # integrate theta
    D = promote(anti_traces(C))

    return D

def get_unique_N_pos(N):
    """
    There are two symmetries in the N_pos obeyed by the probablities. 
    1) orders in N_pos does not matter. e.g. Pr(i, j, k) = Pr(i, k, j) = Pr(k, i, j)
    2) Pr(i, j, k) = Pr(N-i, j, k)
    Here Pr refers to any probablity functions. 
    
    Args:
        N: number of measurements made on each x, y and z axis 
        
    Returns:
        N_pos_uniq: an array with 3 colums and multiple rows. Each row represents a degenerate set of 
            N_pos whose degeneracy is given in degen
        degen: the degeneracy of each row in N_pos_uniq
        
    """
    N_pos = np.array(np.meshgrid(*([range(N+1)]*3))).T.reshape(-1, 3)
    N_pos = (N_pos > N//2)*(N-N_pos) + (N_pos <= N//2)*N_pos
    N_pos = np.sort(N_pos, axis=1)
    N_pos_uniq, degen = np.unique(N_pos, axis=0, return_counts=True)
    
    return N_pos_uniq, degen


class VNE_CoM_Predictor:
    """
    This class compute the statistical properties of the posterior distribution
    of von Neumann entropy. Many efforts are done to prevent recomputing the same quantities.
    """
    def __init__(self, N, f=lambda v: 3/(4*np.pi)):
        """
        Args:
            N: number of measurements made on each x, y and z axis 
            f: the distribution of Bloch length (default is the volumn-uniform distribution )
        """
        self.N = N
        self.f = f
        
        # Create storage to prevent recomputing the same quantuies 
        self.CoM = -np.ones((N+1, N+1, N+1))
        self.Variance = -np.ones((N+1, N+1, N+1))
        self.P_N_pos = -np.ones((N+1, N+1, N+1))
        self.median = -np.ones((N+1, N+1, N+1))
        
        self.counter = 0 # recording keeping only, to keep track of no. of expensive calculations

        # pre-calculations, all numerical integrations are done here 
        # All statistical properties can be computed through the moments 
        # e.g. the 2nd moment is the integral of S_VN(v)**2 * f(v) * some_polynomial(N, N_pos)
        # However, 'some_polynomial' depends on the experiment outcome. But it is possible to 
        # numerically calculate the intgral of S_VN(v)**2 * f(v) * v**i for all interested integers i
        # and then store the result in self.M2[i]. This is what "pre-calculation" means. 
        
        self.M0 = np.zeros(3*N+3)  # 0th Moment of S_VN with rho=v^i
        self.M1 = np.zeros(3*N+3)  # 1st Moment of S_VN with rho=v^i
        self.M2 = np.zeros(3*N+3)  # 2nd Moment of S_VN with rho=v^i
        for i in range(3*N+3):
            self.M0[i] = integrate.quad(lambda v: self.f(v) * v**i, 0, 1)[0]
            self.M1[i] = integrate.quad(lambda v: self.f(v) * v**i * S_VN(v), 0, 1)[0]
            self.M2[i] = integrate.quad(lambda v: self.f(v) * v**i * S_VN(v)**2, 0, 1)[0]

    def get_CoM(self, N_pos):
        """
        Get Centre of Mass (aka mean) of the posterior distribution of von Neumann entropy 
        Args:
            N_pos: the experimental outcome in the form of a 3-tuple of integers. 
        """
        N1, N2, N3 = N_pos
        
        # Check if the Com of this N_pos has been calculated before, 
        # -0.5 is just an arbitrary choice of number smaller than 0
        if self.CoM[N1, N2, N3] > -0.5:
            return self.CoM[N1, N2, N3]

        D = bloch_length_polynomial(self.N, N_pos)
        eta = np.dot(self.M0, D)
        CoM_S = np.dot(self.M1, D)/eta
        self.CoM[N1, N2, N3] = CoM_S
        return CoM_S

    def get_Variance(self, N_pos):
        """
        Get Variance of the posterior distribution of von Neumann entropy 
        Args:
            N_pos: the experimental outcome in the form of a 3-tuple of integers. 
        """
        N1, N2, N3 = N_pos
        
        # Check if the Com of this N_pos has been calculated before, 
        # -0.5 is just an arbitrary choice of number smaller than 0
        if self.Variance[N1, N2, N3] > -0.5:
            return self.Variance[N1, N2, N3], self.P_N_pos[N1, N2, N3]

        D = bloch_length_polynomial(self.N, (N1, N2, N3))
        eta = np.dot(self.M0, D)
        CoM_S = np.dot(self.M1, D)/eta
        m2 = np.dot(self.M2, D)/eta
        P_N_pos = 1/(2**(3*self.N)) * binom(self.N, N1)*binom(self.N, N2)*binom(self.N, N3)*eta
        Var_S = (m2-CoM_S**2)
        self.Variance[N1, N2, N3] = Var_S
        self.P_N_pos[N1, N2, N3] = P_N_pos

        self.counter += 1
        return Var_S, P_N_pos

    def get_min_RMSE(self):
        """
        This method loops through all possible experimental outcomes (N_pos) and gives the minimmum 
        RMSE. It can be shown that the CoMs produce the min RMSE. 
        
        Returns:
            RMSE: root mean squared error 
            total_prob: sum of prob(specific N_pos), supposed to be roughly one if no significal 
            numerical instability
        """
        MSE = 0
        total_prob = 0

        N_pos_uniq, degen = get_unique_N_pos(self.N)

        for i in range(len(N_pos_uniq)):
            var, prob = self.get_Variance(N_pos_uniq[i, :])
            MSE += prob*var*degen[i]
            total_prob += prob*degen[i]

        RMSE = np.sqrt(MSE)/np.log(2)
        return RMSE, total_prob

    def get_anal_RMSE(self):
        """
        This method loops through all possible experimental outcomes (N_pos) and gives the RMSE
        produced by using the scaled direct inversion method.
        
        Returns:
            RMSE: root mean squared error 
            total_prob: sum of prob(specific N_pos), supposed to be roughly one if no significal 
            numerical instability
        """
        MSE = 0
        total_prob = 0

        N_pos_uniq, degen = get_unique_N_pos(self.N)

        for i in range(len(N_pos_uniq)):
            com = self.get_CoM(N_pos_uniq[i, :])
            com_var, prob = self.get_Variance(N_pos_uniq[i, :])
            meas_bl = np.clip(norm(2*N_pos_uniq[i, :]/self.N-1), 0, 1)
            var = com_var + (com-S_VN(meas_bl))**2
            MSE += prob*var*degen[i]
            total_prob += prob*degen[i]

        RMSE = np.sqrt(MSE)/np.log(2)
        return RMSE, total_prob

    def get_max_RMSE(self):
        """
        This method loops through all possible experimental outcomes (N_pos) and gives the RMSE
        produced by using the scaled direct inversion method.
        
        Returns:
            RMSE: root mean squared error 
            total_prob: sum of prob(specific N_pos), supposed to be roughly one if no significal 
            numerical instability
        """
        MSE = 0
        total_prob = 0

        N_pos_uniq, degen = get_unique_N_pos(self.N)

        for i in range(len(N_pos_uniq)):
            com = self.get_CoM(N_pos_uniq[i, :])
            com_var, prob = self.get_Variance(N_pos_uniq[i, :])
            var = com_var + max((com-0)**2, (com-np.log(2))**2)
            MSE += prob*var*degen[i]
            total_prob += prob*degen[i]

        RMSE = np.sqrt(MSE)/np.log(2)
        return RMSE, total_prob

    def get_ann_RMSE(self, model):
        """
        This method loops through all possible experimental outcomes (N_pos) and gives the RMSE
        produced the given keras model. 
        
        Returns:
            RMSE: root mean squared error 
            total_prob: sum of prob(specific N_pos), supposed to be roughly one if no significal 
            numerical instability
        """
        MSE = 0
        total_prob = 0
        
        N_pos_uniq, degen = get_unique_N_pos(self.N)

        for i in range(len(N_pos_uniq)):
            com = self.get_CoM(N_pos_uniq[i, :])
            com_var, prob = self.get_Variance(N_pos_uniq[i, :])
            x = np.array([2.0 * N_pos_uniq[i, :]/self.N - 1])
            vne_pred = np.squeeze(model.predict(x))*np.log(2)
            var = com_var + (com-vne_pred)**2
            MSE += prob*var*degen[i]
            total_prob += prob*degen[i]

        RMSE = np.sqrt(MSE)/np.log(2)
        return RMSE, total_prob


if __name__ == '__main__':
    # specify the distribution of bloch vector
    # f = lambda v: 1/v**2 # uniform bloch length
    # f = lambda v: 3/4/np.pi # uniform bloch sphere
    def f(v): return 3/(4*np.pi)
    #f = lambda v: np.log((1+v)/(1-v))/(2*np.log(2)*4*np.pi*v**2)
    N = 70
    N1, N2, N3 = (50, 30, 60)
    m_v = np.sqrt((2*N1/N-1)**2+(2*N2/N-1)**2+(2*N3/N-1)**2)
    print('measured bloch length = {:}'.format(m_v))

    #eta = (binom(N, N1)/2**N)*(binom(N, N2)/2**N)*(binom(N, N3)/2**N)
    #eta = poly_integrate(D)

    v = np.linspace(0, 1, 50000)
    S = S_VN(v)
    D = bloch_length_polynomial(N, (N1, N2, N3))
    P = np.poly1d(np.flip(D, 0))
    eta = integrate.quad(lambda v: P(v)*f(v), 0, 1)[0]
    rho_v = P(v)*f(v)/eta
    rho_S = 2/(np.log((1+v)/(1-v))) * rho_v
    CoM_v = integrate.quad(lambda v: v*P(v)*f(v), 0, 1)[0]/eta
    CoM_S = integrate.quad(lambda v: S_VN(v)*P(v)*f(v), 0, 1)[0]/eta
    P_N_pos = 1/(2**(3*N)) * binom(N, N1)*binom(N, N2)*binom(N, N3)*eta

    print(P_N_pos)

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.plot(v, rho_v)
    plt.fill_between(v, 0, rho_v, alpha=0.6, label='CoM = {:.4f}'.format(CoM_v))
    plt.axvline(x=CoM_v)
    plt.xlim([0, 1])
    plt.ylim([None, None])
    plt.xlabel('bloch length')
    plt.legend()
    plt.title('\nmeasured_bloch_length={:.4f}, N={:}\n'.format(
        m_v, N) + r'$\rho(v | ' + 'N_1^+={}, N_2^+={}, N_3^+={}'.format(N1, N2, N3)+r')$', loc='left')
    plt.subplot(1, 2, 2)
    plt.plot(S, rho_S)
    plt.fill_between(S, 0, rho_S, alpha=0.6, label='CoM = {:.4f}'.format(CoM_S))
    plt.axvline(x=CoM_S)
    plt.xlim([0, np.log(2)])
    plt.ylim([None, None])
    plt.xlabel('VN entropy')
    plt.legend()
    plt.title(
        r'$\rho(S_{VN} | ' + 'N_1^+={}, N_2^+={}, N_3^+={}'.format(N1, N2, N3)+r')$', loc='left')

    plt.figure(4)
    plt.plot(np.log(np.abs(P.coef[2::2])), 'o')
