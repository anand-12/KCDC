import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import solve_triangular
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def gaussian_kernel(x1, x2, sigma = 1):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

def calculate_mu_embedding_XY(X, Y, kernel):
    K_XY = pairwise_kernels(X, Y, metric=kernel)
    K_YY = pairwise_kernels(Y, metric=kernel)

    alpha = solve_triangular(K_YY + np.eye(len(K_YY)) * 1e-8, K_XY @ X, lower=True)

    mean_embedding = K_XY.T @ alpha / len(X)

    return mean_embedding

def calculate_mu_embedding_YX(X, Y, kernel):

    K_YX = pairwise_kernels(Y, X, metric=kernel)
    K_XX = pairwise_kernels(X, metric=kernel)


    alpha = solve_triangular(K_XX + np.eye(len(K_XX)) * 1e-8, K_YX @ Y, lower=True)

    mean_embedding = K_YX.T @ alpha / len(X)
    
    return mean_embedding

def compute_KCDC_X_Y(X, Y, mu_embedding_XY, mu_embedding_YX):
    n = len(X)
    KCDC = 0
    for i in range(n):
        term2 = (1/n) * np.sum([np.linalg.norm(mu_embedding_YX[j]) for j in range(n)])
        KCDC += (np.linalg.norm(mu_embedding_XY[i]) - term2) ** 2
    KCDC *= (1/n)
    return KCDC

def compute_KCDC_Y_X(Y, X, mu_embedding_YX, mu_embedding_XY):
    n = len(Y)
    KCDC = 0
    for i in range(n):
        term2 = (1/n) * np.sum([np.linalg.norm(mu_embedding_XY[j]) for j in range(n)])
        KCDC += (np.linalg.norm(mu_embedding_YX[i]) - term2) ** 2
    KCDC *= (1/n)
    return KCDC


sigma_sq = 1
'''a = 2
num = 100
X = np.random.normal(loc=0, size=(num, 1))
Y = a * X + np.random.normal(loc=0, scale=sigma_sq, size=(num, 1))
'''

X = [[-0.8206075],
 [ 0.70166955],
 [ 0.88711893],
 [-1.90603594],
 [ 1.52199601],
 [-0.40510117],
 [ 0.3889609 ],
 [ 0.85725729],
 [-0.38874788],
 [ 0.33048288],
 [-0.21270329],
 [-0.0692899 ],
 [ 0.99490638],
 [-0.54288283],
 [ 1.66177956],
 [ 0.50895991],
 [ 2.41485614],
 [ 1.46537558],
 [ 1.07519025],
 [ 0.70830127],
 [ 0.90901799],
 [ 1.76859823],
 [ 2.24608315],
 [-1.37374457],
 [-1.49229479],
 [ 0.68275963],
 [ 1.63905683],
 [ 1.50721521],
 [-1.89750459],
 [ 0.528893  ],
 [ 0.80417438],
 [-0.67483945],
 [ 0.81803387],
 [ 0.04910343],
 [-1.3029399 ],
 [ 0.89187328],
 [ 0.12689707],
 [ 1.28719042],
 [ 1.29562004],
 [-0.47174277],
 [-1.84498112],
 [-1.83419866],
 [-0.25405237],
 [ 1.70009403],
 [-0.83810891],
 [-0.70949585],
 [ 0.84542118],
 [ 1.23978109],
 [-1.57339242],
 [ 1.42513138]]

Y = [[-1.07687244],
 [ 1.47383656],
 [ 1.20123357],
 [-3.84488478],
 [ 3.10633543],
 [-0.09576104],
 [ 1.92360625],
 [ 1.22078987],
 [-0.32757   ],
 [-0.3077323 ],
 [ 1.39849196],
 [-1.18395997],
 [ 2.68312402],
 [-0.15622825],
 [ 1.33653894],
 [ 1.527194  ],
 [ 5.28899572],
 [ 1.79996452],
 [ 3.29379787],
 [ 1.79417338],
 [ 2.38276346],
 [ 5.1046311 ],
 [ 5.14863053],
 [-1.47462563],
 [-2.78188897],
 [ 0.65173236],
 [ 2.42554252],
 [ 3.12608319],
 [-5.35530033],
 [ 1.2282637 ],
 [ 2.5344774 ],
 [-1.43926022],
 [ 2.90957404],
 [-0.33890532],
 [-1.6700398 ],
 [ 2.48465331],
 [ 0.70463625],
 [ 2.33286777],
 [ 1.40091317],
 [ 0.60224009],
 [-4.43675733],
 [-3.0079138 ],
 [-0.92925633],
 [ 3.23419311],
 [-2.46140083],
 [-0.87963335],
 [ 1.40330188],
 [ 1.718697  ],
 [-1.71664816],
 [ 3.06550046]]

mu_embedding_XY = calculate_mu_embedding_XY(X, Y, gaussian_kernel)
mu_embedding_YX = calculate_mu_embedding_YX(X, Y, gaussian_kernel)

KCDC_score_X_Y = compute_KCDC_X_Y(X, Y, mu_embedding_XY, mu_embedding_YX)
KCDC_score_Y_X = compute_KCDC_X_Y(Y, X, mu_embedding_YX, mu_embedding_XY)
print("Variance:", sigma_sq)
print("KCDC(X->Y) score D1:", KCDC_score_X_Y)
print("KCDC(Y->X) score D1:", KCDC_score_Y_X)
print("Delta:", KCDC_score_Y_X - KCDC_score_X_Y)