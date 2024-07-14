import numpy as np

# Función objetivo
def f(x1, x2):
    return ((x1**2 + x2 - 11)**2) + ((x1 + x2**2 - 7)**2)

# Diferencia central para la primera derivada
def primera_derivada(f, x, h):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(*x_forward) - f(*x_backward)) / (2 * h)
    return grad

# Diferencia central para la segunda derivada (matriz Hessiana)
def segunda_derivada(f, x, h):
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                x_forward = np.copy(x)
                x_backward = np.copy(x)
                x_forward[i] += h
                x_backward[i] -= h
                hessian[i, j] = (f(*x_forward) - 2 * f(*x) + f(*x_backward)) / (h**2)
            else:
                x_ij1 = np.copy(x)
                x_ij2 = np.copy(x)
                x_ij3 = np.copy(x)
                x_ij4 = np.copy(x)
                x_ij1[i] += h
                x_ij1[j] += h
                x_ij2[i] += h
                x_ij2[j] -= h
                x_ij3[i] -= h
                x_ij3[j] += h
                x_ij4[i] -= h
                x_ij4[j] -= h
                hessian[i, j] = (f(*x_ij1) - f(*x_ij2) - f(*x_ij3) + f(*x_ij4)) / (4 * h**2)
    return hessian

x_t = np.array([1.0, 1.0])
DeltaX = 0.01

# Cálculo del gradiente y la matriz Hessiana
gradiente = primera_derivada(f, x_t, DeltaX)
hessiana = segunda_derivada(f, x_t, DeltaX)

np.set_printoptions(precision=16)

print("Gradiente en x_t:", gradiente)
print("Hessiana en x_t:")
print(hessiana)
