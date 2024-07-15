import numpy as np

def f(x1, x2):
    """
    Función objetivo de prueba.

    Parámetros:
    x1 : float
        Primer parámetro de entrada.
    x2 : float
        Segundo parámetro de entrada.

    Retorna:
    float
        Valor de la función evaluada en (x1, x2).
    """
    return ((x1**2 + x2 - 11)**2) + ((x1 + x2**2 - 7)**2)

def primera_derivada(f, x, h):
    """
    Calcula el gradiente de una función escalar f en un punto x utilizando
    la diferencia central para la primera derivada.

    Parámetros:
    f : función
        Función escalar de la cual se calculará el gradiente.
    x : array_like
        Punto en el cual se evaluará el gradiente.
    h : float
        Tamaño del paso para la diferencia central.

    Retorna:
    array_like
        Gradiente de f evaluado en x.
    """
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(*x_forward) - f(*x_backward)) / (2 * h)
    return grad

def segunda_derivada(f, x, h):
    """
    Calcula la matriz Hessiana de una función escalar f en un punto x utilizando
    la diferencia central para la segunda derivada.

    Parámetros:
    f : función
        Función escalar de la cual se calculará la matriz Hessiana.
    x : array_like
        Punto en el cual se evaluará la matriz Hessiana.
    h : float
        Tamaño del paso para la diferencia central.

    Retorna:
    array_like
        Matriz Hessiana de f evaluada en x.
    """
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

# Ejemplo de uso
x_t = np.array([1.0, 1.0])
DeltaX = 0.01

# Cálculo del gradiente y la matriz Hessiana
gradiente = primera_derivada(f, x_t, DeltaX)
hessiana = segunda_derivada(f, x_t, DeltaX)

np.set_printoptions(precision=16)

print("Gradiente en x_t:", gradiente)
print("Hessiana en x_t:")
print(hessiana)