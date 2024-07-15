import matplotlib.pyplot as plt
import numpy as np

alpha = 0.5
epsilon = 0.001

xt = np.array([2, 1])
xs = np.array([2, 5])

def derivada(f, x, deltaa_x):
    """
    Calcula la primera derivada de la función f en el punto x utilizando el método de diferencias centrales.

    Args:
    - f: Función a derivar.
    - x: Punto en el cual evaluar la derivada.
    - deltaa_x: Tamaño del paso para calcular la derivada.

    Returns:
    - El valor de la primera derivada de f en x.
    """
    return (f(x + deltaa_x) - f(x - deltaa_x)) / (2 * deltaa_x)

def segunda_derivada(f, x, deltaa_x):
    """
    Calcula la segunda derivada de la función f en el punto x utilizando el método de diferencias centrales.

    Args:
    - f: Función a derivar.
    - x: Punto en el cual evaluar la segunda derivada.
    - deltaa_x: Tamaño del paso para calcular la segunda derivada.

    Returns:
    - El valor de la segunda derivada de f en x.
    """
    return (f(x + deltaa_x) - 2 * f(x) + f(x - deltaa_x)) / (deltaa_x ** 2)

def delta_x(x):
    """
    Determina el tamaño del paso delta_x basado en el valor de x.

    Args:
    - x: Valor del cual calcular el tamaño del paso.

    Returns:
    - El tamaño del paso delta_x.
    """
    if abs(x) > 0.01:
        return 0.01 * abs(x)
    else:
        return 0.0001

def funcioon(x1, x2):
    """
    Define una función de prueba f(x1, x2).

    Args:
    - x1: Primer argumento de la función.
    - x2: Segundo argumento de la función.

    Returns:
    - El valor de la función evaluada en (x1, x2).
    """
    return ((x1 - 10) ** 2) + ((x2 - 10) ** 2)

def newton_method(x0, epsilon, f):
    """
    Implementa el método de Newton para encontrar el mínimo de una función f(x).

    Args:
    - x0: Punto inicial para iniciar la búsqueda.
    - epsilon: Precisión deseada para la convergencia del método.
    - f: Función a minimizar.

    Returns:
    - El valor x donde se encuentra el mínimo local aproximado de la función f.
    """
    x = x0
    while abs(derivada(f, x, delta_x(x))) > epsilon:
        segunda_deriv = segunda_derivada(f, x, delta_x(x))
        if segunda_deriv == 0:
            return x
        x = x - derivada(f, x, delta_x(x)) / segunda_deriv
    return x

def evaluar(alpha):
    """
    Evalúa una función específica utilizando un parámetro alpha.

    Args:
    - alpha: Parámetro de ajuste para la evaluación.

    Returns:
    - El valor de la función evaluada en función de alpha.
    """
    xt = np.array([2, 1])
    xs = np.array([2, 5])
    x_alpha = xt + (alpha * xs)
    x_nuevo = (x_alpha[0] - 10) ** 2 + (x_alpha[1] - 10) ** 2

    return x_nuevo

def busqueda_unidireccional(xt, xs, metodo_optimizacion, funcion_evaluacion):
    """
    Realiza una búsqueda unidireccional para optimizar una función.

    Args:
    - xt: Punto inicial.
    - xs: Punto de dirección.
    - metodo_optimizacion: Método de optimización a utilizar.
    - funcion_evaluacion: Función a evaluar.

    Returns:
    - El resultado optimizado de la función.
    """
    alpha = 0.5
    falpha = metodo_optimizacion(alpha, epsilon, funcion_evaluacion)
    x_alpha = xt + (falpha * xs)
    resultado = funcioon(x_alpha[0], x_alpha[1])

    return resultado

print(busqueda_unidireccional(xt, xs, newton_method, evaluar))
