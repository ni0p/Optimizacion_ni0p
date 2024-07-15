import numpy as np
import math

def gradiente(f, x, deltaX=0.00001):
    """
    Calcula el gradiente de una función escalar f en un punto x dado.

    Parámetros:
    - f: función escalar a la que se le calculará el gradiente.
    - x: punto en el que se evalúa el gradiente.
    - deltaX: tamaño del paso para calcular las diferencias finitas.

    Retorna:
    - numpy.array: vector gradiente de la función f en el punto x.
    """
    grad = []
    for i in range(0, len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i] + deltaX
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return np.array(grad)

def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    """
    Regla de eliminación para comparar dos valores y actualizar un intervalo.

    Parámetros:
    - x1, x2: valores a comparar.
    - fx1, fx2: valores de la función evaluados en x1 y x2, respectivamente.
    - a, b: límites del intervalo actual.

    Retorna:
    - tuple[float, float]: nuevos límites del intervalo actualizado.
    """
    if fx1 > fx2:
        return x1, b

    if fx1 < fx2:
        return a, x2
    
    return x1, x2

def w_to_x(w: float, a, b) -> float:
    """
    Convierte un valor w en el intervalo [0, 1] a un valor en el intervalo [a, b].

    Parámetros:
    - w: valor a convertir.
    - a, b: límites del nuevo intervalo.

    Retorna:
    - float: valor convertido en el nuevo intervalo.
    """
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon: float, a: float = None, b: float = None) -> float:
    """
    Realiza una búsqueda con método de la búsqueda dorada para encontrar el mínimo de una función en un intervalo dado.

    Parámetros:
    - funcion: función escalar a minimizar.
    - epsilon: precisión deseada para la solución.
    - a, b: límites del intervalo inicial (opcional, por defecto [0, 1]).

    Retorna:
    - float: valor aproximado del mínimo de la función en el intervalo [a, b].
    """
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1

    while Lw > epsilon:
        w2 = aw + PHI * Lw
        w1 = bw - PHI * Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)),
                                   funcion(w_to_x(w2, a, b)), aw, bw)
        k += 1
        Lw = bw - aw

    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

def gradiente_conjugado(funcion, x0, epsilon1, epsilon2, M):
    """
    Implementa el método de gradiente conjugado para minimizar una función escalar.

    Parámetros:
    - funcion: función escalar a minimizar.
    - x0: punto inicial de búsqueda.
    - epsilon1: precisión para la norma del gradiente.
    - epsilon2: precisión para la diferencia entre iteraciones.
    - M: máximo número de iteraciones permitidas.

    Retorna:
    - numpy.array: punto aproximado donde se minimiza la función.
    """
    terminar = False
    xk = x0
    k = 0
    s = -1 * (gradiente(funcion, xk))

    while not terminar:

        if np.linalg.norm(s) < epsilon1 or k >= M:
            terminar = True
        else:

            def alpha_funcion(alpha):
                return funcion(xk + alpha * s)

            alpha = busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0.0, b=1.0)
            x_k1 = xk + alpha * s

            s = -1 * (gradiente(funcion, x_k1)) + (np.linalg.norm(gradiente(funcion, x_k1)) ** 2 /
                                                   np.linalg.norm(gradiente(funcion, xk)) ** 2) * s

            if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                terminar = True
            else:
                k = k + 1
                xk = x_k1

    return xk

# Ejemplo de uso con la función de Himmenblau
himmenblau = lambda x: (((x[0] ** 2) + x[1] - 11) ** 2) + ((x[0] + (x[1] ** 2) - 7) ** 2)
print(gradiente_conjugado(himmenblau, np.array([1.0, 1.0]), 0.001, 0.001, 100))
# Debería imprimir un resultado cercano a [3, 2]
