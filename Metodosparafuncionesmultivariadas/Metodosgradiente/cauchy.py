import numpy as np
import math

def regla_eliminacion(x1, x2, fx1, fx2, a, b) -> tuple[float, float]:
    """
    Regla de eliminación para la búsqueda unidimensional.

    Args:
        x1 (float): Punto 1 de evaluación.
        x2 (float): Punto 2 de evaluación.
        fx1 (float): Valor de la función en x1.
        fx2 (float): Valor de la función en x2.
        a (float): Extremo izquierdo del intervalo.
        b (float): Extremo derecho del intervalo.

    Returns:
        tuple[float, float]: Retorna los extremos ajustados del intervalo después de la eliminación.
    """
    if fx1 > fx2:
        return x1, b

    if fx1 < fx2:
        return a, x2
    
    return x1, x2

def w_to_x(w: float, a, b) -> float:
    """
    Transforma un valor w en el intervalo [0, 1] a un valor en el intervalo [a, b].

    Args:
        w (float): Valor en el intervalo [0, 1].
        a (float): Extremo izquierdo del intervalo de salida.
        b (float): Extremo derecho del intervalo de salida.

    Returns:
        float: Valor transformado en el intervalo [a, b].
    """
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon: float, a: float = None, b: float = None) -> float:
    """
    Implementación del método de búsqueda dorada para optimización unidimensional.

    Args:
        funcion (function): Función objetivo a minimizar.
        epsilon (float): Tolerancia de convergencia.
        a (float, optional): Extremo izquierdo del intervalo inicial. Por defecto None.
        b (float, optional): Extremo derecho del intervalo inicial. Por defecto None.

    Returns:
        float: Punto óptimo encontrado dentro del intervalo [a, b].
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

def gradiente(f, x, deltaX=0.001):
    """
    Calcula el gradiente de una función multivariable en un punto dado.

    Args:
        f (function): Función a derivar.
        x (list): Punto de evaluación.
        deltaX (float, optional): Paso de diferencia finita. Por defecto 0.001.

    Returns:
        list: Lista de gradientes parciales en el punto x.
    """
    grad = []
    for i in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] += deltaX
        xn[i] -= deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return grad

def cauchy(funcion, x0, epsilon1, epsilon2, M):
    """
    Método de Cauchy para optimización basada en gradiente.

    Args:
        funcion (function): Función a optimizar.
        x0 (array): Punto inicial.
        epsilon1 (float): Tolerancia para norma del gradiente.
        epsilon2 (float): Tolerancia para cambio relativo en la solución.
        M (int): Máximo número de iteraciones.

    Returns:
        array: Punto óptimo encontrado.
    """
    terminar = False
    xk = x0
    k = 0

    while not terminar:
        grad = np.array(gradiente(funcion, xk))

        if np.linalg.norm(grad) < epsilon1 or k >= M:
            terminar = True
        else:
            def alpha_funcion(alpha):
                return funcion(xk - alpha * grad)

            alpha = busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0.0, b=1.0)
            x_k1 = xk - alpha * grad

            if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                terminar = True
            else:
                k += 1
                xk = x_k1

    return xk

# Ejemplo de uso con la función de Himmenblau
himmenblau = lambda x: (((x[0]**2) + x[1] - 11)**2) + ((x[0] + (x[1]**2) - 7)**2)
print(cauchy(himmenblau, np.array([0.0, 0.0]), 0.001, 0.001, 100))