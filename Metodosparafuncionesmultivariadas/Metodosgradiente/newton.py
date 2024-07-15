import numpy as np
import math

# Definición de la función objetivo
f = lambda x: (((x[0]**2)+x[1]-11)**2) + ((x[0]+(x[1]**2)-7)**2)

def regla_eliminacion(x1, x2, fx1, fx2, a, b) -> tuple[float, float]:
    """
    Aplica la regla de eliminación para actualizar los límites de búsqueda.

    Args:
    - x1, x2: Puntos de prueba en el espacio de búsqueda.
    - fx1, fx2: Valores de la función objetivo en x1 y x2 respectivamente.
    - a, b: Límites de búsqueda actuales.

    Returns:
    - Tupla con los nuevos límites actualizados según la regla de eliminación.
    """
    if fx1 > fx2:
        return x1, b
    elif fx1 < fx2:
        return a, x2
    else:
        return x1, x2

def w_to_x(w: float, a, b) -> float:
    """
    Transforma el parámetro w (en el rango [0, 1]) a un valor en el rango [a, b].

    Args:
    - w: Parámetro en el rango [0, 1].
    - a, b: Límites del rango deseado.

    Returns:
    - Valor transformado en el rango [a, b].
    """
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon: float, a: float = None, b: float = None) -> float:
    """
    Realiza la búsqueda del mínimo de una función utilizando el método de la búsqueda dorada.

    Args:
    - funcion: Función objetivo a minimizar.
    - epsilon: Precisión deseada para la convergencia.
    - a, b: Límites iniciales de búsqueda (por defecto, 0 y 1 respectivamente).

    Returns:
    - Valor aproximado del mínimo de la función dentro del rango [a, b].
    """
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1

    while Lw > epsilon:
        w2 = aw + PHI * Lw
        w1 = bw - PHI * Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
        k += 1
        Lw = bw - aw

    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

def gradiente(f, x, deltaX=0.00001):
    """
    Calcula el gradiente de una función en un punto dado utilizando diferencias finitas.

    Args:
    - f: Función objetivo.
    - x: Punto en el cual se evalúa el gradiente.
    - deltaX: Tamaño del paso para calcular las diferencias finitas.

    Returns:
    - Lista con los componentes del gradiente en el punto x.
    """
    grad = []
    for i in range(0, len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i] + deltaX
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return grad

def hessian_matrix(f, x, deltaX=0.00001):
    """
    Calcula la matriz Hessiana de una función en un punto dado utilizando diferencias finitas.

    Args:
    - f: Función objetivo.
    - x: Punto en el cual se evalúa la matriz Hessiana.
    - deltaX: Tamaño del paso para calcular las diferencias finitas.

    Returns:
    - Matriz Hessiana evaluada en el punto x.
    """
    fx = f(x)
    N = len(x)
    H = []
    for i in range(N):
        hi = []
        for j in range(N):
            if i == j:
                xp = x.copy()
                xn = x.copy()
                xp[i] = xp[i] + deltaX
                xn[i] = xn[i] - deltaX
                hi.append((f(xp) - 2 * fx + f(xn)) / (deltaX ** 2))
            else:
                xpp = x.copy()
                xpn = x.copy()
                xnp = x.copy()
                xnn = x.copy()
                xpp[i] = xpp[i] + deltaX
                xpp[j] = xpp[j] + deltaX
                xpn[i] = xpn[i] + deltaX
                xpn[j] = xpn[j] - deltaX
                xnp[i] = xnp[i] - deltaX
                xnp[j] = xnp[j] + deltaX
                xnn[i] = xnn[i] - deltaX
                xnn[j] = xnn[j] - deltaX
                hi.append((f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * deltaX ** 2))
        H.append(hi)
    return H

def newton(funcion, x0, epsilon1, epsilon2, M):
    """
    Implementa el método de Newton para encontrar el mínimo de una función.

    Args:
    - funcion: Función objetivo a minimizar.
    - x0: Punto inicial de búsqueda.
    - epsilon1: Precisión para la norma del gradiente.
    - epsilon2: Precisión para la convergencia del método.
    - M: Máximo número de iteraciones permitidas.

    Returns:
    - Punto aproximado donde se alcanza el mínimo local de la función.
    """
    terminar = False
    xk = x0
    k = 0

    while not terminar:
        grad = np.array(gradiente(funcion, xk))
        hessiana = hessian_matrix(funcion, xk)
        
        if np.linalg.norm(grad) < epsilon1 or k >= M:
            terminar = True
        else:
            def alpha_funcion(alpha):
                return funcion(xk - alpha * np.dot(np.linalg.inv(hessiana), grad))
            
            alpha = busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0.0, b=1.0)
            x_k1 = xk - alpha * np.dot(np.linalg.inv(hessiana), grad)
            
            if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                terminar = True
            else:
                k += 1
                xk = x_k1

    return xk

# Ejemplo de uso
x = np.array([1.0, 1.0])
print("Gradiente:", gradiente(f, x, 0.01))
print("Matriz Hessiana:", hessian_matrix(f, x, 0.01))

himmenblau = lambda x: (((x[0]**2)+x[1]-11)**2) + ((x[0]+(x[1]**2)-7)**2)
print("Mínimo local encontrado:", newton(himmenblau, np.array([2.0, 2.0]), 0.001, 0.001, 100))