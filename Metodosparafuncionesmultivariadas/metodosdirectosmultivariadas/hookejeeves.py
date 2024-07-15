import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def hooke_jeeves(f, x0, delta, alpha=2, epsilon=1e-6, max_iter=1000):
    """
    Implementación del método de Hooke-Jeeves para la optimización de funciones sin restricciones.

    Parameters:
    ----------
    f : callable
        Función objetivo a minimizar.
    x0 : list or np.array
        Punto inicial de búsqueda.
    delta : float
        Tamaño inicial del paso para el movimiento exploratorio.
    alpha : float, optional
        Factor de escala para actualizar x_new durante la búsqueda (default is 2).
    epsilon : float, optional
        Criterio de convergencia basado en el tamaño del paso delta (default is 1e-6).
    max_iter : int, optional
        Número máximo de iteraciones permitidas (default is 1000).

    Returns:
    -------
    x_best : np.array
        Punto óptimo encontrado que minimiza la función f.
    f_best : float
        Valor mínimo de la función f encontrado en x_best.
    history : list
        Lista que contiene los puntos visitados durante la optimización.
    """
    
    def exploratory_move(x, delta):
        """
        Realiza un movimiento exploratorio para encontrar un mejor punto cercano a x.

        Parameters:
        ----------
        x : np.array
            Punto actual desde el cual se realizará el movimiento exploratorio.
        delta : float
            Tamaño del paso para el movimiento exploratorio.

        Returns:
        -------
        x_new : np.array
            Nuevo punto encontrado después del movimiento exploratorio.
        """
        x_new = np.copy(x)
        f_current = f(x_new)
        for i in range(len(x)):
            x_new[i] += delta
            f_plus = f(x_new)
            if f_plus < f_current:
                f_current = f_plus
            else:
                x_new[i] -= 2 * delta
                f_minus = f(x_new)
                if f_minus < f_current:
                    f_current = f_minus
                else:
                    x_new[i] += delta
        return x_new
    
    x_base = np.array(x0)
    x_best = np.copy(x_base)
    f_best = f(x_best)
    history = [np.copy(x_best)]
    
    for _ in range(max_iter):
        x_new = exploratory_move(np.copy(x_base), delta)
        f_new = f(x_new)
        
        if f_new < f_best:
            while f_new < f_best:
                x_base = np.copy(x_new)
                f_best = f_new
                x_new = x_base + alpha * (x_base - x_best)
                x_new = exploratory_move(np.copy(x_new), delta)
                f_new = f(x_new)
                history.append(np.copy(x_base))
            x_best = np.copy(x_base)
        else:
            delta *= 0.5
            if delta < epsilon:
                break
        history.append(np.copy(x_base))
    
    return x_best, f_best, history

# Funciones de prueba

def sphere_function(x):
    """
    Función esférica: f(x) = sum(x**2)

    Parameters:
    ----------
    x : np.array
        Vector de variables de entrada.

    Returns:
    -------
    float
        Valor de la función esférica evaluada en x.
    """
    x = np.array(x)
    return np.sum(x**2)

def himmelblau_function(x):
    """
    Función de Himmelblau: f(x) = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    Parameters:
    ----------
    x : np.array
        Vector de variables de entrada.

    Returns:
    -------
    float
        Valor de la función de Himmelblau evaluada en x.
    """
    x = np.array(x)
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def rastrigin_function(x):
    """
    Función de Rastrigin: f(x) = A * len(x) + sum(x**2 - A * cos(2 * pi * x))

    Parameters:
    ----------
    x : np.array
        Vector de variables de entrada.

    Returns:
    -------
    float
        Valor de la función de Rastrigin evaluada en x.
    """
    x = np.array(x)
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """
    Función de Rosenbrock: f(x) = sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

    Parameters:
    ----------
    x : np.array
        Vector de variables de entrada.

    Returns:
    -------
    float
        Valor de la función de Rosenbrock evaluada en x.
    """
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Parámetros iniciales
functions = [sphere_function, himmelblau_function, rastrigin_function, rosenbrock_function]
initial_points = [
    [5.0, 5.0],   # Punto inicial para sphere_function
    [0.0, 0.0],   # Punto inicial para himmelblau_function
    [5.12, 5.12], # Punto inicial para rastrigin_function
    [1.2, 1.2]    # Punto inicial para rosenbrock_function
]
delta = 0.5

# Crear animaciones para cada función
for f, x0 in zip(functions, initial_points):
    x_best, f_best, history = hooke_jeeves(f, x0, delta)
    history = np.array(history)

    # Crear una cuadrícula para trazar la superficie de la función
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    images = []

    def update(num):
        ax.clear()
        ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
        ax.plot(history[:num+1, 0], history[:num+1, 1], marker='o', color='red')
        ax.plot(history[num, 0], history[num, 1], marker='o', color='red', markersize=5)
        ax.plot(x_best[0], x_best[1], marker='x', color='blue', markersize=10)
        ax.set_title(f'{f.__name__}\nIteración: {num+1}, Best: {x_best}, f_best: {f_best:.4f}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        # Guardar el frame actual como imagen
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(Image.fromarray(image))

    for num in range(len(history)):
        update(num)
    
    # Guardar la animación como GIF usando Pillow
    images[0].save(f'{f.__name__}_optimization.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

    plt.close(fig)

# Mostrar o guardar figuras estáticas (opcional)
'''
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

for ax, f, x0 in zip(axs.flatten(), functions, initial_points):
    x_best, f_best, history = hooke_jeeves(f, x0, delta)
    history = np.array(history)
    
    # Crear una cuadrícula para trazar la superficie de la función
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    
    # Plotear la superficie de la función
    ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    ax.plot(history[:, 0], history[:, 1], marker='o', color='red')
    ax.plot(x_best[0], x_best[1], marker='x', color='blue', markersize=10)
    ax.set_title(f'{

f.__name__}\nBest: {x_best}, f_best: {f_best:.4f}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

plt.tight_layout()
plt.show()
'''