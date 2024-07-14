import matplotlib.pyplot as plt
import numpy as np

alpha=0.5
epsilon=0.001

xt=np.array([2,1])
xs=np.array([2,5])
#f=((x1-10)**2) + ((x2-10)**2)

#Basadas en Central Difference Method (Scarborough, 1966)
def derivada(f, x, deltaa_x):
    return (f(x + deltaa_x) - f(x - deltaa_x)) / (2 * deltaa_x)

def segunda_derivada(f, x, deltaa_x):
    return (f(x + deltaa_x) - 2 * f(x) + f(x - deltaa_x)) / (deltaa_x ** 2)

def delta_x(x):
    if abs(x) > 0.01:
        return 0.01 * abs(x)
    else:
        return 0.0001

def funcioon(x1,x2):
    return ((x1-10)**2) + ((x2-10)**2)

def newton_method(x0, epsilon, f):
    x = x0
    while abs(derivada(f, x, delta_x(x))) > epsilon:
        segunda_deriv = segunda_derivada(f, x, delta_x(x))
        if segunda_deriv == 0:
            return x
        x = x - derivada(f, x, delta_x(x)) / segunda_deriv
    return x

def evaluar(alpha):
    xt=np.array([2,1])
    xs=np.array([2,5])
    x_alpha= (xt) + (alpha*xs)
    x_nuevo=(x_alpha[0]-10)**2 + ((x_alpha[1]-10)**2)

    return x_nuevo

def busqueda_unidireccional(xt,xs,funcion,evaluar):
    alpha= 0.5
    falpha=funcion(alpha, epsilon, evaluar)
    print(falpha)
    x_alpha= (xt) + (falpha*xs)
    resultado=funcioon(x_alpha[0], x_alpha[1])

    return resultado


#x_alpha=(busqueda_unidireccional(xt,xs,alpha))

'''
x1,x2= x_alpha

print(x1,x2)


def falpha(x_alpha):
    x1,x2=x_alpha
    return evaluar(x1,x2)

#print(falpha(busqueda_unidireccional(xt,xs,alpha)))
'''
print(busqueda_unidireccional(xt,xs, newton_method,evaluar))
