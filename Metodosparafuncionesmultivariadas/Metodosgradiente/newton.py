import numpy as np
import math


f = lambda x: (((x[0]**2)+x[1]-11)**2) + ((x[0]+(x[1]**2)-7)**2)

def regla_eliminacion(x1, x2, fx1, fx2, a, b)->tuple[float, float]:
    if fx1 > fx2:
        return x1, b

    if fx1 < fx2:
        return a, x2
    
    return x1, x2

def w_to_x(w:float, a, b) -> float:
    return w * (b-a) + a

#f = lambda x: x[0]**2- 2*x[1]
def busquedaDorada(funcion, epsilon:float, a:float=None, b:float=None) -> float:
    PHI = ( 1 + math.sqrt(5) ) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1

    while Lw > epsilon:
        w2 = aw + PHI*Lw
        w1 = bw - PHI*Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), 
                                    funcion(w_to_x(w2, a, b)), aw, bw)
        k+=1
        Lw = bw - aw

    return (w_to_x(aw, a, b)+w_to_x(bw, a, b))/2

def gradiente(f, x, deltaX=0.00001):
    grad = []
    for i in range(0, len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i]+deltaX
        xn[i] = xn[i]-deltaX
        print(x, xp, xn, deltaX)
        grad.append((f(xp)-f(xn))/(2*deltaX))
    return grad


def hessian_matrix(f, x, deltaX=0.00001):
    fx = f(x)
    N = len(x)
    H = []
    for i in range(N):
        hi = []
        for j in range(N):
            print(i, j, end=' ')
            if i == j:
                xp = x.copy()
                xn = x.copy()
                xp[i] = xp[i] + deltaX
                xn[i] = xn[i] - deltaX
                hi.append( ( f(xp)- 2*fx + f(xn))/ (deltaX**2) )
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

                hi.append( (f(xpp)-f(xpn)-f(xnp)+f(xnn)) / (4*deltaX**2))
        H.append(hi)
        print()
    return H

def newton(funcion, x0, epsilon1, epsilon2, M):
    terminar = False
    xk = x0
    k=0
    while not terminar:
        grad = np.array(gradiente(funcion, xk))
        hessiana=hessian_matrix(f,xk)
        if np.linalg.norm(grad) < epsilon1 or k >= M:
            terminar = True
        else:
            print("Formula",xk - 0.5*np.dot(np.linalg.inv(hessiana),grad))
            def alpha_funcion(alpha):
                return funcion(xk - alpha*np.dot(np.linalg.inv(hessiana),grad))
            
            alpha = busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0.0, b=1.0)
            x_k1 = xk - alpha*np.dot(np.linalg.inv(hessiana),grad)
            print(xk, alpha, grad, x_k1)
            
            if np.linalg.norm(x_k1-xk)/(np.linalg.norm(xk)+0.00001) <= epsilon2 :
                terminar = True
            else:
                k = k + 1
                xk = x_k1
    return xk

x = np.array([1.0, 1.0])
print(gradiente(f, x, 0.01))
print(hessian_matrix(f, x, 0.01))

himmenblau = lambda x: (((x[0]**2)+x[1]-11)**2) + ((x[0]+(x[1]**2)-7)**2)
print(newton(himmenblau, np.array([2.0,2.0]), 0.001, 0.001, 100))