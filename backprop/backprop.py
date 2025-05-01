import numpy as np

def limit(f, x):
    h = 0.000000001
    return (f(x + h) - f(x)) / h

def equation(x):
    return x**2*np.sin(x)

x = 0.5
approx = limit(equation, x)
#  2xsin(x) + x²cos(x).
exact = 2*x*np.sin(x) + x**2*np.cos(x)
print("Numerical Derivative:", approx)
print("Exact Value:", exact)
print("Difference:", exact-approx)
