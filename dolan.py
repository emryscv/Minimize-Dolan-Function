import numpy as np
import math
from scipy.optimize import minimize
import time

def dolan(x):
    return (x[0] + 1.7 * x[1]) * math.sin(x[0]) - 1.5 * x[2] - 0.1 * x[3] * math.cos(x[3] + x[4] - x[0]) + 0.2 * x[4] * x[4] - x[1] - 1

start_time = time.time()

# Punto de inicio para la optimización
x0 = np.random.uniform(low=-100, high=100, size=5)
x0 = [100, 100, 100, 100, 100]
# x0 = [0, 0, 0, 0, 0]
bounds = [(-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100)]

# Función para la optimización con el método BFGS
result = minimize(dolan, x0, method="BFGS", bounds=bounds)
# Función para la optimización con el método SLSQP
result = minimize(dolan, x0, method="SLSQP", bounds=bounds)

end_time = time.time()

# Imprimir resultados
print("Resultado de la optimización:")
print("Valor óptimo de la función:", result.fun)
print("Número de iteraciones:", result.nit)
print("Tiempo de ejecución: ", end_time - start_time, "segundos")
print("Punto óptimo:", result.x)
