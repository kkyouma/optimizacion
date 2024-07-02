import numpy as np
import time

start_time = time.time()
# Parámetros de la simulación
num_iteraciones = 100000  # Número de iteraciones o muestras
retorno_esperado = [0.03, 0.05, 0.08]  # Rendimientos esperados de los productos
volatilidad = [0.15, 0.22, 0.38]  # Volatilidades de los productos
diversificacion = [0.3333, 0.3333, 0.3333]  # Diversificación de la inversión
monto_inversion = 10000  # Monto de inversión total

ganancias_totales = []
for _ in range(num_iteraciones):
    ganancias = []
    for i in range(len(retorno_esperado)):
        # Generar un rendimiento aleatorio para cada activo
        rendimiento_inversion = np.random.normal(retorno_esperado[i], volatilidad[i])
        ganancias.append(rendimiento_inversion * diversificacion[i] * monto_inversion)
    ganancia_total = sum(ganancias)
    ganancias_totales.append(ganancia_total)

# Calcular la ganancia total después de un año y su desviación estándar
ganancia_total = np.mean(ganancias_totales)  # Promedio de las ganancias
desviacion_estandar = np.std(ganancias_totales)

end_time = time.time()

print("Ganancia total después de un año:", ganancia_total)
print("Desviación estándar del retorno total:", desviacion_estandar)
print("Tiempo de ejecución:", end_time - start_time, "segundos")

# Imprimir el peor y mejor caso
print("Peor caso:", min(ganancias_totales))
print("Mejor caso:", max(ganancias_totales))
