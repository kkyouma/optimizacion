"Modelo de simulación con MonteCarlo"

import numpy as np
import pandas as pd
import time

out = pd.DataFrame()  # Donde iran los resultados de todas las simulaciones
nofsimulation = 100000

# Data: "todos estos datos son ficticios"
# Para simplificar la explicación se pone todo en un diccionario
# en lugar de leerlo desde un archivo input

data = {
    "wind": {"0.1": {"triangular": [30, 50, 100]}},
    "solarpv": {"0.15": {"triangular": [10, 40, 70]}},
    "solarconc": {"0.05": {"triangular": [50, 70, 120]}},
    "hydro": {"0.2": {"uniform": [10, 40]}},
    "coal": {"0.2": {"triangular": [30, 50, 100]}},
    "gas": {"0.3": {"uniform": [40, 120]}},
}


def main():
    initialize()


def initialize():
    start_time = time.perf_counter()
    for i in range(nofsimulation):
        source_cost = {}
        total_cost = 0
        for source, item in data.items():
            source_cost = calculte_cost(item)
            total_cost += source_cost
            label = source + "_cost"
            out.loc[i, label] = source_cost
        label = "total_cost"
        out.loc[i, label] = total_cost
    finish()
    finish_time = time.perf_counter() - start_time
    print("Simulation finished (time)", finish_time)


def finish():
    out.fillna(0, inplace=True)
    out.to_csv("output.csv", sep=";")


def calculte_cost(dict):
    """Calcula la fracción del costo
    por unidad de generación"""
    unit_cost = 0
    for share, values in dict.items():
        for distr, param in values.items():
            if distr.lower() == "triangular":
                x = np.random.random()
                a = param[0]
                b = param[1]
                c = param[2]
                unit_cost = triangular(x, a, b, c)
            elif distr.lower() == "uniform":
                x = np.random.random()
                a = param[0]
                b = param[1]
                unit_cost = uniform(x, a, b)
    return unit_cost * float(share)


# Métodos de distribución
def uniform(x, low, high):
    """x = random number"""
    value = low + (high - low) * x
    return value


def triangular(x, a, b, c):
    """x = random number"""
    k = (b - a) / (c - a)
    if x < k:
        value = a + (x * (c - a) * (b - a)) ** (0.5)
    else:
        value = c - ((1 - x) * (c - a) * (c - b)) ** (0.5)
    return value


if __name__ == "__main__":
    main()
