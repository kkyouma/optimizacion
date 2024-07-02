## Montecarlo

```python
start_time = time.time()

num_iteraciones = 100000
retorno_esperado = [0.03, 0.05, 0.08]
volatilidad = [0.15, 0.22, 0.38]
diversificacion = [0.3333, 0.3333, 0.3333]
monto_inversion = 10000

ganancias_totales = []
for _ in range(num_iteraciones):
    ganancias = []
    for i in range(len(retorno_esperado)):
        rendimiento_inversion = np.random.normal(retorno_esperado[i], volatilidad[i])
        ganancias.append(rendimiento_inversion * diversificacion[i] * monto_inversion)
    ganancia_total = sum(ganancias)
    ganancias_totales.append(ganancia_total)

ganancia_total = np.mean(ganancias_totales)
desviacion_estandar = np.std(ganancias_totales)

end_time = time.time()

print("Ganancia total después de un año:", ganancia_total)
print("Desviación estándar del retorno total:", desviacion_estandar)
print("Tiempo de ejecución:", end_time - start_time, "segundos")

print("Peor caso:", min(ganancias_totales))
print("Mejor caso:", max(ganancias_totales))

```

```python
out = pd.DataFrame()
nofsimulation = 1000

data = {
    "wind": {"0.1": {"triangular": [30, 50, 100]}},
    "solarpv": {"0.15": {"triangular": [10, 40, 70]}},
    "solarconc": {"0.05": {"triangular": [50, 70, 120]}},
    "hydro": {"0.2": {"uniform": [10, 40]}},
    "coal": {"0.2": {"triangular": [30, 50, 100]}},
    "gas": {"0.3": {"uniform": [50, 50]}},
}


def main():
    iterations_range = range(100, max_iterations + 1, 1000)
    variances = []

    for iterations in iterations_range:
        initialize(iterations)
        variance = out["total_cost"].var()
        variances.append(variance)

    plot_convergence(iterations_range, variances)

    # confidence_interval = np.percentile(out["total_cost"], [10, 90])
    # print(f"{confidence_interval}")

    # datos = out.describe()
    # print(datos)


def initialize(iterations):
    start_time = time.perf_counter()
    for i in range(iterations):
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
            elif distr.lower() == "normal":
                mu = param[0]
                sigma = param[1]
                x = np.random.normal(mu, sigma)
                unit_cost = normal(x, mu, sigma) * float(share)

    return unit_cost * float(share)


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


def normal(x, mu, sigma):
   """Returns a value from a normal distribution"""
   return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def plot_convergence(iterations_range, variances):
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_range, variances, marker='o', linestyle='-', color='b')
    plt.title('Convergence of Total Cost Variance with Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Variance of Total Cost')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()

```

### Graficar

Distribuciones

```python
total_cost = out["total_cost"].std()
general_std = out.std()

hist = out.hist("total_cost", bins=100)
hist = out.hist(
    column=[
        "coal_cost",
        "solarpv_cost",
        "hydro_cost",
        "wind_cost",
        "gas_cost",
        "solarconc_cost",
    ],
    bins=50,
    figsize=(10, 10),
)
```

KDE

```python
plt.figure(figsize=(10, 6))
for column in ["coal_cost", "solarpv_cost", "hydro_cost", "wind_cost", "gas_cost", "solarconc_cost"]:
    sns.kdeplot(out[column], label=column)
plt.title('KDE Plot of Source Costs')
plt.legend()
plt.show()
```

Boxplot

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=out[["coal_cost", "solarpv_cost", "hydro_cost", "wind_cost", "gas_cost", "solarconc_cost"]])
plt.title('Box Plot of Source Costs')
plt.show()
```
