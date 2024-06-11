from pandas import *
import pulp

# Defining problem
model = pulp.LpProblem("Scheduling_problem", pulp.LpMinimize)

# Reading data
factory_path = r"C:\Users\13953860\OneDrive - Universidad Catolica del Maule\Escritorio\Opti2024\Ejercicio Pulp PP\factory_variables.csv"
demand_path = r"C:\Users\13953860\OneDrive - Universidad Catolica del Maule\Escritorio\Opti2024\Ejercicio Pulp PP\monthly_demand.csv"

factories = read_csv(factory_path, index_col=["Month", "Factory"])
demand = read_csv(demand_path, index_col=["Month"])


# Variable definition (by month and factory)

production = pulp.LpVariable.dicts(
    "production",
    ((month, factory) for month, factory in factories.index),
    lowBound=0,
    cat="Integer",
)

factory_status = pulp.LpVariable.dicts(
    "factory_status",
    ((month, factory) for month, factory in factories.index),
    cat="Binary",
)


# Objective Function:

model += pulp.lpSum(
    [
        production[month, factory] * factories.loc[(month, factory), "Variable_Costs"]
        for month, factory in factories.index
    ]
    + [
        factory_status[month, factory] * factories.loc[(month, factory), "Fixed_Costs"]
        for month, factory in factories.index
    ]
)


# Restrictions
# Demand: monthly production in A+B = monthly demand
months = demand.index
for month in months:
    model += (
        production[(month, "A")] + production[(month, "B")]
        == demand.loc[month, "Demand"]
    )

# Production in any month must be between minimum and maximum capacity, or zero.
for month, factory in factories.index:
    min_production = factories.loc[(month, factory), "Min_Capacity"]
    max_production = factories.loc[(month, factory), "Max_Capacity"]
    model += (
        production[(month, factory)] >= min_production * factory_status[month, factory]
    )
    model += (
        production[(month, factory)] <= max_production * factory_status[month, factory]
    )

# In May, Factory B is off
model += factory_status[5, "B"] == 0
model += production[5, "B"] == 0


# Solving

model.solve()

print("Status:", pulp.LpStatus[model.status])
# Final results:

output = []
for month, factory in production:
    var_output = {
        "Month": month,
        "Factory": factory,
        "Production": production[(month, factory)].varValue,
        "Factory Status": factory_status[(month, factory)].varValue,
    }
    output.append(var_output)

output_df = DataFrame(output)
print(output_df)
