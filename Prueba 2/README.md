1. Abrir el archivo desde conda

```shell
conda env create -f environment.yml
```

En caso de error se puede ejecutar

```shell
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Y luego instalar `PuLP` por separado

```shell
conda install pulp
```

2. Activar el entorno

```shell
conda activate opt_2
```

3. Instalar Jupyter en caso de que no este instalado

```shell
conda install jupyter
```

4. Abrir Jupyter

```shell
jupyter notebook
```

5. Verificar versiones

```python
import sys
import numpy as np  # Ejemplo de librería
import pandas as pd # Ejemplo de librería

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib", MatplotliborSeaborn3)
```
