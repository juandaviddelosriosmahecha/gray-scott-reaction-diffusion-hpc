# gray-scott-reaction-diffusion-hpc

Simulación del sistema de **reacción-difusión de Gray-Scott** (modelo de tipo
Belousov-Zhabotinsky) en 2D, con **cuatro implementaciones de alto rendimiento
(HPC)** para comparar plataformas de paralelización: **serial (CPU)**, **OpenMP
(CPU multinúcleo)**, **MPI (memoria distribuida)** y **CUDA (GPU)**.

## ¿Qué es esto?

El [modelo de Gray-Scott](https://es.wikipedia.org/wiki/Reacci%C3%B3n-difusi%C3%B3n)
describe dos sustancias químicas `u` y `v` que se difunden y reaccionan sobre una
malla `N×N`, generando patrones espaciales autoorganizados (manchas, laberintos,
ondas) característicos de los sistemas de reacción-difusión. La dinámica se integra
con el esquema explícito:

```
∂u/∂t = Du·∇²u − u·v² + F·(1 − u)
∂v/∂t = Dv·∇²v + u·v² − (F + k)·v
```

El laplaciano `∇²` se discretiza con un estándar de 5 puntos y **condiciones de
frontera periódicas**. Cada versión:

- Permite elegir el **tamaño de malla** `N×N` y una **geometría inicial** para la
  siembra de la sustancia `v`.
- Integra `150000` pasos de tiempo y guarda el campo `v` cada `100` iteraciones.
- Calcula dos métricas por paso guardado: **entropía normalizada** y **gradiente
  promedio**, que cuantifican la formación de estructura.
- Mide y reporta los **tiempos de ejecución** desglosados (inicialización,
  simulación, guardado, métricas), lo que permite estudiar el *speedup* entre
  plataformas y tamaños de malla.

### Geometrías iniciales disponibles

1. Focos circulares (se indica el número de focos)
2. Línea horizontal central
3. Cuadrado central
4. Patrón hexagonal
5. Cruz central

### Parámetros físicos (fijos en el código)

| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `Du`      | 0.20  | Coeficiente de difusión de `u` |
| `Dv`      | 0.10  | Coeficiente de difusión de `v` |
| `F`       | 0.026 | Tasa de alimentación (*feed*) |
| `k`       | 0.053 | Tasa de eliminación (*kill*) |
| `dt`      | 0.1   | Paso temporal |
| `dx`      | 1.0   | Paso espacial |
| `steps`   | 150000 | Iteraciones totales |
| `output_interval` | 100 | Cada cuántos pasos se guarda el estado |

## Estructura del repositorio

```
gray-scott-reaction-diffusion-hpc/
├── Makefile                     # Compila y ejecuta las 4 versiones
├── src/                         # Código fuente de las implementaciones
│   ├── serial.cpp               # Versión serial (CPU, referencia)
│   ├── OMP.cpp                  # Versión OpenMP (CPU multinúcleo)
│   ├── mpi.cpp                  # Versión MPI (memoria distribuida)
│   ├── cuda.cpp / cuda.cu       # Versión CUDA (GPU)
│   ├── cuda_T4.cpp              # Variante CUDA ajustada para GPU T4
│   └── cuda_T4.py               # Utilidad Python asociada a la versión T4
├── scripts/                     # Visualización de resultados (Python)
│   ├── graf_serial.py / graf_openmp.py / graf_cuda.py   # Gráficas de patrones y métricas
│   └── vd_serial.py / vd_cuda.py                         # Generación de vídeos
├── notebooks/                   # Notebooks de arranque (Colab)
│   ├── init_serial.ipynb
│   └── notebook_init.ipynb
├── Results/                     # Resultados de referencia ya generados
│   ├── SERIAL_CPU/   mesh_{50,100,200,400,800,1600}
│   ├── OMP/          mesh_{...}
│   ├── CUDA_1GPUT4/  mesh_{...}   # NVIDIA T4
│   └── CUDA_1GPUA100/mesh_{...}   # NVIDIA A100
├── docs/                        # Documento de referencia (PDF)
├── experimentacion_cuda_1.sh    # Barrido automático de mallas en CUDA
├── organizar_resultados.sh      # Mueve datos/vídeos a sus carpetas
├── select_mode.sh               # Menú interactivo (serial / MPI / CUDA)
├── FIG.jpeg                     # Figura de ejemplo
├── requirements.txt
└── LICENSE                      # MIT
```

## Requisitos

- **C++17** y `make`.
- **g++** (versión serial y OpenMP).
- **OpenMP** (incluido en GCC) para la versión multinúcleo.
- **MPI** (`mpicxx` / `mpirun`, p. ej. OpenMPI) para la versión distribuida.
- **CUDA Toolkit** (`nvcc`) y una GPU NVIDIA para la versión GPU.
- **Python 3** con `numpy` y `matplotlib` para graficar y generar vídeos
  (los scripts de vídeo usan `ffmpeg` a través de Matplotlib).

> **Nota sobre rutas.** El `Makefile` y los scripts `*.sh` usan rutas absolutas de
> **Google Colab** (`/content/gray-scott-reaction-diffusion-hpc`). Si ejecutas en
> local, ajusta `BASE_DIR` en el `Makefile` y las rutas `/content/...` en
> `organizar_resultados.sh` a la ubicación real del repositorio.

## Compilación

```bash
make all        # compila las 4 versiones en bin/
# o individualmente:
make serial
make omp
make mpi
make cuda
make info       # muestra núcleos y versiones de compiladores disponibles
make clean      # elimina binarios y artefactos de profiling
```

Los ejecutables se generan en `bin/` (`gray_scott_serial`, `gray_scott_omp`,
`gray_scott_mpi`, `gray_scott_cuda`).

## Ejecución

Cada programa es **interactivo**: pregunta el tamaño de malla, la geometría y
(para focos circulares) el número de focos.

```bash
make run_serial                 # versión serial
make run_omp                    # OpenMP (usa todos los núcleos: OMP_NUM_THREADS=nproc)
make run_mpi                    # MPI con 4 procesos (mpirun -np 4)
make run_cuda                   # CUDA

make run                        # menú interactivo para elegir versión (1-4)
./select_mode.sh                # menú alternativo (serial / MPI / CUDA)
```

Entrada típica (ejemplo: malla 400×400, geometría de focos circulares, 20 focos):

```
400
1
20
```

Opciones adicionales de OpenMP:

```bash
make run_omp_profile            # ejecuta con gprof → omp_profile_results.txt
make run_omp_debug              # build sin optimizar (-O0 -g) para depurar
```

### Barrido automático de experimentos (CUDA)

`experimentacion_cuda_1.sh` ejecuta la versión CUDA para las mallas
`50, 100, 200, 400, 800, 1600` con geometría de focos circulares (20 focos),
genera las gráficas y organiza las salidas en `results/`:

```bash
bash experimentacion_cuda_1.sh
```

## Salidas que genera

Cada ejecución crea una carpeta `BZ_Geometry_<tipo>/` con:

- `bz_<iter>.csv` — campo `v` (patrón espacial) en cada paso guardado.
- `metrics.csv` — columnas `Paso, Entropia, GradientePromedio`.

Además, por consola se imprime el desglose de **tiempos de ejecución**.

## Visualización

Los scripts de `scripts/` leen los CSV generados y producen figuras y vídeos:

```bash
python3 scripts/graf_serial.py   # gráficas de patrones + métricas (serial)
python3 scripts/graf_openmp.py   # idem para OpenMP
python3 scripts/graf_cuda.py     # idem para CUDA
python3 scripts/vd_serial.py     # vídeo de la evolución temporal (serial)
python3 scripts/vd_cuda.py       # vídeo de la evolución temporal (CUDA)
```

## Resultados incluidos

La carpeta `Results/` contiene salidas de referencia ya generadas para distintos
tamaños de malla y plataformas, útiles para comparar rendimiento sin volver a
ejecutar las simulaciones:

- `SERIAL_CPU/` — versión serial en CPU.
- `OMP/` — versión OpenMP en CPU multinúcleo.
- `CUDA_1GPUT4/` — CUDA en una GPU NVIDIA T4.
- `CUDA_1GPUA100/` — CUDA en una GPU NVIDIA A100.

Cada una incluye subcarpetas `mesh_{50,100,200,400,800,1600}` con las imágenes de
análisis y los tiempos medidos (`tiempos_*_<N>.txt`).

## Documentación

En `docs/` se incluye el documento *Simulación de la Reacción de Belousov
Zhabotinsky* (PDF) con el fundamento teórico y los resultados del estudio.

## Licencia

Distribuido bajo licencia **MIT**. Ver [`LICENSE`](LICENSE).
