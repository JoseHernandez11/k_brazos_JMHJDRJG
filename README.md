# Práctica 1: Bandido de k-brazos

## Información
- **Alumnos:** Hernández Nieto, José María; Rodríguez Garrido, Juan de Dios; García Serrano, Javier
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** JMHJDRJG

## Descripción
En este repositorio, se estudia el rendimiento de distintos algoritmos diseñados para abordar el problema del bandido de k brazos, incluyendo ε-greedy, UCB-1, UCB-2, Softmax y gradiente de preferencias. Estos métodos presentan diferentes estrategias para explorar o explotar el espacio de soluciones, desde enfoques sencillos basados en exploración aleatoria hasta modelos más sofisticados que emplean índices de confianza o aprendizaje basado en gradiente. Además, se analiza cómo el desempeño de estos algoritmos varía en función de la distribución de recompensas de los brazos del bandido, considerando tres tipos de distribuciones: normal, Bernoulli y binomial.

## Estructura
- **algorithms/**: Scripts que implementan los diferentes enfoques de resolución del problema del bandido multibrazo.
- **arms/**: Scripts que implementan las diferentes distribuciones de probabilidad de los brazos.
- **plotting/**: Scripts que implementan las representaciones del estudio.
- **docs/**: incluye el informe documental de la práctica.
- **main.ipynb**: Notebook que permite moverse entre los distintos notebooks del estudio.
- **bandit_experiment_egreedy.ipynb**: Estudio del problema del bandido de k-brazos usando algoritmos ε-greedy, para diferentes valores de ε y diferentes distribuciones de recompensa para los brazos (normal, Bernoulli, Binomial).
- **bandit_experiment_UCB.ipynb**: Estudio del problema del bandido de k-brazos usando algoritmos upper bound confident, concretamente UCB1 y UCB2, para diferentes valores de los hiperparámetros y diferentes distribuciones de recompensa para los brazos (normal, Bernoulli, Binomial).
- **bandit_experiment_gradient.ipynb**: Estudio del problema del bandido de k-brazos usando algoritmos basados en gradientes, concretamente Softmax y gradiente de preferencias, para diferentes valores de los hiperparámetros y diferentes distribuciones de recompensa para los brazos (normal, Bernoulli, Binomial).
- **bandit_experiment_comparison.ipynb**: Comparación de los mejores algoritmos de cada estudio.

## Tecnologías Utilizadas
- **Lenguaje:** Python
- **Librerías:**
  - `numpy==1.26.4`
  - `seaborn==0.13.2`
  - `matplotlib==3.10.0`
