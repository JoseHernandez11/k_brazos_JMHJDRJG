"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from algorithms import Algorithm, EpsilonGreedy, UCB1, UCB2, Softmax, GradientBandit
import os

def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__

    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    elif isinstance(algo, UCB2):
        label += f" (alpha={algo.alpha})"
    elif isinstance(algo, Softmax):
        label += f" (tau={algo.tau})"
    elif isinstance(algo, GradientBandit):
        label += f" (alpha={algo.alpha})"
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    
    return label



def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm], dist = "Normal"):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plot_name = 'Recompensa Promedio vs Pasos de Tiempo '.replace(" ","_") + str (dist) + "-" + str(label)+ ".png"
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo ' + str (dist), fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    #plt.savefig(os.path.join("plots", plot_name))
    plt.show()
    


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm], dist = "Normal"):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plot_name = '% Selecciones óptimas vs Pasos de Tiempo  '.replace(" ","_") + str (dist) + "-" + str(label)+ ".png"
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Porcentaje de selecciones óptimas', fontsize=14)
    plt.title('% Selecciones óptimas vs Pasos de Tiempo (' + str (dist) +  ")", fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    #plt.savefig(os.path.join("plots",plot_name))
    plt.show()


def plot_arm_statistics(arm_stats: List[Dict[int, Dict[str, Any]]],
                        algorithms: List[str], dist= "Normal"):
    """
    Genera gráficas separadas con las estadísticas de cada brazo para distintos algoritmos.
    
    Cada elemento de 'arm_stats' es un diccionario que contiene, para cada brazo (clave entera),
    estadísticas como 'avg_reward', 'times_selected' y 'optimal_arm'. 
    La lista 'algorithms' es un listado de nombres o identificadores de los algoritmos
    correspondientes a cada entrada en 'arm_stats'.
    
    El gráfico de cada algoritmo mostrará:
      - Eje X: cada brazo.
      - Eje Y: promedio de ganancias (avg_reward).
      - Etiqueta de cada barra: número de veces seleccionado + indicador de si es óptimo.
      - Los brazos óptimos se pintan en verde y los no óptimos en rojo.
      - Las etiquetas de eje X se alinean con el centro de las barras.
    
    Args:
        arm_stats (List[Dict[int, Dict[str, Any]]]): 
            Lista de longitud N (una entrada por algoritmo). 
            Cada elemento es un diccionario cuyas claves son los identificadores de los brazos (1..k),
            y los valores son un diccionario con campos:
                {
                  'avg_reward': float,
                  'times_selected': float,
                  'optimal_arm': int (0 o 1)
                }
        algorithms (List[str]):
            Lista de nombres o identificadores de los algoritmos correspondientes a cada
            diccionario en 'arm_stats'.
    
    Returns:
        None. Esta función genera uno o varios gráficos en pantalla.
    """

    # Asegurarnos de que la cantidad de algoritmos coincida con la de datos en arm_stats
    if len(arm_stats) != len(algorithms):
        raise ValueError("La longitud de 'arm_stats' debe coincidir con la de 'algorithms'.")

    # Iterar sobre cada algoritmo y sus estadísticas
    for i, algo_stats in enumerate(arm_stats):
        # 'algo_stats' es un diccionario con las estadísticas por brazo
        # Extraemos los brazos en orden ascendente de su identificador (1..k)
        arms = sorted(algo_stats.keys())
        
        # Extraemos las ganancias promedio
        avg_rewards = [algo_stats[arm]['avg_reward'] for arm in arms]

        # Colores por barra: verde si es óptima, rojo si no
        bar_colors = [
            'green' if algo_stats[arm]['optimal_arm'] == 1 else 'red'
            for arm in arms
        ]
        
        # Construimos las etiquetas de cada barra
        bar_labels = []
        for arm in arms:
            times_sel = int(algo_stats[arm]['times_selected'])
            opt = algo_stats[arm]['optimal_arm']
            label = f"Arm {arm}\nSel: {times_sel}"
            if opt == 1:
                label += "\n(Óptimo)"
            bar_labels.append(label)
        
        # Generamos la figura
        plt.figure(figsize=(10, 6))
        
        # Creamos las posiciones para las barras
        x_positions = range(len(arms))
        # Dibujamos las barras, especificando los colores
        bar_plot = plt.bar(x_positions, avg_rewards, color=bar_colors, edgecolor='black')
        
        # Para alinear las etiquetas de eje X con el centro de cada barra,
        # ajustamos la ubicación de los xticks considerando el ancho de barra por defecto (0.8)
        bar_width = 0.8
        centers = [x + bar_width / 2 for x in x_positions]
        
        # Ajustamos las marcas y las etiquetas en el eje X
        plt.xticks(centers, bar_labels, rotation=45, ha='right')
        
        # Añadimos títulos y etiquetas de ejes
        plt.title(f"Estadísticas de brazos - {get_algorithm_label(algorithms[i])} - {str(dist)}")
        plt.xlabel("Brazo")
        plt.ylabel("Promedio de Ganancias (avg_reward)")
        plot_name = f"Estadísticas de brazos - {get_algorithm_label(algorithms[i])} - {str(dist)}".replace(" ","_")+ ".png"
        # Añadimos una rejilla para el eje Y
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajustamos márgenes para que las etiquetas no se corten
        plt.tight_layout()
        
        # Mostramos la gráfica
        #plt.savefig(os.path.join("plots", plot_name))
        plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], c_theoretical: float = None, dist= "Normal"):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param c_theoretical: la cota teórica Cte * ln(T)
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))

    
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    if c_theoretical is not None:
        # Generamos valores de la cota: c * ln(t),
        # comenzando desde t=1 para evitar log(0)
        t_values = np.arange(1, steps + 1)
        theoretical_bound = np.max (regret_accumulated, axis = 0) + c_theoretical * np.log(t_values)

        # Graficamos la curva de cota en línea punteada
        plt.plot(
            range(steps),
            theoretical_bound,
            label=f'Cota Teórica: {c_theoretical}·ln(t)',
            linestyle='--',
            color='black'
        )

    plot_name = 'Rechazo acumulado Promedio vs Pasos de Tiempo '.replace(" ","_") + str (dist) +   "_" + str(label)+".png"
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Rechazo acumulado promedio', fontsize=14)
    plt.title('Rechazo acumulado Promedio vs Pasos de Tiempo  (' + str (dist) +  ")" , fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    #plt.savefig(os.path.join("plots",plot_name))
    plt.show()
