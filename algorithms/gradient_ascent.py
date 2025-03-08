"""
Module: algorithms/epsilon_greedy.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

import numpy as np
from abc import ABC

class Softmax(Algorithm):
    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo Softmax.
        :param k: Número de brazos.
        :param tau: Parámetro de temperatura (debe ser mayor que 0).
        """
        assert tau > 0, "El parámetro tau debe ser mayor que 0."
        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política Softmax.
        :return: Índice del brazo seleccionado.
        """
        exp_values = np.exp(self.values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(self.k, p=probabilities)

class GradientBandit:
    def __init__(self, k, alpha=0.1):
        self.k = k
        self.alpha = alpha
        self.preferences = np.zeros(k)
        self.average_reward = 0.0
        self.total_pulls = 0  # Si no lo maneja la superclase

        # (O, si usas la superclase, inícialo ahí)
        # super().__init__(k)

    def select_arm(self):
        max_pref = np.max(self.preferences)
        exp_prefs = np.exp(self.preferences - max_pref)  # estabilidad
        probs = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(self.k, p=probs)

    def update(self, chosen_arm, reward):
        self.total_pulls += 1
        # baseline incremental
        self.average_reward += (reward - self.average_reward) / self.total_pulls

        # calcular probs con las preferencias ANTES de modificarlas
        max_pref = np.max(self.preferences)
        exp_prefs = np.exp(self.preferences - max_pref)
        probs = exp_prefs / np.sum(exp_prefs)

        for a in range(self.k):
            if a == chosen_arm:
                self.preferences[a] += self.alpha * (reward - self.average_reward) * (1 - probs[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.average_reward) * probs[a]

        # Si hay una superclase con self.counts, etc.:
        # super().update(chosen_arm, reward)

    def reset(self):
        # Reiniciar todo
        self.preferences[:] = 0.0
        self.average_reward = 0.0
        self.total_pulls = 0