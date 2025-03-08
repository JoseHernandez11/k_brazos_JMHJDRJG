"""
Module: arms/armnormal.py
Description: Contains the implementation of the ArmNormal class for the normal distribution arm.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


import numpy as np

from arms import Arm



class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución de Bernoulli.
        :param p: Probabilidad de éxito (valor entre 0 y 1).
        """
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0, 1]."
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Bernoulli.
        :return: 1 con probabilidad p, 0 con probabilidad (1 - p).
        """
        return np.random.rand() < self.p

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución de Bernoulli.
        :return: Probabilidad de éxito p.
        """
        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.
        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max].
        :param k: Número de brazos a generar.
        :param p_min: Valor mínimo de la probabilidad.
        :param p_max: Valor máximo de la probabilidad.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "Los valores de p_min y p_max deben estar en el rango [0,1]."
        
        # Generar k valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)
        
        return [ArmBernoulli(p) for p in p_values]


