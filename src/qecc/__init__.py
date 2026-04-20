# __init__.py
"""
QECC (Quantum Error Correction Code) package initialization.
"""

__version__ = '0.1.2'
__author__ = 'Omer Sella'
__all__ = ["memBP", "minSum", "polynomialCodes", "funWithMatrices", "osd", "logicals", "polynomialCodes", "gf4", "solver", "arithmetic"]
PACKAGE_NAME = "qecc"

from gymnasium.envs.registration import register

register(
    id="bbcode-v0",
    entry_point="qecc.bb_gym:bb_env",
)