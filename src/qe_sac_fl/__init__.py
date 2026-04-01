"""
QE-SAC-FL: Federated Quantum Reinforcement Learning for Volt-VAR Control

Extends QE-SAC+ with federated learning across multiple utilities.
Each utility keeps its encoder LOCAL (private grid data never shared).
Only the VQC weights (16 parameters) are shared via FedAvg.

Structure:
    fed_config.py       — FL hyperparameters and client registry
    env_34bus.py        — IEEE 34-bus VVC environment (Client B)
    federated_trainer.py — FedAvg loop over VQC weights

Depends on: src/qe_sac/  (reuses all existing agents and environments)
"""
