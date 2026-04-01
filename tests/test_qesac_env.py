"""
Tests for QE-SAC components:
  - VVC environments (13-bus, 123-bus)
  - Classical Autoencoder (CAE)
  - Variational Quantum Circuit (VQC)
  - Classical SAC agent
  - QE-SAC agent (full integration)
"""

import numpy as np
import torch
import pytest

from src.qe_sac.env_utils import VVCEnv13Bus, VVCEnv123Bus
from src.qe_sac.autoencoder import CAE, train_cae
from src.qe_sac.vqc import VQCLayer, N_QUBITS
from src.qe_sac.metrics import count_parameters
from src.qe_sac.sac_baseline import ClassicalSACAgent
from src.qe_sac.qe_sac_policy import QESACActorNetwork, QESACAgent, GNNQESACActorNetwork, GNNQESACAgent
from src.qe_sac.constrained_sac import QESACAgentConstrained
from src.qe_sac.gnn_encoder import GNNEncoder, train_gnn_encoder


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestVVCEnv13Bus:
    def setup_method(self):
        self.env = VVCEnv13Bus(seed=0)

    def test_observation_space(self):
        obs, _ = self.env.reset()
        assert obs.shape == self.env.observation_space.shape
        # 13 buses × 3 (V, P, Q) + 2 caps + 1 reg = 42
        assert obs.shape[0] == 42

    def test_action_space(self):
        # 2 caps (0/1) + 1 reg (0..32) = MultiDiscrete([2,2,33])
        assert len(self.env.action_space.nvec) == 3

    def test_step_returns_correct_types(self):
        self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "v_viol" in info
        assert "voltage" in info

    def test_episode_terminates(self):
        obs, _ = self.env.reset()
        done = False
        steps = 0
        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            steps += 1
            assert steps <= 25  # episode_len=24
        assert steps == 24

    def test_voltages_in_plausible_range(self):
        self.env.reset()
        for _ in range(5):
            action = self.env.action_space.sample()
            _, _, _, _, info = self.env.step(action)
            V = info["voltage"]
            assert np.all(V >= 0.5) and np.all(V <= 1.5)


class TestVVCEnv123Bus:
    def test_observation_shape(self):
        env = VVCEnv123Bus(seed=0)
        obs, _ = env.reset()
        # 123 × 3 + 7 caps + 4 regs = 380
        assert obs.shape[0] == 380

    def test_step_runs(self):
        env = VVCEnv123Bus(seed=1)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape[0] == 380


# ---------------------------------------------------------------------------
# CAE tests
# ---------------------------------------------------------------------------

class TestCAE:
    def test_latent_dimension(self):
        cae = CAE(input_dim=42)
        x = torch.randn(4, 42)
        _, z = cae(x)
        assert z.shape == (4, 8)

    def test_encode_range(self):
        cae = CAE(input_dim=42)
        x = torch.randn(16, 42)
        z = cae.encode(x)
        # tanh * pi → all values in (-π, π)
        assert z.abs().max().item() <= torch.pi + 1e-4

    def test_reconstruction_loss_decreases(self):
        cae = CAE(input_dim=42)
        data = np.random.randn(200, 42).astype(np.float32)
        loss_initial = train_cae(cae, data, n_steps=1)
        loss_final   = train_cae(cae, data, n_steps=100)
        assert loss_final <= loss_initial + 0.5   # allow some slack

    def test_forward_shape(self):
        cae = CAE(input_dim=42)
        x = torch.randn(8, 42)
        x_hat, z = cae(x)
        assert x_hat.shape == (8, 42)
        assert z.shape == (8, 8)


# ---------------------------------------------------------------------------
# VQC tests
# ---------------------------------------------------------------------------

class TestVQCLayer:
    def test_output_shape_single(self):
        vqc = VQCLayer()
        x = torch.randn(N_QUBITS)
        out = vqc(x)
        assert out.shape == (N_QUBITS,)

    def test_output_shape_batched(self):
        vqc = VQCLayer()
        x = torch.randn(3, N_QUBITS)
        out = vqc(x)
        assert out.shape == (3, N_QUBITS)

    def test_output_range(self):
        vqc = VQCLayer()
        x = torch.randn(4, N_QUBITS)
        out = vqc(x)
        # PauliZ expectation values are in [-1, 1]
        assert out.abs().max().item() <= 1.0 + 1e-4

    def test_parameter_count(self):
        vqc = VQCLayer()
        # N_LAYERS × N_QUBITS = 2 × 8 = 16 trainable weights
        assert vqc.n_params == 16

    def test_gradients_exist(self):
        vqc = VQCLayer()
        x = torch.randn(N_QUBITS, requires_grad=False)
        out = vqc(x)
        loss = out.sum()
        loss.backward()
        assert vqc.weights.grad is not None
        assert vqc.weights.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Classical SAC agent test
# ---------------------------------------------------------------------------

class TestClassicalSACAgent:
    def test_param_count_order_of_magnitude(self):
        env = VVCEnv13Bus(seed=0)
        obs_dim  = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.sum())
        agent = ClassicalSACAgent(obs_dim=obs_dim, n_actions=n_actions)
        # Expect on the order of hundreds of thousands of params
        assert agent.param_count() > 10_000

    def test_select_action_shape(self):
        env = VVCEnv13Bus(seed=0)
        obs_dim   = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.sum())
        agent = ClassicalSACAgent(obs_dim=obs_dim, n_actions=n_actions)
        obs, _ = env.reset()
        action = agent.select_action(torch.tensor(obs))
        assert action.shape == ()  # scalar index

    def test_update_after_warmup(self):
        env = VVCEnv13Bus(seed=0)
        obs_dim   = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.sum())
        agent = ClassicalSACAgent(obs_dim=obs_dim, n_actions=n_actions, buffer_size=1000)
        obs, _ = env.reset()
        # Fill buffer
        for _ in range(300):
            action = env.action_space.sample()
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.store(obs, action[0], reward, next_obs, term or trunc)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
        logs = agent.update(batch_size=64)
        assert "actor_loss" in logs
        assert "critic_loss" in logs


# ---------------------------------------------------------------------------
# QE-SAC actor & agent tests
# ---------------------------------------------------------------------------

class TestQESACActorNetwork:
    def test_forward_shape(self):
        obs_dim   = 42
        n_actions = 37  # 2+2+33 for 13-bus
        actor = QESACActorNetwork(obs_dim=obs_dim, n_actions=n_actions)
        obs = torch.randn(obs_dim)
        probs = actor(obs)
        assert probs.shape == (n_actions,)

    def test_probabilities_sum_to_one(self):
        actor = QESACActorNetwork(obs_dim=42, n_actions=37)
        obs = torch.randn(4, 42)
        probs = actor(obs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_param_count_small(self):
        actor = QESACActorNetwork(obs_dim=42, n_actions=37)
        n = count_parameters(actor)
        # Should be much fewer than classical SAC (899K)
        # CAE: ~42*64 + 64*32 + 32*8 + 8*32 + 32*64 + 64*42 ≈ 11K
        # VQC: 16 params
        # head: 8*37 + 37 ≈ 333
        assert n < 50_000


class TestQESACAgent:
    def test_store_and_update(self):
        env = VVCEnv13Bus(seed=0)
        obs_dim   = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.sum())
        agent = QESACAgent(obs_dim=obs_dim, n_actions=n_actions, buffer_size=500)
        obs, _ = env.reset()
        for _ in range(300):
            action = env.action_space.sample()
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.store(obs, action[0], reward, next_obs, term or trunc)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
        logs = agent.update(batch_size=32, cae_update_interval=9999)
        assert "actor_loss" in logs

    def test_param_count_much_smaller_than_classical(self):
        env = VVCEnv13Bus(seed=0)
        obs_dim   = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.sum())
        qe_agent  = QESACAgent(obs_dim=obs_dim, n_actions=n_actions)
        cl_agent  = ClassicalSACAgent(obs_dim=obs_dim, n_actions=n_actions)
        assert qe_agent.param_count() < cl_agent.param_count()


class TestQESACAgentConstrained:
    """Task 2.5 — unit tests for Lagrangian constrained SAC."""

    def _make_agent(self):
        env = VVCEnv13Bus(seed=0)
        obs_dim   = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.sum())
        agent = QESACAgentConstrained(
            obs_dim=obs_dim, n_actions=n_actions,
            buffer_size=500, lambda_lr=0.01,
        )
        return agent, env

    def test_lambda_starts_at_zero(self):
        agent, _ = self._make_agent()
        assert agent.lagrange_lambda == 0.0

    def test_lambda_increases_when_vviol_positive(self):
        agent, _ = self._make_agent()
        agent.update_lambda(mean_vviol=5.0)
        assert agent.lagrange_lambda > 0.0

    def test_lambda_never_goes_below_zero(self):
        agent, _ = self._make_agent()
        agent.lagrange_lambda = 0.0
        agent.update_lambda(mean_vviol=-100.0)
        assert agent.lagrange_lambda == 0.0

    def test_lambda_decreases_when_no_violation(self):
        agent, _ = self._make_agent()
        agent.lagrange_lambda = 1.0
        agent.update_lambda(mean_vviol=0.0)
        assert agent.lagrange_lambda <= 1.0

    def test_actor_loss_includes_lambda_term(self):
        agent, env = self._make_agent()
        obs, _ = env.reset()
        # Fill buffer with some transitions including vviol
        for _ in range(400):
            action = env.action_space.sample()
            next_obs, reward, term, trunc, info = env.step(action)
            agent.store(obs, action[0], reward, next_obs, term or trunc,
                        v_viol=info.get("v_viol", 0))
            obs = next_obs if not (term or trunc) else env.reset()[0]
        agent.lagrange_lambda = 1.0
        logs = agent.update(batch_size=32, cae_update_interval=9999)
        assert "actor_loss" in logs
        assert "constraint_violation" in logs
        assert "lagrange_lambda" in logs

    def test_vviol_buffer_stores_correctly(self):
        agent, env = self._make_agent()
        obs, _ = env.reset()
        next_obs, _, term, trunc, _ = env.step(env.action_space.sample())
        agent.store(obs, 0, -1.0, next_obs, False, v_viol=3.0)
        last_idx = (agent._ptr - 1 + agent._max) % agent._max
        assert agent._buf_vviol_c[last_idx] == 3.0

    def test_save_and_load_preserves_lambda(self, tmp_path):
        agent, _ = self._make_agent()
        agent.lagrange_lambda = 2.718
        path = str(tmp_path / "constrained.pt")
        agent.save(path)
        agent2, _ = self._make_agent()
        agent2.load(path)
        assert abs(agent2.lagrange_lambda - 2.718) < 1e-6


# ---------------------------------------------------------------------------
# GNN Encoder tests (Task 3)
# ---------------------------------------------------------------------------

class TestGNNEncoder:
    def test_latent_dimension(self):
        enc = GNNEncoder()
        obs = torch.randn(42)
        latent = enc.encode(obs)
        assert latent.shape == (8,)

    def test_latent_range(self):
        enc = GNNEncoder()
        obs = torch.randn(16, 42)
        latent = enc.encode(obs)
        # tanh * pi → all values in (-π, π)
        assert latent.abs().max().item() <= torch.pi + 1e-4

    def test_batched_encode(self):
        enc = GNNEncoder()
        obs = torch.randn(4, 42)
        latent = enc.encode(obs)
        assert latent.shape == (4, 8)

    def test_forward_cae_interface(self):
        """GNN encoder exposes same .forward() interface as CAE."""
        enc = GNNEncoder()
        obs = torch.randn(8, 42)
        recon, z = enc(obs)
        assert recon.shape == obs.shape
        assert z.shape == (8, 8)

    def test_param_count_smaller_than_cae(self):
        from src.qe_sac.autoencoder import CAE
        gnn = GNNEncoder()
        cae = CAE(input_dim=42)
        gnn_params = sum(p.numel() for p in gnn.parameters())
        cae_params = sum(p.numel() for p in cae.parameters())
        assert gnn_params < cae_params

    def test_gradients_flow(self):
        enc = GNNEncoder()
        obs = torch.randn(4, 42)
        latent = enc.encode(obs)
        loss = latent.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.grad is not None:
                return  # at least one gradient exists
        pytest.fail("No gradients in GNN encoder")

    def test_train_gnn_encoder_runs(self):
        enc = GNNEncoder()
        data = np.random.randn(100, 42).astype(np.float32)
        loss = train_gnn_encoder(enc, data, n_steps=10)
        assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# GNN QE-SAC actor & agent tests (Task 3)
# ---------------------------------------------------------------------------

class TestGNNQESACActorNetwork:
    def test_forward_shape(self):
        actor = GNNQESACActorNetwork(obs_dim=42, n_actions=132)
        obs = torch.randn(42)
        probs = actor(obs)
        assert probs.shape == (132,)

    def test_probabilities_sum_to_one(self):
        actor = GNNQESACActorNetwork(obs_dim=42, n_actions=132)
        obs = torch.randn(4, 42)
        probs = actor(obs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_param_count_smaller_than_mlp_actor(self):
        gnn_actor = GNNQESACActorNetwork(obs_dim=42, n_actions=132)
        mlp_actor = QESACActorNetwork(obs_dim=42, n_actions=132)
        assert count_parameters(gnn_actor) < count_parameters(mlp_actor)


class TestGNNQESACAgent:
    def test_store_and_update(self):
        env = VVCEnv13Bus(seed=0)
        obs_dim   = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.prod())
        agent = GNNQESACAgent(obs_dim=obs_dim, n_actions=n_actions, buffer_size=500)
        obs, _ = env.reset()
        for _ in range(300):
            action = env.action_space.sample()
            next_obs, reward, term, trunc, _ = env.step(action)
            flat = int(action[0]) * 66 + int(action[1]) * 33 + int(action[2])
            agent.store(obs, flat, reward, next_obs, term or trunc)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
        logs = agent.update(batch_size=32, cae_update_interval=9999)
        assert "actor_loss" in logs

    def test_param_count_smaller_than_cae_agent(self):
        env = VVCEnv13Bus(seed=0)
        obs_dim   = env.observation_space.shape[0]
        n_actions = int(env.action_space.nvec.prod())
        gnn_agent = GNNQESACAgent(obs_dim=obs_dim, n_actions=n_actions)
        cae_agent = QESACAgent(obs_dim=obs_dim, n_actions=n_actions)
        assert gnn_agent.param_count() < cae_agent.param_count()
