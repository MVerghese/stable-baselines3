from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import os

from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.rlpd_sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.type_aliases import ReplayBufferSamples, DictReplayBufferSamples
from tqdm import tqdm

SelfSAC = TypeVar("SelfSAC", bound="SAC")

def load_offline_buffer(offline_buffer_path: str, device: Union[th.device, str], env: GymEnv) -> Union[ReplayBuffer, DictReplayBuffer]:
    if env is not None:
        is_dict = isinstance(env.observation_space, spaces.Dict)
        if is_dict:
            obs = np.load(os.path.join(offline_buffer_path, 'obs.npz'), allow_pickle=True)
            next_obs = np.load(os.path.join(offline_buffer_path, 'next_obs.npz'), allow_pickle=True)
        else:
            obs = np.load(os.path.join(offline_buffer_path, 'obs.npy'))
            next_obs = np.load(os.path.join(offline_buffer_path, 'next_obs.npy'))

        actions = np.load(os.path.join(offline_buffer_path, 'actions.npy'))
        rewards = np.load(os.path.join(offline_buffer_path, 'rewards.npy'))
        dones = np.load(os.path.join(offline_buffer_path, 'dones.npy'))
        buffer_size = actions.shape[0]

        if is_dict:
            offline_buffer = DictReplayBuffer(
                buffer_size=buffer_size,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                optimize_memory_usage=False,
            )
        else:
            offline_buffer = ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                optimize_memory_usage=False,
            )
        # Initialize the buffer with the loaded data. Insert a new dimension in position 1 to match the expected shape.
        if is_dict:
            for key in obs.keys():
                offline_buffer.observations[key] = obs[key][:, None, ...].astype(env.observation_space.spaces[key].dtype)
                offline_buffer.next_observations[key] = next_obs[key][:, None, ...].astype(env.observation_space.spaces[key].dtype)
        else:
            offline_buffer.observations = obs[:, None, ...].astype(env.observation_space.dtype)
            offline_buffer.next_observations = next_obs[:, None, ...].astype(env.observation_space.dtype)
        offline_buffer.actions = actions[:, None, ...].astype(env.action_space.dtype)
        offline_buffer.rewards = rewards[:, None, ...].astype(np.float32)
        offline_buffer.dones = dones[:, None, ...].astype(np.float32)
        offline_buffer.pos = buffer_size
        offline_buffer.full = True

        return offline_buffer
    else:
        # This case gets triggered during eval as the env is not stored in the checkpoint
        return None

def merge_buffer_samples(online_samples: Union[ReplayBufferSamples, DictReplayBufferSamples], offline_samples: Union[ReplayBufferSamples, DictReplayBufferSamples]) -> Union[ReplayBufferSamples, DictReplayBufferSamples]:
    is_dict = isinstance(online_samples, DictReplayBufferSamples) and isinstance(offline_samples, DictReplayBufferSamples)
    if is_dict:
        assert online_samples.observations.keys() == offline_samples.observations.keys(), "Observation keys do not match between online and offline buffers"
        merged_observations = {
            key: th.cat([online_samples.observations[key], offline_samples.observations[key]], dim=0).float()
            for key in online_samples.observations.keys()
        }
        merged_next_observations = {
            key: th.cat([online_samples.next_observations[key], offline_samples.next_observations[key]], dim=0).float()
            for key in online_samples.next_observations.keys()
        }
        merged_samples = DictReplayBufferSamples(
            observations=merged_observations,
            actions=th.cat([online_samples.actions, offline_samples.actions], dim=0).float(),
            next_observations=merged_next_observations,
            dones=th.cat([online_samples.dones, offline_samples.dones], dim=0).float(),
            rewards=th.cat([online_samples.rewards, offline_samples.rewards], dim=0).float(),
            discounts=None if online_samples.discounts is None or offline_samples.discounts is None else th.cat([online_samples.discounts, offline_samples.discounts], dim=0).float()
        )
    else:
        merged_samples = ReplayBufferSamples(
            observations=th.cat([online_samples.observations, offline_samples.observations], dim=0).float(),
            actions=th.cat([online_samples.actions, offline_samples.actions], dim=0).float(),
            next_observations=th.cat([online_samples.next_observations, offline_samples.next_observations], dim=0).float(),
            dones=th.cat([online_samples.dones, offline_samples.dones], dim=0).float(),
            rewards=th.cat([online_samples.rewards, offline_samples.rewards], dim=0).float(),
            discounts=None if online_samples.discounts is None or offline_samples.discounts is None else th.cat([online_samples.discounts, offline_samples.discounts], dim=0).float()
        )
    return merged_samples

class RLPD_SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        offline_ratio: float = 0.0,
        offline_buffer_path: Optional[str] = None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        print("Offline Ratio during initialization:", offline_ratio)
        self.offline_ratio = offline_ratio
        self.offline_buffer_path = offline_buffer_path

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

        print("Offline ratio:", self.offline_ratio)
        print("Offline buffer path:", self.offline_buffer_path)
        print("device:", self.device)
        print("env:", self.env)
        self.offline_buffer = load_offline_buffer(self.offline_buffer_path, self.device, self.env) if self.offline_ratio > 0 else None

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64, only_critic: bool = False) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        with tqdm(range(gradient_steps), desc='Batch') as tqdm_steps:
            for gradient_step in tqdm_steps:
                # Sample replay buffer
                if self.offline_ratio > 0:
                    if self.replay_buffer.pos == 0:
                        # This handles the case where pretraining is performed on the offline buffer before any samples are collected online
                        replay_data = self.offline_buffer.sample(batch_size, env=self._vec_normalize_env)
                    else:
                        replay_data = self.replay_buffer.sample(int(batch_size * (1 - self.offline_ratio)), env=self._vec_normalize_env)  # type: ignore[union-attr]
                        offline_data = self.offline_buffer.sample(int(batch_size * self.offline_ratio), env=self._vec_normalize_env)  # type: ignore[union-attr]
                        replay_data = merge_buffer_samples(replay_data, offline_data)
                    # For n-step replay, discount factor is gamma**n_steps (when no early termination)
                else:
                    replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
                discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                # We need to sample because `log_std` may have changed between two gradient steps
                if self.use_sde:
                    self.actor.reset_noise()

                # Action by the current actor for the sampled state
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    # Important: detach the variable from the graph
                    # so we don't change it with other losses
                    # see https://github.com/rail-berkeley/softlearning/issues/60
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    assert isinstance(self.target_entropy, float)
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor

                ent_coefs.append(ent_coef.item())

                # Optimize entropy coefficient, also called
                # entropy temperature or alpha in the paper
                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    # if gradient_step % 100 == 0:
                    #     print(f"Gradient Step: {gradient_step}, Action Magnitude: {th.norm(next_actions[:,:3], dim=-1).mean().item()}")
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
                current_q_values = self.critic(replay_data.observations, replay_data.actions)

                # Compute critic loss
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                assert isinstance(critic_loss, th.Tensor)  # for type checker
                critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

                # Optimize the critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()


                # Compute actor loss
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Min over all critic networks
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                if not only_critic:
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()

                tqdm_steps.set_postfix(critic_loss=critic_loss.item(), actor_loss=actor_loss.item())


                # Update target networks
                if gradient_step % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    # Copy running stats, see GH issue #996
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
