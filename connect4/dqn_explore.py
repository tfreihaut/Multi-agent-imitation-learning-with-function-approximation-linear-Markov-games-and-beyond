# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use free GPUs

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import copy
from tqdm import tqdm
import logging
from datetime import datetime
import time


import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from connect4.bc import BehavioralCloningSingleAgent
from bitbully import Board, BitBully

from connect4.utils import format_time, Connect4Dataset, ReplayBuffer, TrajectoryBuffer

# Define the structure for storing transitions in the replay buffer
Transition = namedtuple(
    "Transition", ("state", "action", "expert_action", "reward", "next_state")
)


class DQN(nn.Module):
    """
    Optimized Late Fusion DQN
    """

    def __init__(
        self, height, width, in_channels, n_actions, hidden_dim=512, tau=0.005, lr=1e-3
    ):
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.tau = tau

        # --- FIX: Add padding=1 to preserve dimensions ---
        # Input: (C, H, W) -> Output: (32, H, W)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        # Input: (32, H, W) -> Output: (64, H, W)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Input: (64, H, W) -> Output: (128, H, W)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # --- FIX: Calculate Linear Input Size ---
        # Since we use padding=1, the spatial size (H, W) does NOT change.
        # It remains 'image_size' (e.g., 3 for TicTacToe).
        self.feature_size = height * width * 128

        self.fc1 = nn.Linear(self.feature_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + n_actions, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # --- Optimization 1: Persistent Optimizer ---
        # We create the optimizer ONCE. Creating it inside 'train' resets momentum every time!
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _forward_until_action(self, state):
        """Runs the expensive CNN part"""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, state, action):
        """
        Standard forward pass for training (Batch of States + Specific Actions)
        """
        x = self._forward_until_action(state)
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_inner_layers_embedding(self, state, action):
        """
        Restored Helper: Returns the features (phi) used for the Lambda matrix.
        Output is the input to the final Linear layer.
        """
        x = self._forward_until_action(state)
        x = torch.cat((x, action), dim=1)
        # We apply ReLU here so 'phi' represents the actual features seen by the final layer
        x = F.relu(self.fc2(x))
        return x

    def get_all_q_values(self, state):
        """
        get all q_values for a batch of states
        """
        batch_size = state.size(0)

        # 1. Run CNN only ONCE per state (Expensive part)
        # Shape: (batch_size, hidden_dim)
        state_features = self._forward_until_action(state)

        # 2. Expand features for broadcasting
        # Shape: (batch_size, n_actions, hidden_dim)
        state_features_expanded = state_features.unsqueeze(1).expand(
            -1, self.n_actions, -1
        )

        # 3. Create all possible actions batch
        # Shape: (batch_size, n_actions, n_actions)
        all_actions = (
            torch.eye(self.n_actions, device=state.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # 4. Concatenate state features and actions
        # Shape: (batch_size, n_actions, hidden_dim + n_actions)
        combined = torch.cat((state_features_expanded, all_actions), dim=2)

        # 5. Pass through remaining small layers
        x = F.relu(self.fc2(combined))
        q_values = self.fc3(x)  # Shape: (batch_size, n_actions, 1)

        return q_values.squeeze(2)

    def train_on_replay_buffer(
        self,
        replay_buffer,
        target_network,
        epochs=1,
        batch_size=64,
        gamma=0.9,
        device="cuda",
    ):
        # loss_fn = nn.MSELoss()
        loss_fn = (
            nn.SmoothL1Loss()
        )  # Huber Loss: less sensitive to outliers (other libraries use this)

        # Ensure models are on the correct device
        self.to(device)
        target_network.to(device)

        for epoch in range(epochs):
            if len(replay_buffer) < batch_size:
                return  # Exit early if not enough data

            # 1. Sampling
            transitions = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            # 2. Stack and Move to GPU (Optimization: non_blocking=True)
            # Assuming 'Transition' creates tuples/lists. We stack them into tensors here.
            state_batch = torch.stack(batch.state).to(device, non_blocking=True)
            next_state_batch = torch.stack(batch.next_state).to(
                device, non_blocking=True
            )
            reward_batch = (
                torch.stack(batch.reward).to(device, non_blocking=True).unsqueeze(1)
            )
            action_indices = torch.stack(batch.action).to(device, non_blocking=True)

            # Convert action indices to One-Hot
            action_batch = F.one_hot(action_indices, num_classes=self.n_actions).float()

            # 3. Forward Pass (Agent)
            q_values = self(state_batch, action_batch)

            # 4. Target Calculation (Optimized with get_all_q_values)
            with torch.no_grad():
                # This uses the FAST version now
                next_q_values_all = target_network.get_all_q_values(next_state_batch)
                next_q_values, _ = torch.max(next_q_values_all, dim=1, keepdim=True)
                target_q_values = reward_batch + (gamma * next_q_values)

            # 5. Loss & Optimize
            loss = loss_fn(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 6. Soft Update
            for target_param, local_param in zip(
                target_network.parameters(), self.parameters()
            ):
                target_param.data.lerp_(local_param.data, self.tau)


class DeepWARMMAIL:
    def __init__(
        self,
        K,
        num_actions_p1,
        num_actions_p2,
        height,
        width,
        in_channels,
        env,
        dqn_hidden_dim=512,
        replay_buffer_capacity=10000,
        batch_size=64,
        gamma=0.9,
        lr=1e-3,
        tau=0.005,
        beta=10.0,  # regularization parameter
        temperature=1.0,  # exploration parameter,
        bc_cnn=False,
        device="cpu",
        save_buffers=False,
    ):
        self.K = K
        self.num_actions_p1 = num_actions_p1
        self.num_actions_p2 = num_actions_p2
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.beta = beta
        self.temperature = temperature
        self.device = device
        self.feature_dim = dqn_hidden_dim
        self.env = env
        self.target_size = (height, width)  # Target size for image resizing
        self.save_buffers = save_buffers

        self.dqn_p2 = DQN(
            height, width, in_channels, num_actions_p2, dqn_hidden_dim
        ).to(device)
        self.dqn_target_p2 = copy.deepcopy(self.dqn_p2).to(device)
        self.dqn_optimizer_p2 = optim.Adam(self.dqn_p2.parameters(), lr=lr)
        self.dqn_p2 = torch.compile(self.dqn_p2, mode="reduce-overhead")

        # Replay buffers for both players (with capacity for training)
        self.replay_buffer_p2 = ReplayBuffer(replay_buffer_capacity)

        # Trajectory buffers for both players (without capacity, stores all exploration data)
        self.trajectory_buffer_exp_1 = TrajectoryBuffer(player_id=1)

    def _get_reward(self, lambda_matrix_inv, dqn):
        """Compute the reward function using the pre-computed inverse of lambda."""

        def reward_function(state, action):
            with torch.no_grad():
                inner_layers = dqn._get_inner_layers_embedding(state, action)
                # Use torch.einsum for a batched dot product, which is more robust
                reward = torch.sqrt(
                    torch.einsum(
                        "bi,ij,bj->b", inner_layers, lambda_matrix_inv, inner_layers
                    )
                )
            return reward

        return reward_function

    def run(
        self,
        horizon,
        lambda_reg=0.1,
        epochs=1000,
        gradient_updates=10,
        t_max=1000,
        logger=None,
        dataset_sizes=None,
    ):
        """Main training loop for Deep WARM-MAIL.

        Args:
            horizon: Episode horizon
            lambda_reg: Regularization parameter (unused)
            epochs: Number of BC training epochs
            t_max: Max iterations per WARM-MAIL iteration
            logger: Logger instance
            dataset_sizes: List of dataset sizes to train BC on (e.g., [1, 50, 100, 500]).
                          If None, trains once on all collected data.

        Returns:
            If dataset_sizes is None:
                ((hat_mu, hat_nu), (loss_mu, loss_nu), (num_traj_mu, num_traj_nu))
            If dataset_sizes is provided:
                (policies_list, losses_list, trajectory_counts_list)
                where each element corresponds to a dataset size
        """
        if logger is None:
            logger = logging.getLogger("DQN-Explore-BC")

        # Start overall timing
        overall_start_time = time.time()

        logger.info("=" * 80)
        logger.info("Starting DQN-Explore-BC Training")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  K (DQN-Explore iterations): {self.K}")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  BC Epochs: {epochs}")
        logger.info(f"  t_max per iteration: {t_max}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.lr}")
        logger.info(f"  Gradient updates per iteration: {gradient_updates}")
        logger.info(f"  Gamma (discount): {self.gamma}")
        logger.info(f"  Beta (exploration): {self.beta}")
        logger.info(f"  Temperature (exploration): {self.temperature}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Image size: {self.height}x{self.width}")
        logger.info(f"  Feature dim: {self.feature_dim}")
        if dataset_sizes is not None:
            logger.info(f"  Dataset sizes for BC: {dataset_sizes}")
        logger.info("")

        # run exploration for the second player
        logger.info("=" * 80)
        logger.info("PHASE 1: Training Player 1 (Expert is Player 2)")
        logger.info("=" * 80)
        lambda_matrix_p2 = torch.eye(self.feature_dim, device=self.device) * self.beta

        # Track time for Player 2 phase
        p2_start_time = time.time()
        p2_iteration_times = []

        # Load expert agent once
        bb_agent = BitBully()

        # all visited states
        for k in tqdm(range(self.K), desc="Player 1 Training"):
            iteration_start_time = time.time()

            logger.info(f"\n--- Player 1: DQN-Explore Iteration {k + 1}/{self.K} ---")
            old_dqn_p2 = copy.deepcopy(
                self.dqn_p2
            )  # Keep a frozen copy for feature extraction
            lambda_inv_p2 = torch.linalg.inv(lambda_matrix_p2)
            # current_inv = torch.linalg.inv(lambda_matrix_p2)
            reward_function_p2 = self._get_reward(lambda_inv_p2, old_dqn_p2)
            # reward_function_p2 = self._get_reward(current_inv.clone(), old_dqn_p2)

            lambda_new_p2 = lambda_matrix_p2.clone()
            lambda_logdet_p2 = torch.logdet(lambda_matrix_p2).item()
            current_logdet = copy.deepcopy(lambda_logdet_p2)
            logger.info(f"Initial lambda log-determinant: {lambda_logdet_p2:.4f}")

            # Update progress bar only every 10% to prevent terminal spam
            update_interval = max(1, int(t_max * 0.1))
            pbar = tqdm(
                total=t_max,
                desc=f"P1 Epoch {k + 1}/{self.K}",
                leave=False,
                miniters=update_interval,
                mininterval=60.0,
            )
            iter_count = 0
            trajectories_collected = 0
            while iter_count < t_max and current_logdet <= lambda_logdet_p2 + np.log(2):
                new_trajectory_features = []
                trajectories_collected += 1
                # Petting Zoo environment logic should be used here
                self.env.reset()
                # Initialize BitBully expert board
                bb_board = Board()
                for _ in range(horizon):
                    for agent in self.env.agent_iter():
                        state, _, termination, truncation, info = self.env.last()
                        if termination or truncation:
                            self.env.reset()
                            bb_board = Board()
                            # End the current trajectory
                            self.trajectory_buffer_exp_1.end_trajectory()
                            break
                        # Player 1's turn, expert behavior
                        if agent == "player_0":
                            # get expert action for player 1
                            scores = bb_agent.score_next_moves(bb_board)
                            action_p1 = int(np.argmax(scores))

                            # Save preprocessed state and action for trajectory buffer
                            state_preprocessed_1 = (
                                torch.from_numpy(state["observation"])
                                .to(dtype=torch.float32, device=self.device)
                                .permute(2, 0, 1)
                            )
                            action_p1_tensor = torch.tensor(
                                action_p1, dtype=torch.long, device=self.device
                            )

                            self.trajectory_buffer_exp_1.push(
                                state_preprocessed_1,
                                None,  # action (not needed for player 1)
                                action_p1_tensor,
                                None,
                                None,
                            )
                            # Play action on BitBully board
                            bb_board.play(action_p1)

                            # Get next state after player 1's action
                            self.env.step(action_p1)

                        elif agent == "player_1":
                            # Convert to tensor, TODO: check if we need to permute channels here
                            state_tensor = torch.from_numpy(state["observation"]).to(
                                dtype=torch.float32, device=self.device
                            )
                            state_tensor = state_tensor.permute(2, 0, 1)

                            state_tensor = state_tensor.unsqueeze(0)

                            with torch.no_grad():
                                # get q value and use softmax policy to sample action for player 1
                                q_values = self.dqn_p2.get_all_q_values(
                                    state_tensor
                                ).squeeze(0)

                                # Check for NaN or Inf in Q-values
                                if (
                                    torch.isnan(q_values).any()
                                    or torch.isinf(q_values).any()
                                ):
                                    print(
                                        f"Warning: NaN or Inf detected in Q-values at iteration {k + 1}/{self.K}, trajectory {trajectories_collected}"
                                    )
                                    print(f"Q-values: {q_values}")
                                    # Use uniform distribution as fallback
                                    action_p2_probs = (
                                        np.ones(self.num_actions_p2)
                                        / self.num_actions_p2
                                    )
                                else:
                                    # Clip Q-values to prevent overflow in softmax
                                    q_values = torch.clamp(
                                        q_values / self.temperature, min=-50, max=50
                                    )
                                    action_p2_probs = (
                                        F.softmax(q_values, dim=0).cpu().numpy()
                                    )

                                    # Additional safety check
                                    if np.isnan(action_p2_probs).any():
                                        print(
                                            f"Warning: NaN in softmax output at iteration {k + 1}/{self.K}, trajectory {trajectories_collected}"
                                        )
                                        action_p2_probs = (
                                            np.ones(self.num_actions_p2)
                                            / self.num_actions_p2
                                        )

                            # Ensure that action played is legal:
                            action_p2_probs = state["action_mask"] * action_p2_probs
                            action_p2_probs /= action_p2_probs.sum()
                            action_p2 = np.random.choice(
                                self.num_actions_p2, p=action_p2_probs
                            )
                            # compute reward (explorative reward)
                            action_tensor = (
                                F.one_hot(
                                    torch.tensor([action_p2]),
                                    num_classes=self.num_actions_p2,
                                )
                                .float()
                                .to(self.device)
                            )
                            reward = reward_function_p2(
                                state_tensor, action_tensor
                            ).item()
                            with torch.no_grad():
                                phi = old_dqn_p2._get_inner_layers_embedding(
                                    state_tensor, action_tensor
                                )

                                # # We divide by sqrt(dim) so the dot product is roughly 1.0, not huge.
                                # phi_norm = phi / (torch.norm(phi) + 1e-8)
                                new_trajectory_features.append(phi)

                            # Prepare transition data (reuse state_tensor, but remove batch dim for storage)
                            state_preprocessed_2 = state_tensor.squeeze(0)
                            bb_board.play(action_p2)
                            self.env.step(action_p2)

                            action_p2_tensor = torch.tensor(
                                action_p2, dtype=torch.long, device=self.device
                            )
                            reward_tensor = torch.tensor(
                                reward, dtype=torch.float32, device=self.device
                            )

                            next_state, _, _, _, _ = self.env.last()

                            next_state_preprocessed = (
                                torch.from_numpy(next_state["observation"])
                                .to(self.device, dtype=torch.float32)
                                .permute(2, 0, 1)
                            )
                            # Store transition in replay buffer (for DQN training)
                            self.replay_buffer_p2.push(
                                state_preprocessed_2,
                                action_p2_tensor,  #
                                None,
                                reward_tensor,
                                next_state_preprocessed,
                            )

                # Update DQN for player 2
                self.dqn_p2.train_on_replay_buffer(
                    self.replay_buffer_p2,
                    self.dqn_target_p2,
                    epochs=gradient_updates,
                    batch_size=self.batch_size,
                    device=self.device,
                )

                # Update lambda matrix with features from the new trajectory
                if new_trajectory_features:
                    phi_batch = torch.cat(new_trajectory_features, dim=0)
                    lambda_new_p2 += torch.matmul(phi_batch.T, phi_batch)

                iter_count += 1
                pbar.update(1)
                current_logdet = torch.logdet(lambda_new_p2).item()

                # Safety: If logdet explodes to inf, break immediately also if nan
                if np.isinf(current_logdet) or np.isnan(current_logdet):
                    logger.warning(
                        "Log-determinant hit Infinity or NaN. Stopping epoch early."
                    )
                    break

                pbar.set_postfix(
                    {"logdet_diff": f"{current_logdet - lambda_logdet_p2:.2f}"},
                    refresh=False,
                )
            pbar.close()

            final_logdet_p1 = torch.logdet(lambda_new_p2).item()
            logger.info(f"Trajectories collected: {trajectories_collected}")
            logger.info(f"Final lambda log-determinant: {final_logdet_p1:.4f}")
            logger.info(
                f"Log-determinant increase: {final_logdet_p1 - lambda_logdet_p2:.4f}"
            )
            logger.info(f"Replay buffer size: {len(self.replay_buffer_p2)}")

            # Track iteration time and estimate remaining time
            iteration_time = time.time() - iteration_start_time
            p2_iteration_times.append(iteration_time)
            logger.info(f"Iteration time: {format_time(iteration_time)}")

            if len(p2_iteration_times) > 0:
                avg_iteration_time = np.mean(p2_iteration_times)
                remaining_iterations = self.K - (k + 1)
                estimated_remaining = avg_iteration_time * remaining_iterations
                if remaining_iterations > 0:
                    logger.info(
                        f"Estimated time remaining (Player 2): {format_time(estimated_remaining)}"
                    )

            # Rebuild lambda from the entire replay buffer using the updated network
            with torch.no_grad():
                all_transitions = self.replay_buffer_p2.memory
                if not all_transitions:
                    continue

                buffer_batch = Transition(*zip(*all_transitions))
                state_tensors = torch.stack(buffer_batch.state)
                action_tensors = F.one_hot(
                    torch.stack(buffer_batch.action), num_classes=self.num_actions_p2
                ).float()

                phi_vectors = self.dqn_p2._get_inner_layers_embedding(
                    state_tensors, action_tensors
                )
                lambda_matrix_p2 = (
                    torch.matmul(phi_vectors.T, phi_vectors)
                    + torch.eye(self.feature_dim, device=self.device) * self.beta
                )

        num_traj_exp1 = self.trajectory_buffer_exp_1.num_trajectories()

        # Save the data to file for verification
        logger.info(f"Total trajectories collected: Expert Player 1={num_traj_exp1}")
        # Save with timestamp the trajectories of player 1 for BC training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trajectory_buffer_exp_1.save_optimized(
            f"debug_trajectory_buffer_c4_{timestamp}_p2.npy"
        )

        p2_total_time = time.time() - p2_start_time
        logger.info("\nPlayer 2 training completed!")
        logger.info(f"Final replay buffer size: {len(self.replay_buffer_p2)}")
        logger.info(f"Total Player 2 training time: {format_time(p2_total_time)}")
        logger.info("")

        # NOTE: If one also wants to train a policy for the second player, one can revert the roles of player 1 and player 2, but same ideas

        logger.info("=" * 80)
        logger.info("PHASE 3: Behavioral Cloning")
        logger.info("=" * 80)

        bc_start_time = time.time()

        # Save the data to file for verification
        logger.info(f"Total trajectories collected: Expert Player 1={num_traj_exp1}")

        # If dataset_sizes is specified, train BC on multiple subsets
        if dataset_sizes is not None:
            policies_list = []
            losses_list = []
            trajectory_counts_list = []

            for size in dataset_sizes:
                logger.info(
                    f"\n--- Training BC with dataset size: {size} trajectories ---"
                )

                # Subset the data based on number of trajectories
                # Use the first 'size' trajectories from each player
                logger.info("Using CNN-based Behavioral Cloning Subset")
                # Player 1 subset
                trajectories_p1 = self.trajectory_buffer_exp_1.get_trajectories()
                D_nu_muE_subset = Connect4Dataset(trajectories_p1, limit_traj=size)

                logger.info(f"Subset samples: Player 1={len(D_nu_muE_subset)}")
                logger.info(f"Subset trajectories: {min(size, num_traj_exp1)}")

                # Train BC for Player 1
                learner_mu = BehavioralCloningSingleAgent(
                    num_actions=self.num_actions_p2,
                    cnn_policy=True,
                    in_channels=self.in_channels,
                    device=self.device,
                    lr=5e-3,
                )
                hat_mu, loss_mu = learner_mu.train(
                    D_nu_muE_subset, epochs=epochs, device=self.device
                )
                logger.info(
                    f"Player 2 BC training completed. Final loss: {loss_mu:.6f}"
                )

                policies_list.append(hat_mu)
                losses_list.append(loss_mu)
                trajectory_counts_list.append(len(D_nu_muE_subset))

            bc_total_time = time.time() - bc_start_time
            logger.info(
                f"\nTotal BC training time (all sizes): {format_time(bc_total_time)}"
            )
        else:
            logger.info("Using CNN-based Behavioral Cloning")

            # extract data from replay buffer to train behavioral cloning for player 1
            D_nu_muE = Connect4Dataset(trajectories_p1)

            num_traj_p1 = self.trajectory_buffer_exp_1.num_trajectories()
            logger.info(f"Total trajectories collected: Expert Player 1={num_traj_p1}")
            self.trajectory_buffer_exp_1.save("debug_trajectory_buffer_p2.npy")
            # Original behavior: train once on all data
            logger.info(f"Training BC for Player 1 with {len(D_nu_muE)} samples...")
            learner_mu = BehavioralCloningSingleAgent(
                num_actions=self.num_actions_p1,
                cnn_policy=True,
                in_channels=self.in_channels,
                device=self.device,
            )
            hat_mu, loss_mu = learner_mu.train(
                D_nu_muE, epochs=epochs, device=self.device
            )
            logger.info(f"Player 1 BC training completed. Final loss: {loss_mu:.6f}")

            bc_total_time = time.time() - bc_start_time
            logger.info(f"Total BC training time: {format_time(bc_total_time)}")

        # Calculate overall time
        overall_total_time = time.time() - overall_start_time

        logger.info("\n" + "=" * 80)
        logger.info("DQN-Explore-BC Training Completed Successfully!")
        logger.info("=" * 80)
        logger.info("Training Time Summary:")
        logger.info(
            f"  Player 2 training: {format_time(p2_total_time)} ({p2_total_time / overall_total_time * 100:.1f}%)"
        )
        logger.info(
            f"  BC training: {format_time(bc_total_time)} ({bc_total_time / overall_total_time * 100:.1f}%)"
        )
        logger.info(f"  Total time: {format_time(overall_total_time)}")
        logger.info("=" * 80 + "\n")

        # Return based on whether dataset_sizes was specified
        if dataset_sizes is not None:
            return policies_list, losses_list, trajectory_counts_list
        else:
            return hat_mu, loss_mu, len(D_nu_muE)
