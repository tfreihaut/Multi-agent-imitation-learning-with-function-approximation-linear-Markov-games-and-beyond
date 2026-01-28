import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm
import logging
from datetime import datetime
import time


import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from tictactoe.bc import BehavioralCloningSingleAgent
from tictactoe.utils import (
    ReplayBuffer,
    TrajectoryBuffer,
    obs_to_board,
    format_time,
    Transition,
    canonical,
    canonical_action_to_board_action,
    get_vectorized_policy,
)


class DQN(nn.Module):
    """
    Optimized Late Fusion DQN
    """

    def __init__(
        self, image_size, in_channels, n_actions, hidden_dim=512, tau=0.005, lr=1e-3
    ):
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.tau = tau

        # Input: (C, H, W) -> Output: (32, H, W)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        # Input: (32, H, W) -> Output: (64, H, W)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Input: (64, H, W) -> Output: (128, H, W)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.feature_size = image_size * image_size * 128

        self.fc1 = nn.Linear(self.feature_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + n_actions, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

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

        # Shape: (batch_size, hidden_dim)
        state_features = self._forward_until_action(state)

        # Shape: (batch_size, n_actions, hidden_dim)
        state_features_expanded = state_features.unsqueeze(1).expand(
            -1, self.n_actions, -1
        )

        # Shape: (batch_size, n_actions, n_actions)
        all_actions = (
            torch.eye(self.n_actions, device=state.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # Shape: (batch_size, n_actions, hidden_dim + n_actions)
        combined = torch.cat((state_features_expanded, all_actions), dim=2)

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

            # 4. Target Calculation
            with torch.no_grad():
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
        image_size,
        in_channels,
        env,
        dqn_hidden_dim=512,
        replay_buffer_capacity=1000,
        batch_size=64,
        gamma=0.9,
        lr=1e-3,
        tau=0.005,
        beta=10.0,  # regularization parameter
        temperature=1.0,  # exploration parameter,
        bc_cnn=False,
        device="cpu",
        save_buffers=False,
        expert=None,
    ):
        self.K = K
        self.num_actions_p1 = num_actions_p1
        self.num_actions_p2 = num_actions_p2
        self.image_size = image_size
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
        self.target_size = (image_size, image_size)  # Target size for image resizing
        self.bc_cnn = bc_cnn
        self.save_buffers = save_buffers

        self.dqn_p2 = DQN(image_size, in_channels, num_actions_p2, dqn_hidden_dim).to(
            device
        )
        self.dqn_target_p2 = copy.deepcopy(self.dqn_p2).to(device)
        self.dqn_optimizer_p2 = optim.Adam(self.dqn_p2.parameters(), lr=lr)
        self.dqn_p2 = torch.compile(self.dqn_p2, mode="reduce-overhead")

        # Replay buffers for explorin player 2 (with capacity for training)
        self.replay_buffer_p2 = ReplayBuffer(replay_buffer_capacity)

        # Trajectory buffers for expert player
        self.trajectory_buffer_exp_1 = TrajectoryBuffer(player_id=1)

        self.expert = expert  # expert policy for player 1

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
            gradient_updates: Number of gradient updates per DeepQ training iteration
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
            logger = logging.getLogger("DeepMAIL")

        # Start overall timing
        overall_start_time = time.time()

        logger.info("=" * 80)
        logger.info("Starting Deep WARM-MAIL Training")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  K (WARM-MAIL iterations): {self.K}")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  BC Epochs: {epochs}")
        logger.info(f"  Gradient updates per iteration: {gradient_updates}")
        logger.info(f"  t_max per iteration: {t_max}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.lr}")
        logger.info(f"  Gamma (discount): {self.gamma}")
        logger.info(f"  Beta (exploration): {self.beta}")
        logger.info(f"  Temperature (exploration): {self.temperature}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Image size: {self.image_size}")
        logger.info(f"  Feature dim: {self.feature_dim}")
        if dataset_sizes is not None:
            logger.info(f"  Dataset sizes for BC: {dataset_sizes}")
        logger.info("")

        # run exploration for the second player
        logger.info("=" * 80)
        logger.info("PHASE 2: Training Player 2 (Expert is Player 1)")
        logger.info("=" * 80)
        lambda_matrix_p2 = torch.eye(self.feature_dim, device=self.device) * self.beta

        # Track time for Player 2 phase
        p2_start_time = time.time()
        p2_iteration_times = []

        # all visited states
        states_visited_p2 = set()
        for k in tqdm(range(self.K), desc="Player 2 Training"):
            logger.info(
                f"Number of unique states visited so far by P2: {len(states_visited_p2)}"
            )
            iteration_start_time = time.time()

            if (k + 1) % 1000 == 0 or k == 0:
                logger.info(f"\n--- Player 2: WARM-MAIL Iteration {k + 1}/{self.K} ---")
            # 1. Pre-compute matrix inverse
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
                desc=f"P2 Epoch {k + 1}/{self.K}",
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

                for _ in range(horizon):
                    for agent in self.env.agent_iter():
                        state, _, termination, truncation, info = self.env.last()

                        if termination or truncation:
                            self.env.reset()
                            # End the current trajectory
                            self.trajectory_buffer_exp_1.end_trajectory()
                            # End loop if episode is done
                            break
                        # Player 1's turn, expert behavior
                        if agent == self.env.possible_agents[0]:  # player_1
                            # Convert PettingZoo observation to absolute board
                            board = obs_to_board(state["observation"], agent)

                            # Canonicalize
                            canon_board = canonical(board)
                            # Add to visited states
                            states_visited_p2.add(canon_board)

                            # Lookup perfect action
                            canon_action = self.expert[canon_board]

                            # Map back to PettingZoo action space
                            action_p1 = canonical_action_to_board_action(
                                board, canon_board, canon_action
                            )
                            action_p1_tensor = torch.tensor(
                                action_p1, dtype=torch.long, device=self.device
                            )
                            state_preprocessed_1 = (
                                torch.from_numpy(state["observation"])
                                .to(dtype=torch.float32, device=self.device)
                                .permute(2, 0, 1)
                            )

                            # Only push the relevant data for player 2's replay buffer here
                            # Store transition in trajectory buffer (for BC training later)
                            self.trajectory_buffer_exp_1.push(
                                state_preprocessed_1,
                                None,  # action (not needed for player 2)
                                action_p1_tensor,
                                None,
                                None,
                            )

                            # Get next state after player 1's action
                            self.env.step(action_p1)

                        elif agent == self.env.possible_agents[1]:  # player_2
                            state_tensor = torch.from_numpy(state["observation"]).to(
                                dtype=torch.float32, device=self.device
                            )
                            state_tensor = state_tensor.permute(2, 0, 1)

                            state_tensor = state_tensor.unsqueeze(0)

                            with torch.no_grad():
                                # get q value and use softmax policy to sample action for player 2
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
                                            np.ones(self.num_actions_p1)
                                            / self.num_actions_p1
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
                                new_trajectory_features.append(phi)

                            # Prepare transition data (reuse state_tensor, but remove batch dim for storage)
                            state_preprocessed_2 = state_tensor.squeeze(0)

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
                                None,  # expert action (not needed for DQN)
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
                    phi_batch = torch.cat(
                        new_trajectory_features, dim=0
                    )  # Shape: (N, dim)

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

            final_logdet_p2 = torch.logdet(lambda_new_p2).item()
            logger.info(f"Trajectories collected: {trajectories_collected}")
            logger.info(f"Final lambda log-determinant: {final_logdet_p2:.4f}")
            logger.info(
                f"Log-determinant increase: {final_logdet_p2 - lambda_logdet_p2:.4f}"
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

        p2_total_time = time.time() - p2_start_time
        logger.info("\nPlayer 2 training completed!")
        logger.info(f"Final replay buffer size: {len(self.replay_buffer_p2)}")
        logger.info(f"Total Player 2 training time: {format_time(p2_total_time)}")
        logger.info("")

        # extract data from replay buffer to train behavioral cloning for player 2
        logger.info("=" * 80)
        logger.info("PHASE 3: Behavioral Cloning")
        logger.info("=" * 80)

        bc_start_time = time.time()

        num_traj_exp_1 = self.trajectory_buffer_exp_1.num_trajectories()

        # Save the data to file for verification
        logger.info(f"Total trajectories collected: Expert Player 1={num_traj_exp_1}")
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trajectory_buffer_exp_1.save(f"debug_trajectory_buffer_{timestamp}_p1.npy")

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
                if self.bc_cnn:
                    logger.info("Using CNN-based Behavioral Cloning Subset")
                    # Player 1 subset
                    D_nu_muE_subset = []
                    trajectories_p1 = self.trajectory_buffer_exp_1.get_trajectories()
                    for i, trajectory in enumerate(trajectories_p1):
                        if i >= size:
                            break
                        for transition in trajectory:
                            state = transition.state
                            expert_action = transition.expert_action
                            D_nu_muE_subset.append((state, expert_action))

                else:
                    # Player 1 subset
                    D_nu_muE_subset = []
                    trajectories_p1 = self.trajectory_buffer_exp_1.get_trajectories()
                    for i, trajectory in enumerate(trajectories_p1):
                        if i >= size:
                            break
                        for transition in trajectory:
                            state = transition.state_idx
                            expert_action = transition.expert_action
                            D_nu_muE_subset.append((state, expert_action))

                logger.info(f"Subset samples: Player 1={len(D_nu_muE_subset)}")
                logger.info(f"Subset trajectories: {min(size, num_traj_exp_1)}")

                # Train BC for Player 1
                learner_mu = BehavioralCloningSingleAgent(
                    num_actions=self.num_actions_p1,
                    cnn_policy=self.bc_cnn,
                    in_channels=self.in_channels,
                    device=self.device,
                )
                hat_mu, loss_mu = learner_mu.train(
                    D_nu_muE_subset, epochs=epochs, device=self.device
                )
                logger.info(
                    f"Player 1 BC training completed. Final loss: {loss_mu:.6f}"
                )
                hat_mu = get_vectorized_policy(hat_mu, self.expert)

                policies_list.append(hat_mu)
                losses_list.append(loss_mu)
                trajectory_counts_list.append(len(D_nu_muE_subset))

            bc_total_time = time.time() - bc_start_time
            logger.info(
                f"\nTotal BC training time (all sizes): {format_time(bc_total_time)}"
            )
        else:
            if self.bc_cnn:
                logger.info("Using CNN-based Behavioral Cloning")

                # extract data from replay buffer to train behavioral cloning for player 1
                D_nu_muE = []
                for transition in self.trajectory_buffer_p2.iterator():
                    state = transition.state
                    expert_action = transition.expert_action
                    D_nu_muE.append((state, expert_action))

            else:
                logger.info("Using MLP-based Behavioral Cloning")

                # extract data from replay buffer to train behavioral cloning for player 1
                D_nu_muE = []
                for transition in self.trajectory_buffer_p2.iterator():
                    state = transition.state_idx
                    expert_action = transition.expert_action
                    D_nu_muE.append((state, expert_action))

            num_traj_p2 = self.trajectory_buffer_p2.num_trajectories()
            # Save the data to file for verification
            logger.info(f"Total trajectories collected: Expert Player 1={num_traj_p2}")
            self.trajectory_buffer_p2.save("debug_trajectory_buffer_p2.npy")
            # Original behavior: train once on all data
            logger.info(f"Training BC for Player 1 with {len(D_nu_muE)} samples...")
            learner_mu = BehavioralCloningSingleAgent(
                num_actions=self.num_actions_p1,
                cnn_policy=self.bc_cnn,
                in_channels=self.in_channels,
                device=self.device,
            )
            hat_mu, loss_mu = learner_mu.train(
                D_nu_muE, epochs=epochs, device=self.device
            )
            hat_mu = get_vectorized_policy(hat_mu, self.expert)
            logger.info(f"Player 1 BC training completed. Final loss: {loss_mu:.6f}")

            bc_total_time = time.time() - bc_start_time
            logger.info(f"Total BC training time: {format_time(bc_total_time)}")

        # Save trajectory buffers to files if flag is enabled
        if self.save_buffers:
            logger.info("\n" + "=" * 80)
            logger.info("Saving Trajectory Buffers")
            logger.info("=" * 80)

            # Create data directory if it doesn't exist
            data_dir = Path(__file__).parent.parent / "data" / "trajectories"
            data_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for the filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save Player 1 trajectory buffer
            p1_buffer_path = data_dir / f"deep_mail_bufferp1_K{self.K}_{timestamp}.npy"
            self.trajectory_buffer_p1.save(str(p1_buffer_path))
            logger.info(f"Player 1 trajectory buffer saved to: {p1_buffer_path}")

            # Save Player 2 trajectory buffer
            p2_buffer_path = data_dir / f"deep_mail_bufferp2_K{self.K}_{timestamp}.npy"
            self.trajectory_buffer_p2.save(str(p2_buffer_path))
            logger.info(f"Player 2 trajectory buffer saved to: {p2_buffer_path}")
            logger.info("")

        # Calculate overall time
        overall_total_time = time.time() - overall_start_time

        logger.info("\n" + "=" * 80)
        logger.info("Deep WARM-MAIL Training Completed Successfully!")
        logger.info("=" * 80)
        logger.info("Training Time Summary:")
        # logger.info(f"  Player 1 training: {format_time(p1_total_time)} ({p1_total_time/overall_total_time*100:.1f}%)")
        logger.info(
            f"  Player 2 exploring training: {format_time(p2_total_time)} ({p2_total_time / overall_total_time * 100:.1f}%)"
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
            # Original return format: also return the number of trajectories used to train each player
            return hat_mu, loss_mu, len(D_nu_muE)
