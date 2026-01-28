# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use free GPUs
import torch
from tqdm import tqdm
import logging
import time

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from tictactoe.bc import BehavioralCloningSingleAgent
from tictactoe.utils import (
    TrajectoryBuffer,
    obs_to_board,
    format_time,
    canonical,
    canonical_action_to_board_action,
    get_vectorized_policy,
)


class FullBC:
    def __init__(
        self,
        K,
        num_actions_p1,
        num_actions_p2,
        env,
        gamma=0.9,
        device="cpu",
        expert_policy=None,
        in_channels=2,
        num_states=9,
    ):
        self.K = K
        self.num_actions_p1 = num_actions_p1
        self.num_actions_p2 = num_actions_p2
        self.gamma = gamma
        self.env = env
        self.device = device
        self.expert = expert_policy
        # Trajectory buffers for both players (without capacity, stores all exploration data)
        self.trajectory_buffer_exp_p1 = TrajectoryBuffer(player_id=1)
        self.in_channels = in_channels
        self.num_states = num_states

    def run(self, horizon, epochs=1000, logger=None, dataset_sizes=None):
        """Main training loop for Full BC."""
        if logger is None:
            logger = logging.getLogger("Full BC")

        logger.info("=" * 80)
        logger.info("Starting Full BC")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  K (Uniform iterations): {self.K}")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  Gamma (discount): {self.gamma}")
        logger.info("")

        # run exploration for the second player
        logger.info("=" * 80)
        logger.info("PHASE 1: Exploring with Player 2")
        logger.info("=" * 80)

        for k in tqdm(range(self.K)):
            if (k + 1) % 1000 == 0 or k == 0:
                logger.info(f"\n--- Player 2: Iteration {k + 1}/{self.K} ---")
            self.env.reset()

            for _ in range(horizon):
                for agent in self.env.agent_iter():
                    state, reward, termination, truncation, info = self.env.last()
                    if termination or truncation:
                        action = None
                        # Break if episode ended
                        # End the trajectory in the trajectory buffer
                        self.trajectory_buffer_exp_p1.end_trajectory()
                        self.env.reset()
                        break
                    else:
                        if agent == self.env.possible_agents[0]:  # Player 1 (expert)
                            # Convert PettingZoo observation to absolute board
                            board = obs_to_board(state["observation"], agent)

                            # Canonicalize
                            canon_board = canonical(board)

                            # Lookup perfect action
                            canon_action = self.expert[canon_board]

                            # Map back to PettingZoo action space
                            action_p1 = canonical_action_to_board_action(
                                board, canon_board, canon_action
                            )
                            action = action_p1

                            # Preprocessed state for trajectory buffer
                            state_preprocessed_1 = (
                                torch.from_numpy(state["observation"])
                                .to(dtype=torch.float32, device=self.device)
                                .permute(2, 0, 1)
                            )
                            action_p1_tensor = torch.tensor(
                                action_p1, dtype=torch.long, device=self.device
                            )

                            self.trajectory_buffer_exp_p1.push(
                                state_preprocessed_1,
                                None,
                                action_p1_tensor,
                                None,
                                None,
                            )
                        else:
                            # Player 2 plays also according to the expert policy

                            board = obs_to_board(state["observation"], agent)
                            canon_board = canonical(board)
                            canon_action = self.expert[canon_board]
                            action_p2 = canonical_action_to_board_action(
                                board, canon_board, canon_action
                            )
                            action = action_p2

                    self.env.step(action)

        # extract data from replay buffer to train behavioral cloning for player 2
        logger.info("=" * 80)
        logger.info("PHASE 2: Behavioral Cloning")
        logger.info("=" * 80)

        bc_start_time = time.time()

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
                logger.info("Using CNN-based Behavioral Cloning Subset")
                # Player 1 subset
                D_nu_muE_subset = []
                trajectories_p1 = self.trajectory_buffer_exp_p1.get_trajectories()
                for i, trajectory in enumerate(trajectories_p1):
                    if i >= size:
                        break
                    for transition in trajectory:
                        state = transition.state
                        expert_action = transition.expert_action
                        D_nu_muE_subset.append((state, expert_action))

                logger.info(f"Subset samples: Player 1={len(D_nu_muE_subset)}")

                # Train BC for Player 1
                learner_mu = BehavioralCloningSingleAgent(
                    num_actions=self.num_actions_p1,
                    cnn_policy=True,
                    in_channels=2,
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
            # extract data from replay buffer to train behavioral cloning for player 1
            D_nu_muE = []
            for transition in self.trajectory_buffer_exp_p1.iterator():
                state = transition.state
                expert_action = transition.expert_action
                D_nu_muE.append((state, expert_action))

            # Train behavioral cloning policies for both players
            logger.info(f"Training BC for Player 1 with {len(D_nu_muE)} samples...")
            learner_mu = BehavioralCloningSingleAgent(
                num_states=self.num_states,
                num_actions=self.num_actions_p1,
                cnn_policy=True,
                in_channels=3,
                device=self.device,
            )
            policy_mu, loss_mu = learner_mu.train(
                D_nu_muE, epochs=epochs, device=self.device
            )
            policy_mu = get_vectorized_policy(policy_mu, self.expert)
            bc_total_time = time.time() - bc_start_time
            logger.info(f"Total BC training time: {format_time(bc_total_time)}")

        bc_total_time = time.time() - bc_start_time
        logger.info(f"Total BC training time: {format_time(bc_total_time)}")

        # Return based on whether dataset_sizes was specified
        if dataset_sizes is not None:
            return policies_list, losses_list, trajectory_counts_list
        else:
            # Original return format: also return the number of trajectories used to train each player
            return hat_mu, loss_mu, len(D_nu_muE)
