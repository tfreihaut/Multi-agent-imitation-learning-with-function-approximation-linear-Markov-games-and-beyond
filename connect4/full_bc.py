# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Use free GPUs
import torch
import numpy as np
from collections import namedtuple
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
from connect4.utils import format_time, Connect4Dataset, TrajectoryBuffer


# Define the structure for storing transitions in the replay buffer
Transition = namedtuple(
    "Transition", ("state", "action", "expert_action", "reward", "next_state")
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
        num_states=42,
    ):
        self.K = K
        self.num_actions_p1 = num_actions_p1
        self.num_actions_p2 = num_actions_p2
        self.gamma = gamma
        self.env = env
        self.device = device
        self.num_states = num_states

        # Trajectory buffers for expert player 1 (without capacity, stores all exploration data)
        self.trajectory_buffer_exp_p1 = TrajectoryBuffer(player_id=1)

    def run(
        self, horizon, epochs=1000, logger=None, batch_size=4096, dataset_sizes=None
    ):
        """Main training loop for Deep Uniform."""
        if logger is None:
            logger = logging.getLogger("DeepUniform")

        # Start overall timing
        overall_start_time = time.time()

        logger.info("=" * 80)
        logger.info("Starting Deep Uniform")
        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  K (Uniform iterations): {self.K}")
        logger.info(f"  Horizon: {horizon}")
        logger.info(f"  Gamma (discount): {self.gamma}")
        logger.info("")

        # run exploration for the second player
        logger.info("=" * 80)
        logger.info("PHASE: Expert vs Expert")
        logger.info("=" * 80)
        bb_agent = BitBully()
        for k in tqdm(range(self.K)):
            logger.info(f"\n---: Iteration {k + 1}/{self.K} ---")
            self.env.reset()
            bb_board = Board()
            for _ in range(horizon):
                for agent in self.env.agent_iter():
                    state, _, termination, truncation, _ = self.env.last()
                    if termination or truncation:
                        action = None
                        self.trajectory_buffer_exp_p1.end_trajectory()
                        self.env.reset()
                        bb_board = Board()
                        break
                    else:
                        if agent == self.env.possible_agents[0]:
                            # Player 1 plays according to the expert solver
                            # Get scores for all legal moves from the solver and pick the argmax
                            scores = bb_agent.score_next_moves(bb_board)
                            # score_next_moves is expected to return an iterable of move scores
                            # Use numpy argmax to pick the best move index
                            action_p1 = int(np.argmax(scores))
                            bb_board.play(action_p1)
                            action = action_p1
                            # Only Store expert transition in trajectory buffer (for BC training later)

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
                            # Player 2 plays according to the expert solver
                            # Get scores for all legal moves from the solver and pick the argmax
                            scores = bb_agent.score_next_moves(bb_board)
                            # score_next_moves is expected to return an iterable of move scores
                            # Use numpy argmax to pick the best move index
                            action_p2 = int(np.argmax(scores))
                            bb_board.play(action_p2)
                            action = action_p2

                    self.env.step(action)

        # Save the trajectory buffer to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trajectory_buffer_exp_p1.save_optimized(
            f"full_bc_debug_trajectory_buffer_c4_{timestamp}_p1.npy"
        )

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
                trajectories_p1 = self.trajectory_buffer_exp_p1.get_trajectories()
                D_nu_muE_subset = Connect4Dataset(trajectories_p1, limit_traj=size)

                logger.info(f"Subset samples: Player 1={len(D_nu_muE_subset)}")

                # Train BC for Player 1
                learner_mu = BehavioralCloningSingleAgent(
                    num_actions=self.num_actions_p1,
                    cnn_policy=True,
                    in_channels=2,
                    device=self.device,
                    lr=5e-3,
                )
                # Check if epoch is a list (dynamic epochs per size)
                if isinstance(epochs, list):
                    current_epochs = epochs[
                        min(len(epochs) - 1, dataset_sizes.index(size))
                    ]
                else:
                    current_epochs = epochs
                hat_mu, loss_mu = learner_mu.train(
                    D_nu_muE_subset,
                    epochs=current_epochs,
                    device=self.device,
                    batch_size=batch_size,
                )
                logger.info(
                    f"Player 1 BC training completed. Final loss: {loss_mu:.6f}"
                )

                policies_list.append(hat_mu)
                losses_list.append(loss_mu)
                trajectory_counts_list.append(len(D_nu_muE_subset))

            bc_total_time = time.time() - bc_start_time
            logger.info(
                f"\nTotal BC training time (all sizes): {format_time(bc_total_time)}"
            )
        else:
            # extract data from replay buffer to train behavioral cloning for player 1
            trajectories_p1 = self.trajectory_buffer_exp_p1.get_trajectories()
            D_nu_muE = Connect4Dataset(trajectories_p1)

            # Train behavioral cloning policies for both players
            logger.info(f"Training BC for Player 1 with {len(D_nu_muE)} samples...")
            learner_mu = BehavioralCloningSingleAgent(
                num_states=self.num_states,
                num_actions=self.num_actions_p1,
                cnn_policy=True,
                in_channels=2,
                device=self.device,
            )
            policy_mu, loss_mu = learner_mu.train(
                D_nu_muE, epochs=epochs, device=self.device
            )

        bc_total_time = time.time() - bc_start_time
        logger.info(f"Total BC training time: {format_time(bc_total_time)}")

        overall_time = time.time() - overall_start_time
        logger.info(f"Total training time: {format_time(overall_time)}")
        # Return based on whether dataset_sizes was specified
        if dataset_sizes is not None:
            return policies_list, losses_list, trajectory_counts_list
        else:
            return hat_mu, loss_mu, len(D_nu_muE)
