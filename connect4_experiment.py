import numpy as np
from datetime import datetime
import time
import csv
import matplotlib.pyplot as plt

from connect4.dqn_explore import DeepWARMMAIL
from connect4.full_bc import FullBC
from connect4.evaluate_policy import evaluate_policy
from pettingzoo.classic import connect_four_v3
from additional_experiments.utils import setup_experiment_logging
import torch


def main():
    """Run the comparison experiment."""
    # Set up logging
    logger, log_path, timestamp = setup_experiment_logging()

    logger.info("=" * 80)
    logger.info("DQN-Explore-BC vs Deep BC")
    logger.info("=" * 80)

    # =========================================================================
    # Game Setup
    # =========================================================================
    env = connect_four_v3.env(render_mode="rgb_array")
    gamma = 0.99
    NUM_A1 = env.action_space("player_0").n
    NUM_A2 = env.action_space("player_1").n

    logger.info("Game Configuration:")
    logger.info(f"{env.metadata['name']}")
    # logger.info(f"  Number of states: {NUM_STATES}")
    logger.info(f"  Player 1 actions: {7}")
    logger.info(f"  Player 2 actions: {7}")
    logger.info(f"  Gamma: {gamma}")
    logger.info("")

    # ========================================================================
    # Testing parameters
    # ========================================================================
    H = 42  # Horizon
    K = 30_000  # Maximum number of trajectories to collect for unif, outer loop iterations for MAIL
    DATASET_SIZES = [
        1000,
        10_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        3_000_000,
    ]  # Dataset sizes to test
    BC_EPOCHS = 20  # Number of BC training epochs

    SEEDS = [42, 123, 1]  # Random seeds for reproducibility
    BETA_VALUES = [1.5]  # Beta values to test
    TEMPERATURE_VALUES = [1.0]  # Temperature values to test
    # DQN-Explore-BC specific parameters
    HEIGHT = 6
    WIDTH = 7
    DQN_HIDDEN_DIM = 128

    # Check available GPUs and assign them
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        DEVICE_MAIL = torch.device("cuda:0")
        logger.info(
            f"Found {num_gpus} GPUs. Using GPU 0 for DQN-Explore-BC and GPU 1 for DQN-Explore-BC RND"
        )
    elif num_gpus == 1:
        DEVICE_MAIL = torch.device("cuda:0")
        logger.warning("Only 1 GPU found. Both experiments will share GPU 0")
    else:
        DEVICE_MAIL = torch.device("cpu")
        logger.warning(
            "WARNING: No CUDA devices available. Training will be very slow on CPU!"
        )

    logger.info("Experiment Parameters:")
    logger.info(f"  Horizon (H): {H}")
    logger.info(f"  K (max trajectories): {K}")
    logger.info(f"  Dataset sizes to test: {DATASET_SIZES}")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Beta values: {BETA_VALUES}")
    logger.info(f"  Temperature values: {TEMPERATURE_VALUES}")
    logger.info(f"  BC Epochs: {BC_EPOCHS}")
    logger.info(f"  DQN-Explore-BC Device: {DEVICE_MAIL}")
    logger.info(f"  Image Size (DQN-Explore-BC): {HEIGHT}x{WIDTH}")
    logger.info(f"  DQN Hidden Dim (DQN-Explore-BC): {DQN_HIDDEN_DIM}")
    logger.info(
        f"  Total experiments: {len(SEEDS)} seeds × {len(BETA_VALUES)} betas × {len(TEMPERATURE_VALUES)} temps × {len(DATASET_SIZES)} dataset sizes = {len(SEEDS) * len(BETA_VALUES) * len(TEMPERATURE_VALUES) * len(DATASET_SIZES)}"
    )
    logger.info("")

    # =========================================================================
    # Run Experiments
    # =========================================================================
    subopt_levels = [0.0, 1, 2, 3, 4, 5]

    results_data = {
        "MAIL": {
            "win_rate": {s: [] for s in subopt_levels},
            "opponent_entropy": {s: [] for s in subopt_levels},
        },
        "BC": {
            "win_rate": {s: [] for s in subopt_levels},
            "opponent_entropy": {s: [] for s in subopt_levels},
        },
    }

    for seed in SEEDS:
        logger.info("\n" + "=" * 80)
        logger.info(f"SEED: {seed}")
        logger.info("=" * 80)

        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        logger.info(f"\n{'=' * 40}")
        logger.info(
            f"Running DQN-Explore with K={K}, Beta={BETA_VALUES[0]}, Temp={TEMPERATURE_VALUES[0]}, Seed={seed} on {DEVICE_MAIL}"
        )
        logger.info(f"{'=' * 40}")

        mail_start_time = time.time()

        deep_mail_solver = DeepWARMMAIL(
            K=K,
            num_actions_p1=NUM_A1,
            num_actions_p2=NUM_A2,
            height=HEIGHT,
            width=WIDTH,
            in_channels=2,
            env=env,
            dqn_hidden_dim=DQN_HIDDEN_DIM,
            device=DEVICE_MAIL,
            beta=BETA_VALUES[0],
            temperature=1.0,
            batch_size=256,
        )

        policies_mail, _, _ = deep_mail_solver.run(
            horizon=H, epochs=BC_EPOCHS, logger=logger, dataset_sizes=DATASET_SIZES
        )

        mail_time = time.time() - mail_start_time
        logger.info(f"\nDQN-Explore-BC completed in {mail_time / 60:.2f} minutes.")

        # # Full BC baseline
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Running Full BC with K={K}, Seed={seed} on CPU")
        logger.info(f"{'=' * 40}")
        bc_start_time = time.time()
        full_bc_solver = FullBC(
            K=K,
            num_actions_p1=NUM_A1,
            num_actions_p2=NUM_A2,
            env=env,
            gamma=gamma,
            device=DEVICE_MAIL,
            num_states=42,
        )
        policies_bc, _, _ = full_bc_solver.run(
            horizon=H,
            epochs=BC_EPOCHS,
            logger=logger,
            dataset_sizes=DATASET_SIZES,
        )
        bc_time = time.time() - bc_start_time
        logger.info(f"\nFull BC completed in {bc_time / 60:.2f} minutes.")

        # Evaluate the learned policies by having them play against the solver
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING LEARNED POLICIES AGAINST SOLVER")
        logger.info("=" * 80)
        # Now evaluate each policy pair and store results
        for idx, dataset_size in enumerate(DATASET_SIZES):
            eval_episodes = 100
            # REMOVE: DEBUGGING only
            eval_episodes = 2
            logger.info(f"\n--- Evaluating dataset size {dataset_size} ---")

            # DQN-Explore-BC policy
            hat_mu_mail = policies_mail[idx]
            # Create a fresh environment for evaluation
            # Important: Don't reuse the training env as it may be in a corrupted state
            eval_env = connect_four_v3.env(render_mode="rgb_array")
            logger.info("Created fresh evaluation environment")
            logger.info(
                "Evaluating learned policy for player_0 (policy_mu) vs solver as player_1..."
            )
            for subopt in subopt_levels:
                wins_policy_0, wins_policy_1, _, opponent_entropies = evaluate_policy(
                    hat_mu_mail,
                    0,
                    eval_env,
                    num_episodes=eval_episodes,
                    argmax=True,
                    suboptimal=subopt,
                )
                win_rate = wins_policy_0 / eval_episodes
                mean_entropy = (
                    np.mean(opponent_entropies) if opponent_entropies else 0.0
                )

                results_data["MAIL"]["win_rate"][subopt].append(win_rate)
                results_data["MAIL"]["opponent_entropy"][subopt].append(mean_entropy)

            # Close evaluation environment
            eval_env.close()

            # Evaluate BC policy
            print("\n--- BC Policy ---")
            hat_mu_bc = policies_bc[idx]
            # Create a fresh environment for evaluation
            # Important: Don't reuse the training env as it may be in a corrupted state
            eval_env = connect_four_v3.env(render_mode="rgb_array")
            logger.info("Created fresh evaluation environment")
            logger.info(
                "Evaluating learned policy for player_0 (policy_mu) vs solver as player_1..."
            )
            for subopt in subopt_levels:
                wins_policy_0, wins_policy_1, _, opponent_entropies = evaluate_policy(
                    hat_mu_bc,
                    0,
                    eval_env,
                    num_episodes=eval_episodes,
                    argmax=True,
                    suboptimal=subopt,
                )
                win_rate = wins_policy_0 / eval_episodes
                mean_entropy = (
                    np.mean(opponent_entropies) if opponent_entropies else 0.0
                )

                results_data["BC"]["win_rate"][subopt].append(win_rate)
                results_data["BC"]["opponent_entropy"][subopt].append(mean_entropy)

                print(
                    f"  Subopt={subopt:.1f}: WinRate={win_rate:.4f}, Entropy={mean_entropy:.4f}"
                )

            eval_env.close()

            print("\n" + "=" * 80)

    # Calculate mean and std for each subopt level
    mail_winrate_mean = [
        np.mean(results_data["MAIL"]["win_rate"][s]) for s in subopt_levels
    ]
    mail_winrate_std = [
        np.std(results_data["MAIL"]["win_rate"][s]) for s in subopt_levels
    ]
    mail_entropy_mean = [
        np.mean(results_data["MAIL"]["opponent_entropy"][s]) for s in subopt_levels
    ]
    mail_entropy_std = [
        np.std(results_data["MAIL"]["opponent_entropy"][s]) for s in subopt_levels
    ]

    bc_winrate_mean = [
        np.mean(results_data["BC"]["win_rate"][s]) for s in subopt_levels
    ]
    bc_winrate_std = [np.std(results_data["BC"]["win_rate"][s]) for s in subopt_levels]
    bc_entropy_mean = [
        np.mean(results_data["BC"]["opponent_entropy"][s]) for s in subopt_levels
    ]
    bc_entropy_std = [
        np.std(results_data["BC"]["opponent_entropy"][s]) for s in subopt_levels
    ]

    # get total entropy
    entropy_mean = np.array(mail_entropy_mean) + np.array(bc_entropy_mean) / 2
    entropy_std = np.array(mail_entropy_std) + np.array(bc_entropy_std) / 2

    # Save to CSV

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"policy_evaluation_comparison_{timestamp}.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "subopt_level",
            "mail_winrate_mean",
            "mail_winrate_std",
            "mail_opponent_entropy_mean",
            "mail_opponent_entropy_std",
            "bc_winrate_mean",
            "bc_winrate_std",
            "bc_opponent_entropy_mean",
            "bc_opponent_entropy_std",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, subopt in enumerate(subopt_levels):
            writer.writerow(
                {
                    "subopt_level": subopt,
                    "mail_winrate_mean": mail_winrate_mean[i],
                    "mail_winrate_std": mail_winrate_std[i],
                    "mail_opponent_entropy_mean": mail_entropy_mean[i],
                    "mail_opponent_entropy_std": mail_entropy_std[i],
                    "bc_winrate_mean": bc_winrate_mean[i],
                    "bc_winrate_std": bc_winrate_std[i],
                    "bc_opponent_entropy_mean": bc_entropy_mean[i],
                    "bc_opponent_entropy_std": bc_entropy_std[i],
                }
            )

    print(f"Results saved to '{csv_filename}'")

    # Create visualization with error bars
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6)
    )  # Adjusted height: 12 is very tall for 1 row

    x_pos = np.arange(len(subopt_levels))
    width = 0.35

    # --- Plot 1: Win Rate vs Suboptimal Level ---
    # Use axes[0], not axes[0, 0]
    axes[0].bar(
        x_pos - width / 2,
        mail_winrate_mean,
        width,
        label="MAIL",
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
        yerr=mail_winrate_std,
        capsize=5,
    )
    axes[0].bar(
        x_pos + width / 2,
        bc_winrate_mean,
        width,
        label="BC",
        color="coral",
        edgecolor="black",
        alpha=0.7,
        yerr=bc_winrate_std,
        capsize=5,
    )

    axes[0].set_xlabel("Opponent Suboptimal Level", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Win Rate (mean ± std)", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Policy Win Rate vs Opponent Suboptimal Level", fontsize=13, fontweight="bold"
    )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"{s:.1f}" for s in subopt_levels], rotation=45)
    axes[0].set_ylim([0, 1])
    axes[0].legend(fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    # --- Plot 2: Opponent Entropy vs Suboptimal Level ---
    # Use axes[1], not axes[0, 1]
    # NOTE: Check the X-axis logic below (see explanation)
    axes[1].errorbar(
        subopt_levels,
        entropy_mean,
        yerr=entropy_std,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        label="Entropy",
        color="steelblue",
        alpha=0.7,
    )

    axes[1].set_xlabel("Opponent Suboptimal Level", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Opponent Entropy (mean ± std)", fontsize=12, fontweight="bold")
    axes[1].set_title(
        "Opponent Policy Stochasticity vs Suboptimal Level",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_filename = f"policy_evaluation_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as '{plot_filename}'")


if __name__ == "__main__":
    main()
