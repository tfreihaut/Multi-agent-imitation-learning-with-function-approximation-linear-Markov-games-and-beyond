import numpy as np
import logging
from datetime import datetime
import time
from pathlib import Path

from tictactoe.dqn_explore import DeepWARMMAIL
from tictactoe.full_bc import FullBC

from tictactoe.evaluate_policy import evaluate_against_expert
from tictactoe.generate_perfect_policy import generate_policy

from pettingzoo.classic import tictactoe_v3
import torch


def setup_experiment_logging(log_file="deepmail_vs_deepunif.log"):
    """Set up logging for the comparison experiment."""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{log_file}"

    logger = logging.getLogger("DeepMAIL_vs_DeepUnif")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Experiment logging initialized. Log file: {log_path}")

    return logger, log_path, timestamp


def main():
    """Run the comparison experiment (DQN-Explore-BC vs. Full BC)."""
    logger, _, _ = setup_experiment_logging()

    logger.info("=" * 80)
    logger.info("Experiments for TicTacToe with DQN-Explore-BC and Full BC")
    logger.info("=" * 80)

    experiment_start_time = time.time()

    # =========================================================================
    # Game Setup
    # =========================================================================
    # Use human for debugging/visualization; otherwise use None or "rgb_array"
    env = tictactoe_v3.env(render_mode="rgb_array")

    gamma = 0.99
    NUM_A1 = env.action_space("player_1").n
    NUM_A2 = env.action_space("player_2").n

    logger.info("Game Configuration:")
    logger.info(f"{env.metadata['name']}")
    logger.info(f"  Player 1 actions: {NUM_A1}")
    logger.info(f"  Player 2 actions: {NUM_A2}")
    logger.info(f"  Gamma: {gamma}")
    logger.info("")

    # ========================================================================
    # Testing parameters
    # ========================================================================
    H = 10  # Horizon
    K = 1_000  # Max trajectories iterations
    DATASET_SIZES = [10, 50, 100, 250, 500, 1_000]  # Dataset sizes to test
    BC_EPOCHS = 50
    GRADIENT_UPDATES = 10
    SEEDS = [42, 43, 142, 9, 2]
    BETA_VALUES = [1.5]
    TEMPERATURE_VALUES = [1.0]
    IMAGE_SIZE = 3
    DQN_HIDDEN_DIM = 64

    # Device setup
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        DEVICE_MAIL = torch.device("cuda:0")
        logger.info(
            f"Found {num_gpus} GPUs. Using GPU 0 for Deep MAIL and GPU 1 for Full BC"
        )
    elif num_gpus == 1:
        DEVICE_MAIL = torch.device("cuda:0")
        logger.warning("Only 1 GPU found. Both experiments will share GPU 0")
    else:
        DEVICE_MAIL = torch.device("cpu")
        logger.warning(
            "WARNING: No CUDA devices available. Training will be slow on CPU"
        )

    logger.info("Experiment Parameters:")
    logger.info(f"  Horizon (H): {H}")
    logger.info(f"  K (max trajectories): {K}")
    logger.info(f"  Dataset sizes to test: {DATASET_SIZES}")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Beta values: {BETA_VALUES}")
    logger.info(f"  Temperature values: {TEMPERATURE_VALUES}")
    logger.info(f"  BC Epochs: {BC_EPOCHS}")
    logger.info(f"  Deep MAIL Device: {DEVICE_MAIL}")
    logger.info(f"  Image Size (Deep MAIL): {IMAGE_SIZE}")
    logger.info(f"  DQN Hidden Dim (Deep MAIL): {DQN_HIDDEN_DIM}")
    logger.info("")

    results = {
        "seed": [],
        "dataset_size": [],
        "beta": [],
        "temperature": [],
        "deep_mail_average_exploit": [],
        "bc_average_exploit": [],
    }

    expert_policy = generate_policy()

    for seed in SEEDS:
        logger.info("\n" + "=" * 80)
        logger.info(f"SEED: {seed}")
        logger.info("=" * 80)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # ------------------------------------------------------------------
        # DQN-Explore-BC
        # ------------------------------------------------------------------

        logger.info(
            f"Running DQN-Explore-BC with K={K}, Beta={BETA_VALUES[0]}, Temp={TEMPERATURE_VALUES[0]}, Seed={seed} on {DEVICE_MAIL}"
        )

        mail_solver = DeepWARMMAIL(
            K=K,
            num_actions_p1=NUM_A1,
            num_actions_p2=NUM_A2,
            image_size=IMAGE_SIZE,
            in_channels=2,
            env=env,
            dqn_hidden_dim=DQN_HIDDEN_DIM,
            device=DEVICE_MAIL,
            beta=BETA_VALUES[0],
            temperature=1.0,
            bc_cnn=True,
            expert=expert_policy,
        )

        policies_mail, _, _ = mail_solver.run(
            horizon=10,
            epochs=BC_EPOCHS,
            gradient_updates=GRADIENT_UPDATES,
            logger=logger,
            dataset_sizes=DATASET_SIZES,
        )

        # ------------------------------------------------------------------
        # Full BC
        # ------------------------------------------------------------------
        only_bc = FullBC(
            K=K,
            num_actions_p1=NUM_A1,
            num_actions_p2=NUM_A2,
            env=env,
            gamma=gamma,
            device=DEVICE_MAIL,
            expert_policy=expert_policy,
            num_states=9,
            in_channels=2,
        )
        policies_bc, _, _ = only_bc.run(
            horizon=10, epochs=BC_EPOCHS, logger=logger, dataset_sizes=DATASET_SIZES
        )

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING LEARNED POLICIES AGAINST SOLVER")
        logger.info("=" * 80)

        for idx, dataset_size in enumerate(DATASET_SIZES):
            logger.info(f"Evaluating dataset size {dataset_size}")

            # DQN-Explore-BC policy
            hat_mu_mail = policies_mail[idx]
            exploitability_mail = abs(
                evaluate_against_expert(
                    hat_mu_mail,
                    expert_policy,
                    evaluation_episodes=100000,
                    player=1,
                )
            )
            logger.info(f"DQN-Explore-BC exploitability: {exploitability_mail:.4f}")
            results["deep_mail_average_exploit"].append(exploitability_mail)

            # BC policy
            hat_mu_bc = policies_bc[idx]
            exploitability_bc = abs(
                evaluate_against_expert(
                    hat_mu_bc,
                    expert_policy,
                    evaluation_episodes=100000,
                    player=1,
                )
            )
            logger.info(f"  Full BC exploitability: {exploitability_bc:.4f}")
            results["bc_average_exploit"].append(exploitability_bc)

            results["dataset_size"].append(dataset_size)
            results["seed"].append(seed)
            results["beta"].append(BETA_VALUES[0])
            results["temperature"].append(TEMPERATURE_VALUES[0])

    # ----------------------------------------------------------------------
    # Save the policies
    logger.info("\n" + "=" * 80)
    logger.info("SAVING LEARNED POLICIES TO DISK")
    logger.info("=" * 80)
    import pickle

    policies_dir = Path(__file__).parent.parent / "saved_policies"
    policies_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mail_policy_path = policies_dir / f"tictactoe_deep_mail_policy_{timestamp}.pkl"
    with open(mail_policy_path, "wb") as f:
        pickle.dump(policies_mail, f)
    logger.info(f"DQN-Explore-BC policies saved to {mail_policy_path}")

    # Plot the results
    import matplotlib.pyplot as plt

    sizes = sorted(set(results["dataset_size"]))
    deep_mail_mean_exploit = []
    deep_mail_std_exploit = []
    bc_exploitability_mean = []
    bc_exploitability_std = []

    for s in sizes:
        idxs = [i for i, val in enumerate(results["dataset_size"]) if val == s]
        mail_exploitability = [results["deep_mail_average_exploit"][i] for i in idxs]
        bc_exploitability = [results["bc_average_exploit"][i] for i in idxs]

        deep_mail_mean_exploit.append(np.mean(mail_exploitability))
        deep_mail_std_exploit.append(np.std(mail_exploitability))
        bc_exploitability_mean.append(np.mean(bc_exploitability))
        bc_exploitability_std.append(np.std(bc_exploitability))

    plt.figure(figsize=(12, 5))

    # Convert to numpy arrays for element-wise operations
    dm_mean = np.array(deep_mail_mean_exploit)
    dm_std = np.array(deep_mail_std_exploit)
    bc_mean = np.array(bc_exploitability_mean)
    bc_std = np.array(bc_exploitability_std)

    # Plot DQN-Explore-BC
    plt.plot(sizes, dm_mean, marker="o", label="DQN-Explore-BC", color="blue")
    plt.fill_between(sizes, dm_mean - dm_std, dm_mean + dm_std, color="blue", alpha=0.2)

    # Plot Full BC
    plt.plot(sizes, bc_mean, marker="^", label="BC", color="green")
    plt.fill_between(
        sizes, bc_mean - bc_std, bc_mean + bc_std, color="green", alpha=0.2
    )

    plt.xlabel("Dataset size (number of trajectories)")
    plt.ylabel("Average Exploitability")
    plt.title("Average Exploitability vs Dataset Size")
    plt.legend()
    plt.tight_layout()
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"tictactoe_exploitability_{timestamp}.png")
    plt.close()

    # Save results to CSV
    import pandas as pd

    # Save summary (mean + std per dataset size)
    summary = {
        "dataset_size": sizes,
        "deep_mail_mean_exploit": deep_mail_mean_exploit,
        "deep_mail_std_exploit": deep_mail_std_exploit,
        "bc_mean_exploit": bc_exploitability_mean,
        "bc_std_exploit": bc_exploitability_std,
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        f"./tictactoe_summary_{timestamp}.csv", index=False
    )

    logger.info(
        f"Total experiment time: {(time.time() - experiment_start_time) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
