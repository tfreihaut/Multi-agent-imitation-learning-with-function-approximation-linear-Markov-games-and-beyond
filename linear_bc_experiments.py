import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from environments.grid_world import (
    GridZeroSum, 
    convert_gridzero_to_markov_game, 
    state_action_feature_map_interaction,
    state_action_feature_map_tiled_relational,
)
from game_solver import MarkovGameValueIteration
from environments.player import Player
from linear.bc2 import BehavioralCloning
from linear.bc_deep import BehavioralCloningDeep
from additional_experiments.utils import calc_exploitability_true_both


def get_or_compute_expert_policies(num_states, num_a1, num_a2, rewards, transitions, gamma):
    """
    Load expert policies from file if they exist, otherwise compute them via Value Iteration and save.
    
    Args:
        num_states: Number of states in the game
        num_a1: Number of actions for player 1
        num_a2: Number of actions for player 2
        rewards: Reward matrix
        transitions: Transition matrix
        gamma: Discount factor
        
    Returns:
        tuple: (mu_E_vi, nu_E_vi) - Expert policies for both players
    """
    # Create directory if it doesn't exist
    policy_dir = Path(__file__).parent.parent / 'grid_expert_policies'
    policy_dir.mkdir(exist_ok=True)
    
    # Create a unique filename based on game parameters
    policy_filename = f'expert_policies_s{num_states}_a1_{num_a1}_a2_{num_a2}_gamma{gamma:.2f}.npz'
    policy_path = policy_dir / policy_filename
    
    # Try to load existing policies
    if policy_path.exists():
        print(f"Loading expert policies from: {policy_path}")
        try:
            data = np.load(policy_path)
            mu_E_vi = data['mu_E_vi']
            nu_E_vi = data['nu_E_vi']
            print("Expert policies loaded successfully from file")
            return mu_E_vi, nu_E_vi
        except Exception as e:
            print(f"Failed to load policies from file: {e}")
            print("Will recompute policies...")
    
    # Compute policies if not found or loading failed
    print("Computing expert policies via Value Iteration...")
    expert_solver = MarkovGameValueIteration(num_states, num_a1, num_a2, rewards, transitions, discount_factor=gamma)
    _, mu_E_vi, nu_E_vi = expert_solver.value_iteration(max_iterations=1000)
    print("Expert policies computed successfully")
    
    # Save policies to file
    try:
        np.savez(policy_path, mu_E_vi=mu_E_vi, nu_E_vi=nu_E_vi)
        print(f"Expert policies saved to: {policy_path}")
    except Exception as e:
        print(f"Failed to save policies to file: {e}")
    
    return mu_E_vi, nu_E_vi


def test_bc(dataset_size, feature_map, feature_map_args_p1={}, feature_map_args_p2={}, epochs=None):

    bc_solver = BehavioralCloning(
        (mu_E, nu_E), 
        dataset_size, TRANSITIONS, 
        initial_dist_c, 
        REWARDS, 
        GAMMA, 
        feature_map, 
        feature_map,
        feature_map_args_p1=feature_map_args_p1,
        feature_map_args_p2=feature_map_args_p2
    )
    pol1_bc, pol2_bc, (loss_p1, loss_p2), (unique_states_mu, unique_states_nu) = bc_solver.train(epochs=epochs)
    exploitability = calc_exploitability_true_both(
        mu_pi=pol1_bc,
        nu_pi=pol2_bc,
        reward=REWARDS,
        transition=TRANSITIONS,
        initial_dist=initial_dist_c,
        gamma=GAMMA
    )
    # Return exploitability, max loss, and unique states for both players
    max_loss = max(loss_p1, loss_p2)
    return exploitability, max_loss, unique_states_mu, unique_states_nu

def test_deep_bc(dataset_size,lr, eta, use_linear, epochs):
    bc_deep_solver = BehavioralCloningDeep(
        (mu_E, nu_E), 
        grid_game,
        dataset_size=dataset_size,
        transitions=TRANSITIONS,
        initial_dist=initial_dist_c,
        rewards=REWARDS,
        gamma=GAMMA,
        lr=lr,
        eta=eta,
        use_linear=use_linear
    )
    pol1_bc, pol2_bc = bc_deep_solver.train(epochs=epochs)
    exploitability = bc_deep_solver.calc_exploitability_true(
        mu_pi=pol1_bc,
        nu_pi=pol2_bc,
        reward=REWARDS,
        P=TRANSITIONS,
        gamma=GAMMA
    )
    return exploitability


if __name__ == "__main__":
    # Configuration
    GAMMA = 0.9
    SEEDS = [42, 123, 456, 789]  # Multiple seeds for statistical robustness
    NUM_SEEDS = len(SEEDS)
    
    # Initialize game (same for all seeds)
    movements = ["left", "right", "up", "down"]
    grid_game = GridZeroSum(
        length=3, width=3,
        players=[Player(position=None, movements=movements), Player(position=None, movements=movements)],
        reward_coordinates=[[0, 2]], reward_value=1, gamma=GAMMA)
    REWARDS, TRANSITIONS, game_params = convert_gridzero_to_markov_game(grid_game)
    NUM_STATES, NUM_A1, NUM_A2 = game_params['num_states'], game_params['num_actions_p1'], game_params['num_actions_p2']
    
    # Get or compute expert policies (will load from file if available)
    mu_E_vi, nu_E_vi = get_or_compute_expert_policies(
        num_states=NUM_STATES,
        num_a1=NUM_A1,
        num_a2=NUM_A2,
        rewards=REWARDS,
        transitions=TRANSITIONS,
        gamma=GAMMA
    )
    
    configurations = [
        [0.5, 0.5, 0.5, 0.5],
    ]

    # 3. Run the Sweep
    for config in configurations:
        p, q, r, s = config
        print(f"\n===== Running for p={p}, q={q}, r={r}, s={s} =====")
        mu_E = np.zeros((72, 4))
        nu_E = np.zeros((72, 4))
        mu_E[grid_game.all_states.index([[1,0],[2,1]]), grid_game.movements.index("up")] = p
        mu_E[grid_game.all_states.index([[1,0],[2,1]]), grid_game.movements.index("right")] = 1.0 - p
        nu_E[grid_game.all_states.index([[1,0],[2,1]]), grid_game.movements.index("up")] = q
        nu_E[grid_game.all_states.index([[1,0],[2,1]]), grid_game.movements.index("right")] = 1.0 - q
        mu_E[grid_game.all_states.index([[1,1],[2,2]]), grid_game.movements.index("up")] = r
        mu_E[grid_game.all_states.index([[1,1],[2,2]]), grid_game.movements.index("right")] = 1.0 - r
        nu_E[grid_game.all_states.index([[0,0],[1,1]]), grid_game.movements.index("up")] = s
        nu_E[grid_game.all_states.index([[0,0],[1,1]]), grid_game.movements.index("right")] = 1.0 - s
        mu_E[grid_game.all_states.index([[0,1],[1,2]]), grid_game.movements.index("right")] = 1.0
        nu_E[grid_game.all_states.index([[0,1],[1,2]]), grid_game.movements.index("up")] = 1.0
        mu_E[grid_game.all_states.index([[0,0],[2,2]]), grid_game.movements.index("right")] = 1.0
        nu_E[grid_game.all_states.index([[0,0],[2,2]]), grid_game.movements.index("up")] = 1.0
        for state_idx in range(72):
            if mu_E[state_idx].sum() == 0:
                mu_E[state_idx] = mu_E_vi[state_idx]
            if nu_E[state_idx].sum() == 0:
                nu_E[state_idx] = nu_E_vi[state_idx]

    mu_E = mu_E_vi
    nu_E = nu_E_vi
    
    start_state = [[1, 0], [2, 1]]
    start_state_idx = grid_game.all_states.index(start_state)
    initial_dist_c = np.zeros(NUM_STATES); initial_dist_c[start_state_idx] = 1.0
    
    H = 10
    K = [1, 5, 10, 20, 50]
    HK = [H*k for k in K]
    EPOCHS_FOR_BC = 1000
    
    # Store results for all seeds
    all_results = {
        'exploits_tabular_bc': [],
        'losses_tabular_bc': [],
        'exploits_smart_bc': [],
        'losses_smart_bc': [],
        'exploits_deep_linear_bc': [],
        'losses_deep_linear_bc': [],
    }
    
    # Run experiments for each seed
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"Running experiments with seed {seed} ({seed_idx + 1}/{NUM_SEEDS})")
        print(f"{'='*60}\n")
        
        np.random.seed(seed)
        torch.manual_seed(seed)

        os.environ["PYTHONHASHSEED"] = str(seed)
        
        # BC with Tabular Features
        exploits_tabular_bc = []
        losses_tabular_bc = []
        unique_states_mu_tabular_bc = []
        unique_states_nu_tabular_bc = []
        print("BC - Tabular Features:")
        for dataset_size in tqdm(HK):
            exploitability, loss, unique_mu, unique_nu = test_bc(dataset_size, state_action_feature_map_interaction, {}, {}, EPOCHS_FOR_BC)
            exploits_tabular_bc.append(exploitability)
            losses_tabular_bc.append(loss)
            unique_states_mu_tabular_bc.append(unique_mu)
            unique_states_nu_tabular_bc.append(unique_nu)
        all_results['exploits_tabular_bc'].append(exploits_tabular_bc)
        all_results['losses_tabular_bc'].append(losses_tabular_bc)
        
        
        # BC with Relational Features
        exploits_smart_bc = []
        losses_smart_bc = []
        unique_states_mu_smart_bc = []
        unique_states_nu_smart_bc = []
        print("BC - Relational Features:")
        for dataset_size in tqdm(HK):
            exploitability, loss, unique_mu, unique_nu = test_bc(dataset_size, state_action_feature_map_tiled_relational, 
                                     {'grid_game': grid_game, 'player_idx': 0}, 
                                     {'grid_game': grid_game, 'player_idx': 1},
                                     EPOCHS_FOR_BC)
            exploits_smart_bc.append(exploitability)
            losses_smart_bc.append(loss)
            unique_states_mu_smart_bc.append(unique_mu)
            unique_states_nu_smart_bc.append(unique_nu)
        all_results['exploits_smart_bc'].append(exploits_smart_bc)
        all_results['losses_smart_bc'].append(losses_smart_bc)
        
        
        # BC with deep linear features
        print("BC - Deep Linear Features:")
        exploits_deep_linear_bc = []
        losses_deep_linear_bc = []
        unique_states_mu_deep_linear_bc = []
        unique_states_nu_deep_linear_bc = []
        print("BC - deep_linear Features:")
        for dataset_size in tqdm(HK):
            exploitability = test_deep_bc(dataset_size=dataset_size,lr=1e-3,eta=1.0, use_linear=True, epochs=EPOCHS_FOR_BC)
            exploits_deep_linear_bc.append(exploitability)
            losses_deep_linear_bc.append(loss)
        all_results['exploits_deep_linear_bc'].append(exploits_deep_linear_bc)
        
    
    # Compute statistics (mean and std) across seeds
    print(f"\n{'='*60}")
    print("Computing statistics across all seeds...")
    print(f"{'='*60}\n")
    
    stats = {}
    for key in all_results.keys():
        data = np.array(all_results[key])  # Shape: (num_seeds, num_dataset_sizes)
        stats[key] = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'stderr': np.std(data, axis=0) / np.sqrt(NUM_SEEDS)  # Standard error for confidence intervals
        }

    # Create directory for saving results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'linear'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving results to CSV files in '{results_dir}/' directory...")
    print(f"{'='*60}\n")
    
    # Save data to CSV files
    # For each metric, create a CSV with columns: dataset_size, method_mean, method_stderr, method_std
    
    # 1. Exploitability data
    exploit_df = pd.DataFrame({'dataset_size': HK})
    exploit_df['tabular_bc_mean'] = stats['exploits_tabular_bc']['mean']
    exploit_df['tabular_bc_stderr'] = stats['exploits_tabular_bc']['stderr']
    exploit_df['tabular_bc_std'] = stats['exploits_tabular_bc']['std']
    exploit_df['smart_bc_mean'] = stats['exploits_smart_bc']['mean']
    exploit_df['smart_bc_stderr'] = stats['exploits_smart_bc']['stderr']
    exploit_df['smart_bc_std'] = stats['exploits_smart_bc']['std']
    exploit_df['deep_linear_bc_mean'] = stats['exploits_deep_linear_bc']['mean']
    exploit_df['deep_linear_bc_stderr'] = stats['exploits_deep_linear_bc']['stderr']
    exploit_df['deep_linear_bc_std'] = stats['exploits_deep_linear_bc']['std']
    exploit_csv = os.path.join(results_dir, f'exploitability_{timestamp}.csv')
    exploit_df.to_csv(exploit_csv, index=False)
    print(f"✓ Saved exploitability data to: {exploit_csv}")
    
    
    print(f"\n{'='*60}\n")

    # Define consistent colors for each method
    colors = {
        'tabular_bc': '#1f77b4',      # Blue
        'smart_bc': '#ff7f0e',         # Orange
        'tabular_mail': '#d62728',     # Red
        'smart_mail': '#9467bd',       # Purple
        'deep_linear_bc': '#8c564b'          # Brown
    }
    
    # Maximum states that can be visited by each player
    MAX_STATES_PLAYER_MU = 19  # Player 1 (μ)
    MAX_STATES_PLAYER_NU = 17  # Player 2 (ν)
    
    # Plot averaged results with confidence intervals - now with 4 subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(18, 12))
    
    # Plot 1: Exploitability with confidence intervals
    # BC methods
    ax1.plot(HK, stats['exploits_tabular_bc']['mean'], label='Tabular Features BC', 
             marker='o', linewidth=2, color=colors['tabular_bc'])
    ax1.fill_between(HK, 
                     stats['exploits_tabular_bc']['mean'] - 1.96 * stats['exploits_tabular_bc']['stderr'],
                     stats['exploits_tabular_bc']['mean'] + 1.96 * stats['exploits_tabular_bc']['stderr'],
                     alpha=0.2, color=colors['tabular_bc'])
    
    ax1.plot(HK, stats['exploits_smart_bc']['mean'], label='Relational Features BC', 
             marker='o', linewidth=2, color=colors['smart_bc'])
    ax1.fill_between(HK, 
                     stats['exploits_smart_bc']['mean'] - 1.96 * stats['exploits_smart_bc']['stderr'],
                     stats['exploits_smart_bc']['mean'] + 1.96 * stats['exploits_smart_bc']['stderr'],
                     alpha=0.2, color=colors['smart_bc'])
    
    ax1.plot(HK, stats['exploits_deep_linear_bc']['mean'], label='deep_linear Features BC', 
             marker='o', linewidth=2, color=colors['deep_linear_bc'])
    ax1.fill_between(HK, 
                     stats['exploits_deep_linear_bc']['mean'] - 1.96 * stats['exploits_deep_linear_bc']['stderr'],
                     stats['exploits_deep_linear_bc']['mean'] + 1.96 * stats['exploits_deep_linear_bc']['stderr'],
                     alpha=0.2, color=colors['deep_linear_bc'])
    
    
    ax1.set_xlabel('Dataset Size', fontsize=12)
    ax1.set_ylabel('Exploitability', fontsize=12)
    ax1.set_title(f'Exploitability vs Dataset Size\n(Averaged over {NUM_SEEDS} seeds, 95% CI)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    # Save plot to the same directory as CSV files
    plot_path = os.path.join(results_dir, f'bc_results_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_path}")
    print(f"\n{'='*60}\n")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    for method in ['tabular_bc', 'smart_bc', 'deep_linear_bc']:
        print(f"\n{method.upper().replace('_', ' ')}:")
        print(f"  Exploitability - Mean (final): {stats[f'exploits_{method}']['mean'][-1]:.6f} ± {stats[f'exploits_{method}']['stderr'][-1]:.6f}")


