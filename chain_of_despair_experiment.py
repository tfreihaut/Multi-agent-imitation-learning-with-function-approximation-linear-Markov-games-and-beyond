
import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
def setup_logging(log_file='chain_of_despair_experiment.log'):
    log_dir = Path(__file__).parent.parent / 'logs' / 'chain_of_despair'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{timestamp}_{log_file}'
    
    logger = logging.getLogger('ChainOfDespairExp')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path

class ChainOfDespairWrapper:
    """
    Wrapper for Chain of Despair to mimic GridZeroSum interface required by DeepWARMMAIL.
    """
    def __init__(self, num_states, image_size=16, end_state=None):
        self.num_states = num_states
        self.image_size = image_size
        # DeepWARMMAIL checks if player pos is in reward_coordinates
        # In Chain of Despair, goal is the last state
        self.reward_coordinates = end_state # when p2 is non nash this is num_states - 1, when p1 is non nash this is 1
        
    def map_state_idx_to_state(self, state_idx):
        """
        Returns (p1_pos, p2_pos). In Chain of Despair, agents are always in the same state.
        """
        return (state_idx, state_idx)

    def render(self, state):
        """
        Returns an image representation of the state (H, W, C).
        """
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Simple visualization:
        # Divide width by num_states.
        # Color the column corresponding to the current state.
        
        step = max(1, self.image_size // self.num_states)
        start = state * step
        end = min(start + step, self.image_size)
        
        # Current state: Blue-ish
        img[:, start:end, 2] = 255 
        
        return img

def run_single_config(n_states, seed):
    # Local imports to ensure clean state in worker processes
    import os
    # Set CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    
    # Local imports to ensure clean state in worker processes
    import sys
    import os
    import torch
    import numpy as np
    from datetime import datetime
    from unittest.mock import MagicMock
    sys.modules['pygame'] = MagicMock()
    
    from additional_experiments.chain_of_despair import make_chain_of_despair_game
    from game_solver import MarkovGameValueIteration
    from additional_experiments.deep_mail import DeepWARMMAIL
    from additional_experiments.deep_unif_exploration import DeepUniform
    from additional_experiments.utils import calc_exploitability_true_both
    
    # Setup unique logger for this worker
    logger, _ = setup_logging(log_file=f'chain_n{n_states}_s{seed}.log')
    
    logger.info(f"\n" + "="*80)
    logger.info(f"Running Config: N={n_states}, Seed={seed}")
    logger.info("="*80)
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Experiment Parameters
    NUM_STATES = n_states
    IMAGE_SIZE = max(16, NUM_STATES) # Must be large enough for CNN
    HORIZON = 2**(NUM_STATES-2) # Sufficient to reach goal
    EPOCHS = 100
    GAMMA = 0.90
    GRADIENT_STEPS = 10
    
    # Create Environment
    rewards, transitions, game_params = make_chain_of_despair_game(NUM_STATES)
    num_actions_p1 = game_params['num_actions_p1']
    num_actions_p2 = game_params['num_actions_p2']
    
    wrapper_p1 = ChainOfDespairWrapper(NUM_STATES, IMAGE_SIZE, end_state=[NUM_STATES - 1]) # p1 is nash
    
    logger.info(f"Game Configuration:")
    logger.info(f"  Num States: {NUM_STATES}")
    logger.info(f"  Horizon: {HORIZON}")
    logger.info(f"  Gamma: {GAMMA}")
    
    # Compute Expert Policies
    logger.info("Computing expert policies via Value Iteration...")
    expert_solver = MarkovGameValueIteration(NUM_STATES, num_actions_p1, num_actions_p2, rewards, transitions, discount_factor=GAMMA)
    _, mu_E, nu_E = expert_solver.value_iteration(max_iterations=1000)
    
    # Initial state distribution (always start at s0)
    initial_dist = np.zeros(NUM_STATES)
    initial_dist[0] = 1.0
    initial_state_sampler = lambda: 0
    
    # Experiment Settings
    K_values = [1, 2, 5, 10, 20] # Iterations
    
    results = {
        'DeepUniform': {'K': [], 'exploitability': [], 'time': []},
        'DeepMAIL': {'K': [], 'exploitability': [], 'time': []}
    }
    
    for k in K_values:
        logger.info(f"\n--- Iteration K={k} (N={NUM_STATES}, Seed={seed}) ---")

        # --- Deep MAIL ---
        start_time_mail = datetime.now()
        
        solver_mail = DeepWARMMAIL(
            K=k,
            num_states=NUM_STATES,
            num_actions_p1=num_actions_p1,
            num_actions_p2=num_actions_p2,
            expert_policy_p1=mu_E,
            expert_policy_p2=nu_E,
            transition_P=transitions,
            initial_state_sampler=initial_state_sampler,
            image_size=IMAGE_SIZE,
            in_channels=3, 
            grid_game=wrapper_p1,
            dqn_hidden_dim=128, 
            batch_size=32,
            gamma=GAMMA,
            bc_cnn=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # DeepWARMMAIL run returns ((hat_mu, hat_nu), (loss_mu, loss_nu), (count_p1, count_p2))
        (hat_mu_mail, hat_nu_mail), _, (count_p1, count_p2) = solver_mail.run(horizon=HORIZON, epochs=EPOCHS, gradient_steps=GRADIENT_STEPS, t_max=200, logger=logger)
        
        exploitability_mail = calc_exploitability_true_both(
            mu_pi=hat_mu_mail,
            nu_pi=nu_E,
            reward=rewards,
            transition=transitions,
            initial_dist=initial_dist,
            gamma=GAMMA
        )
        
        duration_mail = (datetime.now() - start_time_mail).total_seconds()
        
        results['DeepMAIL']['K'].append(k)
        results['DeepMAIL']['exploitability'].append(exploitability_mail)
        results['DeepMAIL']['time'].append(duration_mail)
        
        logger.info(f"Deep MAIL Result: K={k}, Exploitability={exploitability_mail:.6f}, Time={duration_mail:.2f}s")

        # --- Deep Uniform ---
        dataset_size = max(count_p1, count_p2)
        dataset_size = max(dataset_size, 1)
        
        # logger.info(f"--- Deep Uniform (Matching Dataset Size={dataset_size}) ---")
        start_time_unif = datetime.now()
        
        solver_unif = DeepUniform(
            K=dataset_size, 
            num_states=NUM_STATES,
            num_actions_p1=num_actions_p1,
            num_actions_p2=num_actions_p2,
            expert_policy_p1=mu_E,
            expert_policy_p2=nu_E,
            transition_P=transitions,
            initial_state_sampler=initial_state_sampler,
            grid_game=wrapper_p1,
            gamma=GAMMA,
            bc_cnn=True, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )
        
        (hat_mu_unif, hat_nu_unif), _ = solver_unif.run(horizon=HORIZON, epochs=EPOCHS, logger=logger, max_transitions=dataset_size)
        
        exploitability_unif = calc_exploitability_true_both(
            mu_pi=hat_mu_unif,
            nu_pi=nu_E,
            reward=rewards,
            transition=transitions,
            initial_dist=initial_dist,
            gamma=GAMMA
        )
        
        duration_unif = (datetime.now() - start_time_unif).total_seconds()
        
        results['DeepUniform']['K'].append(k) 
        results['DeepUniform']['exploitability'].append(exploitability_unif)
        results['DeepUniform']['time'].append(duration_unif)
        
        logger.info(f"Deep Uniform Result: DatasetSize={dataset_size}, Exploitability={exploitability_unif:.6f}, Time={duration_unif:.2f}s")
        
    return results

def run_experiment():
    logger, log_path = setup_logging()
    
    logger.info("="*80)
    logger.info("Chain of Despair Experiment: Multi-N, Multi-Seed")
    logger.info("="*80)
    
    N_VALUES = [4, 8, 16]
    SEEDS = [0, 1, 2]
    
    tasks = [(n, s) for n in N_VALUES for s in SEEDS]
    
    logger.info(f"Starting {len(tasks)} experiments in parallel...")
    
    # Run in parallel
    # Run in parallel
    # n_jobs=3 to run one seed set in parallel (or just a safe number).
    # We use backend='loky' (default) which is robust.
    results_list = Parallel(n_jobs=3)(delayed(run_single_config)(n, s) for n, s in tasks)
    
    # Store all raw results
    # raw_data[n][seed] = results_dict
    raw_data = {}
    for n in N_VALUES:
        raw_data[n] = {}
        
    for (n, seed), res in zip(tasks, results_list):
        raw_data[n][seed] = res

    # Average Results and Plot
    logger.info("\nAggregating results and plotting...")
    plot_dir = Path(__file__).parent.parent / 'results' / 'chain_of_despair'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw results first
    results_path = plot_dir / f'chain_results_raw_{timestamp}.json'
    # Convert numpy types to python for json serialization if needed, though K/expl are usually floats/ints
    # We might need a custom encoder if numpy arrays ended up in there, but lists should be fine.
    with open(results_path, 'w') as f:
        json.dump(raw_data, f, indent=4)
        
    # Plotting
    # Plotting
    for n in N_VALUES:
        if n not in raw_data or not raw_data[n]:
            continue
            
        plt.figure(figsize=(10, 6))
        
        color_mail = 'blue'
        color_unif = 'red'
        
        # Aggregate across seeds
        # Assume all runs have same K values
        # Get K list from first successful seed
        first_seed = list(raw_data[n].keys())[0]
        k_values = raw_data[n][first_seed]['DeepMAIL']['K']
        
        # Deep MAIL
        mail_expl_matrix = []
        # Deep Uniform
        unif_expl_matrix = []
        
        seeds = sorted(raw_data[n].keys())
        for seed in seeds:
            mail_expl_matrix.append(raw_data[n][seed]['DeepMAIL']['exploitability'])
            unif_expl_matrix.append(raw_data[n][seed]['DeepUniform']['exploitability'])
                
        mail_mean = np.mean(mail_expl_matrix, axis=0)
        mail_std = np.std(mail_expl_matrix, axis=0)
        
        unif_mean = np.mean(unif_expl_matrix, axis=0)
        unif_std = np.std(unif_expl_matrix, axis=0)
        
        # Plot DeepMAIL (Solid)
        plt.plot(k_values, mail_mean, marker='o', linestyle='-', linewidth=2, 
                 color=color_mail, label=f'DeepMAIL')
        plt.fill_between(k_values, mail_mean - mail_std, mail_mean + mail_std, 
                         color=color_mail, alpha=0.2)
                 
        # Plot DeepUniform (Dashed)
        plt.plot(k_values, unif_mean, marker='s', linestyle='--', linewidth=2, 
                 color=color_unif, label=f'DeepUniform')
        plt.fill_between(k_values, unif_mean - unif_std, unif_mean + unif_std, 
                         color=color_unif, alpha=0.2)
        
        plt.xlabel('K (Iterations)')
        plt.ylabel(f'Exploitability (Averaged over {len(seeds)} seeds)')
        plt.title(f'Chain of Despair Results: N={n}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = plot_dir / f'chain_of_despair_mn_{timestamp}_N{n}.png'
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
        plt.close()

if __name__ == "__main__":
    run_experiment()
