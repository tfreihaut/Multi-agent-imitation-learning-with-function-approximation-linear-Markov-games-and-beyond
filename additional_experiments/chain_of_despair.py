import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def make_chain_of_despair_game(num_states=10):
    """
    Creates the Chain of Despair environment.

    Args:
        num_states (int): The number of states N in the chain.

    Returns:
        tuple:
            - rewards (np.ndarray): Shape (num_states, 2, 2). Rewards for player 1.
            - transitions (np.ndarray): Shape (num_states, 2, 2, num_states). Transition probabilities.
            - game_params (dict): Dictionary containing game parameters.
    """
    # Actions:
    # Player 1: a1 (0), a2 (1)
    # Player 2: b1 (0), b2 (1)
    num_actions_p1 = 2
    num_actions_p2 = 2

    # Initialize matrices
    # transitions: P(s' | s, a1, a2)
    # Shape: (num_states, num_actions_p1, num_actions_p2, num_states)
    transitions = np.zeros((num_states, num_actions_p1, num_actions_p2, num_states))
    
    # rewards: R(s, a1, a2) for player 1
    # Shape: (num_states, num_actions_p1, num_actions_p2)
    rewards = np.zeros((num_states, num_actions_p1, num_actions_p2))

    # Define transitions
    for s in range(num_states):
        # s0
        if s == 0:
            # from s0 you can either go in s1 when agent 2 play b1 (no matter what agent 1 plays)
            # or in s2 when agent 2 plays b2 no matter what agent 1 plays.
            for a1 in range(num_actions_p1):
                # P2 plays b1 (0) -> s1
                transitions[s, a1, 0, 1] = 1.0
                # P2 plays b2 (1) -> s2
                transitions[s, a1, 1, 2] = 1.0
        
        # s1: absorbing
        elif s == 1:
            for a1 in range(num_actions_p1):
                for a2 in range(num_actions_p2):
                    transitions[s, a1, a2, s] = 1.0
        
        # s_N-1: absorbing
        elif s == num_states - 1:
            for a1 in range(num_actions_p1):
                for a2 in range(num_actions_p2):
                    transitions[s, a1, a2, s] = 1.0
            
            # Rewards at s_N-1
            # reward for player 1 is +10 if he plays a1 (and -10 for player 2 no matter what he plays)
            # or -2 if he plays a2 (and +2 for player 2 no matter what he plays)
            # Note: rewards matrix stores player 1's reward.
            for a2 in range(num_actions_p2):
                # P1 plays a1 (0)
                rewards[s, 0, a2] = 10.0
                # P1 plays a2 (1)
                rewards[s, 1, a2] = -2.0

        # s2 ... s_N-2
        else:
            # So for s2..N-2:
            # b2 -> s_i+1 (progress)
            # b1 -> s0 (reset)
            
            for a1 in range(num_actions_p1):
                # P2 plays b1 (0) -> s0
                transitions[s, a1, 0, 0] = 1.0
                # P2 plays b2 (1) -> s_i+1
                transitions[s, a1, 1, s + 1] = 1.0

    game_params = {
        'num_states': num_states,
        'num_actions_p1': num_actions_p1,
        'num_actions_p2': num_actions_p2
    }

    return rewards, transitions, game_params

def render_chain(num_states, current_state=None, output_file="chain_state.png"):
    """
    Visualizes the Chain of Despair environment.
    """
    G = nx.Graph() # Undirected graph for visualization (no arrows)
    
    # Add nodes
    for i in range(num_states):
        G.add_node(f"s{i}")
        
    # Add edges for structure (no labels, undirected)
    # s0 connected to s1 (trap) and s2 (chain start)
    G.add_edge("s0", "s1")
    G.add_edge("s0", "s2")
    
    # Chain s2 ... s_N-1
    for i in range(2, num_states - 1):
        G.add_edge(f"s{i}", f"s{i+1}")
        
    # Layout
    pos = {}
    # s0 at (0, 0)
    pos["s0"] = (0, 0)
    # s1 at (0, -1) (Trap below)
    pos["s1"] = (0, -1)
    
    # Chain s2 ... s_N-1
    for i in range(2, num_states):
        pos[f"s{i}"] = (i - 1, 0)
        
    plt.figure(figsize=(10, 4))
    
    # Define node colors
    node_colors = []
    for i in range(num_states):
        if current_state is not None and i == current_state:
            node_colors.append('orange') # Highlight agent position
        else:
            node_colors.append('lightblue') # Standard states (all same color)
            
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_labels(G, pos)
    
    # Draw edges (simple lines)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
    
    plt.title(f"Chain of Despair (N={num_states})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
