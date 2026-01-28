import torch
import numpy as np
from scipy.stats import entropy

from bitbully import Board, BitBully



def evaluate_policy(policy, policy_idx, env, num_episodes=1, argmax=True, suboptimal=0.0):
    """
    Evaluate `policy` (policy_idx indicates which player index it controls)
    against the C4SolverAgent (which wraps the compiled connect4 solver).
    
    Returns:
        wins_policy_0: number of wins for policy player
        wins_policy_1: number of wins for opponent
        average_length: average game length
        opponent_entropies: list of entropy values from opponent's decisions
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.eval().to(device)

    wins_policy_0 = 0
    wins_policy_1 = 0
    average_length = 0
    opponent_entropies = []  # Track entropy of opponent's action distributions

    for episode in range(num_episodes):
        env.reset()
        bb_board = Board()
        bb_agent = BitBully()
        moves = []

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()

            # --- Episode ended ---
            if term or trunc:
                action = None
                if reward == 1:
                    if agent == "player_0":
                        print(f"Episode {episode+1} ended.")
                        print(f"  Final agent: {agent}, reward: {reward}")
                        wins_policy_0 += 1
                    else:
                        wins_policy_1 += 1
                    

            
            else:
                current_player = 0 if agent == "player_0" else 1

                # --- Policy-controlled player ---
                if current_player == policy_idx:
                    state = obs["observation"]

                    # convert HWC -> CHW and add batch dim
                    obs_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = policy(obs_tensor)

                    mask = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0).to(device)
                    logits[~mask] = -1e10
                    # Sample from masked logits
                    if argmax:
                        action = int(logits.argmax(dim=1).item())
                    else:
                        action = torch.distributions.Categorical(logits=logits).sample().int().item()

                # --- Solver-controlled player ---
                else:
                    scores = bb_agent.score_next_moves(bb_board)
                    legal_mask = np.array(obs["action_mask"], dtype=bool)
                    
                    if suboptimal > 0.0:
                        masked_scores = np.array(scores, dtype=float)
                        masked_scores[~legal_mask] = -1e10
                        # Scale scores with suboptimality factor seen as temperature
                        masked_scores = masked_scores / suboptimal
                        # Sample from softmax
                        exp_scores = np.exp(masked_scores - np.max(masked_scores))
                        probs = exp_scores / exp_scores.sum()
                        
                        # Calculate entropy of the legal action probabilities
                        legal_probs = probs[legal_mask]
                        if len(legal_probs) > 0:
                            action_entropy = entropy(legal_probs)
                            opponent_entropies.append(action_entropy)
                        
                        action = np.random.choice(len(scores), p=probs)

                    elif (np.array(scores) < 0).all():
                        # take scores as probabilities if all negative
                        probs = np.exp(scores - np.max(scores))
                        # illegal moves get zero probability
                        probs *= legal_mask
                        probs /= probs.sum()
                        
                        # Calculate entropy of the legal action probabilities
                        legal_probs = probs[legal_mask]
                        if len(legal_probs) > 0:
                            action_entropy = entropy(legal_probs)
                            opponent_entropies.append(action_entropy)
                        
                        action = np.random.choice(len(scores), p=probs)
               
                    else:
                        sorted_indices = np.argsort(scores)                        
                        # Check if action is losing move
                        copied_board = bb_board.copy()
                        copied_board.play(int(sorted_indices[-1]))
                        if copied_board.can_win_next():
                            action = int(sorted_indices[-1])
                        else:
                            # create softmax to force opponent out of distribution
                            probs = np.exp(scores - np.max(scores))
                            # illegal moves get zero probability
                            probs *= legal_mask
                            probs /= probs.sum()
                        
                            # Calculate entropy of the legal action probabilities
                            legal_probs = probs[legal_mask]

                            action = np.random.choice(len(scores), p=probs)
                        # For deterministic case, entropy is 0
                        opponent_entropies.append(0.0)

                moves.append(action)
                bb_board.play(action)

            env.step(action)

        # print(f"Episode {episode + 1} moves: {moves}")
        average_length += len(moves)
    env.close()
    average_length /= num_episodes
    
    print(f"\nFinal results after {num_episodes} episodes:")
    print(f"  Policy {policy_idx} - Wins: {wins_policy_0}, Losses: {wins_policy_1}, Draws: {num_episodes - wins_policy_0 - wins_policy_1}, Average Length: {average_length:.2f} moves")
    return wins_policy_0, wins_policy_1, average_length, opponent_entropies
