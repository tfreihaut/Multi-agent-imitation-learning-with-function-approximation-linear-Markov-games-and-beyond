import numpy as np
import random

# -----------------------
# Symmetries & utilities
# -----------------------

SYMMETRIES = [
    (0, 1, 2, 3, 4, 5, 6, 7, 8),  # identity
    (6, 3, 0, 7, 4, 1, 8, 5, 2),  # rotate 90
    (8, 7, 6, 5, 4, 3, 2, 1, 0),  # rotate 180
    (2, 5, 8, 1, 4, 7, 0, 3, 6),  # rotate 270
    (2, 1, 0, 5, 4, 3, 8, 7, 6),  # reflect vertical
    (6, 7, 8, 3, 4, 5, 0, 1, 2),  # reflect horizontal
    (0, 3, 6, 1, 4, 7, 2, 5, 8),  # reflect main diagonal
    (8, 5, 2, 7, 4, 1, 6, 3, 0),  # reflect anti-diagonal
]

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def transform(board, sym):
    """Apply symmetry permutation `sym` (a 9-tuple) to board (tuple of length 9)."""
    return tuple(board[i] for i in sym)


def canonical(board):
    """Return lexicographically smallest board across all symmetries."""
    return min(transform(board, s) for s in SYMMETRIES)


def winner(board):
    """Return 1 if X wins, -1 if O wins, 0 otherwise."""
    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def is_terminal(board):
    """Return True if board is terminal (win or full)."""
    return winner(board) != 0 or 0 not in board


def legal_actions(board):
    """Return tuple of legal action indices on the board (empty positions)."""
    return tuple(i for i, v in enumerate(board) if v == 0)


def apply_move(board, action, player):
    """Return new board after player (1 or -1) plays action."""
    if board[action] != 0:
        raise ValueError(f"Illegal move: cell {action} not empty")
    b = list(board)
    b[action] = player
    return tuple(b)


def player_to_move(board):
    """Return player to move: 1 (X) if X_count == O_count else -1 (O)."""
    x = board.count(1)
    o = board.count(-1)
    return 1 if x == o else -1


# -----------------------
# Policy helpers
# -----------------------


def normalize_prob_dict(d):
    """Normalize a dict of {action:prob} to sum to 1. Raises if total prob is 0."""
    total = sum(d.values())
    if total == 0:
        raise ValueError("Policy gives zero total probability across actions")
    return {a: p / total for a, p in d.items()}


def policy_action_probs_from_entry(entry):
    """
    Convert a policy entry to a dict {action_index: prob}.
    Accepts:
      - int -> {int: 1.0}
      - dict -> assumed to be {action: prob}
      - list/tuple -> interpreted as vector length 9 of probs
    """
    if isinstance(entry, (int, np.integer)):
        return {int(entry): 1.0}
    if isinstance(entry, dict):
        return normalize_prob_dict(entry)
    if isinstance(entry, (list, tuple, np.ndarray)):
        if isinstance(entry, np.ndarray):
            entry = entry.tolist()
        if len(entry) != 9:
            raise ValueError(
                "Policy list/tuple must have length 9 (one prob per action)"
            )
        d = {i: float(entry[i]) for i in range(9) if entry[i] > 0}
        return normalize_prob_dict(d)
    raise ValueError(
        f"Unsupported policy entry type: {type(entry)}; must be int, dict, or length-9 list/tuple"
    )


def policy_lookup(policy_canonical, board):
    """
    Lookup action probability dict for a raw board (not canonical).
    policy_canonical: dict mapping canonical_board -> entry (int/dict/list)
    Returns: dict {action_index: prob} in original board indexing.
    """
    canon = canonical(board)
    if canon not in policy_canonical:
        raise KeyError(
            "Policy missing entry for canonical state. Make sure policy covers canonical states."
        )
    entry = policy_canonical[canon]
    probs_canon = policy_action_probs_from_entry(
        entry
    )  # actions w.r.t canonical indexing

    # Verify that canonical actions are actually legal in the canonical board
    legal_canon = legal_actions(canon)
    for a_canon in probs_canon:
        if a_canon not in legal_canon:
            raise ValueError(
                f"Policy has illegal action {a_canon} for canonical board {canon}. Legal: {legal_canon}"
            )

    # Find symmetry that maps original board -> canon, so we can invert action
    for sym in SYMMETRIES:
        if transform(board, sym) == canon:
            # sym[i] tells us: original position i goes to canonical position sym[i]
            # To invert: if canonical action is at position j, find original position i where sym[i] == j
            probs_orig = {}
            for a_canon, p in probs_canon.items():
                a_orig = sym[a_canon]  # ← CORRECT
                if board[a_orig] != 0:
                    raise ValueError(
                        f"Illegal action after symmetry mapping:\n"
                        f"board={board}\n"
                        f"canon={canon}\n"
                        f"sym={sym}\n"
                        f"a_canon={a_canon}, a_orig={a_orig}"
                    )
                probs_orig[a_orig] = probs_orig.get(a_orig, 0.0) + p

            if not probs_orig:
                raise ValueError(
                    f"Policy assigns probability only to illegal actions for board {board}.\nCanonical: {canon}\nPolicy entry: {entry}\nSymmetry: {sym}"
                )

            # normalize (to avoid tiny floating error)
            total = sum(probs_orig.values())
            return {a: (probs_orig[a] / total) for a in probs_orig}
    raise ValueError(
        "No symmetry mapping found from board to canonical (this should not happen)."
    )


# -----------------------
# Value / Exploitability solvers
# -----------------------


def evaluate_against_expert(
    policy_canonical, expert_policy_canonical, player=1, evaluation_episodes=10000
):
    """
    Compute exact value of `policy_canonical` when used by `player`
    against an expert opponent using `expert_policy_canonical`.
    Returns:
      value (float) = expected outcome for `player` when both sides play
                      according to their policies.
    """
    empty_board = tuple([0] * 9)
    wins_player1 = 0
    draws_player1 = 0
    for episode in range(evaluation_episodes):
        board = empty_board
        to_move = 1  # X starts

        while True:
            w = winner(board)
            if w == 1:
                # X wins
                wins_player1 += 1
                break
            elif w == -1:
                # O wins
                break
            elif 0 not in board:
                # Draw
                draws_player1 += 1
                break

            if to_move == player:
                # Our policy
                probs = policy_canonical[board]
            else:
                # Expert policy
                probs = policy_lookup(expert_policy_canonical, board)
                # To be robust against overfitting, add some random exploration, unless expert has winning move
                if random.random() < 0.1 and not any(
                    winner(apply_move(board, a, to_move)) == to_move
                    for a in list(probs.keys())
                ):
                    legal = legal_actions(board)
                    uniform_probs = {a: 1.0 for a in legal}
                    # Mix expert probs with uniform
                    mixed_probs = {}
                    for a in set(probs.keys()).union(uniform_probs.keys()):
                        p_expert = probs.get(a, 0.0)
                        p_uniform = uniform_probs.get(a, 0.0)
                        mixed_probs[a] = 0.6 * p_expert + 0.4 * p_uniform
                    probs = normalize_prob_dict(mixed_probs)

            if type(probs) is int:
                action = probs
            elif isinstance(probs, dict):
                actions = list(probs.keys())
                probabilities = list(probs.values())
                # ENSURE probabilities sum to 1
                probabilities = [p / sum(probabilities) for p in probabilities]
                action = np.random.choice(actions, p=probabilities)
            else:
                # Sample action according to probs
                probs = [p / sum(probs) for p in probs]
                action = np.random.choice(range(9), p=probs)

            # ensure the action is legal
            if board[action] != 0:
                # cheat: choose random legal action
                legal = legal_actions(board)
                action = np.random.choice(legal)
            board = apply_move(board, action, to_move)
            to_move = -to_move  # switch turns
    # Compute expected value for `player`
    total_games = evaluation_episodes
    total_value = total_games - wins_player1 - draws_player1
    expected_exploit = total_value / total_games

    return abs(expected_exploit)
