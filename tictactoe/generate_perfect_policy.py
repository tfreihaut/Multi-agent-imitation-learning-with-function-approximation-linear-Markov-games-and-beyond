from itertools import product
from functools import lru_cache

# -----------------------------
# Board utilities
# -----------------------------

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6),
]

SYMMETRIES = [
    (0,1,2,3,4,5,6,7,8),
    (6,3,0,7,4,1,8,5,2),
    (8,7,6,5,4,3,2,1,0),
    (2,5,8,1,4,7,0,3,6),
    (2,1,0,5,4,3,8,7,6),
    (6,7,8,3,4,5,0,1,2),
    (0,3,6,1,4,7,2,5,8),
    (8,5,2,7,4,1,6,3,0),
]

def transform(board, sym):
    return tuple(board[i] for i in sym)

def canonical(board):
    return min(transform(board, s) for s in SYMMETRIES)

def winner(board):
    for a,b,c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0

def legal(board):
    x = board.count(1)
    o = board.count(-1)
    if not (x == o or x == o + 1):
        return False
    if winner(board) and x == o:
        return False
    return True

# -----------------------------
# Minimax (perfect play)
# -----------------------------

@lru_cache(None)
def minimax(board, player):
    w = winner(board)
    if w:
        # Score is weighted by how "early" the game is.
        # More empty spots (0s) == Earlier in the game.
        empty_spots = board.count(0)
        
        # If I (player) won, return positive score. 
        # If I lost, the score will be negative relative to me.
        # We multiply by 'player' so the score is always relative to the active turn.
        return w * player * (1 + empty_spots), None 

    if 0 not in board:
        return 0, None

    best_score = -float('inf') # Use infinity for comparison
    best_move = None

    for i in range(9):
        if board[i] == 0:
            b = list(board)
            b[i] = player
            score, _ = minimax(tuple(b), -player)
            score = -score 
            
            if score > best_score:
                best_score = score
                best_move = i
    
    return best_score, best_move

# -----------------------------
# Generate perfect policy
# -----------------------------

def generate_policy():
    boards = [
        b for b in product([0,1,-1], repeat=9)
        if legal(b)
    ]

    canonical_states = set(canonical(b) for b in boards)
    print("Canonical states:", len(canonical_states))  # should be 765

    policy = {}

    for state in canonical_states:
        player = 1 if state.count(1) == state.count(-1) else -1
        _, move = minimax(state, player)
        policy[state] = move

    return policy

if __name__ == "__main__":
    policy = generate_policy()
    print("Perfect policy size:", len(policy))

