"""Agent definitions for Rummikub ML training.

Provides base agent class and several example agents.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod


class RummikubAgent(ABC):
    """Base class for Rummikub agents."""
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self.stats = {
            'games_played': 0,
            'wins': 0,
            'total_reward': 0.0,
            'moves_made': 0,
        }
    
    @abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray], 
                     valid_actions: np.ndarray) -> int:
        """Select an action given the current observation.
        
        Args:
            observation: Current game state
            valid_actions: Binary mask of valid actions
            
        Returns:
            Selected action index
        """
        pass
    
    def reset(self):
        """Reset agent state for a new game."""
        pass
    
    def update_stats(self, won: bool, reward: float, moves: int):
        """Update agent statistics after a game."""
        self.stats['games_played'] += 1
        if won:
            self.stats['wins'] += 1
        self.stats['total_reward'] += reward
        self.stats['moves_made'] += moves
    
    def get_win_rate(self) -> float:
        """Get win rate as percentage."""
        if self.stats['games_played'] == 0:
            return 0.0
        return (self.stats['wins'] / self.stats['games_played']) * 100
    
    def __repr__(self):
        return f"{self.name}(win_rate={self.get_win_rate():.1f}%)"


class RandomAgent(RummikubAgent):
    """Agent that selects random valid actions."""
    
    def __init__(self, name: str = "Random"):
        super().__init__(name)
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select random valid action."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0  # Default to DRAW
        return random.choice(valid_indices)


class HeuristicAgent(RummikubAgent):
    """Agent using simple heuristics to play."""
    
    def __init__(self, name: str = "Heuristic"):
        super().__init__(name)
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action based on heuristics."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Priority order:
        # 1. DECLARE_OUT if possible (win)
        if 1 in valid_indices:
            return 1
        
        # 2. PLAY_NEW_MELD (actions 2 to 2+max_hand_size)
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            # Prefer playing melds over adding to existing
            return random.choice(meld_actions)
        
        # 3. ADD_TO_MELD (remaining actions)
        add_actions = [a for a in valid_indices if a >= 2 + 30]
        if add_actions:
            return random.choice(add_actions)
        
        # 4. DRAW as last resort
        if 0 in valid_indices:
            return 0
        
        return random.choice(valid_indices)


class GreedyAgent(RummikubAgent):
    """Agent that tries to play tiles as aggressively as possible."""
    
    def __init__(self, name: str = "Greedy"):
        super().__init__(name)
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action prioritizing playing tiles."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # 1. Win if possible
        if 1 in valid_indices:
            return 1
        
        # 2. Play largest meld first (heuristic: more tiles = better)
        # Actions 2-31 are PLAY_NEW_MELD
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            # Try to find a valid meld - for now just pick one
            # In a more sophisticated version, we'd decode which tiles
            return max(meld_actions)  # Higher index might mean more tiles
        
        # 3. Add to existing meld
        add_actions = [a for a in valid_indices if a >= 2 + 30]
        if add_actions:
            return random.choice(add_actions)
        
        # 4. Draw if nothing else
        return 0 if 0 in valid_indices else random.choice(valid_indices)


class ConservativeAgent(RummikubAgent):
    """Agent that plays defensively, only making safe moves."""
    
    def __init__(self, name: str = "Conservative"):
        super().__init__(name)
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select conservative actions."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Only play if we can win
        if 1 in valid_indices:
            return 1
        
        # Prefer drawing to accumulate more options
        if 0 in valid_indices:
            return 0
        
        # Otherwise play something
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            return random.choice(meld_actions)
        
        add_actions = [a for a in valid_indices if a >= 2 + 30]
        if add_actions:
            return random.choice(add_actions)
        
        return random.choice(valid_indices)


class WormAgent(RummikubAgent):
    """Agent that uses worm.py solver logic for guidance.
    
    This agent attempts to use the constraint propagation logic
    from worm.py to make better decisions.
    """
    
    def __init__(self, name: str = "Worm"):
        super().__init__(name)
        self._import_worm()
    
    def _import_worm(self):
        """Try to import worm.py functions."""
        try:
            import sys
            sys.path.insert(0, 'worm')
            from worm import find_explicit_moves, get_all_runs, get_all_groups
            self.worm_available = True
            self.find_explicit_moves = find_explicit_moves
            self.get_all_runs = get_all_runs
            self.get_all_groups = get_all_groups
        except ImportError:
            self.worm_available = False
            print("Warning: worm.py not available, falling back to heuristic")
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action using worm guidance."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Win if possible
        if 1 in valid_indices:
            return 1
        
        # Try to use worm logic to find definite moves
        # This would require converting observation back to board format
        # For now, fall back to heuristic
        
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            return random.choice(meld_actions)
        
        add_actions = [a for a in valid_indices if a >= 2 + 30]
        if add_actions:
            return random.choice(add_actions)
        
        return 0 if 0 in valid_indices else random.choice(valid_indices)


class QLearningAgent(RummikubAgent):
    """Q-learning agent with simple neural network.
    
    This is a placeholder for a proper RL agent.
    """
    
    def __init__(self, name: str = "QLearning", learning_rate: float = 0.001):
        super().__init__(name)
        self.learning_rate = learning_rate
        self.q_table = {}  # Simple state-action table (not scalable)
        self.epsilon = 0.1  # Exploration rate
        self.last_state = None
        self.last_action = None
    
    def _state_to_key(self, observation: Dict[str, np.ndarray]) -> str:
        """Convert observation to hashable key."""
        # Simplified state representation
        hand_tuple = tuple(observation['hand'][:10])  # First 10 tiles
        return str(hand_tuple)
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        state_key = self._state_to_key(observation)
        valid_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_indices) == 0:
            return 0
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(valid_indices)
        else:
            # Choose best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            q_values = self.q_table[state_key]
            best_action = None
            best_value = float('-inf')
            
            for action in valid_indices:
                value = q_values.get(action, 0.0)
                if value > best_value:
                    best_value = value
                    best_action = action
            
            action = best_action if best_action is not None else random.choice(valid_indices)
        
        self.last_state = state_key
        self.last_action = action
        return action
    
    def update(self, reward: float, next_observation: Dict[str, np.ndarray]):
        """Update Q-value based on reward."""
        if self.last_state is None or self.last_action is None:
            return
        
        # Simple Q-learning update
        if self.last_state not in self.q_table:
            self.q_table[self.last_state] = {}
        
        current_q = self.q_table[self.last_state].get(self.last_action, 0.0)
        
        # Estimate future value (simplified)
        next_state_key = self._state_to_key(next_observation)
        if next_state_key in self.q_table:
            next_q = max(self.q_table[next_state_key].values(), default=0.0)
        else:
            next_q = 0.0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + 0.99 * next_q - current_q)
        self.q_table[self.last_state][self.last_action] = new_q
    
    def reset(self):
        """Reset episode state."""
        self.last_state = None
        self.last_action = None


class HoardingAgent(RummikubAgent):
    """Agent that hoards tiles and tries to go out in one big play.
    
    Uses worm logic to find moves but prefers to wait until it can
    make a significant play (long runs, multiple groups) or go out entirely.
    """
    
    def __init__(self, name: str = "Hoarding", tiles_to_hoard: int = 10):
        super().__init__(name)
        self.tiles_to_hoard = tiles_to_hoard
        self._import_worm_integration()
    
    def _import_worm_integration(self):
        """Try to import worm integration functions."""
        try:
            from worm_integration import (
                find_hand_only_moves,
                get_best_move_for_strategy,
                can_go_out_in_one_move,
                score_move_for_hoarding
            )
            self.worm_available = True
            self.find_hand_only_moves = find_hand_only_moves
            self.get_best_move_for_strategy = get_best_move_for_strategy
            self.can_go_out_in_one_move = can_go_out_in_one_move
            self.score_move_for_hoarding = score_move_for_hoarding
        except ImportError:
            self.worm_available = False
            print(f"Warning: worm_integration not available for {self.name}")
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action - hoard tiles until big play possible."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Always win if possible
        if 1 in valid_indices:
            return 1
        
        # Count tiles in hand
        hand_size = int(observation['hand_mask'].sum())
        has_initial = observation['has_initial_meld'][0] > 0
        
        # If we can go out in one move, do it!
        if self.worm_available and hand_size >= 3:
            # Decode hand from observation
            hand = self._decode_hand(observation)
            if self.can_go_out_in_one_move(hand, []):
                # Find the best move
                best_move = self.get_best_move_for_strategy(hand, [], 'hoarding')
                if best_move:
                    # Try to play it
                    meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
                    if meld_actions:
                        return meld_actions[0]  # Pick first valid meld action
        
        # If we haven't reached hoarding threshold and don't have initial meld, draw more
        if hand_size < self.tiles_to_hoard and not has_initial:
            if 0 in valid_indices:
                return 0
        
        # If we have initial meld and can make a big play (5+ tiles), do it
        if has_initial and hand_size >= 5 and self.worm_available:
            hand = self._decode_hand(observation)
            moves = self.find_hand_only_moves(hand)
            
            # Find moves with 5+ tiles
            big_moves = []
            for move in moves:
                if move[0] == 'run':
                    _, row, start, end = move
                    if end - start + 1 >= 5:
                        big_moves.append(move)
                elif move[0] == 'group':
                    _, col, rows = move
                    if len(rows) >= 4:
                        big_moves.append(move)
            
            if big_moves:
                meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
                if meld_actions:
                    return meld_actions[0]
        
        # Otherwise, prefer drawing to accumulate more tiles
        if 0 in valid_indices and not has_initial:
            return 0
        
        # If must play, pick a valid meld
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            return random.choice(meld_actions)
        
        add_actions = [a for a in valid_indices if a >= 2 + 30]
        if add_actions:
            return random.choice(add_actions)
        
        return 0 if 0 in valid_indices else random.choice(valid_indices)
    
    def _decode_hand(self, observation: Dict[str, np.ndarray]) -> List[Any]:
        """Decode hand tiles from observation."""
        from tile import Tile
        hand = []
        for i, tile_code in enumerate(observation['hand']):
            if observation['hand_mask'][i] > 0:
                try:
                    hand.append(Tile.decode(int(tile_code)))
                except:
                    pass
        return hand


class AggressiveAgent(RummikubAgent):
    """Agent that plays tiles as aggressively as possible.
    
    Uses worm logic to find and play the best available move immediately.
    Never draws if it can play something.
    """
    
    def __init__(self, name: str = "Aggressive"):
        super().__init__(name)
        self._import_worm_integration()
    
    def _import_worm_integration(self):
        """Try to import worm integration functions."""
        try:
            from worm_integration import (
                find_hand_only_moves,
                get_best_move_for_strategy,
                score_move_for_aggressive
            )
            self.worm_available = True
            self.find_hand_only_moves = find_hand_only_moves
            self.get_best_move_for_strategy = get_best_move_for_strategy
            self.score_move_for_aggressive = score_move_for_aggressive
        except ImportError:
            self.worm_available = False
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action - play aggressively."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Always win if possible
        if 1 in valid_indices:
            return 1
        
        # Prefer playing new melds over anything else
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            if self.worm_available:
                # Use worm to find best aggressive move
                hand = self._decode_hand(observation)
                best_move = self.get_best_move_for_strategy(hand, [], 'aggressive')
                if best_move:
                    return meld_actions[0]  # Play the best move found
            return random.choice(meld_actions)
        
        # Next preference: add to existing melds
        add_actions = [a for a in valid_indices if a >= 2 + 30]
        if add_actions:
            return random.choice(add_actions)
        
        # Only draw if absolutely necessary
        if 0 in valid_indices:
            return 0
        
        return random.choice(valid_indices)
    
    def _decode_hand(self, observation: Dict[str, np.ndarray]) -> List[Any]:
        """Decode hand tiles from observation."""
        from tile import Tile
        hand = []
        for i, tile_code in enumerate(observation['hand']):
            if observation['hand_mask'][i] > 0:
                try:
                    hand.append(Tile.decode(int(tile_code)))
                except:
                    pass
        return hand


class BalancedAgent(RummikubAgent):
    """Agent with balanced strategy between hoarding and aggressive.
    
    Uses worm logic to evaluate moves and makes intelligent decisions
    about when to play vs when to draw.
    """
    
    def __init__(self, name: str = "Balanced", play_threshold: float = 0.6):
        super().__init__(name)
        self.play_threshold = play_threshold
        self._import_worm_integration()
    
    def _import_worm_integration(self):
        """Try to import worm integration functions."""
        try:
            from worm_integration import (
                find_hand_only_moves,
                get_best_move_for_strategy,
                score_move_for_balanced,
                can_go_out_in_one_move
            )
            self.worm_available = True
            self.find_hand_only_moves = find_hand_only_moves
            self.get_best_move_for_strategy = get_best_move_for_strategy
            self.score_move_for_balanced = score_move_for_balanced
            self.can_go_out_in_one_move = can_go_out_in_one_move
        except ImportError:
            self.worm_available = False
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action using balanced strategy."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Always win if possible
        if 1 in valid_indices:
            return 1
        
        hand_size = int(observation['hand_mask'].sum())
        has_initial = observation['has_initial_meld'][0] > 0
        
        # Use worm logic to evaluate best move
        if self.worm_available and hand_size >= 3:
            hand = self._decode_hand(observation)
            moves = self.find_hand_only_moves(hand)
            
            if moves:
                # Score each move
                scored = [(move, self.score_move_for_balanced(move, hand, [])) 
                         for move in moves]
                scored.sort(key=lambda x: x[1], reverse=True)
                
                best_move, best_score = scored[0]
                
                # If score is above threshold, play it
                if best_score >= self.play_threshold * 10:  # Normalize threshold
                    meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
                    if meld_actions:
                        return meld_actions[0]
                
                # If we can go out in one move, always play
                if self.can_go_out_in_one_move(hand, []):
                    meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
                    if meld_actions:
                        return meld_actions[0]
        
        # If we don't have initial meld and can play one, do it
        if not has_initial:
            meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
            if meld_actions:
                return random.choice(meld_actions)
            # Need to draw to get initial meld
            if 0 in valid_indices:
                return 0
        
        # With initial meld, prefer playing but will draw occasionally
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions and random.random() < 0.8:  # 80% chance to play
            return random.choice(meld_actions)
        
        add_actions = [a for a in valid_indices if a >= 2 + 30]
        if add_actions and random.random() < 0.9:  # 90% chance to add
            return random.choice(add_actions)
        
        # Occasionally draw to improve hand
        if 0 in valid_indices:
            return 0
        
        return random.choice(valid_indices)
    
    def _decode_hand(self, observation: Dict[str, np.ndarray]) -> List[Any]:
        """Decode hand tiles from observation."""
        from tile import Tile
        hand = []
        for i, tile_code in enumerate(observation['hand']):
            if observation['hand_mask'][i] > 0:
                try:
                    hand.append(Tile.decode(int(tile_code)))
                except:
                    pass
        return hand


class SmartWormAgent(RummikubAgent):
    """Agent that deeply integrates worm.py solver logic.
    
    Uses constraint propagation and move analysis to make optimal decisions.
    """
    
    def __init__(self, name: str = "SmartWorm", strategy: str = "balanced"):
        super().__init__(name)
        self.strategy = strategy
        self._import_worm()
    
    def _import_worm(self):
        """Import worm functions."""
        try:
            from worm_integration import (
                find_hand_only_moves,
                find_combined_moves,
                get_best_move_for_strategy,
                hand_to_board_matrix,
                table_to_board_matrix,
                combine_boards
            )
            from worm import solve, find_explicit_moves
            
            self.worm_available = True
            self.find_hand_only_moves = find_hand_only_moves
            self.find_combined_moves = find_combined_moves
            self.get_best_move_for_strategy = get_best_move_for_strategy
            self.hand_to_board_matrix = hand_to_board_matrix
            self.table_to_board_matrix = table_to_board_matrix
            self.combine_boards = combine_boards
            self.worm_solve = solve
            self.find_explicit_moves = find_explicit_moves
        except ImportError as e:
            self.worm_available = False
            print(f"Warning: worm/worm_integration not available: {e}")
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action using worm solver logic."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Always win if possible
        if 1 in valid_indices:
            return 1
        
        if not self.worm_available:
            # Fall back to heuristic
            meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
            if meld_actions:
                return random.choice(meld_actions)
            return 0 if 0 in valid_indices else random.choice(valid_indices)
        
        # Decode game state
        hand = self._decode_hand(observation)
        table = self._decode_table(observation)
        
        # Use worm to find definite moves
        hand_board = self.hand_to_board_matrix(hand)
        definite = self.find_explicit_moves(hand_board)
        
        if definite:
            # Definite moves are forced - play them
            meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
            if meld_actions:
                return meld_actions[0]
        
        # Note: worm_solve can be very slow on complex hands
        # Skipping full solve for performance - using heuristic instead
        try:
            solution = self.worm_solve(hand_board)
            if solution and len(solution) > 0:
                meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
                if meld_actions:
                    return meld_actions[0]
        except:
            pass
        
        # Find best move based on strategy
        best_move = self.get_best_move_for_strategy(hand, table, self.strategy)
        if best_move:
            meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
            if meld_actions:
                return random.choice(meld_actions)
        
        # Try add-to-meld
        combined = self.find_combined_moves(hand, table)
        if combined['add_to_meld']:
            add_actions = [a for a in valid_indices if a >= 2 + 30]
            if add_actions:
                return random.choice(add_actions)
        
        # Draw if nothing else
        if 0 in valid_indices:
            return 0
        
        return random.choice(valid_indices)
    
    def _decode_hand(self, observation: Dict[str, np.ndarray]) -> List[Any]:
        """Decode hand tiles from observation."""
        from tile import Tile
        hand = []
        for i, tile_code in enumerate(observation['hand']):
            if observation['hand_mask'][i] > 0:
                try:
                    hand.append(Tile.decode(int(tile_code)))
                except:
                    pass
        return hand
    
    def _decode_table(self, observation: Dict[str, np.ndarray]) -> List[Any]:
        """Decode table melds from observation."""
        from meld import Meld
        from tile import Tile
        
        melds = []
        for meld_idx in range(20):
            tiles = []
            for tile_idx in range(13):
                if observation['table_mask'][meld_idx, tile_idx] > 0:
                    try:
                        tile_code = int(observation['table'][meld_idx, tile_idx])
                        tiles.append(Tile.decode(tile_code))
                    except:
                        pass
            if len(tiles) >= 3:
                try:
                    melds.append(Meld(tiles))
                except:
                    pass
        return melds


if __name__ == "__main__":
    # Test agents
    print("Testing agents...")
    
    agents = [
        RandomAgent(),
        HeuristicAgent(),
        GreedyAgent(),
        ConservativeAgent(),
        WormAgent(),
        QLearningAgent(),
        HoardingAgent("Hoarding", tiles_to_hoard=10),
        AggressiveAgent("Aggressive"),
        BalancedAgent("Balanced"),
        SmartWormAgent("SmartWorm"),
    ]
    
    # Mock observation
    obs = {
        'hand': np.zeros(30, dtype=np.int32),
        'hand_mask': np.zeros(30, dtype=np.float32),
        'table': np.zeros((20, 13), dtype=np.int32),
        'table_mask': np.zeros((20, 13), dtype=np.float32),
        'pool_size': np.array([50], dtype=np.int32),
        'has_initial_meld': np.array([0], dtype=np.float32),
        'opponent_hands': np.array([14], dtype=np.int32),
        'valid_actions_mask': np.zeros(652, dtype=np.float32),
        'current_player': np.array([0], dtype=np.int32),
        'turn_count': np.array([0], dtype=np.int32),
    }
    
    # Set some valid actions
    obs['valid_actions_mask'][0] = 1.0  # DRAW
    obs['valid_actions_mask'][2] = 1.0  # PLAY_NEW_MELD
    obs['valid_actions_mask'][3] = 1.0  # PLAY_NEW_MELD
    
    print("\nAgent actions:")
    for agent in agents:
        action = agent.select_action(obs, obs['valid_actions_mask'])
        print(f"  {agent.name}: action={action}")
