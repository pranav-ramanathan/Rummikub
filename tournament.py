"""Tournament system for training and evaluating Rummikub agents.

Allows multiple agents to play against each other and tracks statistics.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

from ml_environment import RummikubMLEnv
from agent import RummikubAgent


class ProgressTracker:
    """Tracks and displays tournament progress with statistics."""
    
    def __init__(self, total_games: int, update_interval: float = 1.0):
        """Initialize progress tracker.
        
        Args:
            total_games: Total number of games to run
            update_interval: How often to update display (seconds)
        """
        self.total_games = total_games
        self.completed_games = 0
        self.start_time = None
        self.update_interval = update_interval
        self.last_update = 0
        self.agent_stats = defaultdict(lambda: {'wins': 0, 'games': 0})
        self.current_matchup = ""
        self.win_streaks = defaultdict(int)
        self.last_winner = None
        
    def start(self):
        """Start tracking progress."""
        self.start_time = time.time()
        self.last_update = self.start_time
        self._print_header()
        
    def _print_header(self):
        """Print progress header."""
        print("\n" + "="*80)
        print(f"{'TOURNAMENT PROGRESS':^80}")
        print("="*80)
        print(f"Total Games: {self.total_games}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80)
        
    def update(self, game_result: Dict):
        """Update progress with a completed game.
        
        Args:
            game_result: Result from run_game()
        """
        self.completed_games += 1
        
        # Update agent stats
        agents = game_result['agents']
        winner = game_result['winner_name']
        
        for agent in agents:
            self.agent_stats[agent]['games'] += 1
        
        if winner != 'None':
            self.agent_stats[winner]['wins'] += 1
            
            # Track win streaks
            if winner == self.last_winner:
                self.win_streaks[winner] += 1
            else:
                if self.last_winner:
                    self.win_streaks[self.last_winner] = 0
                self.win_streaks[winner] = 1
                self.last_winner = winner
        
        # Update current matchup info
        self.current_matchup = " vs ".join(agents)
        
        # Check if we should display update
        # Update on: first game, interval elapsed, or last game
        current_time = time.time()
        if self.completed_games == 1 or \
           current_time - self.last_update >= self.update_interval or \
           self.completed_games == self.total_games:
            self._display_progress()
            self.last_update = current_time
            
    def _display_progress(self):
        """Display current progress."""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        progress_pct = (self.completed_games / self.total_games) * 100
        
        # Calculate ETA
        if self.completed_games > 0 and elapsed > 0:
            games_per_sec = self.completed_games / elapsed
            remaining_games = self.total_games - self.completed_games
            eta_seconds = remaining_games / games_per_sec if games_per_sec > 0 else 0
            eta_str = self._format_time(eta_seconds)
        else:
            games_per_sec = 0
            eta_str = "--:--:--"
        
        # Build progress bar
        bar_width = 40
        filled = int(bar_width * self.completed_games / self.total_games)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Clear screen and print progress
        sys.stdout.write("\033[2J\033[H")  # Clear screen and move to top
        sys.stdout.flush()
        
        print("\n" + "="*80)
        print(f"{'TOURNAMENT PROGRESS':^80}")
        print("="*80)
        print(f"Progress: [{bar}] {progress_pct:.1f}%")
        print(f"Games: {self.completed_games}/{self.total_games} | "
              f"Rate: {games_per_sec:.1f} games/sec | "
              f"ETA: {eta_str}")
        print(f"Elapsed: {self._format_time(elapsed)}")
        print("-"*80)
        
        # Current matchup
        if self.current_matchup:
            print(f"Current: {self.current_matchup}")
            print()
        
        # Agent standings
        print("Current Standings:")
        print(f"{'Rank':<6} {'Agent':<20} {'Wins':<8} {'Games':<8} {'Win %':<10} {'Streak':<8}")
        print("-"*80)
        
        # Sort by win rate
        standings = []
        for agent, stats in self.agent_stats.items():
            win_rate = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
            standings.append((agent, stats['wins'], stats['games'], win_rate, 
                            self.win_streaks[agent]))
        
        standings.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (agent, wins, games, win_rate, streak) in enumerate(standings, 1):
            streak_str = f"{streak}W" if streak > 1 else "-"
            print(f"{rank:<6} {agent:<20} {wins:<8} {games:<8} {win_rate:>6.1f}%   {streak_str:<8}")
        
        print()
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
    def finish(self):
        """Display final summary."""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*80)
        print(f"{'TOURNAMENT COMPLETE':^80}")
        print("="*80)
        print(f"Total Games: {self.completed_games}/{self.total_games}")
        print(f"Total Time: {self._format_time(elapsed)}")
        if elapsed > 0:
            print(f"Average: {self.completed_games/elapsed:.2f} games/sec")
        print()
        

class Tournament:
    """Manages games between multiple agents."""
    
    def __init__(self, agents: List[RummikubAgent], num_players: int = 2):
        """Initialize tournament.
        
        Args:
            agents: List of agents to compete
            num_players: Number of players per game
        """
        self.agents = agents
        self.num_players = num_players
        self.env = RummikubMLEnv(num_players=num_players)
        
        # Statistics
        self.game_results = []
        self.agent_stats = {agent.name: {
            'wins': 0,
            'games': 0,
            'total_reward': 0.0,
            'avg_game_length': 0,
        } for agent in agents}
    
    def run_game(self, agent_indices: List[int], seed: Optional[int] = None,
                 verbose: bool = False) -> Dict:
        """Run a single game with specified agents.
        
        Args:
            agent_indices: Indices of agents to use (length must match num_players)
            seed: Random seed
            verbose: Print game details
            
        Returns:
            Game result dict
        """
        print(f"[DEBUG] Starting run_game with agents: {agent_indices}, seed={seed}")
        
        if len(agent_indices) != self.num_players:
            raise ValueError(f"Need {self.num_players} agents, got {len(agent_indices)}")
        
        game_agents = [self.agents[i] for i in agent_indices]
        print(f"[DEBUG] Game agents: {[a.name for a in game_agents]}")
        
        # Reset environment and agents
        print("[DEBUG] Calling env.reset()...")
        obs = self.env.reset(seed=seed)
        print("[DEBUG] env.reset() completed")
        for agent in game_agents:
            agent.reset()
        
        done = False
        step = 0
        current_player = 0
        game_history = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting game with: {[a.name for a in game_agents]}")
            print(f"{'='*60}")
        
        info = {}
        last_progress_step = 0
        import time
        step_start_time = time.time()
        
        while not done and step < self.env.max_steps:
            agent = game_agents[current_player]
            valid_actions = obs['valid_actions_mask']
            
            # Debug: Check if we're stuck
            elapsed = time.time() - step_start_time
            if elapsed > 5.0:  # 5 seconds per step timeout
                print(f"\n[ERROR] Step timeout at step {step}! Breaking.")
                print(f"[ERROR] Current player: {current_player} ({agent.name})")
                print(f"[ERROR] Last action: {info.get('action_taken', 'UNKNOWN')}")
                break
            
            if step - last_progress_step > 100:
                print(f"\n[DEBUG] Possible hang at step {step}")
                print(f"[DEBUG] Current player: {current_player} ({agent.name})")
                print(f"[DEBUG] Valid actions count: {np.sum(valid_actions)}")
                print(f"[DEBUG] Hand size: {np.sum(obs['hand_mask'])}")
                print(f"[DEBUG] Pool size: {obs['pool_size'][0]}")
                print(f"[DEBUG] Has initial meld: {obs['has_initial_meld'][0]}")
                last_progress_step = step
            
            # Agent selects action
            print(f"[Step {step}] {agent.name} selecting action...")
            action = agent.select_action(obs, valid_actions)
            print(f"[Step {step}] {agent.name} selected Action {action}")
            
            # Execute action
            print(f"[Step {step}] Executing action...")
            next_obs, reward, done, info = self.env.step(action)
            print(f"[Step {step}] Result: {info.get('action_taken', 'UNKNOWN')}, reward={reward:.1f}, done={done}")
            
            step_start_time = time.time()  # Reset timer after successful step
            
            # Update Q-learning agents if applicable
            if hasattr(agent, 'update'):
                agent.update(reward, next_obs)
            
            game_history.append({
                'step': step,
                'player': current_player,
                'agent': agent.name,
                'action': action,
                'action_type': info.get('action_taken', 'UNKNOWN'),
                'reward': reward,
            })
            
            if verbose and info.get('action_success'):
                print(f"Step {step}: {agent.name} took {info['action_taken']} "
                      f"(reward: {reward:.2f})")
            
            obs = next_obs
            current_player = obs['current_player'][0]
            step += 1
            
            if done:
                break
        
        # Determine winner
        winner_idx = info.get('winner')
        if winner_idx is None:
            winner_idx = -1
        
        result = {
            'agents': [a.name for a in game_agents],
            'winner': winner_idx,
            'winner_name': game_agents[winner_idx].name if winner_idx >= 0 else 'None',
            'steps': step,
            'seed': seed,
            'history': game_history if verbose else None,
        }
        
        # Update stats
        for i, agent in enumerate(game_agents):
            self.agent_stats[agent.name]['games'] += 1
            if i == winner_idx:
                self.agent_stats[agent.name]['wins'] += 1
        
        self.game_results.append(result)
        
        if verbose:
            print(f"\nGame ended after {step} steps")
            print(f"Winner: {result['winner_name']}")
        
        return result
    
    def run_round_robin(self, games_per_matchup: int = 10,
                        verbose: bool = False,
                        show_progress: bool = True,
                        progress_interval: float = 2.0) -> Dict:
        """Run round-robin tournament where each agent plays each other.
        
        Args:
            games_per_matchup: Number of games per agent pair
            verbose: Print progress text
            show_progress: Show visual progress tracker
            progress_interval: How often to update progress display (seconds)
            
        Returns:
            Tournament results
        """
        if self.num_players != 2:
            raise ValueError("Round-robin only supported for 2 players")
        
        num_agents = len(self.agents)
        total_games = num_agents * (num_agents - 1) * games_per_matchup
        
        print(f"\nRunning Round-Robin Tournament")
        print(f"Agents: {[a.name for a in self.agents]}")
        print(f"Games per matchup: {games_per_matchup}")
        print(f"Total games: {total_games}")
        print()
        
        # Initialize progress tracker
        tracker = None
        if show_progress:
            tracker = ProgressTracker(total_games, update_interval=progress_interval)
            tracker.start()
        
        games_completed = 0
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                
                for game_num in range(games_per_matchup):
                    seed = random.randint(0, 1000000)
                    result = self.run_game([i, j], seed=seed, verbose=False)
                    games_completed += 1
                    
                    if tracker:
                        tracker.update(result)
                    
                    if verbose and not show_progress and game_num == 0:
                        print(f"{self.agents[i].name} vs {self.agents[j].name}: "
                              f"Winner = {result['winner_name']}")
        
        if tracker:
            tracker.finish()
        
        return self.get_results()
    
    def run_random_matchups(self, num_games: int = 100,
                            verbose: bool = False,
                            show_progress: bool = True,
                            progress_interval: float = 2.0) -> Dict:
        """Run random matchups between agents.
        
        Args:
            num_games: Total number of games to run
            verbose: Print progress text
            show_progress: Show visual progress tracker
            progress_interval: How often to update progress display (seconds)
            
        Returns:
            Tournament results
        """
        print(f"\nRunning Random Matchups")
        print(f"Total games: {num_games}")
        print()
        
        # Initialize progress tracker
        tracker = None
        if show_progress:
            tracker = ProgressTracker(num_games, update_interval=progress_interval)
            tracker.start()
        
        for game_num in range(num_games):
            # Randomly select agents
            agent_indices = random.sample(range(len(self.agents)), self.num_players)
            seed = random.randint(0, 1000000)
            
            result = self.run_game(agent_indices, seed=seed, verbose=False)
            
            if tracker:
                tracker.update(result)
            
            if verbose and not show_progress and game_num % 10 == 0:
                print(f"Game {game_num + 1}/{num_games}: "
                      f"Winner = {result['winner_name']}")
        
        if tracker:
            tracker.finish()
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """Get tournament results and statistics."""
        results = {
            'total_games': len(self.game_results),
            'agent_stats': {},
            'win_matrix': defaultdict(lambda: defaultdict(int)),
        }
        
        for agent_name, stats in self.agent_stats.items():
            games = stats['games']
            wins = stats['wins']
            win_rate = (wins / games * 100) if games > 0 else 0
            
            results['agent_stats'][agent_name] = {
                'games': games,
                'wins': wins,
                'losses': games - wins,
                'win_rate': win_rate,
            }
        
        # Build win matrix
        for game in self.game_results:
            if game['winner'] >= 0:
                winner = game['winner_name']
                loser = game['agents'][1 - game['winner']]
                results['win_matrix'][winner][loser] += 1
        
        return results
    
    def print_results(self):
        """Print tournament results."""
        results = self.get_results()
        
        print("\n" + "="*60)
        print("TOURNAMENT RESULTS")
        print("="*60)
        
        print(f"\nTotal Games: {results['total_games']}")
        print()
        
        # Agent statistics
        print("Agent Performance:")
        print(f"{'Agent':<20} {'Games':<8} {'Wins':<8} {'Losses':<8} {'Win Rate':<10}")
        print("-" * 60)
        
        sorted_agents = sorted(
            results['agent_stats'].items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        for agent_name, stats in sorted_agents:
            print(f"{agent_name:<20} {stats['games']:<8} {stats['wins']:<8} "
                  f"{stats['losses']:<8} {stats['win_rate']:>6.1f}%")
        
        print()
    
    def save_results(self, filename: Optional[str] = None):
        """Save tournament results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tournament_results_{timestamp}.json"
        
        results = self.get_results()
        results['game_results'] = self.game_results
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")


def train_agents(agents: List[RummikubAgent], num_episodes: int = 1000,
                eval_interval: int = 100, verbose: bool = True,
                show_progress: bool = True):
    """Train agents through self-play.
    
    Args:
        agents: List of agents to train
        num_episodes: Number of training episodes
        eval_interval: Evaluate every N episodes
        verbose: Print progress
        show_progress: Show visual progress tracker
    """
    print("="*60)
    print("TRAINING AGENTS")
    print("="*60)
    print(f"Agents: {[a.name for a in agents]}")
    print(f"Episodes: {num_episodes}")
    print()
    
    # Separate trainable and fixed agents
    trainable = [a for a in agents if hasattr(a, 'update')]
    
    if not trainable:
        print("No trainable agents found!")
        return
    
    print(f"Trainable agents: {[a.name for a in trainable]}")
    print()
    
    tournament = Tournament(agents, num_players=2)
    
    # Initialize progress tracker
    tracker = None
    if show_progress:
        tracker = ProgressTracker(num_episodes, update_interval=5.0)
        tracker.start()
    
    for episode in range(num_episodes):
        # Run training game
        agent_indices = random.sample(range(len(agents)), 2)
        result = tournament.run_game(agent_indices, verbose=False)
        
        if tracker:
            tracker.update(result)
        
        # Periodic text evaluation
        if not show_progress and (episode + 1) % eval_interval == 0:
            if verbose:
                print(f"Episode {episode + 1}/{num_episodes}")
                results = tournament.get_results()
                for agent_name, stats in results['agent_stats'].items():
                    if stats['games'] > 0:
                        print(f"  {agent_name}: {stats['win_rate']:.1f}% win rate "
                              f"({stats['wins']}/{stats['games']})")
                print()
    
    if tracker:
        tracker.finish()
    
    # Final results
    tournament.print_results()
    return tournament


if __name__ == "__main__":
    from agent import (RandomAgent, HeuristicAgent, GreedyAgent,
                       ConservativeAgent, QLearningAgent, HoardingAgent,
                       AggressiveAgent, BalancedAgent, SmartWormAgent)

    # Create 3 agents: SmartWorm + Random + Heuristic
    agents = [
        RandomAgent("Random"),
        HeuristicAgent("Heuristic"),
        SmartWormAgent("SmartWorm"),  # Now has solve() disabled - should be fast
    ]

    worm_agents = [
        SmartWormAgent("SmartWorm_1"),
        SmartWormAgent("SmartWorm_2")
    ]

    print("Rummikub ML Tournament System")
    print("="*60)
    print("\nAgent Strategies:")
    print("  - Random: Completely random valid moves")
    print("  - Heuristic: Simple priority-based selection")
    print("  - Greedy: Aggressive tile playing")
    print("  - Conservative: Defensive, prefers drawing")
    print("  - Hoarding10/15: Saves tiles until big play (10 or 15 tiles)")
    print("  - Aggressive: Uses worm logic to play immediately")
    print("  - Balanced: Middle ground strategy with worm scoring")
    print("  - SmartWorm: Deep worm.py solver integration")
    print("  - QLearning: Tabular Q-learning")

    # Option 1: Tournament with SmartWorm + 2 others, 100 games
    print("\n" + "="*60)
    print("1. Running Tournament: SmartWorm vs Random vs Heuristic (100 games)")
    print("="*60)
    tournament = Tournament(worm_agents, num_players=2)
    tournament.run_random_matchups(num_games=10, verbose=False, show_progress=True, progress_interval=2.0)
    tournament.print_results()

    # Option 2: Ultra-fast test (5 games only)
    # print("\n" + "="*60)
    # print("2. Ultra-Fast Test (5 games)...")
    # print("="*60)
    # fast_agents = [
    #     RandomAgent("Random"),
    #     HeuristicAgent("Heuristic"),
    #     AggressiveAgent("Aggressive"),
    # ]
    # tournament2 = Tournament(fast_agents, num_players=2)
    # tournament2.run_random_matchups(num_games=5, show_progress=True, progress_interval=0.5)
    # tournament2.print_results()

    # Option 3: Training with progress tracker (if you have trainable agents)
    # print("\n3. Training Q-Learning Agent with progress tracking...")
    # train_agents(agents, num_episodes=500, eval_interval=100, show_progress=True)
