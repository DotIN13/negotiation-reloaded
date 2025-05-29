"""
ABM Simulation Framework for Conflict Resolution Modeling

This module implements an agent-based model for simulating negotiation and conflict dynamics
between multiple parties. The model tracks issue resolution and battlefield control over time.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import yaml
import numpy as np

from agent import Agent

# -------------------------------
# Data Classes for State Tracking
# -------------------------------
    
@dataclass
class AgentParameters:
    """
    Container for per-agent decision-making parameters.
    """
    issue_weights: Dict[str, float]
    battle_weights: Dict[str, float]
    issue_betas: Dict[str, np.ndarray]
    battle_betas: Dict[str, np.ndarray]
    issue_bottomlines: Dict[str, float]

    @staticmethod
    def random(
        issue_names: List[str],
        battle_names: List[str],
        seed: Optional[int] = None
    ) -> 'AgentParameters':
        """
        Generate random parameters for an agent.

        Args:
            issue_names: List of issue identifiers.
            battle_names: List of battlefield identifiers.
            seed: Optional random seed for reproducibility.

        Returns:
            An AgentParameters instance with randomized weights, betas, and bottom-lines.
        """
        rng = np.random.default_rng(seed)

        issue_weights = {iss: float(rng.random()) for iss in issue_names}
        battle_weights = {bf: float(rng.random()) for bf in battle_names}

        issue_betas = {
            iss: rng.standard_normal((len(Agent.ACTIONS_ISSUE), Agent.NUM_ISSUE_FEATURES))
            for iss in issue_names
        }
        battle_betas = {
            bf: rng.standard_normal((len(Agent.ACTIONS_BATTLE), Agent.NUM_BATTLE_FEATURES))
            for bf in battle_names
        }

        issue_bottomlines = {
            iss: float(rng.random() * 50.0) for iss in issue_names
        }

        return AgentParameters(
            issue_weights=issue_weights,
            battle_weights=battle_weights,
            issue_betas=issue_betas,
            battle_betas=battle_betas,
            issue_bottomlines=issue_bottomlines
        )

@dataclass
class Issue:
    """Represents a negotiable issue between parties with current proposal state.
    
    Attributes:
        name: Unique identifier for the issue
        proposal: Current allocation proposal as (A_share, B_share) percentages
        settled: Flag indicating if issue has been resolved
        accepted_by: Set of agent IDs who have accepted current proposal
    """
    name: str
    proposal: Optional[Tuple[float, float]] = None
    settled: bool = False
    accepted_by: set = field(default_factory=set)

@dataclass
class Battlefield:
    """Represents a contested area with military control percentages.
    
    Attributes:
        name: Unique identifier for the battlefield  
        lat: Latitude coordinate (optional)
        lng: Longitude coordinate (optional)
    """
    name: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    intensity: float = 0.0

# -------------------------------
# Main Simulation Environment
# -------------------------------

class World:
    """Central simulation environment managing agents, issues and battlefields.
    
    The world coordinates:
    - Agent negotiations on issues
    - Military resource allocation
    - Battlefield control updates
    - Conflict intensity tracking
    
    Parameters:
        config_path: Path to YAML configuration file
        control_transition_factor: Scaling factor for control changes (default: 50.0)
        initial_conflict_intensity: Starting conflict level (default: 0.3)
        max_steps: Maximum simulation steps (default: 50)
        npz_file: Path to NPZ file with model parameters (optional)
    """

    def __init__(
        self,
        config_path: str,
        agent_params: Optional[Dict[str, AgentParameters]] = None,
        initial_conflict_intensity: float = 1.0,
        initial_negotiation_tension: float = 0.3,
        negotiation_tension_factor: float = 0.05,
        initial_fatigue: float = 0.0,
        fatigue_change_factor: float = 0.01,
        fatigue_factor: float = 0.3,
        surplus_factor: float = 0.3,
        external_pressure_factor: float = 0.3,
        proposal_std: float = 5.0,  # Standard deviation for proposal generation
        resolved_threshold: float = 0.75, # Threshold for early termination
        max_steps: int = 50,
        seed: Optional[int] = None,
    ):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Load simulation configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Simulation parameters
        self.conflict_intensity = initial_conflict_intensity
        self.negotiation_tension = initial_negotiation_tension
        self.negotiation_tension_factor = negotiation_tension_factor
        
        # Fatigue model parameters
        self.fatigue = initial_fatigue
        self.fatigue_change_factor = fatigue_change_factor
        
        # Decision-making parameters
        self.fatigue_factor = fatigue_factor
        self.surplus_factor = surplus_factor
        self.external_pressure_factor = external_pressure_factor
        self.proposal_std = proposal_std
        
        # Termination conditions
        self.max_steps = max_steps
        self.resolved_threshold = resolved_threshold
        
        # Initialize data structures
        self._initialize_logging()
        self._initialize_world_entities()
        self.camps = defaultdict(int)  # Track number of agents per camp
        self.agents = self._create_agents(agent_params)
        
        # Calculate initial global metrics
        self.total_resources = sum(agent.resources for agent in self.agents)
        self.total_allocations = 0.0  # Tracks military spending per step
        self.max_resources = max(agent.resources for agent in self.agents)
        self.last_step_actions = self._new_last_step_actions()
        self.current_step_actions = self._new_last_step_actions()

    def _new_last_step_actions(self):
        """Reset last step actions for negotiation and battlefield domains."""
        actions = {
            "negotiation": {},
            "battlefield": {}
        }
        
        for issue in self.issues:
            actions["negotiation"][issue.name] = {}
            for camp in self.camps:
                actions["negotiation"][issue.name][camp] = defaultdict(int)
        
        for bf in self.battles:
            actions["battlefield"][bf.name] = {}
            for camp in self.camps:
                actions["battlefield"][bf.name][camp] = defaultdict(int)

        return actions
    
    def build_external_pressure_offsets(
        self,
        camp: str,
        action_type: str,
        name: str,
        actions: List[str],
    ) -> np.ndarray:
        """
        Build offsets based on the percentage of allies who took each action
        in the previous step.

        Args:
            camp: The camp of the agent (e.g. 'A', 'B', 'neutral').
            action_type: Either 'negotiation' or 'battlefield'.
            name: Issue or battlefield name.
            actions: List of possible actions (e.g. ['compromise', 'accept', 'demand', 'idle']).
            scale: Scaling factor for the offsets (default: 1.0).

        Returns:
            np.ndarray of offsets aligned with the action order.
        """
        if self.camps[camp] == 0:
            # No allies, no offset
            return np.zeros(len(actions))

        offsets = []
        for action in actions:
            # Percentage of allies who chose this action
            count = self.last_step_actions[action_type][name][camp][action]
            fraction = count / self.camps[camp]
            offsets.append(fraction)  # Positive offset if allies did it

        return np.array(offsets)


    def _initialize_logging(self):
        """Set up data structures for recording simulation history."""
        self.logs = {
            'agent_actions': [],       # All agent decisions
            'resolved_issues': [],      # Successful negotiations
            'conflict_intensity': [],    # Global tension metrics
            'battle_actions': [],       # Battlefield engagement decisions
            'battle_outcomes': []       # Results of military engagements
        }

    def _initialize_world_entities(self):
        """Create issues and battlefields from configuration."""
        self.issues = [Issue(**i) for i in self.config['issues']]
        self.battles = [Battlefield(**b) for b in self.config['battlefields']]

    # -------------------------------
    # Model Parameter Initialization
    # -------------------------------

    def _create_agents(self, agent_params) -> List[Agent]:
        """
        Instantiate Agent objects using provided parameters.
        """
        issue_names = [issue.name for issue in self.issues]
        battle_names = [bf.name for bf in self.battles]
        
        agents: List[Agent] = []
        for agent_cfg in self.config['agents']:
            aid = agent_cfg['id']

            if agent_params and aid in agent_params:
                params = agent_params[aid]
            else:
                params = AgentParameters.random(issue_names, battle_names, self.seed)

            agents.append(Agent(
                agent_id=aid,
                camp=agent_cfg['camp'],
                resources=float(agent_cfg['resources']),
                combat=agent_cfg['combat'],
                issue_weights=params.issue_weights,
                battle_weights=params.battle_weights,
                issue_betas=params.issue_betas,
                battle_betas=params.battle_betas,
                issue_bottomlines=params.issue_bottomlines
            ))
            self.camps[agent_cfg['camp']] += 1
        return agents

    # -------------------------------
    # Game Mechanics
    # -------------------------------

    def get_ally_support(self, agent: Agent, action_type: str, name: str) -> float:
        """Calculate support from allies"""
        if self.camps[agent.camp] == 0:
            return 1.0
        if action_type == "negotiation":
            return self.last_step_actions[action_type][name][agent.camp]["accept"] / self.camps[agent.camp]
        if action_type == "battlefield":
            return self.last_step_actions[action_type][name][agent.camp]["engage"] / self.camps[agent.camp]

    def get_neutral_support(self, action_type: str, name: str) -> float:
        """Calculate support from neutral parties"""
        neutral_camp = 'neutral'
        if self.camps[neutral_camp] == 0:
            return 1.0
        if action_type == "negotiation":
            return self.last_step_actions[action_type][name][neutral_camp]["accept"] / self.camps[neutral_camp]
        if action_type == "battlefield":
            return self.last_step_actions[action_type][name][neutral_camp]["engage"] / self.camps[neutral_camp]

    # -------------------------------
    # Logging Helpers
    # -------------------------------

    def _log_agent_action(self, step: int, agent: str,
                          issue: str, action: str, proposal: Optional[Tuple[float, float]]):
        """Record an agent's decision in the simulation logs."""
        self.current_step_actions["negotiation"][issue][agent.camp][action] += 1
        self.logs['agent_actions'].append({
            'step': step,
            'agent_id': agent.id,
            'action_type': "negotiation",
            'issue': issue,
            'action': action,
            'proposal': proposal
        })
        
    def _log_battle_action(self, step: int, agent: str, action: str, battlefield_name: str):
        """Record an agent's battlefield action in the logs."""
        self.current_step_actions["battlefield"][battlefield_name][agent.camp][action] += 1
        self.logs['battle_actions'].append({
            'step': step,
            'agent_id': agent.id,
            'action_type': "battlefield",
            'action': action,
            'battlefield': battlefield_name
        })

    def _log_resolved_issue(self, step: int, issue_name: str, final_proposal: Tuple[float, float]):
        """Record successful issue resolution."""
        self.logs['resolved_issues'].append({
            'step': step,
            'issue': issue_name,
            'final_proposal': final_proposal
        })

    def _log_conflict_intensity(self, step: int):
        """Record current global conflict intensity metric."""
        self.logs['conflict_intensity'].append({
            'step': step,
            'conflict_intensity': self.conflict_intensity
        })

    # -------------------------------
    # Step Execution
    # -------------------------------

    def _process_issues(self, step: int) -> None:
        """Handle all issue negotiation logic for one simulation step.
        
        Processes:
        - Proposal generation
        - Acceptance/demandion decisions
        - Resolution tracking
        """
        for agent in self.agents:
            for issue in self.issues:
                if issue.settled:
                    continue  # Skip already resolved issues

                # Get agent's decision
                action, proposal = agent.decide_issue(issue, self)
                
                # Skip if no valid proposal exists yet
                if action != 'compromise' and not issue.proposal:
                    continue

                # Log the action
                self._log_agent_action(
                    step, agent, issue.name,
                    action, proposal or issue.proposal
                )

                # Handle different action types
                if action == 'compromise' and proposal:
                    issue.proposal = proposal
                    
                    # Remove same camp's acceptance
                    same_camp_agents = [a.id for a in self.agents if a.camp == agent.camp]
                    issue.accepted_by.difference_update(same_camp_agents)
                    
                    # Add agent's acceptance
                    issue.accepted_by.add(agent.id)

                if action == 'demand' and proposal:
                    issue.proposal = proposal
                    issue.accepted_by.add(agent.id)

                    # Remove opposite camp's acceptance
                    opposite_agents = [a.id for a in self.agents if a.camp != agent.camp]
                    issue.accepted_by.difference_update(opposite_agents)

                if action == 'accept':
                    issue.accepted_by.add(agent.id)
                    # Check if all agents have accepted
                    if len(issue.accepted_by) == len(self.agents):
                        issue.settled = True
                        self._log_resolved_issue(step, issue.name, issue.proposal)
                        # Update agent surpluses
                        for a in self.agents:
                            share = issue.proposal[0] if a.camp == 'A' else issue.proposal[1]
                            a.total_surplus += a.issue_weights[issue.name] * share

    def _process_battles(self, step: int) -> None:
        for bf in self.battles:
            contributions = {'A': 0.0, 'B': 0.0}
            # Agents decide to engage
            for agent in self.agents:
                if not agent.combat:
                    continue
                
                action = agent.decide_battle(bf, self)

                if action == 'engage':
                    contributions[agent.camp] += agent.resources
                    self.total_allocations += agent.resources
                    self._log_battle_action(step, agent, action, bf.name)

            total = contributions['A'] + contributions['B']
            if total > 0:
                p_A = contributions['A'] / total
                winner = 'A' if np.random.rand() < p_A else 'B'
                self.logs['battle_outcomes'].append({
                    'step': step,
                    'battlefield': bf.name,
                    'winner': winner
                })
            
            bf.intensity = total / self.max_resources  # Normalize intensity
    
    def resolved_ratio(self) -> float:
        """Calculate the ratio of resolved issues to total issues."""
        return sum(issue.settled for issue in self.issues) / len(self.issues)

    def _update_conflict_intensity(self, step_num: int) -> None:
        """Update global conflict intensity metric based on recent military spending.
        
        Conflict intensity is calculated as:
        (total military allocations) / (total available resources)
        """
        self._log_conflict_intensity(step_num)
        self.conflict_intensity = self.total_allocations / (self.total_resources * len(self.battles))
        self.total_allocations = 0.0  # Reset for next step
        
    def _update_negotiation_tension(self) -> None:
        """Update negotiation tension based on unresolved issues.
        
        Tension increases with unresolved issues, simulating negotiation pressure.
        """
        self.negotiation_tension += self.negotiation_tension_factor * (1 - self.resolved_ratio())
        
    def _update_fatigue(self) -> None:
        """Update agent fatigue based on negotiation and battle actions.
        
        Fatigue increases with each step, simulating resource depletion.
        """
        self.fatigue += self.fatigue_change_factor
        # Cap fatigue to a maximum value
        self.fatigue = min(self.fatigue, 1.0)

    def step(self, step_num: int) -> None:
        """Execute one complete simulation step.
        
        Processing order:
        1. Issue negotiations
        2. Battlefield allocations
        3. Global metrics update
        """
        self._process_issues(step_num)
        self._process_battles(step_num)
        self._update_conflict_intensity(step_num)
        self._update_negotiation_tension()
        self._update_fatigue()
        self.last_step_actions = self.current_step_actions
        self.current_step_actions = self._new_last_step_actions()

    def run(self) -> Dict[str, List[Dict]]:
        """Execute full simulation sequence.
        
        Runs until either:
        - All issues are resolved, or
        - Maximum steps reached
        
        Returns complete simulation logs.
        """
        for step in range(self.max_steps):
            self.step(step)
            if self.resolved_ratio() >= self.resolved_threshold:
                break  # Early termination if all issues resolved

        # Save logs to file
        with open('logs/simulation_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2)

        return self.logs

# -------------------------------
# Script Entry Point
# -------------------------------

if __name__ == '__main__':
    import sys
    # Load configuration from command line or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/bosnian_war.yml'
    npz_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Initialize and run simulation
    world = World(config_file, max_steps=1000)
    logs = world.run()
