"""
ABM Simulation Framework for Conflict Resolution Modeling

This module implements an agent-based model for simulating negotiation and conflict dynamics
between multiple parties. The model tracks issue resolution and battlefield control over time.
"""

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
        control: Current control as (A_control, B_control) percentages
    """
    name: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    control: Tuple[float, float] = (50.0, 50.0)

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
        control_transition_factor: float = 50.0,
        initial_conflict_intensity: float = 0.3,
        max_steps: int = 50,
        npz_file: Optional[str] = None
    ):
        # Load simulation configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Simulation parameters
        self.max_steps = max_steps
        self.control_transition_factor = control_transition_factor
        self.conflict_intensity = initial_conflict_intensity
        
        # Initialize data structures
        self._initialize_logging()
        self._initialize_world_entities()
        self._load_parameters(npz_file)
        self.agents = self._create_agents()
        
        # Calculate initial global metrics
        self.total_resources = sum(agent.resources for agent in self.agents)
        self.total_allocations = 0.0  # Tracks military spending per step

    def _initialize_logging(self):
        """Set up data structures for recording simulation history."""
        self.logs = {
            'agent_actions': [],       # All agent decisions
            'resource_allocations': [], # Military spending details  
            'battle_controls': [],      # Battlefield state changes
            'resolved_issues': [],      # Successful negotiations
            'conflict_intensity': []    # Global tension metrics
        }

    def _initialize_world_entities(self):
        """Create issues and battlefields from configuration."""
        self.issues = [Issue(**i) for i in self.config['issues']]
        self.battles = [Battlefield(**b) for b in self.config['battlefields']]

    # -------------------------------
    # Model Parameter Initialization
    # -------------------------------

    def _load_parameters(self, npz_file: Optional[str]) -> None:
        """Load or generate model parameters for agents.
        
        Parameters can be loaded from NPZ file or generated randomly.
        Includes:
        - Issue importance weights
        - Battlefield importance weights  
        - Decision model coefficients (betas)
        - Negotiation bottom lines
        """
        agent_ids = [a['id'] for a in self.config['agents']]
        issue_names = [issue.name for issue in self.issues]
        battle_names = [bf.name for bf in self.battles]

        if npz_file:
            # Load pre-trained parameters from file
            data = np.load(npz_file, allow_pickle=True)
            self.issue_weights = data['weights_issues'].item()
            self.battle_weights = data['weights_battlefield'].item()
            self.issue_betas = data['betas_issues'].item()
            self.resource_betas = data['betas_resource'].item()
            self.battle_betas = data['betas_battlefield'].item()
            self.issue_bottomlines = data['issue_bottomlines'].item()
        else:
            # Generate random parameters if no file provided
            self.issue_weights = {
                aid: {iss: np.random.rand() for iss in issue_names}
                for aid in agent_ids
            }
            self.battle_weights = {
                aid: {bf: np.random.rand() for bf in battle_names}
                for aid in agent_ids
            }
            self.issue_betas = {
                aid: {iss: np.random.randn(Agent.NUM_ISSUE_FEATURES) for iss in issue_names}
                for aid in agent_ids
            }
            self.resource_betas = {
                aid: np.random.randn(Agent.NUM_RESOURCE_FEATURES)
                for aid in agent_ids
            }
            self.battle_betas = {
                aid: np.random.randn(Agent.NUM_BATTLE_FEATURES)
                for aid in agent_ids
            }
            self.issue_bottomlines = {
                aid: {iss: np.random.rand() * 75 for iss in issue_names}
                for aid in agent_ids
            }


    def _create_agents(self) -> List[Agent]:
        """Instantiate Agent objects with their configured parameters."""
        agents: List[Agent] = []
        for agent_cfg in self.config['agents']:
            agent_id = agent_cfg['id']
            agents.append(Agent(
                id=agent_id,
                camp=agent_cfg['camp'],
                resources=float(agent_cfg['resources']),
                combat=agent_cfg['combat'],
                issue_weights=self.issue_weights[agent_id],
                battle_weights=self.battle_weights[agent_id],
                issue_betas=self.issue_betas[agent_id],
                resource_beta=self.resource_betas[agent_id],
                battle_beta=self.battle_betas[agent_id],
                issue_bottomlines=self.issue_bottomlines[agent_id]
            ))
        return agents

    # -------------------------------
    # Game Mechanics
    # -------------------------------

    def get_military_advantage(self, agent: Agent) -> float:
        """Calculate agent's relative military advantage across all battlefields.
        
        Computes weighted sum of control differences from agent's perspective.
        Positive values indicate advantage, negative values indicate disadvantage.
        """
        total_advantage = 0.0
        for bf in self.battles:
            control_A, control_B = bf.control
            weight = agent.battle_weights[bf.name]
            # Calculate advantage from this agent's perspective
            diff = (control_A - control_B) if agent.camp == 'A' else (control_B - control_A)
            total_advantage += weight * diff
        return total_advantage

    def get_ally_support(self, agent: Agent, domain: str, name: str) -> float:
        """TODO: Calculate support from allies (placeholder for future implementation)."""
        return 0.0

    def get_neutral_support(self, agent: Agent, domain: str, name: str) -> float:
        """TODO: Calculate support from neutral parties (placeholder for future implementation)."""
        return 0.0

    # -------------------------------
    # Logging Helpers
    # -------------------------------

    def _log_agent_action(self, step: int, agent_id: str, domain: str, 
                         name: str, action: str, proposal: Optional[Tuple[float, float]]):
        """Record an agent's decision in the simulation logs."""
        self.logs['agent_actions'].append({
            'step': step,
            'agent': agent_id,
            'domain': domain,
            'name': name,
            'action': action,
            'proposal': proposal
        })

    def _log_resource_allocation(self, step: int, agent_id: str, battlefield: str,
                               allocated: float, total_to_spend: float, resources: float):
        """Record military resource allocation details."""
        self.logs['resource_allocations'].append({
            'step': step,
            'agent': agent_id,
            'battlefield': battlefield,
            'allocated': allocated,
            'total_to_spend': total_to_spend,
            'resources': resources
        })

    def _log_battle_control(self, step: int, battlefield: str, control: Tuple[float, float]):
        """Record updated battlefield control state."""
        self.logs['battle_controls'].append({
            'step': step,
            'battlefield': battlefield,
            'control': control
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
        - Acceptance/rejection decisions
        - Resolution tracking
        """
        for agent in self.agents:
            for issue in self.issues:
                if issue.settled:
                    continue  # Skip already resolved issues

                # Get agent's decision
                action, proposal = agent.decide_issue(issue, self)
                
                # Skip if no valid proposal exists yet
                if action != 'propose' and not issue.proposal:
                    continue

                # Log the action
                self._log_agent_action(
                    step, agent.id, 'issue', issue.name, 
                    action, proposal or issue.proposal
                )

                # Handle different action types
                if action == 'propose' and proposal:
                    issue.proposal = proposal
                    issue.accepted_by.clear()
                elif action == 'accept':
                    issue.accepted_by.add(agent.id)
                    # Check if all agents have accepted
                    if len(issue.accepted_by) == len(self.agents):
                        issue.settled = True
                        self._log_resolved_issue(step, issue.name, issue.proposal)
                        # Update agent surpluses
                        for a in self.agents:
                            share = issue.proposal[0] if a.camp == 'A' else issue.proposal[1]
                            a.total_surplus += a.issue_weights[issue.name] * share
                elif action == 'reject':
                    issue.accepted_by.clear()

    def _process_battles(self, step: int) -> None:
        """Handle military resource allocation and battlefield updates.
        
        Processes:
        - Agent resource allocation decisions
        - Battlefield control updates
        - Resource spending tracking
        """
        # Track contributions from each camp per battlefield
        contributions = {bf.name: {'A': 0.0, 'B': 0.0} for bf in self.battles}

        for agent in self.agents:
            if not agent.combat:
                continue  # Skip non-combatants

            # Get agent's allocation decisions
            allocations, total_to_spend = agent.allocate_battle(self)
            self.total_allocations += total_to_spend

            # Record allocations
            for bf_name, amount in allocations.items():
                contributions[bf_name][agent.camp] += amount
                if amount > 1e-2:  # Only log significant allocations
                    self._log_resource_allocation(
                        step, agent.id, bf_name, 
                        amount, total_to_spend, agent.resources
                    )

        # Update battlefield control based on contributions
        for bf in self.battles:
            ra = contributions[bf.name]['A']
            rb = contributions[bf.name]['B']
            
            if ra + rb > 0:
                # Calculate control change based on relative contributions
                delta = (ra - rb) / (ra + rb) * self.control_transition_factor
                control_A = max(0.0, min(100.0, bf.control[0] + delta))
                bf.control = (control_A, 100.0 - control_A)
            
            self._log_battle_control(step, bf.name, bf.control)

    def _update_conflict_intensity(self, step_num: int) -> None:
        """Update global conflict intensity metric based on recent military spending.
        
        Conflict intensity is calculated as:
        (total military allocations) / (total available resources)
        """
        self._log_conflict_intensity(step_num)
        self.conflict_intensity = self.total_allocations / self.total_resources
        self.total_allocations = 0.0  # Reset for next step

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

    def run(self) -> Dict[str, List[Dict]]:
        """Execute full simulation sequence.
        
        Runs until either:
        - All issues are resolved, or
        - Maximum steps reached
        
        Returns complete simulation logs.
        """
        for step in range(self.max_steps):
            self.step(step)
            if all(issue.settled for issue in self.issues):
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
    world = World(config_file, max_steps=20, npz_file=npz_file)
    logs = world.run()
