# world.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import json
import yaml
import numpy as np

from agent import Agent


@dataclass
class Issue:
    name: str
    proposal: Optional[Tuple[float, float]] = None
    settled: bool = False
    accepted_by: set = field(default_factory=set)


@dataclass
class Battlefield:
    name: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    control: Tuple[float, float] = (50.0, 50.0)


class World:
    """
    Manages a turn-based simulation of issue negotiations and battlefield resource allocations.
    Logs are stored as JSON with four top-level categories:
      - agent_actions
      - resource_allocations
      - battle_controls
      - resolved_issues
    """

    def __init__(
        self,
        config_path: str,
        control_transition_factor: float = 50.0,
        initial_conflict_intensity: float = 0.3,
        max_steps: int = 50,
        npz_file: Optional[str] = None
    ):
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.max_steps = max_steps
        self.control_transition_factor = control_transition_factor
        self.logs: Dict[str, List[Dict]] = {
            'agent_actions': [],
            'resource_allocations': [],
            'battle_controls': [],
            'resolved_issues': [],
            'conflict_intensity': []
        }

        # Initialize simulation entities
        self.issues = [Issue(**i) for i in self.config['issues']]
        self.battles = [Battlefield(**b) for b in self.config['battlefields']]

        # Load or initialize model parameters
        self._load_parameters(npz_file)

        # Instantiate agents
        self.agents = self._create_agents()

        # Initialize conflict intensity
        self.total_resources = sum(agent.resources for agent in self.agents)
        self.total_allocations = 0.0
        self.conflict_intensity = initial_conflict_intensity

    def _load_parameters(self, npz_file: Optional[str]) -> None:
        """Load weights, betas, and bottom lines from .npz or random-init."""
        ids = [a['id'] for a in self.config['agents']]
        issue_names = [issue.name for issue in self.issues]
        battle_names = [bf.name for bf in self.battles]

        if npz_file:
            data = np.load(npz_file, allow_pickle=True)
            self.issue_weights = data['weights_issues'].item()
            self.battle_weights = data['weights_battlefield'].item()
            self.issue_betas = data['betas_issues'].item()
            self.resource_betas = data['betas_resource'].item()
            self.battle_betas = data['betas_battlefield'].item()
            self.issue_bottomlines = data['issue_bottomlines'].item()
        else:
            # Random initialization for all agents
            self.issue_weights = {
                aid: {iss: np.random.rand() for iss in issue_names}
                for aid in ids
            }
            self.battle_weights = {
                aid: {bf: np.random.rand() for bf in battle_names}
                for aid in ids
            }
            self.issue_betas = {
                aid: {iss: np.random.randn(Agent.NUM_ISSUE_FEATURES) for iss in issue_names}
                for aid in ids
            }
            self.resource_betas = {
                aid: np.random.randn(Agent.NUM_RESOURCE_FEATURES)
                for aid in ids
            }
            self.battle_betas = {
                aid: np.random.randn(Agent.NUM_BATTLE_FEATURES)
                for aid in ids
            }
            self.issue_bottomlines = {
                aid: {iss: np.random.rand() * 75 for iss in issue_names}
                for aid in ids
            }

    def _create_agents(self) -> List[Agent]:
        """Construct Agent instances using loaded parameters."""
        agents: List[Agent] = []
        for agent_cfg in self.config['agents']:
            aid = agent_cfg['id']
            agents.append(
                Agent(
                    id=aid,
                    camp=agent_cfg['camp'],
                    resources=float(agent_cfg['resources']),
                    combat=agent_cfg['combat'],
                    issue_weights=self.issue_weights[aid],
                    battle_weights=self.battle_weights[aid],
                    issue_betas=self.issue_betas[aid],
                    resource_beta=self.resource_betas[aid],
                    battle_beta=self.battle_betas[aid],
                    issue_bottomlines=self.issue_bottomlines[aid]
                )
            )
        return agents

    def get_military_advantage(self, agent: Agent) -> float:
        """Compute weighted battlefield control difference for an agent's camp."""
        total = 0.0
        for bf in self.battles:
            control_A, control_B = bf.control
            weight = agent.battle_weights[bf.name]
            diff = (control_A - control_B) if agent.camp == 'A' else (control_B - control_A)
            total += weight * diff
        return total

    def get_ally_support(self, agent: Agent, domain: str, name: str) -> float:
        # Placeholder for future ally-support calculations
        return 0.0

    def get_neutral_support(self, agent: Agent, domain: str, name: str) -> float:
        # Placeholder for future neutral-support calculations
        return 0.0

    # --- Logging Helpers ---
    def _log_agent_action(
        self,
        step: int,
        agent_id: str,
        domain: str,
        name: str,
        action: str,
        proposal: Optional[Tuple[float, float]]
    ) -> None:
        self.logs['agent_actions'].append({
            'step': step,
            'agent': agent_id,
            'domain': domain,
            'name': name,
            'action': action,
            'proposal': proposal
        })

    def _log_resource_allocation(
        self,
        step: int,
        agent_id: str,
        battlefield: str,
        allocated: float,
        total_to_spend: float,
        resources: float
    ) -> None:
        self.logs['resource_allocations'].append({
            'step': step,
            'agent': agent_id,
            'battlefield': battlefield,
            'allocated': allocated,
            'total_to_spend': total_to_spend,
            'resources': resources
        })

    def _log_battle_control(
        self,
        step: int,
        battlefield: str,
        control: Tuple[float, float]
    ) -> None:
        self.logs['battle_controls'].append({
            'step': step,
            'battlefield': battlefield,
            'control': control
        })

    def _log_resolved_issue(
        self,
        step: int,
        issue_name: str,
        final_proposal: Tuple[float, float]
    ) -> None:
        self.logs['resolved_issues'].append({
            'step': step,
            'issue': issue_name,
            'final_proposal': final_proposal
        })

    def _log_conflict_intensity(self, step: int) -> None:
        self.logs['conflict_intensity'].append({
            'step': step,
            'conflict_intensity': self.conflict_intensity
        })

    # --- Simulation Steps ---
    def _process_issues(self, step: int) -> None:
        for agent in self.agents:
            for issue in self.issues:
                if issue.settled:
                    continue

                action, proposal = agent.decide_issue(issue, self)
                # Check if the agent is allowed to act on this issue
                if action != 'propose' and not issue.proposal:
                    continue

                self._log_agent_action(
                    step, agent.id, 'issue', issue.name, action, proposal or issue.proposal)

                if action == 'propose' and proposal:
                    issue.proposal = proposal
                    issue.accepted_by.clear()
                elif action == 'accept':
                    issue.accepted_by.add(agent.id)
                    if len(issue.accepted_by) == len(self.agents):
                        issue.settled = True
                        self._log_resolved_issue(step, issue.name, issue.proposal)
                        for a in self.agents:
                            share = issue.proposal[0] if a.camp == 'A' else issue.proposal[1]
                            a.total_surplus += a.issue_weights[issue.name] * share
                elif action == 'reject':
                    issue.accepted_by.clear()

    def _process_battles(self, step: int) -> None:
        contributions: Dict[str, Dict[str, float]] = {
            bf.name: {'A': 0.0, 'B': 0.0} for bf in self.battles
        }

        for agent in self.agents:
            if not agent.combat:
                continue

            allocations, total_to_spend = agent.allocate_battle(self)
            self.total_allocations += total_to_spend
            for bf_name, amount in allocations.items():
                contributions[bf_name][agent.camp] += amount
                if amount > 1e-2:
                    self._log_resource_allocation(
                        step, agent.id, bf_name, amount, total_to_spend, agent.resources
                    )

        for bf in self.battles:
            ra = contributions[bf.name]['A']
            rb = contributions[bf.name]['B']
            if ra + rb != 0:
                control_A = bf.control[0]
                control_A += (ra - rb) / (ra + rb) * self.control_transition_factor
                control_A = max(0.0, min(100.0, control_A))
                bf.control = (control_A, 100.0 - control_A)
            self._log_battle_control(step, bf.name, bf.control)
    
    def _update_conflict_intensity(self, step_num) -> None:
        """Update the conflict intensity based on resource allocations."""
        self._log_conflict_intensity(step_num)
        self.conflict_intensity = self.total_allocations / self.total_resources
        self.total_allocations = 0.0

    def step(self, step_num: int) -> None:
        self._process_issues(step_num)
        self._process_battles(step_num)
        self._update_conflict_intensity(step_num)

    def run(self) -> Dict[str, List[Dict]]:
        for step in range(self.max_steps):
            self.step(step)
            if all(issue.settled for issue in self.issues):
                break

        # Write logs to JSON
        with open('logs/simulation_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2)

        return self.logs


if __name__ == '__main__':
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else 'configs/bosnian_war.yml'
    npz = sys.argv[2] if len(sys.argv) > 2 else None

    world = World(cfg, max_steps=20, npz_file=npz)
    logs = world.run()
