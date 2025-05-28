"""
Agent module for the ABM simulation framework.

Defines the Agent class which represents actors in the simulation with decision-making capabilities
for both negotiation (issues) and conflict (battlefields) domains.
"""

from typing import Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from world import World

class Agent:
    """Represents an autonomous actor in the simulation with decision-making capabilities.
    
    Each agent has:
    - Unique identity and faction affiliation
    - Military resources to allocate
    - Preferences over issues and battlefields
    - Behavioral parameters for decision-making
    
    Attributes:
        id (str): Unique identifier for the agent
        camp (str): Faction affiliation ('A', 'B', or 'neutral')
        resources (float): Current military resource amount
        combat (bool): Whether the agent can participate in military actions
        issue_weights (Dict[str, float]): Importance weights for each negotiable issue
        battle_weights (Dict[str, float]): Importance weights for each battlefield
        total_surplus (float): Cumulative surplus from successful negotiations
        issue_betas (Dict[str, np.ndarray]): Decision parameters for issue actions
        battle_beta (np.ndarray): Parameters for battlefield allocation decisions
        issue_bottomlines (Dict[str, float]): Minimum acceptable shares for each issue
    """

    # Constants defining feature space dimensions
    NUM_ISSUE_FEATURES = 6      # Number of features for issue decisions
    NUM_BATTLE_FEATURES = 6     # Number of features for allocation decisions

    def __init__(
        self,
        agent_id: str,
        camp: str,
        resources: float,
        combat: bool,
        issue_weights: Dict[str, float], # Sum to 1.0 across issues
        battle_weights: Dict[str, float], # Sum to 1.0 across battlefields
        issue_betas: Dict[str, np.ndarray],
        battle_betas: Dict[str, np.ndarray],
        issue_bottomlines: Dict[str, float]
    ):
        """Initialize an agent with given parameters and attributes.
        
        Args:
            agent_id: Unique identifier string
            camp: Faction identifier ('A', 'B', or 'neutral')
            resources: Starting military resources
            combat: Whether agent can participate in military actions
            issue_weights: Importance weights for each issue
            battle_weights: Importance weights for each battlefield
            issue_betas: Decision parameters for issue actions
            battle_betas: Parameters for battlefield allocations
            issue_bottomlines: Minimum acceptable shares per issue
        """
        self.id = agent_id
        self.camp = camp
        self.resources = resources
        self.combat = combat

        # Decision model parameters
        self.issue_weights = issue_weights
        self.battle_weights = battle_weights
        self.issue_betas = issue_betas
        self.battle_betas = battle_betas
        self.issue_bottomlines = issue_bottomlines

        # Initialize properties
        self.total_surplus = 0.0
        self.total_surplus_possible = 100 * len(issue_bottomlines) - sum(issue_bottomlines.values())

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        """Compute numerically stable softmax probabilities.
        
        Args:
            values: Input array of raw values
            
        Returns:
            Array of probabilities summing to 1
        """
        shifted = values - np.max(values)  # For numerical stability
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    def decide_issue(
        self,
        issue: str,
        world: 'World'
    ) -> Tuple[str, Optional[Tuple[float, float]]]:
        """Make a decision about an issue in the current state.
        
        Agents can:
        - Propose a new allocation (must exceed their bottomline)
        - Accept the current proposal
        - Reject the current proposal
        - Remain idle
        
        Args:
            issue: The issue object being decided on
            world: Reference to the world state
            
        Returns:
            Tuple containing:
            - Action string ('propose', 'accept', 'reject', or 'idle')
            - New proposal tuple if proposing, else None
        """
        if issue.settled:
            return 'idle', None  # No action if issue already resolved
        
        actions = ['propose', 'accept', 'reject', 'idle']

        # Calculate current surplus relative to bottom line
        bottomline = self.issue_bottomlines[issue.name]
        if issue.proposal:
            share_A, share_B = issue.proposal
            my_share = share_A if self.camp == 'A' else share_B
            surplus = (my_share - bottomline) / (100.0 - bottomline)  # Normalize to [0, 1]
        else:
            surplus = 0.0

        # Construct features for decision making
        features = {
            'conflict_intensity': world.conflict_intensity,
            'weight': self.issue_weights[issue.name],
            'surplus': surplus,
            'total_surplus': self.total_surplus / self.total_surplus_possible,
            'ally_support': world.get_ally_support(self, 'negotiation', issue.name),
            'neutral_support': world.get_neutral_support('negotiation', issue.name)
        }
        
        feature_vec = [
            features['conflict_intensity'],
            features['weight'],
            features['surplus'],
            features['total_surplus'],
            features['ally_support'],
            features['neutral_support']
        ]

        # Compute action probabilities using softmax
        beta = self.issue_betas[issue.name] # (4, 6)
        utilities = beta.dot(feature_vec) # (4,)
        probs = self._softmax(utilities)

        # Select action probabilistically
        action = np.random.choice(actions, p=probs)

        if action == 'propose':
            # Generate new proposal that exceeds bottom line
            if self.camp == 'A':
                new_A = np.random.uniform(bottomline, 100.0)
                return 'propose', (new_A, 100.0 - new_A)

            new_B = np.random.uniform(bottomline, 100.0)
            return 'propose', (100.0 - new_B, new_B)

        return action, None

    def decide_battle(
        self,
        battlefield: str,
        world: 'World'
    ) -> str:
        """Make a decision about military action in a battlefield.
        
        Agents can:
        - Engage conflict (commit resources)
        - Remain idle (do nothing)

        Args:
            battlefield: The battlefield object being decided on
            world: Reference to the world state
            
        Returns:
            Action string ('engage' or 'idle')
        """
        actions = ['engage', 'idle']

        # Gather features parallel to decide_issue
        features = {
            'conflict_intensity': world.conflict_intensity,
            'weight': self.battle_weights.get(battlefield.name, 0.0),
            'resources': self.resources / world.max_resources,
            'total_surplus': self.total_surplus / self.total_surplus_possible,
            'ally_support': world.get_ally_support(self, 'battlefield', battlefield.name),
            'neutral_support': world.get_neutral_support('battlefield', battlefield.name),
        }

        feature_vec = np.array([
            features['conflict_intensity'],
            features['weight'],
            features['resources'],
            features['total_surplus'],
            features['ally_support'],
            features['neutral_support']
        ])

        # Compute utilities for both actions
        beta = self.battle_betas[battlefield.name]  # (2, 6)
        utils = beta.dot(feature_vec)  # (2,)
        probs = self._softmax(utils)

        # Sample action
        decision = np.random.choice(actions, p=probs)
        return decision
