"""
Agent module for the ABM simulation framework.

Defines the Agent class which represents actors in the simulation with decision-making capabilities
for both negotiation (issues) and conflict (battlefields) domains.
"""

from typing import Dict, Tuple, Optional
import numpy as np


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
        resource_beta (np.ndarray): Parameters for resource spending decisions
        battle_beta (np.ndarray): Parameters for battlefield allocation decisions
        issue_bottomlines (Dict[str, float]): Minimum acceptable shares for each issue
    """
    
    # Constants defining feature space dimensions
    NUM_ISSUE_FEATURES = 8      # Number of features for issue decisions
    NUM_RESOURCE_FEATURES = 4    # Number of features for spending decisions  
    NUM_BATTLE_FEATURES = 8     # Number of features for allocation decisions

    def __init__(
        self,
        id: str,
        camp: str,
        resources: float,
        combat: bool,
        issue_weights: Dict[str, float],
        battle_weights: Dict[str, float],
        issue_betas: Dict[str, np.ndarray],
        resource_beta: np.ndarray,
        battle_beta: np.ndarray,
        issue_bottomlines: Dict[str, float]
    ):
        """Initialize an agent with given parameters and attributes.
        
        Args:
            id: Unique identifier string
            camp: Faction identifier ('A', 'B', or 'neutral')
            resources: Starting military resources
            combat: Whether agent can participate in military actions
            issue_weights: Importance weights for each issue
            battle_weights: Importance weights for each battlefield
            issue_betas: Decision parameters for issue actions
            resource_beta: Parameters for spending fraction calculation
            battle_beta: Parameters for battlefield allocations
            issue_bottomlines: Minimum acceptable shares per issue
        """
        self.id = id
        self.camp = camp
        self.resources = resources
        self.combat = combat
        self.issue_weights = issue_weights
        self.battle_weights = battle_weights
        self.total_surplus = 0.0

        # Decision model parameters
        self.issue_betas = issue_betas
        self.resource_beta = resource_beta
        self.battle_beta = battle_beta
        self.issue_bottomlines = issue_bottomlines

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

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Compute numerically stable logistic sigmoid.
        
        Uses different formulations for positive/negative x to avoid overflow.
        
        Args:
            x: Input value
            
        Returns:
            Sigmoid output between 0 and 1
        """
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            ex = np.exp(x)
            return ex / (1.0 + ex)

    def decide_issue(
        self,
        issue,
        world
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

        # Calculate current surplus relative to bottom line
        bottomline = self.issue_bottomlines.get(issue.name, 0.0)
        if issue.proposal:
            share_A, share_B = issue.proposal
            my_share = share_A if self.camp == 'A' else share_B
            surplus = my_share - bottomline
        else:
            surplus = 0.0

        # Construct features for decision making
        features = {
            'conflict_intensity': world.conflict_intensity,
            'advantage': world.get_military_advantage(self),
            'resources': self.resources,
            'weight': self.issue_weights[issue.name],
            'surplus': surplus,
            'total_surplus': self.total_surplus,
            'ally_support': world.get_ally_support(self, 'issue', issue.name),
            'neutral_support': world.get_neutral_support(self, 'issue', issue.name)
        }

        # Build feature matrix (each column is an action's features)
        feature_matrix = np.zeros((self.NUM_ISSUE_FEATURES, 4))
        actions = ['propose', 'accept', 'reject', 'idle']
        
        for idx, action in enumerate(actions):
            feature_matrix[:, idx] = [
                features['conflict_intensity'],
                features['advantage'],
                features['resources'],
                features['weight'],
                features['surplus'],
                features['total_surplus'],
                features['ally_support'],
                features['neutral_support']
            ]

        # Compute action probabilities using softmax
        beta = self.issue_betas[issue.name]
        utilities = beta.dot(feature_matrix)
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

    def compute_spend_fraction(self, world) -> float:
        """Calculate fraction of resources to spend on military actions.
        
        Uses logistic regression with current state features to determine
        what portion of resources to allocate to battlefields.
        
        Args:
            world: Reference to world state
            
        Returns:
            Fraction between 0 and 1 representing portion of resources to spend
        """
        features = np.array([
            world.conflict_intensity,
            world.get_military_advantage(self),
            self.resources,
            self.total_surplus
        ])
        
        # Compute utility and convert to probability via sigmoid
        utility = self.resource_beta.dot(features)
        return self._sigmoid(utility)

    def allocate_battle(
        self,
        world
    ) -> Dict[str, float]:
        """Allocate military resources across battlefields.
        
        Determines:
        1. Total amount to spend (fraction of resources)
        2. Allocation of that amount across battlefields
        
        Args:
            world: Reference to world state
            
        Returns:
            Tuple containing:
            - Dict mapping battlefield names to allocated amounts
            - Total amount being spent
        """
        # Determine total spending amount
        spend_frac = self.compute_spend_fraction(world)
        total_to_spend = spend_frac * self.resources

        # Calculate utilities for each battlefield
        utilities = []
        battlefield_names = []
        base_advantage = world.get_military_advantage(self)
        
        for bf in world.battles:
            # Calculate battlefield-specific advantage
            cA, cB = bf.control
            area_advantage = (cA - cB) if self.camp == 'A' else (cB - cA)
            
            # Build feature vector
            features = np.array([
                world.conflict_intensity,
                base_advantage,
                area_advantage,
                self.battle_weights[bf.name],
                self.resources,
                self.total_surplus,
                world.get_ally_support(self, 'battle', bf.name),
                world.get_neutral_support(self, 'battle', bf.name)
            ])
            
            # Compute utility and store
            utilities.append(self.battle_beta.dot(features))
            battlefield_names.append(bf.name)

        # Convert utilities to allocation proportions
        proportions = self._softmax(np.array(utilities))
        
        # Create allocation dictionary
        allocation = {
            name: total_to_spend * prop 
            for name, prop in zip(battlefield_names, proportions)
        }
        
        return allocation, total_to_spend
