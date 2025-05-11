# agent.py
from typing import Dict, Tuple, Optional, List

import numpy as np


class Agent:
    """
    Represents an actor in the simulation.

    Attributes:
        id: Unique identifier.
        camp: 'A', 'B', or 'neutral'.
        resources: Current military resources.
        total_surplus: Cumulative total_surplus from negotiations.
        issue_weights: Importance weight per issue.
        battle_weights: Importance weight per battlefield.
        issue_betas: Logistic regression parameters for issue decisions.
        resource_beta: Parameters to determine total resource spend fraction.
        battle_beta: Parameters to allocate spent resources across battlefields.
        issue_bottomlines: Minimum acceptable shares for proposals.
    """
    NUM_ISSUE_FEATURES = 7
    NUM_RESOURCE_FEATURES = 3  # [military_advantage, resources, total_surplus]
    NUM_BATTLE_FEATURES = 7    # [overall_adv, area_adv, resources, total_surplus, ally_support, neutral_support]

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
        self.id = id
        self.camp = camp
        self.resources = resources
        self.combat = combat
        self.issue_weights = issue_weights
        self.battle_weights = battle_weights
        self.total_surplus = 0.0

        # Decision parameters
        self.issue_betas = issue_betas
        self.resource_beta = resource_beta
        self.battle_beta = battle_beta
        self.issue_bottomlines = issue_bottomlines

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = values - np.max(values)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable logistic sigmoid."""
        if x >= 0:
            # for x >= 0, exp(-x) is at most 1
            return 1.0 / (1.0 + np.exp(-x))
        else:
            # for x < 0, exp(x) is small, compute exp(x)/(1+exp(x)) instead
            ex = np.exp(x)
            return ex / (1.0 + ex)

    def decide_issue(
        self,
        issue,
        world
    ) -> Tuple[str, Optional[Tuple[float, float]]]:
        """
        Decide whether to propose, accept, reject, or stay idle on an issue.
        Proposals always exceed the agent's bottomline for that issue.
        Returns:
            action: One of 'propose', 'accept', 'reject', 'idle'.
            proposal: (share_A, share_B) if action is 'propose', else None.
        """
        if issue.settled:
            return 'idle', None

        # Current bottomline
        bottomline = self.issue_bottomlines.get(issue.name, 0.0)
        
        # Calculate surplus
        if issue.proposal:
            share_A, share_B = issue.proposal
            my_share = share_A if self.camp == 'A' else share_B
            surplus = my_share - bottomline
        else:
            surplus = 0.0

        # Build feature vector per action
        base_features = {
            'advantage': world.get_military_advantage(self),
            'resources': self.resources,
            'weight': self.issue_weights[issue.name],
            'surplus': surplus,
            'total_surplus': self.total_surplus,
            'ally_support': world.get_ally_support(self, 'issue', issue.name),
            'neutral_support': world.get_neutral_support(self, 'issue', issue.name)
        }

        feature_matrix = np.zeros((self.NUM_ISSUE_FEATURES, 4))
        actions = ['propose', 'accept', 'reject', 'idle']
        for idx, action in enumerate(actions):
            feature_matrix[:, idx] = [
                base_features['advantage'],
                base_features['resources'],
                base_features['weight'],
                base_features['surplus'],
                base_features['total_surplus'],
                base_features['ally_support'],
                base_features['neutral_support']
            ]

        # Compute utilities and probabilities
        beta = self.issue_betas[issue.name]
        utilities = beta.dot(feature_matrix)
        probs = self._softmax(utilities)
        action = np.random.choice(actions, p=probs)

        if action == 'propose':
            # generate new proposal above bottomline
            if self.camp == 'A':
                new_A = np.random.uniform(bottomline, 100.0)
                return 'propose', (new_A, 100.0 - new_A)
            new_B = np.random.uniform(bottomline, 100.0)
            return 'propose', (100.0 - new_B, new_B)

        return action, None

    def compute_spend_fraction(self, world) -> float:
        """
        Determine fraction of resources to invest in all battlefields this step.
        Uses sigmoid(resource_beta · [military_advantage, resources, total_surplus]).
        """
        adv = world.get_military_advantage(self)
        resources = self.resources
        x = self.resource_beta.dot(np.array([adv, resources, self.total_surplus]))
        return self._sigmoid(x)

    def allocate_battle(
        self,
        world
    ) -> Dict[str, float]:
        """
        Allocate a portion of resources across battlefields.
        Returns:
            Mapping battlefield_name -> resource_amount.
        """
        spend_frac = self.compute_spend_fraction(world)
        total_to_spend = spend_frac * self.resources

        utilities: List[float] = []
        names: List[str] = []
        base_adv = world.get_military_advantage(self)
        resources = self.resources

        for bf in world.battles:
            cA, cB = bf.control
            area_adv = (cA - cB) if self.camp == 'A' else (cB - cA)
            area_weight = self.battle_weights[bf.name]
            features = np.array([
                base_adv,
                area_adv,
                area_weight,
                resources,
                self.total_surplus,
                world.get_ally_support(self, 'battle', bf.name),
                world.get_neutral_support(self, 'battle', bf.name)
            ])
            utilities.append(self.battle_beta.dot(features))
            names.append(bf.name)

        utilities = np.array(utilities)
        proportions = self._softmax(utilities)
        allocation = {name: total_to_spend * prop for name, prop in zip(names, proportions)}
        return allocation
