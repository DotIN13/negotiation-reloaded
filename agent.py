from typing import Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from world import World


class Agent:
    """Agent with decision logic for both negotiation (issues) and conflict (battlefields)."""

    ACTIONS_ISSUE = ['compromise', 'accept', 'demand', 'idle']
    ACTIONS_BATTLE = ['engage', 'idle']
    NUM_ISSUE_FEATURES = 4
    NUM_BATTLE_FEATURES = 4

    def __init__(
        self,
        agent_id: str,
        camp: str,
        resources: float,
        combat: bool,
        issue_weights: Dict[str, float],
        battle_weights: Dict[str, float],
        issue_betas: Dict[str, np.ndarray],
        battle_betas: Dict[str, np.ndarray],
        issue_bottomlines: Dict[str, float]
    ):
        self.id = agent_id
        self.camp = camp
        self.resources = resources
        self.combat = combat

        # Decision parameters
        self.issue_weights = issue_weights
        self.battle_weights = battle_weights
        self.issue_betas = issue_betas
        self.battle_betas = battle_betas
        self.issue_bottomlines = issue_bottomlines

        # Track surpluses
        self.total_surplus = 0.0
        max_surplus = 100 * len(issue_bottomlines) - \
            sum(issue_bottomlines.values())
        self.total_surplus_possible = max_surplus if max_surplus > 0 else 1.0

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    def decide_issue(
        self,
        issue,
        world: 'World'
    ) -> Tuple[str, Optional[Tuple[float, float]]]:
        if issue.settled:
            return 'idle', None

        # 1) Compute normalized surpluses
        surplus = self._compute_normalized_surplus(issue)
        total_surplus = self.total_surplus / self.total_surplus_possible

        # 2) Build base utilities via feature vector
        features = self._issue_feature_vector(issue, world)
        beta = self.issue_betas[issue.name]                    # shape (4, 4)
        utilities = beta.dot(features)                         # shape (4,)

        # 3) Compute decision offsets
        offsets = self._issue_offsets(world, surplus, total_surplus)
        ally_offsets = world.build_external_pressure_offsets(
            self.camp, "negotiation", issue.name, self.ACTIONS_ISSUE
        )
        offsets += world.external_pressure_factor * ally_offsets

        # 4) Softmax + offsets → final probabilities
        probs = self._softmax(utilities) + offsets
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()

        # 5) Sample action
        action = np.random.choice(self.ACTIONS_ISSUE, p=probs)

        if action in ['compromise', 'demand']:
            return self._make_proposal(issue, action, world)

        return action, None

    def decide_battle(
        self,
        battlefield,
        world: 'World'
    ) -> str:
        if not self.combat:
            return 'idle'

        # 1) Feature vector for battle
        features = self._battle_feature_vector(battlefield, world)
        beta = self.battle_betas[battlefield.name]              # shape (2, 4)
        utilities = beta.dot(features)                         # shape (2,)

        # 3) Softmax + offsets → probabilities
        probs = self._softmax(utilities)
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()

        # 4) Sample and return
        return np.random.choice(self.ACTIONS_BATTLE, p=probs)

    # -------------------------------
    # Private helpers
    # -------------------------------

    def _compute_normalized_surplus(self, issue) -> float:
        """Surplus relative to bottomline, normalized to [0,1]."""
        bottom = self.issue_bottomlines[issue.name]
        if not issue.proposal:
            return 0.0

        share = issue.proposal[0] if self.camp == 'A' else issue.proposal[1]
        raw = max(0.0, share - bottom)
        return raw / (100.0 - bottom) if bottom < 100.0 else 1.0

    def _issue_feature_vector(self, issue, world: 'World') -> np.ndarray:
        """Construct feature vector for issue utilities."""
        return np.array([
            world.negotiation_tension,
            world.conflict_intensity,
            self.issue_weights[issue.name],
            self.resources / world.max_resources
        ])

    def _battle_feature_vector(self, battlefield, world: 'World') -> np.ndarray:
        """Construct feature vector for battle utilities."""
        return np.array([
            world.negotiation_tension,
            world.conflict_intensity,
            self.battle_weights.get(battlefield.name, 0.0),
            self.resources / world.max_resources
        ])

    def _issue_offsets(self, world, surplus: float, total_surplus: float) -> np.ndarray:
        """Compute surplus- and fatigue-based offsets for issue decisions."""
        sf, ff = world.surplus_factor, world.fatigue_factor
        # compromise/demand get negative boost when surplus high; accept gets positive
        return np.array([
            (sf * surplus + sf * total_surplus + ff * world.fatigue),  # compromise
            (sf * surplus + sf * total_surplus + ff * world.fatigue),  # accept
            (-sf * surplus - sf * total_surplus),                     # demand
            sf * total_surplus                                        # idle
        ])

    def _make_proposal(
        self,
        issue,
        action: str,
        world: 'World'
    ) -> Tuple[str, Tuple[float, float]]:
        """
        For compromise: pick a new share ~ N(mean(bottom, current), σ), 
        for demand: ~ N(mean(current,100), σ), truncated [0,100].
        """
        bottom = self.issue_bottomlines[issue.name]

        # 1) what was current share?
        if issue.proposal:
            current = issue.proposal[0] if self.camp == 'A' else issue.proposal[1]
        else:
            # if no current proposal, start at bottom‐line
            current = bottom

        # 2) compute the target mean
        if action == 'compromise':
            mean = 0.5 * (bottom + current)
        else:  # 'demand'
            mean = 0.5 * (current + 100.0)

        # 3) sample (using a normal with σ=10 here, but you can tweak σ)
        raw = np.random.normal(loc=mean, scale=world.proposal_std)
        proposal_value = float(np.clip(raw, 0.0, 100.0))

        # 4) build the (A,B) tuple
        if self.camp == 'A':
            A, B = proposal_value, 100.0 - proposal_value
        else:
            B, A = proposal_value, 100.0 - proposal_value

        return action, (A, B)
