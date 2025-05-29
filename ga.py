import random
import yaml
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from scoop import futures

from world import World, AgentParameters
from agent import Agent

# --- Configuration ---
CONFIG_PATH = 'configs/bosnian_war.yml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    
GROUND_TRUTH_PATH = 'data/bosnian_war.csv'
ground_truth = pd.read_csv(GROUND_TRUTH_PATH, index_col=0)
ground_truth = ground_truth["camp_A"].to_dict()

agent_ids = [a['id'] for a in config['agents']]
issue_names = [i['name'] for i in config['issues']]
battle_names = [b['name'] for b in config['battlefields']]

# Dimensions from Agent
A_ISSUE = len(Agent.ACTIONS_ISSUE)
F_ISSUE = Agent.NUM_ISSUE_FEATURES
A_BATTLE = len(Agent.ACTIONS_BATTLE)
F_BATTLE = Agent.NUM_BATTLE_FEATURES

# Hyperparameter keys and bounds
hyper_keys = [
    'initial_conflict_intensity', 'initial_negotiation_tension',
    'negotiation_tension_factor', 'initial_fatigue', 'fatigue_change_factor',
    'fatigue_factor', 'surplus_factor', 'external_pressure_factor', 'proposal_std',
    'resolved_threshold'
]
bounds = {
    'initial_conflict_intensity': (0.0, 1.0),
    'initial_negotiation_tension': (0.0, 1.0),
    'negotiation_tension_factor': (0.0, 0.5),
    'initial_fatigue': (0.0, 1.0),
    'fatigue_change_factor': (0.0, 0.5),
    'fatigue_factor': (0.0, 0.5),
    'surplus_factor': (0.0, 0.5),
    'external_pressure_factor': (0.0, 0.5),
    'proposal_std': (1.0, 20.0),
    'resolved_threshold': (0.75, 1.0)
}

# Calculate gene length
H = len(hyper_keys)
P_iw = len(agent_ids) * len(issue_names)
P_bw = len(agent_ids) * len(battle_names)
P_bl = len(agent_ids) * len(issue_names)
P_ib = len(agent_ids) * len(issue_names) * A_ISSUE * F_ISSUE
P_bb = len(agent_ids) * len(battle_names) * A_BATTLE * F_BATTLE
GENE_LENGTH = H + P_iw + P_bw + P_bl + P_ib + P_bb

# GA Setup
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register('attr_float', random.random)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_LENGTH)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def decode(individual):
    hyp = {}
    idx = 0
    # Decode hypers
    for key in hyper_keys:
        lo, hi = bounds[key]
        ind = max(0.0, min(1.0, individual[idx]))
        hyp[key] = lo + ind * (hi - lo)
        idx += 1

    # Decode agent-specific parameters
    agent_params = {}
    # Issue weights
    iw_matrix = np.array(individual[idx:idx+P_iw]).reshape(len(agent_ids), len(issue_names))
    idx += P_iw
    # Battle weights
    bw_matrix = np.array(individual[idx:idx+P_bw]).reshape(len(agent_ids), len(battle_names))
    idx += P_bw
    # Bottom-lines (0-100)
    bl_matrix = np.array(individual[idx:idx+P_bl]).reshape(len(agent_ids), len(issue_names)) * 100.0
    idx += P_bl
    # Issue betas
    ib_matrix = np.array(individual[idx:idx+P_ib]).reshape(len(agent_ids), len(issue_names), A_ISSUE, F_ISSUE)
    # scale from [0,1]
    ib_matrix = ib_matrix * 20.0 - 10.0
    idx += P_ib
    # Battle betas
    bb_matrix = np.array(individual[idx:idx+P_bb]).reshape(len(agent_ids), len(battle_names), A_BATTLE, F_BATTLE)
    bb_matrix = bb_matrix * 20.0 - 10.0
    idx += P_bb

    # Build AgentParameters for each agent
    for ai, aid in enumerate(agent_ids):
        iw = {iss: float(iw_matrix[ai, ji]) for ji, iss in enumerate(issue_names)}
        bw = {bf: float(bw_matrix[ai, jb]) for jb, bf in enumerate(battle_names)}
        bottom = {iss: float(bl_matrix[ai, ji]) for ji, iss in enumerate(issue_names)}
        issue_betas = {
            iss: ib_matrix[ai, ji] for ji, iss in enumerate(issue_names)
        }
        battle_betas = {
            bf: bb_matrix[ai, jb] for jb, bf in enumerate(battle_names)
        }
        agent_params[aid] = AgentParameters(
            issue_weights=iw,
            battle_weights=bw,
            issue_betas=issue_betas,
            battle_betas=battle_betas,
            issue_bottomlines=bottom
        )

    return hyp, agent_params


def evaluate(individual):
    hyp, agent_params = decode(individual)
    world_kwargs = {k: hyp[k] for k in hyper_keys}
    world = World(CONFIG_PATH, agent_params, max_steps=500, **world_kwargs)
    logs = world.run()
    
    predicted = {}
    for issue in logs["resolved_issues"]:
        predicted[issue["issue"]] = issue["final_proposal"][0]
    
    mse = 0.0
    for issue in ground_truth:
        pred = predicted.get(issue, 0.0)
        mse += (ground_truth[issue] - pred) ** 2

    mse /= len(ground_truth)
    return (-mse,)  # Minimize MSE, so return negative value

# GA operators
toolbox.register('evaluate', evaluate)
toolbox.register('mate', tools.cxBlend, alpha=0.5)
toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register("map", futures.map)

def main():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(20)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('best', np.max)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.6, # Crossover probability
        mutpb=0.3, # Mutation probability
        ngen=50, # Number of generations
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    return hof

if __name__ == '__main__':
    best = main()
    np.save('models/best_individuals.npy', [list(ind) for ind in best])
