# Multi-Agent Simulation of Crisis Resolution

This project implements a calibrated agent-based model (ABM) for simulating international conflict resolution. The model integrates bargaining over political issues with dynamic military resource allocation, enabling researchers to explore how states and actors balance coercion and negotiation during crises.

## Features

- Dual-domain decision-making: diplomacy and battlefield.
- Agent preferences modeled using interpretable linear functions over domain-specific features.
- Calibratable parameters using evolutionary methods (e.g., genetic algorithms).
- Extensible YAML-based configuration format.
- Structured JSON logs for analysis and validation.

## Directory Structure

```

.
├── agent.py                    # Agent class with negotiation and combat logic
├── world.py                    # Simulation environment and world state
├── configs/
│   └── bosnian_war.yml         # Example conflict scenario configuration
├── logs/
│   └── simulation_results.json # Simulation output
├── calibrate.py                # (Not implemented yet) parameter fitting using evolutionary search
├── README.md

````

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DotIN13/negotiation-reloaded
   cd negotiation-reloaded
   ```

2. Install dependencies: Ensure you have Python 3.10 or later installed.
   ```bash
   pip install numpy pyyaml
   ```

## Running the Simulation

To run the simulation with the Bosnian War configuration:

```bash
python world.py configs/bosnian_war.yml
```

Simulation results will be saved in `logs/simulation_results.json`.

## Output Format

The output JSON contains logs for:

* `agent_actions`: proposals, acceptances, and rejections on issues.
* `resource_allocations`: battlefield-specific resource use by agents.
* `battle_controls`: control shares over contested areas over time.
* `resolved_issues`: final settled outcomes and shares.
* `conflict_intensity`: ratio of total resources spent to capacity, per step.

These logs can be used for evaluation, visualization, or further modeling.

## Research Applications

This model supports research in:

* Crisis bargaining and conflict escalation.
* Learning strategic preferences from historical data.
* Evaluating the impact of third-party mediators or military interventions.

## Citation

If using this work in academic research, please cite:

```bibtex
@misc{zhang2025crisisabm,
  author = {Tianyi Zhang},
  title = {Bargains Under Fire: A Calibrated Agent-Based Model of International Crisis Behavior},
  year = {2025},
  note = {https://github.com/DotIN13/negotiation-reloaded}
}
```

## Contact

For questions or collaborations, contact Tianyi Zhang at [tianyiz@uchicago.edu](mailto:tzhang3@uchicago.edu) or open an issue on GitHub.
