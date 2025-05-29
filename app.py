import eventlet
eventlet.monkey_patch()

from flask_socketio import SocketIO
from flask import Flask, render_template

from world import World

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# We'll keep `world` in module scope and recreate it on each restart.
world: World = None
simulation_task = None


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('start_simulation')
def handle_start(data):
    """Run simulation and push live updates to the client."""
    global world
    global simulation_task

    # kill any previous run
    if simulation_task is not None:
        try:
            simulation_task.kill()
        except Exception:
            pass
    
    world = World(
        config_path='configs/bosnian_war.yml',
        initial_conflict_intensity=float(data['initial_conflict_intensity']),
        initial_negotiation_tension=float(data['initial_negotiation_tension']),
        negotiation_tension_factor=float(data['negotiation_tension_factor']),
        initial_fatigue=float(data['initial_fatigue']),
        fatigue_change_factor=float(data['fatigue_change_factor']),
        fatigue_factor=float(data['fatigue_factor']),
        surplus_factor=float(data['surplus_factor']),
        external_pressure_factor=float(data['external_pressure_factor']),
        proposal_std=float(data['proposal_std']),
        resolved_threshold=float(data['resolved_threshold']),
        max_steps=int(data['max_steps']),
        seed=int(data['seed'])
    )
    
    def run():
        for step in range(world.max_steps):
            world.step(step)

            # Build the per‐step payload
            payload = {
                'step': step,
                'conflict_intensity': world.conflict_intensity,
                'resolved_ratio': world.resolved_ratio(),
                'proposals': [
                    {
                        'issue': iss.name,
                        'A': iss.proposal[0] if iss.proposal else None,
                        'B': iss.proposal[1] if iss.proposal else None,
                        'accepted_by': sorted(list(iss.accepted_by)) if iss.accepted_by else None
                    } for iss in world.issues
                ],
                'battles': [
                    {
                        'name': bf.name,
                        'lat': bf.lat, 'lng': bf.lng,
                        'intensity': bf.intensity
                    } for bf in world.battles
                ]
            }

            socketio.emit('step_update', payload)

            if world.resolved_ratio() >= 1.0 or step >= world.max_steps - 1:
                break
        
        socketio.emit('simulation_complete', {'total_steps': step + 1})

    simulation_task = socketio.start_background_task(run)


if __name__ == '__main__':
    socketio.run(app, debug=True)
