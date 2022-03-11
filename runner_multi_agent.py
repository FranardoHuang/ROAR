import logging
from pathlib import Path
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
from ROAR.agent_module.pure_pursuit_agent import PurePursuitAgent
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.special_agents.recording_agent import RecordingAgent
from ROAR.agent_module.potential_field_agent import PotentialFieldAgent
from ROAR.agent_module.occupancy_map_agent import OccupancyMapAgent
from ROAR.agent_module.michael_pid_agent import PIDAgent
# from ROAR.agent_module.special_agents.waypoint_generating_agent import WaypointGeneratigAgent
from pydantic import BaseModel, Field
from carla import *

class PitStop(BaseModel):
    carla_config: CarlaConfig = Field(default=CarlaConfig())
    agent_config: AgentConfig = Field(default=AgentConfig())


def main():
    """Starts game loop"""
    agent_config = AgentConfig.parse_file(Path("./ROAR_Sim/configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("./ROAR_Sim/configurations/configuration.json"))

    carla_runner = CarlaRunner(carla_settings=carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent)

    my_vehicle = carla_runner.set_carla_world()
    world = carla_runner.get_world()

    new_vehicle = world.spawn_actor(spawn_point_id=0)
    new_vehicle.set_autopilot()

    new_vehicle_2 = world.spawn_actor(spawn_point_id=2)#Cannot spawn actor at ID [0]. Error: Spawn failed because of collision at spawn position
    new_vehicle_2.set_autopilot()


    agent = PIDAgent(vehicle=my_vehicle, agent_settings=agent_config)
    carla_runner.start_game_loop(agent=agent, use_manual_control=True)

    agent2 = PIDAgent(vehicle=new_vehicle, agent_settings=agent_config)
    carla_runner.start_game_loop(agent=agent2, use_manual_control=False)# is grey tesla

    agent3 = PIDAgent(vehicle=new_vehicle_2, agent_settings=agent_config)
    carla_runner.start_game_loop(agent=agent3, use_manual_control=False)  # is grey tesla

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s '
                               '- %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)
    import warnings

    warnings.filterwarnings("ignore", module="carla")
    main()