import time

import carla
import numpy as np

from stars_agent import StarsAgent

VEHICLE_NAME = "vehicle.lincoln.mkz2017"
PRESET_WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    # 2: carla.WeatherParameters.CloudyNoon,
    # 3: carla.WeatherParameters.WetNoon,
    # 5: carla.WeatherParameters.MidRainyNoon,
    # # 4: carla.WeatherParameters.WetCloudyNoon,
    # # 6: carla.WeatherParameters.HardRainNoon,
    # # 7: carla.WeatherParameters.SoftRainNoon,
    # 8: carla.WeatherParameters.ClearSunset,
    # 9: carla.WeatherParameters.CloudySunset,
    # 10: carla.WeatherParameters.WetSunset,
    # 12: carla.WeatherParameters.MidRainSunset,
    # 11: carla.WeatherParameters.WetCloudySunset,
    # 13: carla.WeatherParameters.HardRainSunset,
    # 14: carla.WeatherParameters.SoftRainSunset,
}

WEATHERS = list(PRESET_WEATHERS.values())
NUM_AGENTS = 1


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


class VehiclePool(object):
    def __init__(self, client, n_vehicles):
        self.client = client
        self.world = client.get_world()

        veh_bp = self.world.get_blueprint_library().filter("vehicle.*")
        spawn_points = np.random.choice(
            self.world.get_map().get_spawn_points(), n_vehicles
        )
        batch = list()

        for i, transform in enumerate(spawn_points):
            bp = np.random.choice(veh_bp)
            bp.set_attribute("role_name", "autopilot")

            batch.append(
                carla.command.SpawnActor(bp, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True)
                )
            )

        self.vehicles = list()
        errors = set()

        for msg in self.client.apply_batch_sync(batch):
            if msg.error:
                errors.add(msg.error)
            else:
                self.vehicles.append(msg.actor_id)

        if errors:
            print("\n".join(errors))

        print("%d / %d vehicles spawned." % (len(self.vehicles), n_vehicles))

    def __del__(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])


class PedestrianPool(object):
    def __init__(self, client, n_pedestrians):
        self.client = client
        self.world = client.get_world()

        ped_bp = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        con_bp = self.world.get_blueprint_library().find("controller.ai.walker")

        spawn_points = [self._get_spawn_point() for _ in range(n_pedestrians)]
        batch = [
            carla.command.SpawnActor(np.random.choice(ped_bp), spawn)
            for spawn in spawn_points
        ]
        walkers = list()
        errors = set()

        for msg in client.apply_batch_sync(batch, True):
            if msg.error:
                errors.add(msg.error)
            else:
                walkers.append(msg.actor_id)

        if errors:
            print("\n".join(errors))

        batch = [
            carla.command.SpawnActor(con_bp, carla.Transform(), walker_id)
            for walker_id in walkers
        ]
        controllers = list()
        errors = set()

        for msg in client.apply_batch_sync(batch, True):
            if msg.error:
                errors.add(msg.error)
            else:
                controllers.append(msg.actor_id)

        if errors:
            print("\n".join(errors))

        self.walkers = self.world.get_actors(walkers)
        self.controllers = self.world.get_actors(controllers)

        for controller in self.controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.4 + np.random.randn())

        self.timers = [np.random.randint(60, 600) * 20 for _ in self.controllers]

        print("%d / %d pedestrians spawned." % (len(self.walkers), n_pedestrians))

    def _get_spawn_point(self, n_retry=10):
        for _ in range(n_retry):
            spawn = carla.Transform()
            spawn.location = self.world.get_random_location_from_navigation()

            if spawn.location is not None:
                return spawn

        raise ValueError("No valid spawns.")

    def tick(self):
        for i, controller in enumerate(self.controllers):
            self.timers[i] -= 1

            if self.timers[i] <= 0:
                self.timers[i] = np.random.randint(60, 600) * 20
                controller.go_to_location(
                    self.world.get_random_location_from_navigation()
                )

    def __del__(self):
        for controller in self.controllers:
            controller.stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers])
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.controllers]
        )


class CarlaEnv(object):
    def __init__(self, town="Town03", port=2000, **kwargs):
        self._client = carla.Client("localhost", port)
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        # self._town_name = town
        # self._world = self._client.load_world(town)
        self._world = self._client.get_world()
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()

        self._tick = 0

        # vehicle, sensor
        # self._actor_dict = collections.defaultdict(list)
        self._players = []
        self._cameras = dict()

    def _set_weather(self, weather_string):
        if weather_string == "random":
            weather = np.random.choice(WEATHERS)
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def reset(self, weather="random", n_vehicles=0, n_pedestrians=0, seed=0):
        is_ready = False

        while not is_ready:
            np.random.seed(seed)

            self._clean_up()
            ### make this a multi agent thing(take num_agents as args)
            ### Initialize the ActorClass object inside this and add to a list called self._players
            num_agents = NUM_AGENTS
            self._spawn_players(self._map.get_spawn_points(), num_agents)
            # self._spawn_player(np.random.choice(self._map.get_spawn_points()))
            # self._setup_sensors()

            self._set_weather(weather)
            self._pedestrian_pool = PedestrianPool(self._client, n_pedestrians)
            self._vehicle_pool = VehiclePool(self._client, n_vehicles)

            is_ready = self.ready()

    def _spawn_players(self, spawn_points, num_agents):
        ### make this a multi agent thing(take num_agents as args) and take spawn points
        vehicle_bp = np.random.choice(self._blueprints.filter(VEHICLE_NAME))
        ###
        for i in range(num_agents):
            self._players.append(
                StarsAgent(self._client, vehicle_bp, spawn_points, i)
            )  # internally set role_name as well
        # vehicle_bp.set_attribute("role_name", "hero")

        # self._player = self._world.spawn_actor(vehicle_bp, start_pose)

        # self._actor_dict["player"].append(self._player)

    def ready(self, ticks=10):
        for _ in range(ticks):
            self.step()

        # for x in self._actor_dict["camera"]:
        #     x.get()

        self._time_start = time.time()
        self._tick = 0

        return True

    def step(self, control=None):
        # if control is not None:
        #     self._player.apply_control(control)

        # self._world.tick()
        self._tick += 1
        self._pedestrian_pool.tick()
        for player in self._players:
            player.run_step()
            # ?player.get_state()
        # batch inference

        # for player in self._players:
        #    player.get_command(inferred)

        # batch apply command
        # transform = self._player.get_transform()
        # velocity = self._player.get_velocity()

        ### Update this to use the ActorClass's camera object
        ### incorporate the transforms within the ActorClass
        ### [actor.get_map_image() for actor in self._players]
        ### can ignore the result that is returned since we are the ones who will call step
        # Put here for speed (get() busy polls queue).

        ## TODO: Get the map and target here
        # for agent in self.pl

        # result = {key: val.get() for key, val in self._cameras.items()}
        # result.update(
        #     {
        #         "wall": time.time() - self._time_start,
        #         "tick": self._tick,
        #         "x": transform.location.x,
        #         "y": transform.location.y,
        #         "theta": transform.rotation.yaw,
        #         "speed": np.linalg.norm([velocity.x, velocity.y, velocity.z]),
        #     }
        # )

    def __enter__(self):
        set_sync_mode(self._client, True)

        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self._clean_up()

        set_sync_mode(self._client, False)

    def _clean_up(self):
        self._pedestrian_pool = None
        self._vehicle_pool = None
        self._cameras.clear()
        print("cleaning up")
        # for actor_type in list(self._actor_dict.keys()):
        #     self._client.apply_batch(
        #         [carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]]
        #     )
        # self._actor_dict[actor_type].clear()

        # self._actor_dict.clear()

        for player in self._players:
            player.destroy()

        self._tick = 0
        self._time_start = time.time()

        self._player = None
