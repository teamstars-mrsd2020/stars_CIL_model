# set up a carla runner environment

import os
import queue
import random
import time
from collections import deque

import carla
import cv2
import ipdb
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

import utils.common as common
from models.map_model import MapModel
from utils.agents.navigation.global_route_planner import GlobalRoutePlanner
from utils.agents.navigation.global_route_planner_dao import \
    GlobalRoutePlannerDAO
from utils.pid_controller import PIDController

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
VEHICLE_NAME = "vehicle.lincoln.mkz2017"
COLLISION_THRESHOLD = 20000
PIXELS_PER_WORLD = 5.5

DEBUG = int(os.environ.get("HAS_DISPLAY", 0))

DEBUG_MAP_VIEW = False


class ActorClass:
    # represents a single actor
    def __init__(
        self,
        client,
        blueprint,
        spawn_points,
        player_id,
        checkpoint_path="./weights/model.ckpt",
    ):
        ## initialize the actor, the MapCamera
        self.client = client
        self.world = client.get_world()
        self._map = self.world.get_map()
        self.spawn_points = spawn_points
        blueprint.set_attribute("role_name", "hero")
        self.checkpoint_path = checkpoint_path
        self.player_id = player_id

        spawn_point = np.random.choice(spawn_points)
        self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_point = np.random.choice(spawn_points)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        print(f"[ActorClass]spawned Player {self.player_id}")

        self.waypoints_queue = deque()
        self._grp = None

        self.waypoints_queue.append(self._map.get_waypoint(spawn_point.location))
        self.reroute(self.spawn_points)
        _ = self.get_next_waypoint()
        self.cur_target = self.get_next_waypoint()
        # from behaviour_Agent.py
        # Add global plan and goal and get_target
        # Add global planner
        # Add PID controller(next 4 points to control object )
        # self.init_planner(spawn_point)
        self.init_PID_controller()
        self.init_camera()
        self.init_model()

    # def init_planner(self, start_location):
    #     self.set_destination(
    #         start_location, np.random.choice(self.spawn_points), clean=True
    #     )

    def init_PID_controller(self):
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def init_model(self):
        self.model = MapModel.load_from_checkpoint(self.checkpoint_path)
        self.model.cuda()
        self.model.eval()

    def reroute(self, spawn_points):
        """
        This method implements re-routing for vehicles approaching its destination.
        It finds a new target and computes another path to reach it.

            :param spawn_points: list of possible destinations for the agent
        """

        # print("Target almost reached, setting new destination...")
        random.shuffle(spawn_points)
        new_start = self.waypoints_queue[-1].transform.location
        destination = (
            spawn_points[0].location
            if spawn_points[0].location != new_start
            else spawn_points[1].location
        )
        # print("New destination: " + str(destination))

        self.set_destination(
            new_start, destination, start_waypoint=self.waypoints_queue[-1]
        )

    def set_destination(
        self, start_location, end_location, start_waypoint=None, clean=False
    ):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router.

            :param start_location: initial position
            :param end_location: final position
            :param clean: boolean to clean the waypoint queue
        """
        if clean:
            self.waypoints_queue.clear()
        if start_waypoint is None:
            self.start_waypoint = self._map.get_waypoint(start_location)
        else:
            self.start_waypoint = start_waypoint
        self.end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self._trace_route(self.start_waypoint, self.end_waypoint)

        self.waypoints_queue.extend([x[0] for x in route_trace])

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the
        optimal route from start_waypoint to end_waypoint.

            :param start_waypoint: initial position
            :param end_waypoint: final position
        """
        # Setting up global router
        if self._grp is None:

            dao = GlobalRoutePlannerDAO(self._map, sampling_resolution=4.5)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location, end_waypoint.transform.location
        )

        return route

    def get_next_waypoint(self):
        """Will be used for the lbc model's input along with the image
        """
        return self.waypoints_queue.popleft()

    def get_control_command(self, points_world, speed):
        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        # desired_speed *= (1 - abs(angle)) ** 2

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        # if DEBUG:
        #     debug_display(
        #         tick_data,
        #         target_cam.squeeze(),
        #         points.cpu().squeeze(),
        #         steer,
        #         throttle,
        #         brake,
        #         desired_speed,
        #         self.step,
        #     )

        return control

    def apply_command(self, control):
        self.player.apply_control(control)

    def get_map_image(self):
        # topdown =
        # topdown = topdown.crop((128, 0, 128 + 256, 256))
        # topdown = np.array(topdown)
        return self.camera_topdown.get()

    def init_camera(self):
        # self.camera_rgb = Camera(
        #     self.world, self.player, 256, 144, 90, 1.2, 0.0, 1.3, 0.0, 0.0
        # )
        # self.camera_rgb_left = Camera(
        #     self.world, self.player, 256, 144, 90, 1.2, -0.25, 1.3, 0.0, -45.0
        # )
        # self.camera_rgb_right = Camera(
        #     self.world, self.player, 256, 144, 90, 1.2, 0.25, 1.3, 0.0, 45.0
        # )
        self.camera_topdown = MapCamera(
            self.world, self.player, 512, 5 * 10.0, 100.0, 5.5, 5
        )

    def transform_to_world(self, points):
        map_size = 256
        points[..., 0] = (points[..., 0] + 1) / 2 * map_size
        points[..., 1] = (points[..., 1] + 1) / 2 * map_size
        points = points.squeeze()
        # points_world = self.converter.cam_to_world(points_cam).numpy()
        position = np.array([map_size // 2, map_size + 1])
        relative_pixel = points - position
        relative_pixel[..., 1] *= -1

        return relative_pixel / PIXELS_PER_WORLD

    def transform_target_waypoint(self, target_waypoint):
        """Transform the Carla Waypoint to a x,y target object in the image space

        Args:
            target_waypoint (Carla.Waypoint): the next waypoint from the plan
        """
        theta = self.player.get_transform().rotation.yaw * np.pi / 180
        loc = self.player.get_location()
        target_loc = target_waypoint.transform.location
        current_position = np.array([loc.x, loc.y])
        if np.isnan(theta):
            theta = 0.0
        theta = theta + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)],])
        target = np.array([target_loc.x, target_loc.y])
        target = R.T.dot(target - current_position)
        target *= PIXELS_PER_WORLD
        target += [128, 256]
        target = np.clip(target, 0, 256)
        target = torch.FloatTensor(target)
        return target

    def debug_view(self, topdown, target, points):
        _topdown = Image.fromarray(
            common.COLOR[topdown.argmax(0).detach().cpu().numpy()]
        )
        _draw = ImageDraw.Draw(_topdown)

        _draw.ellipse(
            (target[0] - 2, target[1] - 2, target[0] + 2, target[1] + 2),
            (255, 255, 255),
        )

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x - 2, y - 2, x + 2, y + 2), (0, 0, 255))

        return _topdown

        # for x, y in _out:
        #     x = (x + 1) / 2 * 256
        #     y = (y + 1) / 2 * 256

        #     _draw.ellipse((x - 2, y - 2, x + 2, y + 2), (255, 0, 0))

        # for x, y in _between:
        #     x = (x + 1) / 2 * 256
        #     y = (y + 1) / 2 * 256

        # _draw.ellipse((x - 1, y - 1, x + 1, y + 1), (0, 255, 0))

    def run_step(self):
        with torch.no_grad():
            # decouple the tasks below into separate functions so that its easy to batch the execution later
            # get image
            topdown = self.get_map_image()

            # might need to scale the points according to the real-world units
            velocity = self.player.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
            DT = 2.5
            # get target
            # print("Speed, Topdown shape", speed, topdown.shape)
            if self.player.get_location().distance(
                self.cur_target.transform.location
            ) < (speed * DT):
                # print("assigning new target")
                self.cur_target = self.get_next_waypoint()
                if len(self.waypoints_queue) < 4:
                    self.reroute(self.spawn_points)
            # print(
            #     "target_wp ",
            #     self.cur_target.transform.location,
            #     self.player.get_location(),
            # )
            topdown = torch.FloatTensor(topdown)

            # self.world.debug.draw_point(
            #     self.cur_target.transform.location, life_time=10
            # )
            # import ipdb

            # ipdb.set_trace()

            target = self.transform_target_waypoint(self.cur_target)
            # print("target_tf ", target)
            # ------------------
            # model forward pass - can move this outside for batch inference
            # out_points = (
            #     self.model(topdown.cuda()[None], target.unsqueeze(0)).cpu().squeeze()
            # )
            ## debug

            if DEBUG_MAP_VIEW:
                out_points, heatmap = self.model(
                    topdown.cuda()[None], target.unsqueeze(0), debug=True
                )

                debugimg = self.debug_view(
                    topdown, target, out_points.squeeze().cpu().numpy()
                )

                # hm = heatmap[0].cpu().squeeze().numpy()
                # import matplotlib.pyplot as plt

                # _ = plt.figure(1)
                cv2.imshow(
                    f"debugview_{self.player_id}", np.array(debugimg)[:, :, ::-1]
                )
                cv2.waitKey(1)
            else:
                # model forward pass - can move this outside for batch inference
                # t0 = time.time()
                out_points = (
                    self.model(topdown.cuda()[None], target.unsqueeze(0))
                    .cpu()
                    .squeeze()
                )
                # t1 = time.time() - t0
                # print("Player Time=", self.player_id, t1 * 1000)
            # plt.imshow(debugimg)
            # plt.show()
            # import ipdb

            # ipdb.set_trace()
            # ------------------
            # print(out_points)
            points_world = self.transform_to_world(out_points.cpu().numpy())
            # print(points_world)

            # generate control command from model output
            control = self.get_control_command(points_world, speed)
            # print(control)
            # apply command to agent
            self.apply_command(control)

    def destroy(self):
        self.player.destroy()


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


class Camera(object):
    def __init__(self, world, player, w, h, fov, x, y, z, pitch, yaw, type="rgb"):
        bp = world.get_blueprint_library().find("sensor.camera.%s" % type)
        bp.set_attribute("image_size_x", str(w))
        bp.set_attribute("image_size_y", str(h))
        bp.set_attribute("fov", str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.type = type
        self.queue = queue.Queue()

        self.camera = world.spawn_actor(bp, transform, attach_to=player)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None

        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.type == "semantic_segmentation":
            return array[:, :, 0]

        return array

    def __del__(self):
        self.camera.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


class MapCamera(Camera):
    def __init__(self, world, player, size, fov, z, pixels_per_meter, radius):
        super().__init__(
            world, player, size, size, fov, 0, 0, z, -90, 0, "semantic_segmentation"
        )

        self.world = world
        self.player = player
        self.pixels_per_meter = pixels_per_meter
        self.size = size
        self.radius = radius

    def preprocess_semantic(self, semantic_np):
        topdown = common.CONVERTER[semantic_np]
        topdown = torch.LongTensor(topdown)
        topdown = (
            torch.nn.functional.one_hot(topdown, len(common.COLOR))
            .permute(2, 0, 1)
            .float()
        )

        return topdown

    def get(self):
        image = Image.fromarray(super().get())
        draw = ImageDraw.Draw(image)

        transform = self.player.get_transform()
        pos = transform.location
        theta = np.radians(90 + transform.rotation.yaw)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)],])

        ### Draw traffic lights here
        for light in self.world.get_actors().filter("*traffic_light*"):
            delta = light.get_transform().location - pos

            target = R.T.dot([delta.x, delta.y])
            target *= self.pixels_per_meter
            target += self.size // 2

            if min(target) < 0 or max(target) >= self.size:
                continue

            trigger = light.trigger_volume
            light.get_transform().transform(trigger.location)
            dist = trigger.location.distance(self.player.get_location())
            a = np.sqrt(
                trigger.extent.x ** 2 + trigger.extent.y ** 2 + trigger.extent.z ** 2
            )
            b = np.sqrt(
                self.player.bounding_box.extent.x ** 2
                + self.player.bounding_box.extent.y ** 2
                + self.player.bounding_box.extent.z ** 2
            )

            if dist > a + b:
                continue
            x, y = target
            draw.ellipse(
                (x - self.radius, y - self.radius, x + self.radius, y + self.radius),
                13 + light.state.real,
            )
        ## user needs to transform/crop this depending on the model
        topdown = image
        topdown = topdown.crop((128, 0, 128 + 256, 256))
        topdown = np.array(topdown)
        return self.preprocess_semantic(topdown)


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
            num_agents = 10
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
                ActorClass(self._client, vehicle_bp, spawn_points, i)
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

        self._world.tick()
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


if __name__ == "__main__":
    print("Run.py")
    with CarlaEnv() as env:
        print("Env activated")
        env.reset()
        for i in tqdm(range(10000)):
            # print("IN Step ", i)
            env.step()
            # time.sleep(0.2)
