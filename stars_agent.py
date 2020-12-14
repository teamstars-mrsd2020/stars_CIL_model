import time
import random
from collections import deque

import carla
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import utils.common as common
from models.map_model import MapModel
from utils.agents.navigation.global_route_planner import GlobalRoutePlanner
from utils.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from utils.pid_controller import PIDController

from map_camera import MapCamera


PIXELS_PER_WORLD = 5.5
DEBUG_MAP_VIEW = True


class StarsAgent:
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

        print(f"[ActorClass] Spawned Player {self.player_id}")

        self.waypoints_queue = deque()
        self._grp = None

        self.waypoints_queue.append(self._map.get_waypoint(spawn_point.location))
        self.reroute()
        self.cur_target = self.get_next_waypoint()

        self.init_PID_controller()
        self.init_camera()
        self.init_model()

    def init_PID_controller(self):
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def init_model(self):
        self.model = MapModel.load_from_checkpoint(self.checkpoint_path)
        self.model.cuda()
        self.model.eval()

    def reroute(self):
        """
        This method implements re-routing for vehicles approaching its destination.
        It finds a new target and computes another path to reach it.

            :param spawn_points: list of possible destinations for the agent
        """

        # print("Target almost reached, setting new destination...")
        random.shuffle(self.spawn_points)
        new_start = self.waypoints_queue[-1].transform.location
        destination = (
            self.spawn_points[0].location
            if self.spawn_points[0].location != new_start
            else self.spawn_points[1].location
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

        return control

    def apply_command(self, control):
        self.player.apply_control(control)

    def get_map_image(self):
        return self.camera_topdown.get()

    def init_camera(self):
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

    def run_step(self):
        with torch.no_grad():
            t0 = time.time()
            topdown = self.get_map_image()
            # print(f"Player {self.player_id}, image time = ", 1000 * (time.time() - t0))
            velocity = self.player.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

            DT = 3.5  # speed-time window for rerouting

            if self.player.get_location().distance(
                self.cur_target.transform.location
            ) < (speed * DT):
                # Assigning new target
                self.cur_target = self.get_next_waypoint()
                if len(self.waypoints_queue) < 4:
                    self.reroute()

            topdown = torch.FloatTensor(topdown)
            target = self.transform_target_waypoint(self.cur_target)

            if DEBUG_MAP_VIEW:

                self.world.debug.draw_point(
                    self.cur_target.transform.location, life_time=2
                )
                self.world.debug.draw_point(
                    self.player.get_location(),
                    color=carla.Color(0, 0, 255),
                    life_time=2,
                )

                out_points, heatmap = self.model(
                    topdown.cuda()[None], target.unsqueeze(0), debug=True
                )

                debugimg = self.debug_view(
                    topdown, target, out_points.squeeze().cpu().numpy()
                )

                # bigger_img = cv2.resize(np.array(debugimg)[:, :, ::-1],(800,800))
                # cv2.imshow(
                #     f"debugview_{self.player_id}", bigger_img
                # )
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

            points_world = self.transform_to_world(out_points.cpu().numpy())

            control = self.get_control_command(points_world, speed)

            self.apply_command(control)
            # print(f"Player {self.player_id}, total time = ", 1000 * (time.time() - t0))

    def destroy(self):
        self.player.destroy()

