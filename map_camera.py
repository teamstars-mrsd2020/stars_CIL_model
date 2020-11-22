import queue

import carla
import numpy as np
import torch
from PIL import Image, ImageDraw

import utils.common as common


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
