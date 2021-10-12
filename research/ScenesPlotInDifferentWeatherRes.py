import glob
import copy
import inspect
import traceback
import os
import sys
import numpy as np
import utils.simu_utils as utils
import math
import queue
import matplotlib.pyplot as plt
from carla import Transform, Location, Rotation
import torch
from compares.cmp_model import NVIDIA_ORIGIN
import cv2
import time


try:
    sys.path.append(glob.glob('../carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

seq_len = 5

IM_WIDTH = 320
IM_HEIGHT = 180


def display(image):
    cv2.imshow("display", image)
    cv2.waitKey(1)


def get_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    # i3 = i3.transpose(1, 2, 0)
    image_queue.queue.clear()
    image_queue.put_nowait(i3)


def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))



actor_list = []
client = carla.Client('localhost', 2000)
client.set_timeout(10)
world = client.load_world('Town01')

# set the weather
# weather = world.get_weather()
# weather.sun_altitude_angle = 90
# world.set_weather(weather)
weathers = [carla.WeatherParameters.ClearNight, carla.WeatherParameters.ClearNoon, carla.WeatherParameters.ClearSunset, carla.WeatherParameters.CloudyNight, carla.WeatherParameters.CloudyNoon, carla.WeatherParameters.CloudySunset, carla.WeatherParameters.HardRainNight, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.HardRainSunset, carla.WeatherParameters.MidRainSunset, carla.WeatherParameters.MidRainyNight, carla.WeatherParameters.MidRainyNoon, carla.WeatherParameters.SoftRainNight, carla.WeatherParameters.SoftRainNoon, carla.WeatherParameters.SoftRainSunset, carla.WeatherParameters.WetCloudyNight, carla.WeatherParameters.WetCloudyNoon, carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.WetNight, carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetSunset]
wea = np.random.choice(weathers)
world.set_weather(wea)

spectator = world.get_spectator()
blueprint_library = world.get_blueprint_library()
vehicle_blueprints = blueprint_library.filter('model3')
blueprint = vehicle_blueprints[0]

# point = world.get_map().get_spawn_points()[2]
# vehicle = world.spawn_actor(blueprint, point)
spawn_point = np.random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(blueprint, spawn_point)

print(vehicle.type_id)
actor_list.append(vehicle)

world.wait_for_tick()
# spectator.set_transform(get_transform(vehicle.get_location(), 0))
cam_bp = blueprint_library.filter("sensor.camera.rgb")[0]
cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
cam_bp.set_attribute("fov", "110")

spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
actor_list.append(sensor)
time.sleep(1)
image_queue = queue.Queue(1)
sensor.listen(lambda data: get_img(data))

dect_bp = blueprint_library.filter("sensor.camera.rgb")[0]
dect_bp.set_attribute("image_size_x", "1200")
dect_bp.set_attribute("image_size_y", "800")
dect_bp.set_attribute("fov", "110")

billboard_width = 590
billboard_height = 400
billboard = np.array([[0, 0], [billboard_width, 0], [0, billboard_height], [billboard_width, billboard_height]], dtype=np.float32)

Image_Array = None
collision_count = 0
simu_time = 80
dect_time = 0
begin_time = int(time.time())
collision = False
weather_index = 0
try:
    batch_img = []
    for i in range(0, 1000000):
        control = vehicle.get_control()
        control.throttle = 0
        control.brake = 0.5
        vehicle.apply_control(control)

        if Image_Array is not None:
            image = copy.copy(Image_Array)
            cv2.imwrite("weather_example/" + str(i) + ".jpg", image)

        if i % 3 == 0:
            wea = weathers[weather_index]
            weather_index += 1
            world.set_weather(wea)

        if int(time.time()) - dect_time > 2:
            dect_time = int(time.time())
            trans = vehicle.get_transform()
            spectator.set_transform(carla.Transform(trans.location + carla.Location(z=25), carla.Rotation(pitch=-90)))

        if int(time.time()) - begin_time > simu_time:
            print("End of simulation. simu-time: " + str(simu_time) + " collision num: " + str(collision_count))
            print("Time-collision ratio: " + str(simu_time / (collision_count + 1)))
            break

except Exception as e:
    print(traceback.format_exc())
    print(repr(e))
    for ac in actor_list:
        ac.destroy()
