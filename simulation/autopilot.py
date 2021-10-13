#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from models.model import CSATNet, CSATNet_multitask
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


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
# print(device)
mynet = CSATNet_multitask(128, 4, 5, 3, 2, 3, 3, 32, 1, 0.05, 32, False, True)
mynet.load_state_dict(torch.load('../checkpoints/CSATNet_multitask-ADHDataset-nhi128-nhe4-sl5-cl1n3-cl2n2-eln3-dln3-vn32-is(180, 320)-ls1-do0.05-mos32-aFalse-lapTrue-0/checkpoint.pth'))
# mynet = NVIDIA_ORIGIN()
# mynet.load_state_dict(torch.load('../checkpoints/NVIDIA_ORIGIN-ADHDataset-nhi128-nhe4-sl1-cl1n3-cl2n2-eln3-dln3-vn32-is(180, 320)-ls1-do0.05-mos32-aFalse-lapTrue-0/checkpoint.pth'))
seq_len = 5
mynet.eval()
if use_gpu:
    mynet = mynet.cuda()
# mynet = torch.load("model/Train_4_NVIDIA_ORIGIN/Epoch8_Val_loss0.00981.pkl")    # the best util now
# mynet.to(device)

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


def dect_collision(data):
    global collision_count
    global vehicle, control, collision
    if data.timestamp - collision > 3:
        collision = data.timestamp
        print("collision happend! total collision num: " + str(collision_count))
        all_new_pos = world.get_map().get_spawn_points()
        new_pos = np.random.choice(all_new_pos)
        vehicle.set_transform(new_pos)
        vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        collision_count += 1
        time.sleep(1)
        # spectator.set_transform(get_transform(vehicle.get_location(), 0))



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

collision_sensor_bp = blueprint_library.filter("sensor.other.collision")[0]
collision_point = carla.Transform(carla.Location())
collision_sensor = world.spawn_actor(collision_sensor_bp, collision_point, attach_to=vehicle)
collision_sensor.listen(lambda data: dect_collision(data))

billboard_width = 590
billboard_height = 400
billboard = np.array([[0, 0], [billboard_width, 0], [0, billboard_height], [billboard_width, billboard_height]], dtype=np.float32)

Image_Array = None
collision_count = 0
simu_time = 1200
dect_time = 0
begin_time = int(time.time())
collision = False
try:
    batch_img = []
    for i in range(0, 1000000):
        image = image_queue.get()
        # image = np.array(image.raw_data)
        # image = image.reshape((IM_HEIGHT, IM_WIDTH, 4))
        #
        # image = image[:, :, 3]
        # image = utils.IN(image)

        if i == 0:
            for j in range(seq_len):
                batch_img.append(image)
        else:
            batch_img = batch_img[1:]
            batch_img.append(image)

        image = torch.from_numpy(np.array(batch_img)).float()
        image = image.permute(0, 3, 1, 2)
        if use_gpu:
            image = image.cuda()
        # image = image.to(device)

        # compute the steer
        image.requires_grad_(True)
        image = image.unsqueeze(0)
        predict_steer = float(mynet(image)[0, -1, 0])
        v = vehicle.get_velocity()
        speed_m_per_s = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        control = vehicle.get_control()
        control.steer = float(predict_steer)
        if speed_m_per_s < 5:
            control.throttle = 0.5
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = 0.5
        vehicle.apply_control(control)

        if Image_Array is not None:
            image = copy.copy(Image_Array)
            cv2.imshow('Spector View', image)

        if i % 1000 == 0:
            wea = np.random.choice(weathers)
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
#
# for ac in actor_list:
#     ac.destroy()


# In[ ]:




