#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import sys
import numpy as np
import utils.simu_utils as utils
import math
import queue
import torch
from models.model import CSATNet
import cv2
import time


# In[2]:


try:
    sys.path.append(glob.glob('../carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


# In[3]:


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
# print(device)
mynet = CSATNet(128, 4, 16, 3, 2, 3, 3, (88, 200), 1, 0.1, 32, True)
mynet.load_state_dict(torch.load(''))
if use_gpu:
    mynet = mynet.cuda()
# mynet = torch.load("model/Train_4_NVIDIA_ORIGIN/Epoch8_Val_loss0.00981.pkl")    # the best util now
# mynet.to(device)


# In[ ]:


IM_WIDTH = 200
IM_HEIGHT = 99


def display(image):
    cv2.imshow("display", image)
    cv2.waitKey(1)


def get_img(image):
    image_queue.queue.clear()
    image_queue.put_nowait(image)


def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))


actor_list = []
client = carla.Client('localhost', 2000)
client.set_timeout(10)
world = client.get_world()

# set the weather
weather = world.get_weather()
weather.sun_altitude_angle = 90
world.set_weather(weather)

spectator = world.get_spectator()
blueprint_library = world.get_blueprint_library()
vehicle_blueprints = blueprint_library.filter('model3')
blueprint = vehicle_blueprints[0]
point = world.get_map().get_spawn_points()[2]

vehicle = world.spawn_actor(blueprint, point)
print(vehicle.type_id)
actor_list.append(vehicle)

world.wait_for_tick()
spectator.set_transform(get_transform(vehicle.get_location(), 0))
cam_bp = blueprint_library.filter("sensor.camera.rgb")[0]
cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
cam_bp.set_attribute("fov", "110") #"fov" field of view
spawn_point = carla.Transform(carla.Location(x=2, z=1))#locate the camera
sensor = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
actor_list.append(sensor)
time.sleep(3)
image_queue = queue.Queue(1)
sensor.listen(lambda data: get_img(data))

billboard_width = 590
billboard_height = 400
billboard = np.array([[0, 0], [billboard_width, 0], [0, billboard_height], [billboard_width, billboard_height]], dtype=np.float32)

seq_len = 4
try:
    batch_img = []
    for i in range(0, 100000):
        image = image_queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((IM_HEIGHT, IM_WIDTH, 4))
        image = image[0:200][33:99]

        image = image[:, :, [2, 1, 0]]
        image = utils.IN(image)

        if i == 0:
            for i in range(seq_len):
                batch_img.append(image)
        else:
            batch_img = batch_img[1:]
            batch_img.append(image)

        # image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(np.array(batch_img)).float()
        image = image.permute(0, 3, 1, 2)
        if use_gpu:
            image = image.cuda()
        # image = image.to(device)

        # compute the steer
        image.requires_grad_(True)
        predict_steer = float(mynet(image))
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
except BaseException:
    for ac in actor_list:
        ac.destroy()

for ac in actor_list:
    ac.destroy()


# In[ ]:




