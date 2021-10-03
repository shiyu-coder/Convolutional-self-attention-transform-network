import glob
import os
import sys
import time
import random
import time
import numpy as np
import cv2
import h5py

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 320
IM_HEIIGHT = 180
count = 0
imgs = []
steers = []



def save_h5(imgs, steers):
    index = "%04d" % (count // 200)
    root_dir = '../../town01/SeqTrain'
    path = os.path.join(root_dir, index + '.h5')
    print("saving " + path)
    f = h5py.File(path, 'w')  # 创建一个h5文件，文件指针是f
    f['img'] = imgs  # 将数据写入文件的主键data下面
    f['steer'] = steers  # 将数据写入文件的主键labels下面
    f.close()


def process_img(image):
    global count, steers
    global imgs
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(10)
    i3 = i3.transpose(2, 0, 1)
    control = vehicle.get_control()
    if vehicle.is_at_traffic_light():
        if (count + 1) % 2 == 0:
            imgs.append(i3)
            steers.append(float(control.steer))
            count += 1
        if (count + 1) % 200 == 0:
            save_h5(imgs, steers)
            imgs = []
            steers = []
    return i3


actor_list = []
try:
    # create client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    # world connection
    world = client.load_world('Town04')
    world.set_weather(carla.WeatherParameters.ClearNoon)
    # get blueprint libarary
    blueprint_library = world.get_blueprint_library()
    # Choose a vehicle blueprint which name is model3 and select the first one
    bp = blueprint_library.filter("model3")[0]
    print(bp)
    # Returns a list of recommended spawning points and random choice one from it
    spawn_point = random.choice(world.get_map().get_spawn_points())
    # spawn vehicle to the world by spawn_actor method
    vehicle = world.spawn_actor(bp, spawn_point)
    # control the vehicle
    vehicle.set_autopilot(enabled=True)
    # vehicle.apply_control(carla.VehicleControl(throttle=0.1,steer=0.0))
    # add vehicle to the actor list
    actor_list.append(vehicle)
    # as the same use find method to find a sensor actor.
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    # set the attribute of camera
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIIGHT}")
    cam_bp.set_attribute("fov", "110")
    # add camera sensor to the vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)

    sensor.listen(lambda data: process_img(data))

    time.sleep(4000)
finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")