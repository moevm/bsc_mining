import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

from vehicle import VehicleSimulator
from fmcwradar import Fmcw

show_animation = True
now = datetime.datetime.now()
path = now.strftime("%d-%m-%Y %H-%M-%S")
os.mkdir(path)
vr_path = './' + path + '/vel-rad/'
os.mkdir(vr_path)
ar_path = './' + path + '/azim-rad/'
os.mkdir(ar_path)

def main():

    while (os.path.exists(path) and os.path.exists(vr_path) and os.path.exists(ar_path)) == False:
        continue
    # simulation parameters
    sim_time = 40.0  # simulation time
    dt = 0.1  # time tick

    with open("radar_input.txt", "r") as f:
        radar_data = f.readlines()
        radar_data = [line.rstrip() for line in radar_data]

    with open("object_input.txt", "r") as f:
        objects_data = f.readlines()
        objects_data = [line.rstrip() for line in objects_data]

    radar = Fmcw(*(radar_data[0].split(' ')), show_animation, ar_path, vr_path)

    vehs = []

    for i in range(len(objects_data)):
        v1 = VehicleSimulator(*(objects_data[i].split(' ')), radar.d)
        vehs.append(v1)

    time = 0.0
    while time <= sim_time:
        time += dt

        radar.update(dt)

        if show_animation:
            plt.figure(1)
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.axis("equal")
            plt.plot(radar.x, radar.y, "*r")
            plt.quiver(radar.x, radar.y, np.cos(radar.yaw), np.sin(radar.yaw), color='r', width=0.002)

        radar.angle_dots = [] 
        radar.dist_dots = []
        radar.rad_v_dots = []

        for veh in vehs:
            veh.update(dt)
            dots = veh.visible_coords
            if show_animation:
                veh.plot()
                for i in range(len(dots)):
                    plt.plot(*dots[i], "o", color='b')
            
            for i in range(len(dots)):
                x = dots[i][0]
                y = dots[i][1]
                v = veh.v
                tet = veh.yaw
                if radar.y > y:
                    continue
                dist = np.hypot(x-radar.x, y - radar.y)
                angle =  np.rad2deg(np.arcsin((x-radar.x) / dist)) - (90 - np.rad2deg(radar.yaw))

                if np.abs(angle) > 80:
                    continue

                radar.angle_dots.append(round(angle))
                radar.dist_dots.append(dist)
                radar.rad_v_dots.append(((x - radar.x) * (v * np.cos(tet) - radar.v * np.cos(radar.yaw)) + (y - radar.y) * (v * np.sin(tet) - radar.v * np.sin(radar.yaw))) / np.hypot(x - radar.x, y - radar.y))

        radar.find_velocity_range_map(time)
        radar.find_angle_range_map(time)

        if show_animation:
            plt.pause(0.1)

    print("Done")


if __name__ == '__main__':
    main()
