import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, concat

# parameters
IN_X, IN_Y = 51.03, -27.96
NUM_RAYS = 120 # Must be between 1 and 360
RAY_LENGTH = 150 # Length of each Ray 
RAY_SPREAD = 120 # Value of Ray Spread in degree
RAY_SPREAD_INIT = - RAY_SPREAD/2
RAY_SPREAD_END = RAY_SPREAD/2

# file 
file_colms = ['recordingId', 'trackId', 'frame', 'trackLifetime', 'xCenter', 'yCenter',
                'heading', 'width', 'length', 'xVelocity', 'yVelocity', 'xAcceleration',
                'yAcceleration', 'lonVelocity', 'latVelocity', 'lonAcceleration', 'latAcceleration']

def get_data(file_path, veh_id, ped_id, srv_id, offset):
    data_array = read_csv(file_path, sep=',').values
    df = DataFrame(data_array, columns=file_colms)

    # Ego vehicle
    ev_df = df.loc[df['trackId'] == veh_id].copy()

    # Stationary Remote vehicle
    srv_df = df.loc[df['trackId'] == srv_id].copy()

    # Remote vehicles
    rvs_df = df.loc[np.logical_and(np.isin(df['frame'], ev_df['frame'].values), df['trackId'] != veh_id)].copy()

    # Pedestrain
    pd_df = df.loc[df['trackId'] == ped_id].copy()

    # Reset frames
    original_offset = ev_df.iloc[0].frame - pd_df.iloc[0].frame
    srv_df['frame'] -= ev_df.iloc[0].frame # For veh55 -> 0 to 13420frames
    rvs_df['frame'] -= ev_df.iloc[0].frame
    ev_df['frame'] -= ev_df.iloc[0].frame # For veh172 -> 0 to 349frames
    pd_df['frame'] -= (pd_df.iloc[0].frame + offset) # For +ve offset -> start from 0, for -ve offset -> starts from -ve value
    # pd_df = pd_df[pd_df['frame'].isin(ev_df['frame'])]
    if offset >=0:
        pass
        # pd_df = pd_df[pd_df['frame'].isin(ev_df['frame'])]
    else:
        pass
        # ev_df = ev_df[ev_df['frame'].isin(pd_df['frame'])]
        # srv_df = srv_df[srv_df['frame'].isin(pd_df['frame'])]
        # rvs_df = rvs_df[rvs_df['frame'].isin(pd_df['frame'])]

    if pd_df.empty:
        raise('Offset is larger than data size.')

    return ev_df, rvs_df, pd_df, srv_df


class Ray:
    def __init__(self, x, y, angle, range):
        self.xstart = x
        self.ystart = y
        self.length = range
        self.xend = x + range*np.cos(angle)
        self.yend = y + range*np.sin(angle)

    def update_end_pnt(self, pnt):
        self.xend, self.yend = pnt

    def check_intersection(self, line):
            x1, y1 = (self.xstart, self.ystart)
            x2, y2 = (self.xend, self.yend)
            x3, y3 = line[0]
            x4, y4 = line[1]

            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
            if denominator == 0:
                return None
            
            t = numerator / denominator
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

            if 1 >= t >= 0 and 1 >= u >= 0:
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                collide_pos = [x, y]
                return collide_pos

class Sensor:
    def __init__(self, x, y, heading, range, fov, num_rays) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.num_rays = num_rays
        self.range = range
        self.fov = fov
        self.create_rays()

    def update(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading
        self.create_rays()

    def create_rays(self):
        self.rays = []
        for i in np.linspace(-self.fov/2, self.fov/2, self.num_rays):
            self.rays.append(Ray(self.x, self.y, np.deg2rad(i+self.heading), self.range))

    def plot(self, ax, clr='b'):
        for ray in self.rays:
            ax.plot([ray.xstart, ray.xend], [ray.ystart, ray.yend], clr, linewidth=0.5, alpha=0.7)
        
class Vehicle:
    def __init__(self, x, y, heading, width, length, id) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.width = width
        self.length = length
        self.id = id
        self.radar = Sensor(x, y, heading, range=100, fov=120, num_rays=121)
        self.create_box()

    def update(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading
        self.radar.update(x, y, heading)
        self.create_box()

    def get_detections(self, rvs_df, buildings):
        obj_list = []
        obj_polys = []
        obj_infos = []

        # buildings/structures
        for obj in buildings:
            obj_list.append(9999)
            obj_polys.append(obj.box)

        # pedestrian and rvs
        for obj_id in np.unique(rvs_df.loc[rvs_df['trackId'] != self.id, 'trackId']):
            rv_x, rv_y, rv_heading, rv_speed, rv_length, rv_width = rvs_df.loc[rvs_df['trackId'] == obj_id, ['xCenter', 'yCenter', 'heading', 'lonVelocity', 'length', 'width']].values[0]
            rv_box = np.array([[-rv_length/2, -rv_length/2, +rv_length/2, +rv_length/2],
                                [-rv_width/2, +rv_width/2, +rv_width/2, -rv_width/2]])
            rv_box = rotation_matrix(rv_heading) @ rv_box + np.array([[rv_x], [rv_y]])
            obj_list.append(int(obj_id))
            obj_polys.append(rv_box)
            obj_infos.append([rv_x, rv_y, rv_heading, rv_speed])

        # detections
        det_id_list = []
        det_poly_list = []
        det_info = []
        for ray in self.radar.rays:
            closest = 1000000
            ray_det_obj = None
            ray_det_info = None
            for obj_id, obj_poly, obj_info in zip(obj_list, obj_polys, obj_infos):
                s = obj_poly.T
                N = len(s)
                for i in range(N): # assume closed shape, if open stop at N-1
                    intersect_pnt = ray.check_intersection([s[i], s[(i+1)%N]])
                    if intersect_pnt is not None:
                        dx = ray.xstart - intersect_pnt[0]
                        dy = ray.ystart - intersect_pnt[1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < closest:
                            closest = dist
                            ray.update_end_pnt(intersect_pnt)
                            if obj_id != 9999:
                                ray_det_obj = obj_id
                                ray_det_poly = obj_poly
                                ray_det_info = obj_info
            if ray_det_obj is not None and ray_det_obj not in det_id_list:
                det_id_list.append(ray_det_obj)
                det_poly_list.append(ray_det_poly)
                det_info.append(ray_det_info)

        return det_id_list, det_poly_list, det_info, obj_list, obj_polys

    def create_box(self):
            self.box = np.array([[-self.length/2, -self.length/2, self.length/2, self.length/2],
                        [-self.width/2, self.width/2, self.width/2, -self.width/2]])
            self.box = self.rotation_matrix() @ self.box + np.array([[self.x], [self.y]])

    def rotation_matrix(self):
        theta = np.deg2rad(self.heading)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))
    
    def plot(self, ax, clr='b'):
        ax.plot(*np.append(self.box, self.box[:,[0]], axis=1), clr)
    
class ConstructionZone:
    def __init__(self) -> None:
        self.box = np.array([[65.11, -25.32], 
                             [62.56, -28.15], 
                             [61.10, -30.70], 
                             [61.10, -33.04], 
                             [63.05, -34.41], 
                             [64.71, -33.04], 
                             [67.06, -26.69],
                             [65.20, -25.32],
                             [65.11, -25.32]]).T

        background_image = plt.imread(r'13_background_g.png')
        scaling_down_factor = 12.0
        ortho_px_to_meter = 0.00814636091724916
        background_lim = ortho_px_to_meter*np.array([2000, 11500, -9450, 0])

        cz = plt.Polygon(self.box.T, edgecolor='k', facecolor='y', hatch=r'XXX')
        image_height, image_width = background_image.shape[:2]
        background_extent=[0, image_width*ortho_px_to_meter*scaling_down_factor, -image_height*ortho_px_to_meter*scaling_down_factor, 0]

        fig, ax = plt.subplots(1, 1)
        ax.set_xticklabels([])
        ax.set_autoscale_on(False)
        ax.set_yticklabels([])
        ax.axis('off')
        ax.imshow(background_image, extent=background_extent)
        # ax.add_patch(cz)
        ax.set_xlim(background_lim[:2])
        ax.set_ylim(background_lim[2:])

        self.ax = ax

    def plot(self, ax, clr='k'):
        ax.plot(*np.append(self.box, self.box[:,[0]], axis=1), clr)

class InterSectionBoundary:
    def __init__(self) -> None:
        self.box = np.array([[21.15, -6.49], 
                             [28.78, -0.02], 
                             [35.73, -0.05], 
                             [45.12, -10.32], 
                             [55.57, -1.79], 
                             [75.05, -22.22], 
                             [70.05, -28.03],
                             [68.48, -28.20],
                             [67.55, -29.13],
                             [68.23, -29.91],
                             [68.14, -35.62],
                             [87.88, -57.08],
                             [76.25, -68.13],
                             [56.60, -46.34],
                             [53.62, -49.02],
                             [52.39, -47.90],
                             [45.65, -54.89],
                             [34.85, -42.76],
                             [42.72, -35.09],
                             [43.26, -31.03],
                             [21.15, -6.49]]).T

    def plot(self, ax, clr='k'):
        ax.plot(*np.append(self.box, self.box[:,[0]], axis=1), clr)

def rotation_matrix(theta):
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def check_cpm_inclusion(cur_time, det_list, det_info, tracker):
    cnt = 0
    for obj_id, obj_info in zip(det_list, det_info):
        rv_x, rv_y, rv_h, rv_v = obj_info
        check_inclusion = False
        if obj_id in tracker:
            last_cpm = tracker[obj_id]
            d = np.sqrt((rv_x - last_cpm[1])**2 + (rv_y - last_cpm[2])**2)  # distance since last cpm
            check_inclusion = (abs(last_cpm[3] - rv_v) > 0.5) or (abs(d) > 4) or ((timestep - last_cpm[0]) > 1)
        else:
            check_inclusion = True

        if check_inclusion:
            tracker[obj_id] = obj_info
            cnt += 1

    return cnt, tracker