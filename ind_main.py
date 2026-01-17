import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.optimize as so
from ind_utils import get_data, Vehicle, ConstructionZone, InterSectionBoundary, check_cpm_inclusion, IN_X, IN_Y

show_animation = True

VEH_ID = 63
PED_ID = 1
SRV_ID = 55
OFFSET = 70 # does not work well with negative values
SERVICE = ['NC', 'CP', 'IBCP'][0] # non-cooperative, CPM, Intent-based

file_path = r'13_tracks.csv'

D_CONFLICT = 3.0 # conflict distance threshold
TTC_THRESHOLD = 3.0 # TTC threshold 
PREDICTION_HORIZON = 4.0 # prediction horizon for the ibcp
SIGMA = 0.6 # reaction time
COMFORT_DECEL = 4
MAX_DECEL = 8
FRAME_RATE = 25.0

ev_df, rvs_df, pd_df, srv_df = get_data(file_path, VEH_ID, PED_ID, SRV_ID, OFFSET)
ev_x, ev_y, ev_v, ev_a, ev_h, ev_frames = ev_df['xCenter'].values, ev_df['yCenter'].values, np.abs(ev_df['lonVelocity'].values), ev_df['lonAcceleration'].values, ev_df['heading'].values, ev_df['frame'].values
pd_x, pd_y, pd_v , pd_h, pd_frames = pd_df['xCenter'].values, pd_df['yCenter'].values, pd_df['lonVelocity'].values, pd_df['heading'].values, pd_df['frame'].values
srv_x, srv_y, srv_h, srv_frames = srv_df['xCenter'].values, srv_df['yCenter'].values, srv_df['heading'].values, srv_df['frame'].values

# create paths (splines)
ev_u = np.cumsum(np.append([0], np.sqrt(np.diff(ev_x)**2 + np.diff(ev_y)**2)))
ev_tck, ev_u = interpolate.splprep([ev_x, ev_y], s=0, u=ev_u)
EV_A_MIN = np.min(ev_a)
EV_A_MAX = np.max(ev_a)
EV_V_MIN = np.min(ev_v)
EV_V_MAX = np.max(ev_v)

pd_u = np.cumsum(np.append([0], np.sqrt(np.diff(pd_x)**2 + np.diff(pd_y)**2)))
pd_tck, pd_u = interpolate.splprep([pd_x, pd_y], s=0, u=pd_u)

ev = Vehicle(ev_x[0], ev_y[0], ev_h[0], ev_df.iloc[0].width, ev_df.iloc[0].length, ev_df.iloc[0].trackId)
srv = Vehicle(srv_x[0], srv_y[0], srv_h[0], srv_df.iloc[0].width, srv_df.iloc[0].length, srv_df.iloc[0].trackId)
cz = ConstructionZone()
ib = InterSectionBoundary()
ax = cz.ax
ax.plot(ev_x, ev_y, 'r', linewidth=1)
ax.plot(pd_x, pd_y, 'b', linewidth=1)

# returns the sum of distance from point P
def dist_to_p(u, P, tck):
    P_u = interpolate.splev(u, tck)
    return np.sqrt(((P-P_u)**2).sum())

ev_dist_to_in  = so.fminbound(dist_to_p, ev_u.min(), ev_u.max(), args=(np.array([IN_X, IN_Y]), ev_tck))

ev_ped_detection = False
srv_ped_detection = False
conflict_detected_cp = False
conflict_detected_icbp = False
take_over = False
reaction_timer = 0.0
cur_time = 0.0
decel_in = np.zeros_like(ev_x)
decel_pd = np.zeros_like(ev_x)
ttc = 100*np.ones_like(ev_x)

ev_cpm_txed = False
srv_cpm_txed = False
ev_tracker = {}
srv_tracker = {}
last_cpm_time = -1.0
ev_last_cpm_time = -1.0
srv_last_cpm_time = -1.0
ev_cpm_count = []
srv_cpm_count = []
last_cam_time = -1.0
cam_count = 0
last_intent_time = -1.0
intent_count = 0
last_alert_time = -1.0
alert_count = 0

for idx, frame in enumerate(ev_frames):
    ev.update(ev_x[idx], ev_y[idx], ev_h[idx])
    srv.update(srv_x[idx], srv_y[idx], srv_h[idx])
    pd_idx = pd_frames == frame
    if not np.any(pd_idx):
        continue

    P = np.array([pd_x[pd_idx], pd_y[pd_idx]]).flatten()
    pd_on_ev_sp_dist, pd_on_ev_min_dist, _, _  = so.fminbound(dist_to_p, ev_u.min(), ev_u.max(), args=(P, ev_tck), full_output=True) # Distance from s=0 to pd_on_ev_sp_dist, nearest point on the ev spline
    closest_pnt = np.ravel(interpolate.splev(pd_on_ev_sp_dist, ev_tck))

    # Deceleration calculation to stop at the intersection
    stopping_dist = ev_dist_to_in - ev_u[idx] - SIGMA * ev_v[idx]
    if stopping_dist > 0:
        decel_in[idx] = (ev_v[idx])**2/(2*(ev_dist_to_in - ev_u[idx] - SIGMA * ev_v[idx]))
    else:
        decel_in[idx] = 100

    #  Deceleration calculation to stop before pedestrian
    conflict_dist = pd_on_ev_sp_dist - ev_u[idx] - D_CONFLICT - SIGMA * ev_v[idx]
    if conflict_dist > 0: 
        decel_pd[idx] = np.minimum(MAX_DECEL, (ev_v[idx])**2/(2*(pd_on_ev_sp_dist - ev_u[idx] - D_CONFLICT - SIGMA * ev_v[idx])))
    else:
        decel_pd[idx] = MAX_DECEL

    # Current time to conflict
    if pd_on_ev_sp_dist - ev_u[idx] > D_CONFLICT and pd_on_ev_min_dist <= D_CONFLICT:
        ttc[idx] = (pd_on_ev_sp_dist - ev_u[idx] - D_CONFLICT)/ev_v[idx]
    elif pd_on_ev_min_dist > D_CONFLICT:
        ttc[idx] = 100.0
    else:
        ttc[idx] = 0.0

    # sensor detection
    frame_data = pd.concat([pd_df.loc[pd_idx], rvs_df.loc[rvs_df['frame']==frame]])
    det_list, det_polys, det_info, obj_list, obj_polys = ev.get_detections(frame_data, [cz, ib])
    srv_det_list, srv_det_polys, srv_det_info, srv_obj_list, srv_obj_polys = srv.get_detections(frame_data, [cz, ib])

    if PED_ID in det_list:
        ev_ped_detection = True
    else:
        ev_ped_detection = False

    if PED_ID in srv_det_list:
        srv_ped_detection = True
    else:
        srv_ped_detection = False

    # CA service
    if cur_time - last_cam_time >= 0.1:
        cam_count += 1
        last_cam_time = cur_time

    if cur_time - last_intent_time >= 1.0:
        intent_count += 1
        last_intent_time = cur_time

    # CP service
    if cur_time - last_cpm_time >= 0.1:
        ev_cpm_size, ev_tracker = check_cpm_inclusion(cur_time, det_list, det_info, ev_tracker)
        ev_cpm_count.append(ev_cpm_size)
        srv_cpm_size, srv_tracker = check_cpm_inclusion(cur_time, srv_det_list, srv_det_info, srv_tracker)
        srv_cpm_count.append(srv_cpm_size)
        last_cpm_time = cur_time

    ev_cpm_txed = False
    if cur_time - ev_last_cpm_time >= 0.5 and ev_ped_detection:
        ev_cpm_txed = True
        ev_last_cpm_time = cur_time

    srv_cpm_txed = False
    if cur_time - srv_last_cpm_time >= 0.5 and srv_ped_detection:
        srv_cpm_txed = True
        srv_last_cpm_time = cur_time
    
    # CP calculating using current info for vehicle
    for tau in np.arange(0.05, PREDICTION_HORIZON, 0.05):
        pd_x_pred = pd_x[pd_idx] + pd_v[pd_idx] * np.cos(np.deg2rad(pd_h[pd_idx])) * tau
        pd_y_pred = pd_y[pd_idx] + pd_v[pd_idx] * np.sin(np.deg2rad(pd_h[pd_idx])) * tau
        ev_s_pred = ev_u[idx] + ev_v[idx] * tau
        ev_x_pred, ev_y_pred = interpolate.splev(ev_s_pred, ev_tck)
        min_dist = np.sqrt((ev_x_pred-pd_x_pred)**2 + (ev_y_pred-pd_y_pred)**2)

        if min_dist <= D_CONFLICT and tau < TTC_THRESHOLD:
            conflict_detected_cp = True
            break

    # Intent Sharing Calculating ev_s_min and ev_s_max for vehicle
    if srv_ped_detection and not take_over:
        for tau in np.arange(0.05, PREDICTION_HORIZON, 0.05):
            pd_u_pred = pd_u[pd_idx] + pd_v[pd_idx] * tau
            P_pred = np.ravel(interpolate.splev(pd_u_pred, pd_tck))
            ev_v_min = max(EV_V_MIN, ev_v[idx] + EV_A_MIN * tau)
            ev_v_max = min(EV_V_MAX, ev_v[idx] + EV_A_MAX * tau)
            ev_s_min = ev_u[idx] + ev_v_min * tau
            ev_s_max = ev_u[idx] + ev_v_max * tau

            closest_u, min_dist, _, _ = so.fminbound(dist_to_p, ev_s_min, ev_s_max, args=(P_pred, ev_tck), full_output=True)
            if min_dist <= D_CONFLICT and tau < TTC_THRESHOLD:
                conflict_detected_icbp = True
                break

    # plots
    if show_animation:
        ev.radar.plot(ax, 'b')
        # srv.radar.plot(ax, 'b')
        ax.plot(*np.append(ev.box, ev.box[:,[0]], axis=1), color='k')
        ax.fill(*np.append(ev.box, ev.box[:,[0]], axis=1), color='r')
        ax.plot(*np.append(srv.box, srv.box[:,[0]], axis=1), color='k')
        ax.fill(*np.append(srv.box, srv.box[:,[0]], axis=1), color='b')
        # ax.text(ev.box[0, 0], ev.box[1, 0], 'HV')
        for id, poly in zip(obj_list, obj_polys):
            if id == 9999:
                continue
            ax.plot(*np.append(poly, poly[:,[0]], axis=1), 'k')
            # ax.text(poly[0, 0], poly[1, 0], id)
            if id == PED_ID:
                cfz = plt.Circle((pd_x[pd_idx], pd_y[pd_idx]), D_CONFLICT, facecolor='w', edgecolor='b', alpha=0.7)
                ax.add_patch(cfz)
        for id, poly in zip(det_list, det_polys):
            if id == SRV_ID:
                continue
            ax.fill(*np.append(poly, poly[:,[0]], axis=1), color='g')
            # ax.text(poly[0, 0], poly[1, 0], int(id))
        msg = ''
        if ev_ped_detection:
            msg += 'EV: Pedestrian Detected\n'
        else:
            msg += 'EV: Pedestrian Not Detected\n'
        if srv_ped_detection:
            msg += 'RV: Pedestrian Detected'
        else:
            msg += 'RV: Pedestrian Not Detected'
        ax.text(20, -70, msg, color='k', bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=1'))

        plt.pause(0.01)
        # plt.savefig(f'ind_frames/frame_{int(frame):d}.png',bbox_inches='tight', dpi=300)
        for artist in plt.gca().lines[2:] + plt.gca().patches + plt.gca().texts:
            artist.remove()

    if not take_over:
        if decel_in[idx] < COMFORT_DECEL:
            stopping_decel = decel_in[idx]
            stopping_loc = 'intersection'
            potential_decision = f'{SERVICE}: Stop at intersection (frame {int(frame)} - {frame/FRAME_RATE} s): Decel: {decel_in[idx]}, TTC: {ttc[idx]}, Min. TTC: {ttc.min()}'
        elif decel_pd[idx] < MAX_DECEL:
            stopping_decel = decel_pd[idx]
            stopping_loc = 'intersection'
            potential_decision = f'{SERVICE}: Stop at pedestrian (frame {int(frame)} - {frame/FRAME_RATE} s): Decel: {decel_pd[idx]}, TTC: {ttc[idx]}, Min. TTC: {ttc.min()}'
        else:
            stopping_decel = MAX_DECEL
            stopping_loc = 'conflict'
            potential_decision = f'{SERVICE}: Conflict with pedestrian (frame {int(frame)} - {frame/FRAME_RATE} s): Decel: {decel_pd[idx]}, TTC: {ttc[idx]}, Min. TTC: {ttc.min()}, Impact Speed: {np.maximum(0, ev_v[idx]-MAX_DECEL*ttc[idx])}'

        if SERVICE == 'NC':
            if ttc[idx] <= TTC_THRESHOLD and ev_ped_detection:
                print(potential_decision)
                take_over = True
        elif SERVICE == 'CP':
            if conflict_detected_cp and (srv_cpm_txed or ev_ped_detection):
                print(potential_decision)
                take_over = True
        elif SERVICE == 'IBCP':
            if conflict_detected_icbp or (conflict_detected_cp and (srv_cpm_txed or ev_ped_detection)):
                print(potential_decision)
                take_over = True

    if take_over and idx < len(ev_frames)-1:
        if cur_time - last_alert_time >= 0.1:
            alert_count += 1
            last_alert_time = cur_time

        if reaction_timer <= SIGMA:
            reaction_timer += 1.0/FRAME_RATE
            continue

        ev_v[idx+1] = np.maximum(0.0, ev_v[idx] - stopping_decel / FRAME_RATE)
        ev_u[idx+1] = ev_u[idx] + ev_v[idx] / FRAME_RATE
        ev_x[idx+1], ev_y[idx+1] = interpolate.splev(ev_u[idx+1], ev_tck)
        ev_h[idx+1] = np.rad2deg(np.arctan2(ev_y[idx+1]-ev_y[idx], ev_x[idx+1]-ev_x[idx]))
        if ev_v[idx+1] <= 0.0:
            print(f'Stopped at frame {int(frame)} - {frame/FRAME_RATE} s')
            break

    cur_time += 1.0/FRAME_RATE

data_size = cam_count*350*2 # 2 for ev and srv
max_data_size_per_sec = 350*2*10
if SERVICE in ['CP', 'IBCP']:
    data_size += (121+35)*(len(ev_cpm_count)+len(srv_cpm_count)) # header and sensor info
    data_size += 35*(sum(ev_cpm_count)+sum(srv_cpm_count)) # objects
    max_data_size_per_sec += (121+35)*2*10
    tmp = np.cumsum(np.add(ev_cpm_count, srv_cpm_count))
    tmp[10:] = tmp[10:] - tmp[:-10]
    max_data_size_per_sec += 35*max(tmp)
elif SERVICE == 'IBCP':
    data_size += 100*2*intent_count # intent
    data_size += 100*alert_count # alert until the vehicle stops
    max_data_size_per_sec += 100*1
    max_data_size_per_sec += 100*10

print(f'Average channel load is {data_size/cur_time} Byte/sec and maximum is {max_data_size_per_sec} over {cur_time}')

fig, ax = plt.subplots(3, 1)
ax[0].plot(ttc)
ax[0].set_xlabel('frame')
ax[0].set_ylabel('TTC')
ax[1].plot(decel_in)
ax[1].set_xlabel('frame')
ax[1].set_ylabel('Decel (IN)')
ax[2].plot(decel_pd)
ax[2].set_xlabel('frame')
ax[2].set_ylabel('Decel (PD)')
plt.show()

# run the command below in terminal to get the animation
# ffmpeg -i frame_%d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 15 -pix_fmt yuv420p out.mp4 -y
