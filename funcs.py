#Sreehari Premkumar
#Northeastern University

import numpy as np

def lidar2camTransform(calib):
    
    P_lidar2cam = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    
    R_ref2rect = np.eye(4)
    R_ref2rect[:3, :3] = calib['R0_rect'].reshape(3, 3)
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_lidar2cam
    return proj_mat


def project_to_image(points, proj_mat):

    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


def project_camera_to_lidar(points, proj_mat):

    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    return points[:3, :]


def calibration(filepath): #Read calibration to dictionary
    
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
