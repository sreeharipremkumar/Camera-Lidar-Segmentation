#Sreehari Premkumar
#Northeastern University

import os
import cv2
import matplotlib.pyplot as plt
import open3d
import time

from funcs import *


def render_lidar_as_video(folder_path, calib, width, height):

    file_list = os.listdir(folder_path)
    file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name="Lidar")
    #vis.get_render_option().point_size = 1.5
    #vis.get_render_option().background_color = np.asarray([0.95, 0.95, 0.95])

    pcd = open3d.geometry.PointCloud()

    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    first_run = True

    # Start streaming loop
    while True:
        # Loop through files in folder
        file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for file_name in file_list:
            # Check if file is a .bin file
            if file_name.endswith(".bin"):
                print(file_name)
                # Load point cloud from file
                file_path = os.path.join(folder_path, file_name)
                
                point_cloud = np.fromfile(file_path, dtype=np.float32)
                point_cloud = point_cloud.reshape((-1, 4))[:, :3]

                # projection matrix (project from velo2cam2)
                proj_velo2cam2 = lidar2camTransform(calib)

                # apply projection
                pts_2d = project_to_image(point_cloud.transpose(), proj_velo2cam2)

                # Filter lidar points to be within image FOV
                new_fov = np.where((pts_2d[0, :] < width) & (pts_2d[0, :] >= 0) & (pts_2d[1, :] < height) & (pts_2d[1, :] >= 0) & (point_cloud[:, 0] > 0))[0]
                pc_new_fov = point_cloud[new_fov, :]

                # create open3d point cloud and axis
                pcd.points = open3d.utility.Vector3dVector(pc_new_fov)

                if first_run == True:
                    vis.add_geometry(pcd)
                    first_run = False
                else:
                    vis.update_geometry(pcd)
                    
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.1)                         


def render_lidar_on_image(lidar_folder, camera_folder, calib, width, height):

    lidar_list = os.listdir(lidar_folder)
    
    camera_list = os.listdir(camera_folder)
    camera_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    while True:
        # Loop through files in folder
        lidar_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for i,lidar_name in enumerate(lidar_list):
            camera_path = os.path.join(camera_folder, camera_list[i])
            lidar_path = os.path.join(lidar_folder, lidar_name)
            img = cv2.cvtColor(cv2.imread(camera_path), cv2.COLOR_BGR2RGB)
            
            # projection matrix (project from velo2cam2)
            proj_velo2cam2 = lidar2camTransform(calib)

            # apply projection
            point_cloud = np.fromfile(lidar_path, dtype=np.float32)
            point_cloud = point_cloud.reshape((-1, 4))[:, :3]
            pts_2d = project_to_image(point_cloud.transpose(), proj_velo2cam2)

            # Filter lidar points to be within image FOV
            new_fov = np.where((pts_2d[0, :] < width) & (pts_2d[0, :] >= 0) &
                            (pts_2d[1, :] < height) & (pts_2d[1, :] >= 0) &
                            (point_cloud[:, 0] > 0)
                            )[0]

            # Filter out pixels points
            imgfov_pc_pixel = pts_2d[:, new_fov]

            # Retrieve depth from lidar
            pc_new_fov = point_cloud[new_fov, :]
            pc_new_fov = np.hstack((pc_new_fov, np.ones((pc_new_fov.shape[0], 1))))
            imgfov_pc_cam2 = proj_velo2cam2 @ pc_new_fov.transpose()

            cmap = plt.cm.get_cmap('hsv', 256)
            cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

            for i in range(imgfov_pc_pixel.shape[1]):
                depth = imgfov_pc_cam2[2, i]
                color = cmap[int(640.0 / depth)%256, :]
                cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                                 int(np.round(imgfov_pc_pixel[1, i]))),
                           2, color=tuple(color), thickness=-1)
            cv2.imshow("image",img)
            key = cv2.waitKey(1)

            if(key == ord('q')):
                exit()



if __name__ == '__main__':

    # Load test image to get width, height
    rgb = cv2.cvtColor(cv2.imread(os.path.join('dataset/data/image_02/data/0000000000.png')), cv2.COLOR_BGR2RGB)
    height, width,_ = rgb.shape

    # Load calibration
    calib = calibration('dataset/calib/calib.txt')

    # Run 1 of the following at a time

    # # Load Lidar PC
    # lidar_path = "dataset/data/velodyne_points/data"
    # render_lidar_as_video(folder_path=lidar_path, calib= calib, width=width, height=height)

    #Load Lidar and Camera folder Path
    lidar_path = "dataset/data/velodyne_points/data"
    camera_path = "dataset/data/image_02/data"
    render_lidar_on_image(lidar_path,camera_path, calib, width, height)

    

