import os
import math
import pandas
import numpy as np
from python.pandaset import DataSet
from python.pandaset import geometry


def cuboids_to_boxes_transform_coord(cuboids: pandas.DataFrame, poses, rotate_yaw=None):
    str_ret = ''
    numb_ob = 0
    sensor1_num = 0
    for i, row in cuboids.iterrows():
        # cuboids.sensor_id值为-1,0,1，
        # 对于两个雷达重复区域的框用0表示(mechanical 360° LiDAR)用1表示 (front-facing LiDAR)，其它区域用-1表示
        sensor_id = row["cuboids.sensor_id"]
        if sensor_id == 1:
            sensor1_num += 1
            continue
        # if row["label"] == 'Pedestrian with Object':
        #     ped_with_obj_num += 1
        #     continue
        # 坐标偏移
        # pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, yaw = row["position.x"], row["position.y"], row["position.z"], row["dimensions.x"], row["dimensions.y"],  row["dimensions.z"], row["yaw"]
        w, l, h = row["dimensions.x"], row["dimensions.y"],  row["dimensions.z"]
        yaw = row["yaw"] + rotate_yaw
        center_xyz = np.array([[row["position.x"], row["position.y"], row["position.z"]]])
        rotate_corners, _ = geometry.lidar_points_to_ego(center_xyz, poses)
        x, y, z = rotate_corners[0, 0], rotate_corners[0, 1], rotate_corners[0, 2]

        while yaw < -np.pi:
            yaw = np.pi * 2 + yaw
        while yaw > np.pi:
            yaw = yaw - np.pi * 2

        str = '{} {} {} {} {} {} {} {}\n'.format(row["label"], y, x, z, l, w, h, -yaw)

        str_ret += str
        numb_ob = numb_ob + 1
    return str_ret, numb_ob

def gen_pandaset_kitti():
    des_path = '/data/Documents/auto-drive/datasets'
    pandas_root_path = '/data/Documents/auto-drive/PandaSet_pandaset_0'

    ds = DataSet(pandas_root_path)
    pkl_dir = os.listdir(pandas_root_path)
    pkl_dir.sort()
    for index_nu in pkl_dir:
        print("index_num =", index_nu)
        sequence = ds[index_nu]
        sequence.load_lidar().load_cuboids().load_semseg()

        for i in range(80):
            print("frame =", i)
            frame = i

            bin_points = sequence.lidar[frame].to_numpy()[..., :4]

            # 获取每一帧传感器信息
            poses = sequence.lidar.poses[frame]
            # 将点云的xyz从世界坐标系下转换成传感器自身坐标系
            ego_pandar_points, rotate_yaw = geometry.lidar_points_to_ego(bin_points[:, :3], poses)

            labels, numb_ob = cuboids_to_boxes_transform_coord(sequence.cuboids[frame], poses, rotate_yaw)

            pl = bin_points.reshape(-1, 4).astype(np.float32)
            pl[:, 0] = ego_pandar_points[:, 1]
            pl[:, 1] = ego_pandar_points[:, 0]
            pl[:, 2] = ego_pandar_points[:, 2]
            pl[:, 3] /= 255.


            filename = str(i).zfill(6)
            bin_dir = os.path.join(des_path, "{}_bin".format(index_nu).zfill(3))
            txt_dir = os.path.join(des_path, "{}_txt".format(index_nu).zfill(3))
            if not os.path.exists(bin_dir):
                os.mkdir(bin_dir)
            if not os.path.exists(txt_dir):
                os.mkdir(txt_dir)
            velodyne_file_new = os.path.join(bin_dir, filename) + '.bin'
            pl.tofile(velodyne_file_new)

            velodyne_file_label_txt = os.path.join(txt_dir, filename) + '.txt'
            with open(velodyne_file_label_txt, 'w') as f:
                f.writelines(labels)
    print('done')

if __name__ == '__main__':
    gen_pandaset_kitti()
