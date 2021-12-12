import open3d as o3d
import numpy as np
import math

def grid_map(k, interval = 10):
    temp_0 = np.arange(-k, k + interval, interval).reshape((-1, 1))
    temp_1 = np.array([-k for i in range(len(temp_0))]).reshape((-1, 1))
    temp_2 = np.array([k for i in range(len(temp_0))]).reshape((-1, 1))

    loc_left = np.concatenate((temp_1, temp_0), 1)
    loc_right = np.concatenate((temp_2, temp_0), 1)
    loc_lower = np.concatenate((temp_0, temp_1), 1)
    loc_upper = np.concatenate((temp_0, temp_2), 1)
    # Vertical and horizontal
    lineset_vertical = o3d.geometry.LineSet()
    points_vertical = np.concatenate((loc_lower, loc_upper), 0)
    points_vertical = np.concatenate((points_vertical, np.zeros((len(points_vertical), 1))), 1)
    lines_box_vertical = []
    for i in range(len(points_vertical) // 2):
        lines_box_vertical.append([i, i + len(loc_lower)])
    colors = np.array([[0.2, 0.2, 0.2] for j in range(len(lines_box_vertical))])
    lineset_vertical.points = o3d.utility.Vector3dVector(points_vertical)
    lineset_vertical.lines = o3d.utility.Vector2iVector(np.array(lines_box_vertical))
    lineset_vertical.colors = o3d.utility.Vector3dVector(colors)

    lineset_horizontal = o3d.geometry.LineSet()
    points_horizontal = np.concatenate((loc_left, loc_right), 0)
    points_horizontal = np.concatenate((points_horizontal, np.zeros((len(points_horizontal), 1))), 1)
    lines_box_horizontal = []
    for j in range(len(points_horizontal) // 2):
        lines_box_horizontal.append([j, j + len(loc_lower)])
    colors = np.array([[0.2, 0.2, 0.2] for j in range(len(lines_box_horizontal))])
    lineset_horizontal.points = o3d.utility.Vector3dVector(points_horizontal)
    lineset_horizontal.lines = o3d.utility.Vector2iVector(np.array(lines_box_horizontal))
    lineset_horizontal.colors = o3d.utility.Vector3dVector(colors)
    return [lineset_vertical, lineset_horizontal]



def corners_nd(dims, origin=0.5):
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
        axis=1).astype(dims.dtype)

    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])
    return corners
def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)
def center_to_corner_box3d(bbox,
                           origin=(0.5, 0.5, 0.5),
                           axis=2):
    centers = bbox[:, :3]
    dims = bbox[:, 3:6]
    angles = bbox[:, 6]
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

def custom_draw_geometry(pcd,linesets):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    grid_lines = grid_map(300)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(axis_pcd)
    for i in linesets:
        vis.add_geometry(i)
    for i in grid_lines:
        vis.add_geometry(i)

    render_option = vis.get_render_option()
    render_option.point_size = 2
    render_option.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()

def show(point_path, label_path):
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    bbox = []
    for line in open(label_path):
        line = line.strip().split(" ")
        line = [float(i) for i in line[-7:]]
        bbox.append(line)
    bbox = np.array(bbox)
    bbox = bbox[:, [0, 1, 2, 4, 3, 5, 6]]
    bbox[:, -1] *= -1
    bbox[:, -1] += math.pi * 1.5
    box3d = center_to_corner_box3d(bbox)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points[:, :3])
    line_sets = [o3d.geometry.LineSet() for _ in range(len(box3d))]
    for j in range(len(box3d)):
        points_box = box3d[j]
        lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])
        colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
        colors[4, :] = np.array([1, 0, 0])
        colors[8, :] = np.array([1, 0, 0])
        colors[9, :] = np.array([1, 0, 0])
        line_sets[j].points = o3d.utility.Vector3dVector(points_box)
        line_sets[j].lines = o3d.utility.Vector2iVector(lines_box)
        line_sets[j].colors = o3d.utility.Vector3dVector(colors)
    custom_draw_geometry(pointcloud, line_sets)

if __name__ == "__main__":
    label_path = "../open3d_demo/000000.txt"
    point_path = "../open3d_demo/000000.bin"
    show(point_path, label_path)
