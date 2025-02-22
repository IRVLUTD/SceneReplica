import argparse, os

import numpy as np
import trimesh
from shapely.geometry import Polygon
from transforms3d.quaternions import mat2quat, quat2mat

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


def generate_grid(
    div_x,
    div_y,
    table_position,
    table_side,
    table_height=0.8,
    table_front_margin=0.15,
    table_rear_margin=0.4,
    table_left_margin=0.2,
    table_right_margin=0.2,
):
    """
    Arguments: number of required division along x, along y,
               list of table position, value of table side length, table height,
               front, rear, left, right margins
    Return: numpy array of 3d vertices i.e., np.array([[x1,y1,z1],[x2,y2,z2],...])
    """

    # compute margins on table
    start_x = table_position[0] - table_side / 2 + table_front_margin
    end_x = table_position[0] + table_side / 2 - table_rear_margin

    start_y = table_position[1] - table_side / 2 + table_left_margin
    end_y = table_position[1] + table_side / 2 - table_right_margin

    # Adjust margins to exclude the points on the margin before diving the grid
    x_divison_size = abs(end_x - start_x) / div_x / 2
    y_divison_size = abs(end_y - start_y) / div_y / 2

    start_x = start_x + x_divison_size
    start_y = start_y + y_divison_size

    end_x = end_x - x_divison_size
    end_y = end_y - y_divison_size

    # Generating spawn locations
    x_div = np.linspace(start_x, end_x, div_x)
    y_div = np.linspace(start_y, end_y, div_y)

    x_grid, y_grid = np.meshgrid(x_div, y_div)

    print(f"spawn locations generated . ")

    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    z_grid = x_grid.copy() * 0 + table_height
    vertices = np.column_stack((x_grid, y_grid, z_grid))
    print(f"{vertices.shape}")
    # print(f"vertices {vertices}")
    return vertices


def visualise2d(vertices, table_position, table_side):
    start_x = table_position[0] - table_side / 2
    start_y = table_position[1] - table_side / 2

    rect = Rectangle(
        (start_x, start_y), table_side, table_side, alpha=0.1, label="Table"
    )

    fig, ax = plt.subplots()

    # Spawn positions
    spwan_positions = ax.scatter(
        vertices[:, 0], vertices[:, 1], color="red", label="object spawn positions"
    )

    # Fetch position
    fetch_position = ax.scatter(0, 0, color="green", label="fetch position")

    # Table
    ax.add_patch(rect)

    # Custom legend items
    ax.legend(handles=[rect, spwan_positions, fetch_position])

    ax.set_ylim([2, -2])
    ax.set_xlim([2, -2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.show()


def visualise3d(vertices, table_position, table_side):
    start_x = table_position[0] - table_side / 2
    start_y = table_position[1] - table_side / 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Spawn positions
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="red")

    # Fetch position
    ax.scatter(0, 0, 0, color="green")

    # Table
    x_range = np.linspace(start_x, start_x + table_side, 2)
    y_range = np.linspace(start_y, start_y + table_side, 2)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, 0.8)
    ax.plot_surface(X, Y, Z, alpha=0.1, color="blue")

    # Custom legend items
    spawn_positions_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Spawn positions",
        markerfacecolor="red",
        markersize=8,
    )
    fetch_position_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Fetch position",
        markerfacecolor="green",
        markersize=8,
    )
    table_legend = Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="Table",
        markerfacecolor="blue",
        markersize=8,
    )

    ax.legend(handles=[spawn_positions_legend, fetch_position_legend, table_legend])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def rt_to_ros_qt(rt):
    """
    rt: 4x4 transform matrix
    Returns:
        quat: quaternion x,y,z,w
        trans: x,y,z location
    """
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = list(rt[:3, 3])
    return quat, trans


def load_mesh(models_path, ycb_id):
    mesh_path = os.path.join(models_path, ycb_id, "textured_simple.obj")
    return trimesh.load(mesh_path)


def create_rectange_from_bbox(mesh, x_offset, y_offset):
    """
    Input:
        mesh: trimesh mesh object
        x_offset: x coordinate of com
        y_offset: y coordinate of com
    Returns:
        rect: Rectangle as a shapely.Polygon
    """
    verts_xy = np.array(mesh.bounding_box.vertices[:, :-1])
    verts_xy += np.array([x_offset, y_offset])
    xy_locs = set((v[0], v[1]) for v in verts_xy)
    return Polygon([locn for locn in xy_locs])


def check_collision(curr_rect, placed_rects):
    """
    Checks if the provided rectangle is collision free with all of the placed
    rectangles. Returns True if all are collision free else False
    Input:
        curr_rect: shapely.Polygon rectangle
        placed_rects: list of rectangles
    Returns: bool
    """
    for other_rect in placed_rects:
        if curr_rect.intersects(other_rect):
            return False
    return True


def is_valid(loc: tuple, bounds: tuple):
    """
    loc: tuple with location indices into the grid
    bounds: tuple with bounds for the grid in terms of the indices
    """
    bound_x, bound_y = bounds
    return loc[0] >= 0 and loc[0] < bound_x and loc[1] >= 0 and loc[1] < bound_y


def valid_neighbors(loc, bounds, seen):
    ngs = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            tmp = (loc[0] + i, loc[1] + j)
            if is_valid(tmp, bounds) and (tmp not in seen):
                ngs.append(tmp)
    return ngs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="args to visualise plots", add_help=True
    )
    parser.add_argument("-vis_2d", action="store_true")
    parser.add_argument("-vis_3d", action="store_true")

    args = parser.parse_args()

    vertices = generate_grid(4, 4, [1, 0, 0], 1)

    if args.vis_2d:
        visualise2d(vertices, [1, 0, 0], 1)
    if args.vis_3d:
        visualise3d(vertices, [1, 0, 0], 1)
