import argparse
import os
import open3d as o3d
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Compute inertial for SDF files',
    usage="\
    python modify_gazebo_sdf.py -f /PATH/TO/root_DATA_FOLDER/")

parser.add_argument('-f', '--root_dir', type=str, default='/mnt/Data/rootScannedObjects/SampleObjects/', help='Path to rootScan3d Models Directory.')


def main(args):

    if not args.root_dir:
        print(f"models list not specified")
        exit(0)

    if not os.path.isdir(args.root_dir):
        print(f"models directory (containing meshes): {args.root_dir} is incorrect")
        exit(0)

    model_names = ['003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', \
                '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '037_scissors', '040_large_marker', \
                '052_extra_large_clamp']
    masses = [0.41, 0.48, 0.348, 0.612, 0.176, 0.184, 0.096, 0.366, 0.062, 1.1, 0.128, 0.104, 0.612, 0.084, 0.014, 0.12]  

    for i, model in enumerate(model_names):
        mass = masses[i]
        mesh_name = 'textured_simple.obj'
        print(mass)

        # read obj file to compute inertia
        filename = os.path.join(args.root_dir, model, mesh_name)
        mesh = o3d.io.read_triangle_mesh(filename)
        w, h, l = mesh.get_axis_aligned_bounding_box().get_extent()
        ixx = mass * (w*w + h*h) / 12
        iyy = mass * (l*l + h*h) / 12
        izz = mass * (l*l + w*w) / 12

        # read sdf file
        filename = os.path.join(args.root_dir, model, 'model.sdf')
        tree = ET.parse(filename)
        root = tree.getroot()
        mass_node = root.find('./model/link/inertial/mass')
        mass_node.text = str(mass)

        node = root.find('./model/link/inertial/inertia/ixx')
        node.text = str(ixx)

        node = root.find('./model/link/inertial/inertia/iyy')
        node.text = str(iyy)

        node = root.find('./model/link/inertial/inertia/izz')
        node.text = str(izz)
        
        print(filename)
        tree.write(filename)


if __name__ == '__main__':
    main(parser.parse_args())