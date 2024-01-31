import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# gto planner path
gto_path = '/home/ninad/Projects/GraspTrajOpt'
add_path(gto_path)

gto_path = '/home/yuxiang/Projects/GraspTrajOpt'
add_path(gto_path)
