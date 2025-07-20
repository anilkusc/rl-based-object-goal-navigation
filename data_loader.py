import os

def scene_loader(root_path):
    fl_list = get_folder_list(root_path)
    files = []
    for fl in fl_list:
        fp = root_path+fl+"/"+fl.split("-")[1]
        files.append(
            {"glb":fp+".glb","basis.glb":fp+".basis.glb","navmesh":fp+".basis.navmesh"}
        )
    return files

def get_folder_list(path):
    folder_list = []
    for _, dirs, _ in os.walk(path):
        for directory in dirs:
            folder_list.append(directory)
    return folder_list