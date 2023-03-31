import os


def create_dir(dir_path):
    """ Helper function to create a directory in given path """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)