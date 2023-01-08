#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_DIR = os.path.dirname(SCRIPT_DIR)
CONTAINER_NAME = "stellar_mesh_integration"
DEFAULT_DOCKER_IMAGE = "stellar_mesh_integration/docker/x86_64/linux/stellar_mesh_integration"
DEFAULT_DOCKER_IMAGE_VERSION = "1.0"


def process_input_parameters():
    program_description = """ Creates a docker container by executing docker run and mounting a /repo volume 
    nside the container."""
    image_description = "Specify an image to use instead of the default."
    path_description = "Specify a dir path to mount under /repo inside the container. By default it is one level below of this script"
    nvidia_description = "Run container with GPU access"

    parser = argparse.ArgumentParser(description=program_description)

    parser.add_argument(
        "-c", "--cpu", help=nvidia_description, action="store_false")
    parser.add_argument(
        "-i", "--image", help=image_description)
    parser.add_argument(
        "-p", "--path", type=dir_path, help=path_description, default=DEFAULT_REPO_DIR)
    args = parser.parse_args()

    return args


def dir_path(path):
    abs_path = os.path.abspath(path)
    if os.path.isdir(abs_path):
        return abs_path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{abs_path} is not a valid path")


def get_option_args():
    return get_nvidia_dependent_options() + get_system_dependent_options() + get_common_options() + get_docker_image()


def get_nvidia_dependent_options():
    if parameters.cpu:
        return []
    return [ "--gpus", "all", "--runtime=nvidia"]


def get_system_dependent_options():
    if sys.platform == "linux":
        return ["--volume", "/tmp/.X11-unix:/tmp/.X11-unix",
                "--volume", os.path.join(os.environ['HOME'],
                                         ".ssh") + ":/home/developer/.ssh",
                "--env", "DISPLAY=" + os.environ['DISPLAY']]

    elif sys.platform == "win32":
        return ["--volume", os.path.join(os.environ['HOMEDRIVE'],
                                         os.environ['HOMEPATH'], ".ssh") + ":/home/developer/.ssh",
                "--env", "DISPLAY=host.docker.internal:0"]
    return []


def get_common_options():
    return ["--rm",
            "--volume", parameters.path + ":/home/developer/workdir",
            "--privileged",
            # "--net=host",
            "--interactive",
            "--cap-add", "sys_ptrace",
            "--tty",
            "-p", "8888:8888",
            "-p", "6006:6006",
            "--name", CONTAINER_NAME,
            "-it"]


def get_docker_image():
    if parameters.image:
        return [parameters.image]
    return [DEFAULT_DOCKER_IMAGE + ":" + DEFAULT_DOCKER_IMAGE_VERSION]


def enable_gui_apps():
    if sys.platform == "linux":
        subprocess.run(["xhost", "+local:docker"])


if __name__ == '__main__':
    parameters = process_input_parameters()
    enable_gui_apps()
    subprocess.run(["docker", "container", "run"] + get_option_args())
