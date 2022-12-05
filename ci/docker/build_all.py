#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTRMEDIATE_LAYERS_DIR = os.path.join(SCRIPT_DIR, "intermediate_images")
TF_BASE_IMAGE = "tensorflow/tensorflow:2.7.0-gpu-jupyter"
IMAGE_NAME_PREFIX = "stellar_mesh_integration/docker/x86_64/linux/"
DEFAULT_USER_UID = 1000
DEFAULT_USER_GID = 1000

# List here the successive layers of the docker image in the build order.


def get_layers():
    layers = [IntermediateLayer(image_name="base_python", version="1.0", file_name="base_image_python.dockerfile"),
              FinalLayer(image_name="stellar_mesh_integration", version="1.0", file_name="stellar_mesh_integration.dockerfile", uid=str(parameters.uid), gid=str(parameters.gid))]

    return layers


def get_base_image():
    return TF_BASE_IMAGE


def process_input_parameters():
    program_description = "Build docker container for ArielML 2021 Data Challenge."
    uid_description = "Set UID number for developer user"
    gid_description = "Set GID number for developer user"

    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument("-u", "--uid", help=uid_description,
                        type=int, default=get_uid())
    parser.add_argument("-g", "--gid", help=gid_description,
                        type=int, default=get_gid())
    args = parser.parse_args()

    return args


def get_uid():
    if sys.platform == "linux":
        return os.getuid()
    return DEFAULT_USER_UID


def get_gid():
    if sys.platform == "linux":
        return os.getgid()
    return DEFAULT_USER_GID


class IntermediateLayer:
    def __init__(self, image_name, version, file_name, path_to_file=INTRMEDIATE_LAYERS_DIR):

        self.image_name = image_name
        self.version = version
        self.file_name = file_name
        self.path_to_file = path_to_file
        self.image_prefix = IMAGE_NAME_PREFIX

    def get_tagged_image_name(self):
        return self.image_prefix + self.image_name + ":" + self.version

    def get_image_name_options(self):
        return ["-t", self.get_tagged_image_name()]

    def get_source_file_options(self):
        return ["-f", os.path.join(self.path_to_file, self.file_name), self.path_to_file]

    def get_build_options(self):
        return self.get_image_name_options() + self.get_source_file_options()


class FinalLayer(IntermediateLayer):
    def __init__(self, image_name, version, file_name, uid=str(DEFAULT_USER_UID), gid=str(DEFAULT_USER_GID), path_to_file=SCRIPT_DIR):
        self.uid = uid
        self.gid = gid
        super().__init__(image_name, version, file_name, path_to_file)

    def get_uid_options(self):
        return ["--build-arg", "UID=" + self.uid]

    def get_gid_options(self):
        return ["--build-arg", "GID=" + self.gid]

    def get_build_options(self):
        return super().get_build_options() + self.get_uid_options() + self.get_gid_options()


class DockerBuilder:
    def __init__(self, base_image=TF_BASE_IMAGE):
        self.base_image = base_image

    def get_base_image(self):
        return ["--build-arg", "BASE_IMAGE=" + self.base_image]

    def build_images(self, layers):
        for layer in layers:
            build = subprocess.run(
                ["docker", "build"] + self.get_base_image() + layer.get_build_options())
            self.base_image = layer.get_tagged_image_name()
            if build.returncode != 0:
                sys.exit(1)


if __name__ == '__main__':
    parameters = process_input_parameters()
    DockerBuilder(get_base_image()).build_images(get_layers())
