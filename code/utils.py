"""
Utility methods for a synchronous distributed optimizatio in PyTorch.
"""

import fcntl

import os

import socket

import struct

import sys

import torch
import torch.autograd
import torch.distributed as D
import torch.nn
import torch.utils
import torch.utils.data

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms



def allocate_buffer(model):
    """Returns a list of cloned tensors."""
    return [p.data.clone() for p in model.parameters()]


def allocate_rank_tensor():
    """Returns a tensor which holds the rank of the current process."""
    return torch.FloatTensor([float(rank())])


def batch_size(default=64):
    """
    Returns the requested batch size.

    The default batch-size is set to 64. This can be modified through the method
    argument, a CMD argument `--batch-size` or through the environment.
    """
    flag = "--batch-size"
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    environment_key = "BATCH_SIZE"
    if environment_key in os.environ:
        return int(os.environ[environment_key])

    return default


def broadcast(model):
    """
    Broadcasts the model to all workers.

    Using D.broadcast causes crashes sometimes. That's why we use this
    naive implementation.
    """
    num_processes = world_size()
    for rank in range(1, num_processes):
        for p in model.parameters():
            D.send(p.data, rank)


def epochs(default=1):
    """Returns the number of data epochs to train on."""
    flag = "--epochs"
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    environment_key = "EPOCHS"
    if environment_key in os.environ:
        return int(os.environ[environment_key])

    return default


def initialize_environment_variables(**kwargs):
    for k in kwargs.keys():
        os.environ[k] = kwargs[k]


def initialize_process_group(backend="tcp"):
    """Initializes PyTorch's distributed backend with the specified backend."""
    D.init_process_group(
        backend=backend,
        init_method=backend + "://" + master_address(),
        rank=rank(),
        world_size=world_size()
    )


def ip_address(ifname):
    """Fetches the IP address assigned to the specified interface name."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,
        struct.pack("256s", str.encode(ifname[:15]))
    )[20:24])


def is_parameter_server():
    return rank() == 0


def is_worker():
    return rank() != 0


def learning_rate(default=0.01):
    flag = "--learning-rate"
    if flag in sys.argv:
        return float(sys.argv[sys.argv.index(flag) + 1])
    environment_key = "LEARNING_RATE"
    if environment_key in os.environ:
        return float(os.environ[environment_key])

    return default


def master_address(default="127.0.0.1:5000"):
    flag = "--master"
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    environment_key = "MASTER"
    if environment_key in os.environ:
        return os.environ[environment_key]

    return default


def momentum(default=0.9):
    flag = "--momentum"
    if flag in sys.argv:
        return float(sys.argv[sys.argv.index(flag) + 1])
    environment_key = "MOMENTUM"
    if environment_key in os.environ:
        return float(os.environ[environment_key])

    return default


def network_interface(default="lo"):
    flag = "--ifname"
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    environment_key = "IFNAME"
    if environment_key in os.environ:
        return os.environ[environment_key]

    return default


def rank():
    flag = "--rank"
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    environment_key = "RANK"
    if environment_key not in os.environ:
        raise ValueError("No rank specified through the environment or by program argument.")

    return int(os.environ[environment_key])


def store_model(model, name="model", extension=".pth"):
    torch.save(model.state_dict(), name + extension)


def world_size():
    flag = "--world-size"
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    environment_key = "WORLD_SIZE"
    if environment_key not in os.environ:
        raise ValueError("No world size specified through the environment or by program argument.")

    return int(os.environ[environment_key])


def zero_gradients(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.zero_()


def zero_tensors(tensors):
    for t in tensors:
        t.zero_()


def kappa(default=1.):
    flag = "--kappa"
    if flag in sys.argv:
        return float(sys.argv[sys.argv.index(flag) + 1])
    environment_key = "KAPPA"
    if environment_key in os.environ:
        return float(os.environ[environment_key])

    return default
