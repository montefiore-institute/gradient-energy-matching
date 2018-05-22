"""
Gradient Energy Matching
"""

import os

import socket

import sys

import torch

import torch.autograd
import torch.distributed as D
import torch.nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import utils as u



class Model(torch.nn.Module):
    """
    Adopted from the PyTorch examples.

    Increased the dimensionality of the fully connected layers to
    make the computations more "expensive".
    """

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 300)
        self.fc2 = torch.nn.Linear(300, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


def main():
    """Main entry point of the optimization procedure."""
    download_dataset()
    # Initializes PyTorch's distributed backend.
    u.initialize_process_group()
    if u.is_parameter_server():
        process = ParameterServer()
    else:
        process = Worker()
    process.run()
    # Clean exit by waiting for all involved processes.
    process.cleanup()
    del process

def download_dataset(worker_download=False):
    """
    Downloads the MNIST dataset.

    By default, only the parameter server will download the dataset unless
    specified otherwise. This is especially helpfull when running GEM on
    a single machine without causing conflicts.
    """
    if u.is_parameter_server() or worker_download:
        # This can probably be done in a cleaner way...
        dataset = datasets.MNIST("../data", train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        # Cleanup the memory.
        del dataset


def allocate_model():
    """Allocate your PyTorch model here."""
    return Model()


def allocate_criterion():
    """Allocates the desired criterion to train your model for."""
    return torch.nn.CrossEntropyLoss()


def allocate_data_loader(num_workers=1, train=True):
    """Allocates the data loader."""
    # Allocate the dataset object with the appropriate transformations.
    dataset = datasets.MNIST("../data", train=train, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    # Allocates the batch loader.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=u.batch_size(),
        shuffle=True,
        num_workers=num_workers
    )

    return data_loader


def total_training_batches():
    """Yields the total number of batches in the training dataset."""
    data_loader = allocate_data_loader(train=True)
    num_samples = len(data_loader.dataset)
    num_batches = int(num_samples / u.batch_size())
    del data_loader # Cleanup the memory.

    return num_batches


class OptimizationProcess(object):

    def __init__(self, rank):
        self._rank = rank
        self._model = allocate_model()
        self._epochs = u.epochs()
        self._num_workers = u.world_size() - 1
        self._step = 0

    def num_steps(self):
        return self._step

    def next_step(self):
        self._step += 1

    def allocate_buffer(self):
        return u.allocate_buffer(self._model)

    def run(self):
        raise NotImplementedError

    def cleanup(self):
        """Waits for all involved processes before exiting."""
        D.barrier()


class ParameterServer(OptimizationProcess):

    def __init__(self):
        super(ParameterServer, self).__init__(u.rank())
        self._buffer = self.allocate_buffer()
        self._update_budget = self._epochs * self._num_workers * total_training_batches()
        self._sock = None

    def initialize(self):
        # Broadcast the model to the workers.
        u.broadcast(self._model)
        # Start the UDP server for the queueing mechanism.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (u.ip_address(u.network_interface()), 8000) # Fixed port.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(server_address)

    def next_in_queue(self):
        data, address = self._sock.recvfrom(10)
        data = data.decode().split('.')[0]
        rank = int(data)

        return rank

    def run(self):
        self.initialize()
        rank_tensor = u.allocate_rank_tensor()
        # Wait for all processes to complete the initialization.
        D.barrier()
        for update in range(self._update_budget):
            worker_rank = self.next_in_queue()
            D.send(rank_tensor, dst=worker_rank)
            # Send the central variable to the worker.
            for p in self._model.parameters():
                D.send(p.data, dst=worker_rank)
            # Receive the update into the local buffer.
            for p in self._buffer:
                D.recv(p, src=worker_rank)
            # Incorporate the update into the central variable.
            for parameter_index, p in enumerate(self._model.parameters()):
                p.data.add_(self._buffer[parameter_index])

    def cleanup(self):
        self._sock.close()
        super(ParameterServer, self).cleanup()
        self.validate_model()

    def validate_model(self):
        self._model.eval()
        data_loader = allocate_data_loader(train=False)
        # Start the validation process.
        validation_loss = 0
        correct = 0
        for x, y in data_loader:
            x, y = torch.autograd.Variable(x), torch.autograd.Variable(y)
            output = self._model(x)
            validation_loss += F.nll_loss(output, y, size_average=False).item()
            prediction = output.data.max(1)[1]
            correct += prediction.eq(y.data).sum().item()
        # Compute the statistics.
        num_samples = len(data_loader.dataset)
        validation_loss /= num_samples
        validation_accuracy = correct / num_samples
        # Show the statistics.
        print("Validation loss: " + str(validation_loss))
        print("Validation accuracy: " + str(validation_accuracy))


class Worker(OptimizationProcess):

    def __init__(self):
        super(Worker, self).__init__(u.rank())
        self._learning_rate = u.learning_rate()
        self._momentum = u.momentum()
        self._criterion = allocate_criterion()
        self._data_loader = allocate_data_loader(train=True)
        self._num_batches = total_training_batches()
        self._epsilon = 10e-13
        self._moment = self.allocate_buffer()
        self._central_variable_moment = self.allocate_buffer()
        self._stale = self.allocate_buffer()
        self._kappa = u.kappa()
        self._sock = None
        self._master_address = None
        # Bytes representation of the rank for the UDP socket.
        self._rank_tensor = u.allocate_rank_tensor()
        self._rank_bytes = str(self._rank).zfill(10).encode()

    def initialize(self):
        # Clear the buffers.
        u.zero_tensors(self._moment)
        u.zero_tensors(self._central_variable_moment)
        u.zero_tensors(self._stale)
        # Initialize a UDP socket (fixes PyTorch recv bug) for the queueing mechanism.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._master_address = (u.master_address().split(':')[0], 8000) # Fixed port.

    def queue_parameter_server(self):
        # We are ready to send an update, notify parameter server.
        self._sock.sendto(self._rank_bytes, self._master_address)
        # Receive an acknowledgement of the parameter server.
        D.recv(self._rank_tensor, src=0)

    def pull(self):
        for parameter_index, p in enumerate(self._model.parameters()):
            D.recv(p.data, src=0)
            self._central_variable_moment[parameter_index] = (p.data - self._stale[parameter_index])
            self._stale[parameter_index].copy_(p.data)

    def commit(self):
        for p in self._model.parameters():
            D.send(p.grad.data, dst=0)

    def gradient_energy_matching(self):
        Pi = [] # Pi tensors for all parameters.

        for parameter_index, p in enumerate(self._model.parameters()):
            proxy = self._kappa * self._moment[parameter_index].abs()
            central_variable = self._central_variable_moment[parameter_index].abs()
            update = (p.grad.data + self._epsilon).abs()
            pi = (proxy - central_variable) / update
            pi.clamp_(min=0., max=5.) # For numerical stability.
            Pi.append(pi)

        return Pi

    def step(self):
        # Update the states with the current gradient.
        for parameter_index, p in enumerate(self._model.parameters()):
            p.grad.data *= -self._learning_rate
            self._moment[parameter_index] *= self._momentum
            self._moment[parameter_index] += p.grad.data
        # Receive the new central variable.
        self.queue_parameter_server()
        self.pull()
        # Compute GEM's pi scalars.
        pi = self.gradient_energy_matching()
        # Apply the scalars to the update.
        for parameter_index, p in enumerate(self._model.parameters()):
            p.grad.data.mul_(pi[parameter_index])
        # Send the update to the parameter server.
        self.commit()
        # Add the update to the parameters.
        for p in self._model.parameters():
            p.data.add_(p.grad.data)
        # Update the current tick.
        self.next_step()

    def train(self):
        data_iterator = iter(self._data_loader)
        for batch_index in range(self._num_batches):
            # Fetch the next mini-batch.
            try:
                x, y = next(data_iterator)
                x = torch.autograd.Variable(x)
                y = torch.autograd.Variable(y)
            except Exception as e:
                # Issue with underlaying files or file-system.
                print(e)
                continue
            # Zero the old gradients of the model, and compute the new ones.
            u.zero_gradients(self._model)
            output = self._model(x)
            loss = self._criterion(output, y)
            loss.backward()
            # Apply a step of the optimizer.
            self.step()
        del data_iterator

    def run(self):
        # Initialize the worker state.
        self.initialize()
        # Synchronize the parameterization with the parameter server.
        self.pull()
        # Wait for all workers to be initialized.
        D.barrier()
        # Start the optimization procedure.
        for epoch in range(self._epochs):
            self._model.train() # Set model in training mode.
            self.train()

    def cleanup(self):
        self._sock.close()
        super(Worker, self).cleanup()


if __name__ == "__main__":
    main()
