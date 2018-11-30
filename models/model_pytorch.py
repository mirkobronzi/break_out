import logging
import os

import torch
from torch import nn


logger = logging.getLogger(__name__)


class ModelImpl(nn.Module):

    def __init__(self, n_actions, x, y):
        super(ModelImpl, self).__init__()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.conv1 = nn.Conv2d(n_actions, 16, 8, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(32 * 23 * 17, 256)
        self.fc2 = nn.Linear(256, 4)

        self.fc1_test = nn.Linear(105 * 80 * n_actions, 256)
        self.fc2_test = nn.Linear(256, 4)

    def forward(self, frames, actions):
        batch_size = frames.shape[0]
        torch_input = torch.from_numpy(frames).cuda().float()
        after_conv1 = self.conv1(torch_input)
        after_conv2 = self.conv2(after_conv1)
        flat = after_conv2.view(batch_size, -1)
        hidden = torch.relu(self.fc1(flat))
        output = torch.sigmoid(self.fc2(hidden))
        filtered_output = output * torch.from_numpy(actions).cuda().float()

        # flat = torch_input.view(batch_size, -1)
        # hidden = torch.relu(self.fc1_test(flat))
        # filtered_output = torch.sigmoid(self.fc2_test(hidden))
        return filtered_output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ModelPyTorch:

    def __init__(self, n_actions, x_shape, y_shape, batch_size,
                 model_path=None):
        self.n_actions = n_actions
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.model_impl = ModelImpl(n_actions, x_shape, y_shape)
        self.optimizer = torch.optim.Adam(self.model_impl.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        if model_path is None or not os.path.isfile(model_path):
            logger.info('initializing new models')
        else:
            logger.info('loading new models from {}'.format(model_path))
            self.model_impl.load_state_dict(torch.load(model_path))

    def predict(self, frames, actions, return_numpy=True):
        result = self.model_impl.forward(frames, actions)
        if return_numpy:
            return result.cpu().detach().numpy()
        else:
            return result

    def fit(self, frames, actions, target):
        self.optimizer.zero_grad()
        predictions = self.predict(frames, actions, return_numpy=False)
        loss = self.loss_fn(predictions, torch.from_numpy(target).cuda().float())
        loss.backward()
        self.optimizer.step()

    def evaluate(self, frames, actions, target):
        predictions = self.predict(frames, actions, return_numpy=False)
        loss = self.loss_fn(predictions, torch.from_numpy(target).float())
        return loss

    def save(self, path):
        logger.info('saving models to {}'.format(path))
        torch.save(self.model_impl.state_dict(), path)
