#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.simplefilter('default')

import os
import torch

from maddpg.core.agent_base import AgentBase
from maddpg.core.torch.algorithm import Algorithm
# from parl.utils import machine_info

__all__ = ['Agent']
torch.set_num_threads(1)


class Agent(AgentBase):
    """
    | `alias`: ``parl.Agent``
    | `alias`: ``parl.core.torch.agent.Agent``

    | Agent is one of the three basic classes of PARL.

    | It is responsible for interacting with the environment and collecting data for training the policy.
    | To implement a customized ``Agent``, users can:

      .. code-block:: python

        import parl

        class MyAgent(parl.Agent):
            def __init__(self, algorithm, act_dim):
                super(MyAgent, self).__init__(algorithm)
                self.act_dim = act_dim

    Attributes:
        device (torch.device): select GPU/CPU to be used.
        alg (parl.Algorithm): algorithm of this agent.

    Public Functions:
        - ``sample``: return a noisy action to perform exploration according to the policy.
        - ``predict``: return an estimate Q function given current observation.
        - ``learn``: update the parameters of self.alg.
        - ``save``: save parameters of the ``agent`` to a given path.
        - ``restore``: restore previous saved parameters from a given path.
        - ``train``: set the agent in training mode.
        - ``eval``: set the agent in evaluation mode.

    Todo:
        - allow users to get parameters of a specified model by specifying the model's name in ``get_weights()``.
    """

    def __init__(self, algorithm):
        """.

        Args:
            algorithm (parl.Algorithm): an instance of `parl.Algorithm`. This algorithm is then passed to `self.alg`.
            device (torch.device): specify which GPU/CPU to be used.
        """

        assert isinstance(algorithm, Algorithm)
        super(Agent, self).__init__(algorithm)
        # agent mode (bool): True is in training mode, False is in evaluation mode.
        self.training = True

    def learn(self, *args, **kwargs):
        """The training interface for ``Agent``.

        It is often used in the training stage.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """Predict an estimated Q value when given the observation of the environment.

        It is often used in the evaluation stage.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """Return an action with noise when given the observation of the environment.

        In general, this function is used in train process as noise is added to the action to preform exploration.

        """
        raise NotImplementedError

    def save(self, save_path, model=None):
        """Save parameters.

        Args:
            save_path(str): where to save the parameters.
            model(parl.Model): model that describes the neural network structure. If None, will use self.alg.model.

        Raises:
            ValueError: if model is None and self.alg.model does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model.ckpt')

        """
        if model is None:
            model = self.alg.model
        sep = os.sep
        dirname = sep.join(save_path.split(sep)[:-1])
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(model.state_dict(), save_path)

    def restore(self, save_path, model=None, map_location=None):
        """Restore previously saved parameters.
        This method requires a model that describes the network structure.
        The save_path argument is typically a value previously passed to ``save()``.

        Args:
            save_path(str): path where parameters were previously saved.
            model(parl.Model): model that describes the neural network structure. If None, will use self.alg.model.
            map_location: a function, torch.device, string or a dict specifying how to remap storage locations

        Raises:
            ValueError: if model is None and self.alg does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model.ckpt')
            agent.restore('./model.ckpt')
            
            agent.restore('./model.ckpt', map_location=torch.device('cpu')) # load gpu-trained model in cpu machine
        """

        if model is None:
            model = self.alg.model
        checkpoint = torch.load(save_path, map_location=map_location)
        model.load_state_dict(checkpoint)

    def train(self):
        """Sets the agent in training mode, which is the default setting.
        Model of agent will be affected if it has some modules (e.g. Dropout, BatchNorm) that behave differently in train/evaluation mode.

        Example:

        .. code-block:: python

            agent.train()   # default setting
            assert (agent.training is True)
            agent.eval()
            assert (agent.training is False)

        """
        self.alg.model.train()
        self.training = True

    def eval(self):
        """Sets the agent in evaluation mode.
        """
        self.alg.model.eval()
        self.training = False
