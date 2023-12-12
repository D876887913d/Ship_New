'''
Author: gongweijing 876887913@qq.com
Date: 2023-12-08 22:28:46
LastEditors: gongweijing 876887913@qq.com
LastEditTime: 2023-12-11 22:12:06
FilePath: /root/Ship_New/maddpg/core/torch/algorithm.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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

from maddpg.core.algorithm_base import AlgorithmBase
from maddpg.core.torch.model import Model

__all__ = ['Algorithm']


class Algorithm(AlgorithmBase):
    """
    | `alias`: ``parl.Algorithm``
    | `alias`: ``parl.core.torch.algorithm.Algorithm``

    | ``Algorithm`` defines the way how to update the parameters of the
    ``Model``. This is where we define loss functions and the optimizer of the
    neural network. An ``Algorithm`` has at least a model.

    | PARL has implemented various algorithms(DQN/DDPG/PPO/A3C/IMPALA) that can
    be reused quickly, which can be accessed with ``parl.algorithms``.

    Example:

    .. code-block:: python

        import parl

        model = Model()
        dqn = parl.algorithms.DQN(model, lr=1e-3)

    Attributes:
        model(``parl.Model``): a neural network that represents a policy or a
        Q-value function.

    Pulic Functions:
        - ``predict``: return an estimate q value given current observation.
        - ``learn``: define the loss function and create an optimizer to
        minimize the loss.

    """

    def __init__(self, model=None):
        """
        Args:
            model(``parl.Model``): a neural network that represents a policy or
            a Q-value function.
        """
        assert isinstance(model, Model)
        self.model = model

    def get_weights(self):
        """ Get weights of self.model.

        Returns:
            weights (dict): a Python dict containing the parameters of
            self.model.
        """
        return self.model.get_weights()

    def set_weights(self, params):
        """ Set weights from ``get_weights`` to the model.

        Args:
            weights (dict): a Python dict containing the parameters of
            self.model.
        """
        self.model.set_weights(params)

    def learn(self, *args, **kwargs):
        """ Define the loss function and create an optimizer to minimize the loss.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """ Refine the predicting process, e.g,. use the policy model to
        predict actions.
        """
        raise NotImplementedError
