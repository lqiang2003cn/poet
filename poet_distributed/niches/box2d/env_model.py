# The following code is modified from hardmaru/estool (https://github.com/hardmaru/estool/) under the MIT License.

# Modifications Copyright (c) 2019 Uber Technologies, Inc.
from collections import namedtuple

import numpy as np
import logging

from bipedal_walker_custom import Env_config

logger = logging.getLogger(__name__)


EnvModelConfig = namedtuple('EnvModelConfig', ['env_id','input_size','output_size','layers'])

env_model_config = EnvModelConfig(
    env_id='1',
    input_size=2804,
    output_size=1,
    layers=[100, 100]
)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)

class EnvModel:
    ''' the model for the env generator'''

    def __init__(self, envModelConfig):
        self.env_id = envModelConfig.env_id
        self.layer_1 = envModelConfig.layers[0]
        self.layer_2 = envModelConfig.layers[1]
        self.input_size = envModelConfig.input_size
        self.output_size = envModelConfig.output_size
        #shapes of the env network
        self.shapes = [(self.input_size, self.layer_1),
                       (self.layer_1, self.layer_2),
                       (self.layer_2, self.output_size)]

        self.activations = [relu,relu,relu]#three layers;
        self.weight = []
        self.bias = []
        self.param_count = 0

        idx = 0
        for shape in self.shapes:
            #or start with random weights and bias
            # rand_weight = np.random.uniform(0,1,shape)
            # rand_bias = np.random.uniform(0,1,shape[1])
            # self.weight.append(rand_weight)
            # self.bias.append(rand_bias)

            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))

            # self.weight.append(np.ones(shape=shape))
            # self.bias.append(np.ones(shape=shape[1]))

            self.param_count += (np.product(shape) + shape[1])
            idx += 1


    def feed_forward(self, agent_theta):
        # if mean_mode = True, ignore sampling.
        h = np.array(agent_theta).flatten()
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            #print('h for the layer', i, 'before activation is :', h)
            h = self.activations[i](h)
            #print('h for the layer',i ,'is :',h)
        h=(h/2805010.1)#to make h a number between [0,10]
        print('model output:',h)
        return h

    # model_params is theta, a flattened array of weights and bias,need to rewrite
    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s


def get_env_neural_output(theta,env_param,env_name):
    env_model = EnvModel(env_model_config)
    env_model.set_model_params(env_param)
    ground_roughness = env_model.feed_forward(theta)
    env = Env_config(
        name=env_name,  # the default name of the env
        ground_roughness=ground_roughness[0],
        pit_gap=[],
        stump_width=[],
        stump_height=[],
        stump_float=[],
        stair_height=[],
        stair_width=[],
        stair_steps=[])
    return env

    #
    # #you may need to load a env model from a json file?
    # def load_model(self, filename):
    #     with open(filename) as f:
    #         data = json.load(f)
    #     print('loading file %s' % (filename))
    #     self.data = data
    #     model_params = np.array(data[0])  # assuming other stuff is in data
    #     self.set_model_params(model_params)
    #
    # def saveModel(self,filename):
    #     with open(filename,'w') as f:
    #         json.dump(record, f)
    #
    # def get_random_model_params(self, stdev=0.1):
    #     return np.random.randn(self.param_count) * stdev
