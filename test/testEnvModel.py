from poet_distributed.niches.box2d.env_model import EnvModel
from poet_distributed.niches.box2d.env_model import env_model_config
import numpy as np

def test_env_model():
    env_model = EnvModel(env_model_config)
    #theta = np.ones(2804)
    theta = np.random.uniform(0, 1, 2804)
    env_param = np.random.uniform(0,1,env_model.param_count)
    #env_param = np.ones(env_model.param_count)
    env_model.set_model_params(env_param)
    print('theta: ',theta)
    print('model output: ',env_model.feed_forward(theta))

test_env_model()
