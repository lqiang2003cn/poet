
from poet_distributed.niches.box2d.env import make_env
import json
from poet_distributed.niches.box2d.bipedal_walker_custom import Env_config

def loadEnvFromJson(jsonFilePath):
    with open(jsonFilePath) as f:
        start_from_config = json.load(f)
    print(start_from_config['config'])
    config = Env_config(**start_from_config['config'])
    env = make_env('BipedalWalkerCustom', seed=start_from_config['seed'], render_mode='human', env_config=config)
    print('hold')

loadEnvFromJson('/home/qiangliu/logs/poet_final_test/poet_final_test.r0.2.p0.4_0.8.b1_0.2_0.8.env.json')