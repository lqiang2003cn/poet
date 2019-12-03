from poet_distributed.niches.box2d.env import make_env
from poet_distributed.niches.box2d.bipedal_walker_custom import Env_config
import numpy as np

def run_test():
    my_Env_Config = Env_config(
        name='default_env',
        ground_roughness=10,
        pit_gap=[10,10],
        stump_width=[],
        stump_height=[5,5],
        stump_float=[],
        stair_height=[5,5],
        stair_width=[],
        stair_steps=[9])
    seed = np.random.randint(1000000000,9000000000)
    env = make_env('BipedalWalkerCustom', seed=seed,render_mode='human', env_config=my_Env_Config)
    # base_dir = "/home/qiangliu/ai/recordedVideos/test"
    # t = time.time()
    # dyn_dir = base_dir + str(int(t))
    # env = gym.wrappers.Monitor(env, directory=dyn_dir)
    print('hold');

run_test()