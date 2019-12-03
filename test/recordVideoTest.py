import gym
import uuid
import time
import datetime

import time

from numpy import long


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    uuid  = uuid.uuid1()
    base_dir = "/home/qiangliu/ai/recordedVideos/test"
    t = time.time()
    dyn_dir = base_dir+str(int(t))
    env = gym.wrappers.Monitor(env, directory=dyn_dir)

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        #随机地取一个动作，没有任何智能
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
    env.close()