from tqdm import tqdm

from carla_env import CarlaEnv

if __name__ == "__main__":
    print("Run.py")
    env = CarlaEnv()
    print("Env activated")
    env.reset()
    for i in tqdm(range(10000)):
        # print("IN Step ", i)
        env.step()
        # time.sleep(0.2)
