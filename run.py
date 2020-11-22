from tqdm import tqdm

from carla_env import CarlaEnv

SYNCMODE = True
NUM_PLAYERS = 2

if __name__ == "__main__":
    print("Run.py")
    if not SYNCMODE:
        env = CarlaEnv(sync=SYNCMODE, num_players=NUM_PLAYERS)
        print("Env activated")
        env.reset()
        for i in tqdm(range(10000)):
            # print("IN Step ", i)
            env.step()
            # time.sleep(0.2)
    else:
        with CarlaEnv(sync=SYNCMODE, num_players=NUM_PLAYERS) as env:
            print("Env activated")
            env.reset()
            for i in tqdm(range(10000)):
                # print("IN Step ", i)
                env.step()
