## Team Stars - MRSD 2020

This system is a way to test autonomous vehicle software with the other agents running a learned model. These learned agents provide the ego vehicle with a realistic testing environment and is better that rule based autopilots in many ways(such as realistic reactions and better generalization to newer maps and scenarios).

The learned agents run an imitation Learning based model based on [Learning By Cheating](https://arxiv.org/abs/1912.12294) (By Dian Chen, Brady Zhou, Vladlen Koltun, Philipp Krähenbühl). The original paper is used to control a single ego vehicle in the simulator and it drives in specific scenarios. Our work incorporates this model into the non-ego vehicles in CARLA and gives them dynamic and random goals, such that these agents are continously driving in the simulator. This kind of setup is good for general testing of autonomous vehicle software. It is also possible to extend our software to create multi agent dynamic scenarios and use them for more complex, scenario-specific tests. 

More details on the Team STARS' MRSD 2020 Project website - https://mrsdprojects.ri.cmu.edu/2020teamh/
