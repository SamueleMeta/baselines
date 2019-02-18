'''
    Testing the TRPO algorithm from RLLab. To be integrated with
    POIS baselines to be run also on sacred.
'''
import rllab

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

HORIZON = 500
BATCH_SIZE = 100
N_ITR = 500
GAMMA = 0.99
STEP_SIZE = 0.1

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=[]
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=BATCH_SIZE,
    whole_paths=True,
    max_path_length=HORIZON,
    n_itr=N_ITR,
    discount=GAMMA,
    step_size=STEP_SIZE,
)
algo.train()
