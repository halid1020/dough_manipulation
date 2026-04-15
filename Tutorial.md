# Training a SAC Policy on DiffSkill Environments Using Actoris Harena

**Prerequisites:** Before starting this tutorial, please ensure you have followed the instructions in the `README.md` of the current directory to set up your environment.
>
**Note:** The exact code and configuration files shown in this tutorial are already provided within the repository. The purpose of this guide is to walk you through the methodological pipeline so you understand the framework's mechanics and can replicate these steps to integrate other RL policies on new environments.

In this tutorial, you will learn how to train a Soft Actor-Critic (`SAC`) Reinforcement Learning (RL) policy within the `DiffSkill` environment using the Actoris Harena framework.

## 1\. Build the Evaluation Pipeline and Test with a Random Policy

Before training a complex policy, it is best practice to build an evaluation pipeline for our policy (agent) and environment (arena) pair. We will start by verifying that the environment can successfully run a simple random policy within our framework.

### Step 1a: Create a Random Agent Configuration

First, create a YAML configuration file for the random agent. Create a file named `random_policy.yaml` under the `conf/agent` directory with the following content:

```yaml
name: random
```

This tells the Actoris Harena framework to create an instance of the [`RandomPolicy`](https://github.com/halid1020/actoris_harena/blob/py3.8/actoris_harena/agent/random/random_policy.py) class. The framework expects to find this class in the `random_policy.py` file under the `actoris_harena/agent/random` directory.

**Note:** For your convenience, here is what the underlying `RandomPolicy` implementation looks like, which inherits from the base [`Agent`](https://www.google.com/search?q=%5Bhttps://github.com/halid1020/actoris_harena/blob/py3.8/actoris_harena/agent/agent.py) class:

```python
from actoris_harena import Agent

class RandomPolicy(Agent):

    def __init__(self, config):
        super().__init__(config)

    def act(self, info_list, update=False):
        actions = []
        for info in info_list:
            if 'action_space' in info:
                action_space = info['action_space']
                actions.append(action_space.sample())
            else:
                raise ValueError('action_space not found in info')
        
        return actions
    
    def single_act(self, info, update=False):
        action_space = info['action_space']
        return action_space.sample()
    
    def get_name(self):
        return 'random'
```

### Step 1b: Create the DiffSkill Environment Configuration

Next, we need to create a YAML file to initialise the DiffSkill environment. Create a file named `diffskill_LiftSpread-v1.yaml` under the `conf/arena` directory. The content should be:

```yaml
name: diffskill
task: 'LiftSpread-v1'
display: False
resolution: [84, 84]
```

This instructs the framework to initialise an instance of our environment adapter (which we will build next) using the argument `task: 'LiftSpread-v1'`.

### Step 1c: Implement the DiffSkill Arena Adapter

Now we need to create the actual adapter class that inherits from the [`Arena`](https://github.com/halid1020/actoris_harena/blob/py3.8/actoris_harena/arena/arena.py) interface in the Actoris Harena framework and wraps the `DiffSkill` content. Create this file under the `env` folder.

Here is the content for the `diffskill_arena.py` adapter:

```python
import numpy as np
import cv2
from plb.envs import make
from core.diffskill.env_spec import set_render_mode
from actoris_harena import Arena, StandardLogger

class DiffSkillArena(Arena):
    def __init__(self, config):
        super().__init__(config)
        self.task_name = config.get('task', 'LiftSpread-v1')
        self._env = make(self.task_name)
        set_render_mode(self._env, self.task_name, 'mesh')
        self.resolution = config.get('resolution', [256, 256])
        self.action_space = self._env.action_space
        print('action space', self.action_space)
        self._action_repeat = config.get('action_repeat', 1)
        self.action_horizon = getattr(self._env, '_max_episode_steps', 50)
        self.set_disp(bool(config.get('display', False)))
        self.logger = StandardLogger()
     
    def reset(self, episode_config=None):
        self._sim_step, self._total_reward, self.video_frames = 0, 0, []
        conf = episode_config or {}
        self._save_frame = conf.get('save_video', False)

        offsets = {'eval': 0, 'val': self.num_eval_trials,
                   'train': self.num_eval_trials + self.num_val_trials}

        self.eid = conf.get('eid', np.random.randint(0, self.get_num_episodes()))
        seed = offsets[self.mode] + self.eid
       
        self._env.seed(int(seed))
        self._last_obs, self._last_info = self._env.reset(), {}
        if self.disp: self._display()
        return self._format_info(done=False, term=False)

    def step(self, action):
        action = action['default'] if isinstance(action, dict) else action
        r_sum = 0
        
        for _ in range(self._action_repeat):
            self._last_obs, r, done, self._last_info = self._env.step(action)
            r_sum += r; self._sim_step += 1
            if self.disp: self._display()
            if self._save_frame: self.video_frames.append(self._get_rgb())
            if done or self._sim_step >= self.action_horizon:
                done = True; break
                
        self._total_reward += r_sum
        return self._format_info(done=done, term=done, reward=r_sum)

    def _format_info(self, done, term, reward=0):
        """Helper to DRY up the return dictionaries for reset and step."""
        obs = dict(self._last_obs) if isinstance(self._last_obs, dict) else {'state': self._last_obs}
        if 'desired_goal' in obs:
            obs['state'] = np.concatenate([obs.get('observation', []), obs.get('achieved_goal', []), obs.get('desired_goal', [])])
        obs['rgb'] = self._get_rgb()
        
        suc = bool(self._last_info.get('is_success', self._last_info.get('success', False)))
        if not suc and 'achieved_goal' in obs:
            suc = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) < 0.05

        return {'done': done, 'terminated': term, 'reward': reward, 'evaluation': {'total_reward': self._total_reward},
                'action_space': self._env.action_space, 'observation': obs, 'arena_id': self.aid, 
                'sim_steps': self._action_repeat if reward else 0, 'success': suc}

    def _get_rgb(self): 
        """Get RGB frame, scale to 0-255 uint8 format, and optionally resize."""
        img = np.clip(self._env.render(mode='rgb') * 255.0, 0, 255).astype(np.uint8)
        return cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)[:, :, :3]
    
    def _display(self):
        px = self._get_rgb()
        if px is not None:
            cv2.imshow('simulation', px[:, :, ::-1]) # Slicing [::-1] instantly converts RGB to BGR
            cv2.waitKey(1)
        else: print('[DiffSkillArena] px is None')

    def evaluate(self): return {'total_reward': self._total_reward}

    def compare(self, r1, r2):
        """Ultra-compact comparison using list comprehensions and tuple looping."""
        def stats(res):
            g = lambda r, k: r.get(k, [0])[-1] if isinstance(r.get(k), list) else r.get(k, 0)
            suc = [g(r, 'success') for r in res]
            rwd = [g(r, 'total_reward') for r in res]
            stp = [r.get('length', r.get('steps', len(r.get('success', [])) if type(r.get('success'))==list else 0)) for r, s in zip(res, suc) if s 0.5]
            return np.mean(suc or [0]), np.std(rwd or [0]), np.mean(stp or [float('inf')])
        
        (s1, r1, st1), (s2, r2, st2) = stats(r1), stats(r2)
        
        # Loop over criteria in priority order: (+ means higher is better, - means lower is better)
        for diff, eps in [(s1 - s2, 1e-4), (r2 - r1, 1e-4), (st2 - st1, 0.5)]:
            if diff eps: return 1
            if diff < -eps: return -1
        return 0
    
    def success(self):
        return bool(self._last_info.get('is_success', self._last_info.get('success', False)))
```

### Step 1d: Register the Arena

To make the Actoris Harena framework aware of our new arena, we need to register it. Open the `registration/sim_arena.py` file and add the following:

```python
import actoris_harena as athar
from env.diffskill_arena import DiffSkillArena

def register_arenas():
    athar.register_arena('diffskill', DiffSkillArena)
```

*(Note: We do not have to register the random policy because it is already registered by default in the `actoris_harena` repository).*

### Step 1e: Create the Experiment Configuration

Now, we need an experiment YAML file to link the initialization of our agent and arena, and to dictate where to save the results. The `train_and_eval: train_and_evaluate_single` key specifically points to the core API function [`ag_ar.train_and_evaluate_single`](https://github.com/halid1020/actoris_harena/blob/py3.8/actoris_harena/api.py#L445) used to run the experiment.

Create the file `conf/sim_exp/random_policy_on_diffskill_LiftSpread-v1.yaml` with the following configuration:

```yaml
# @package _global_
defaults:
  - /agent@agent: random_policy
  - /arena@arena: diffskill_LiftSpread-v1
  - /task@task: dummy # Included automatically by the environment

project_name: dough_manipulation
save_root: /media/hcv530/T7/
train_and_eval: train_and_evaluate_single
```

### Step 1f: Run the Evaluation Pipeline

Finally, we can test our evaluation pipeline by running the random policy on our newly adapted DiffSkill environment. Execute the run via Hydra using your terminal (this relies on the underlying [`ag_ar.evaluate`](https://github.com/halid1020/actoris_harena/blob/py3.8/actoris_harena/api.py#L319) API):

```bash
source ./setup.sh
python tool/hydra_eval.py --config-name sim_exp/random_policy_on_diffskill_LiftSpread-v1
```

-----

## 2\. Train SAC Policy on DiffSkill

The SAC implementation is already provided in the Actoris Harena framework under the `actoris_harena/agent/drl/sac` folder. Here, we will use the image-based version to train on the `LiftSpread-v1` environment.

First, under `conf/agent`, create a yaml file named `vanilla_image_sac.yaml` with the following content:

```yaml
name: vanilla-image-sac
device: ${oc.env:DEVICE,cuda:0}

# Observation and Reward routing
obs_keys: ['rgb']            # Keys used to extract the observation from the environment info dict
reward_key: 'reward'         # Key mapped to the reward signal

# Architecture Configuration
context_horizon: 1           # Number of previous observations to stack as context
each_image_shape: [3, 84, 84] # Image dimensions: [Channels, Height, Width]
hidden_dim: 256              # Size of the hidden layers in the MLP networks
feature_dim: 512             # Output size of the Convolutional feature encoder
action_dim: 12               # Dimensionality of the action space for this specific task
action_range: 1.0            # Boundary for the action space output (e.g., [-1.0, 1.0])

# Learning Rates and Hyperparameters
actor_lr: 3e-4               # Learning rate for the Actor network (Policy)
critic_lr: 3e-4              # Learning rate for the Critic network (Q-function)
alpha_lr: 3e-4               # Learning rate for the entropy temperature parameter
tau: 0.005                   # Polyak averaging coefficient for target network soft updates
gamma: 0.99                  # Discount factor for future rewards
batch_size: 256              # Number of transitions sampled per gradient step

# Training Logistics
replay_capacity: 1000000     # Maximum size of the experience replay buffer
initial_act_steps: 100       # Number of random exploration steps before training begins
train_freq: 1                # Frequency of network updates relative to environment steps
gradient_steps: 1            # Number of gradient updates per training frequency
target_update_interval: 1    # How often to update the target networks

total_update_steps: 1000000  # Total number of network update steps to train for
validation_interval: 10000   # Frequency of running the evaluation pipeline
eval_checkpoint: -1          # Checkpoint index to load for evaluation (-1 usually loads the latest)

save_replay: False           # Whether to save the replay buffer to disk
```

*(Note: We do not need to register the SAC policy or create a new arena, as SAC is natively supported by the framework and we already built the arena in Section 1).*

Next, we need to create a new experimental config file under `conf/sim_exp/` named `vanilla_image_sac_on_diffskill_LiftSpread-v1.yaml` with the following content:

```yaml
# @package _global_
defaults:
  - /agent@agent: vanilla_image_sac
  - /arena@arena: diffskill_LiftSpread-v1
  - /task@task: dummy 

project_name: dough_manipulation
save_root: /media/hcv530/T7/ # Change this path according to your local machine
train_and_eval: train_and_evaluate_single
```

Before starting the training process, you need to set your Weights & Biases (`wandb`) API key to monitor the training progress and logs. You can log in by running `wandb login` in your terminal, or by directly exporting your key:

```bash
source ./setup.sh
export WANDB_API_KEY="your_api_key_here" # This only needs to be set once
```

You can now start training the image-based SAC policy on the `LiftSpread-v1` environment. To run it in the background:

```bash
source ./setup.sh
./submit_training_locally.sh vanilla_image_sac_on_diffskill_LiftSpread-v1
```

If you prefer to run it in the foreground to monitor logs directly, use:

```bash
source ./setup.sh
./submit_training_locally.sh vanilla_image_sac_on_diffskill_LiftSpread-v1 f
```

Alternatively, you can run it via standard Python execution:

```bash
source ./setup.sh
python tool/hydra_train.py --config-name sim_exp/vanilla_image_sac_on_diffskill_LiftSpread-v1
```

-----

## 3\. Summary

In this tutorial, we successfully bridged a third-party physics environment (`DiffSkill`) into the Actoris Harena framework and trained an image-based Reinforcement Learning agent on it.

Here is the methodological pipeline you can follow when integrating and replicating RL policies on **new environments** in the future:

1.  **Build the Arena Wrapper:** Create an adapter class inheriting from `Arena`. Ensure the `reset()` and `step()` functions cleanly map the new environment's outputs into the standardized dictionary format expected by the framework.
2.  **Register the Environment:** Add your new wrapper class to the framework's registry via `registration/sim_arena.py`.
3.  **Configure the Environment Settings:** Create a YAML file in `conf/arena/` defining the initialization arguments for your new environment.
4.  **Sanity Check with Random Policy:** Before introducing learning algorithms, always create an experiment YAML linking your new Arena with the native `RandomPolicy`. Run this to catch any shape mismatches, rendering bugs, or data formatting errors.
5.  **Configure the RL Agent:** Create a YAML file in `conf/agent/` for the RL policy you want to use (e.g., SAC, PPO). Map the observation dimensions (`obs_keys`, `each_image_shape`) and action space (`action_dim`) correctly.
6.  **Deploy Training:** Create your final experiment YAML linking the Agent and the Arena, and launch training via the Hydra command line or bash scripts.