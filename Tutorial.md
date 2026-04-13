# Training a SAC Policy on DiffSkill environments using Actoris Harena

In this tutorial, you will learn how to train a `SAC` Reinforcement Learning (RL) policy within the `DiffSkill` environment using the Actoris Harena framework. 

## 1. Build the Evaluation Pipeline

Before training a complex policy, we need to build an evaluation pipeline for our policy (agent) and environment (arena) pair. We will start by verifying that the environment can successfully run a simple random policy within our framework.

### Step 1a: Create a Random Agent Configuration

First, create a YAML configuration file for the random agent. Create a file named `random_policy.yaml` under the `conf/agent` directory with the following content:

```yaml
name: random
```

This tells the Actoris Harena framework to create an instance of the `RandomPolicy` class. The framework expects to find this class in the `random_policy.py` file under the `actoris_harena/agent/random` directory.

For your convenience, here is what the underlying `RandomPolicy` implementation looks like:

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

Next, we need to create a YAML file to initialize the DiffSkill environment. Create a file named `diffskill-LiftSpread-v1.yaml` under the `conf/arena` directory. The content should be:

```yaml
name: diffskill
task: 'LiftSpread-v1'
```

This instructs the framework to initialize an instance of our environment adapter (which we will build next) with the class argument `task: 'LiftSpread-v1'`.

### Step 1c: Implement the DiffSkill Arena Adapter

Now we need to create the actual adapter class that inherits from the `Arena` interface in the Actoris Harena framework and wraps the `DiffSkill` content. Create this file under the `env` folder.

Here is the content for the `DiffSkillArena` adapter:

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
        self.action_space = self._env.action_space
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

    def _get_rgb(self): # RGB from the envirnment must be between 0 and 255.
        return np.clip(self._env.render(mode='rgb') * 255.0, 0, 255).astype(np.uint8)

    
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
            stp = [r.get('length', r.get('steps', len(r.get('success', [])) if type(r.get('success'))==list else 0)) for r, s in zip(res, suc) if s > 0.5]
            return np.mean(suc or [0]), np.std(rwd or [0]), np.mean(stp or [float('inf')])
        
        (s1, r1, st1), (s2, r2, st2) = stats(r1), stats(r2)
        
        # Loop over criteria in priority order: (+ means higher is better, - means lower is better)
        for diff, eps in [(s1 - s2, 1e-4), (r2 - r1, 1e-4), (st2 - st1, 0.5)]:
            if diff > eps: return 1
            if diff < -eps: return -1
        return 0
    
    def success(self):
        return bool(self._last_info.get('is_success', self._last_info.get('success', False)))
```

### Step 1d: Register the Arena

To make the Actoris Harena framework aware of our new arena, we need to register it. Open the `registration/sim_arena.py` file and add the following:


import actoris_harena as ag_ar
from env.diffskill_arena import DiffSkillArena

def register_arenas():
    ag_ar.register_arena('diffskill', DiffSkillArena)
```

*(Note: We do not have to register the random policy because it is already registered by default in the `actoris_harena` repository).*

### Step 1e: Create the Experiment Configuration

Now, we need an experiment YAML file to link the initialization of our agent and arena, and to dictate where to save the results. 

Create the file `conf/sim_exp/random_policy_on_diffskill_LiftSpread-v1.yaml` with the following configuration:

```yaml
# @package _global_
defaults:
  - /agent@agent: random_policy
  - /arena@arena: diffskill_LiftSpread-v1
  - /task@task: dummy #because it is included already in the environment.

project_name: dough_manipulation
save_root: /media/hcv530/T7/
train_and_eval: train_and_evaluate_single

```

### Step 1f: Run the Evaluation Pipeline

Finally, we can test our evaluation pipeline by running the random policy on our newly adapted DiffSkill environment. Execute the run via Hydra using the terminal:

```bash
source ./setup.sh
python tool/hydra_eval.py --config-name sim_exp/random_policy_on_diffskill_LiftSpread-v1
```

# TO BE CONTINUE ...