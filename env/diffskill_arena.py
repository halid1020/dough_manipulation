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