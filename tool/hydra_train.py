# hydra_train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig  # <-- 1. Import HydraConfig
import os
import actoris_harena.api as ag_ar

# from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.build_task import build_task
from tool.utils import resolve_save_root


@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    
    # register_agents()
    register_arenas()

    new_save_root = resolve_save_root(cfg.save_root)
    print(f"[tool.hydra_train] Using Save Root: {new_save_root}")

    # 2. Get the name of the yaml config file (without the .yaml extension)
    config_name = HydraConfig.get().job.config_name

    # 3. Use the config file name as the fallback for exp_name
    project_name = cfg.get("project_name", "dough_manipulation")
    exp_name = cfg.get("exp_name", config_name) 

    # 4. Update the config object safely
    OmegaConf.set_struct(cfg, False)
    cfg.save_root = new_save_root
    cfg.exp_name = exp_name
    OmegaConf.set_struct(cfg, True)
    # -------------------------------------

    print("[tool.hydra_train] --- Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("[tool.hydra_train] ---------------------")


    save_dir = os.path.join(cfg.save_root, project_name, exp_name)
    
    agent_name = cfg.agent.get('name', exp_name) 
    arena_name = cfg.arena.get('name', 'default_arena')

    # Build Agent
    print(f"[hydra eval] Building Agent: {agent_name}")
    agent = ag_ar.build_agent(
        agent_name, 
        cfg.agent,
        project_name=project_name,
        exp_name=exp_name,
        save_dir=save_dir,
        disable_wandb=True
    )

    
    # Build Arena
    print(f"[hydra eval] Building Arena: {arena_name}")
    arena = ag_ar.build_arena(
        arena_name, 
        cfg.arena,
        project_name=project_name,
        exp_name=exp_name,
        save_dir=save_dir
    )
    
    # Build Task
    if hasattr(cfg, 'task') and cfg.task is not None:
        task = build_task(cfg.task)
        arena.set_task(task)


    res = ag_ar.train_and_evaluate_single(
        agent,
        arena,
        cfg.agent.validation_interval,
        cfg.agent.total_update_steps,
        eval_last_check=True,
        eval_best_check=True,
        policy_terminate=False,
        env_success_stop=False
    )

   

if __name__ == "__main__":
    main()
