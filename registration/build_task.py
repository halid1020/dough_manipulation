from dotmap import DotMap

def build_task(task_cfg):
    if task_cfg.name == 'dummy':
        task = None
    else:
        raise NotImplementedError(f"Task {task_cfg.name} not supported")
    
    return task