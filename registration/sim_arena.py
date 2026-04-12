import actoris_harena as ag_ar
from env.diffskill_arena import DiffSkillArena

def register_arenas():
    ag_ar.register_arena('diffskill', DiffSkillArena)