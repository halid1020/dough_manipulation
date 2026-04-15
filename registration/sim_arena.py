import actoris_harena as athar
from env.diffskill_arena import DiffSkillArena

def register_arenas():
    athar.register_arena('diffskill', DiffSkillArena)