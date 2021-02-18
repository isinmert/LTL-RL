from gym.envs.registration import register

# OfficeWorld
register(
    id='OfficeWorldDoorsTask1-v0',
    entry_point='gym_LTL_RL.envs.officeworld:OfficeWorldDoorsTask1',
)

register(
    id='OfficeWorldDoorsTask2-v0',
    entry_point='gym_LTL_RL.envs.officeworld:OfficeWorldDoorsTask2',
)

register(
    id='OfficeWorldDoorsTask3-v0',
    entry_point='gym_LTL_RL.envs.officeworld:OfficeWorldDoorsTask3',
)

register(
    id='OfficeWorldBigTask1-v0',
    entry_point='gym_LTL_RL.envs.officeworld:OfficeWorldBigTask1',
)

register(
    id='OfficeWorldTaskBCA-v0',
    entry_point='gym_LTL_RL.envs.officeworld:OfficeWorldTaskBCA',
)
