from gym.envs.registration import register

#Classic Control Environments with varying difficulty 

#Mountain Car environment with an adjustable hill height
register(
    id='Var_MC-v0',
    entry_point='Var_Games.Var_MountainCar.VarMC:VMC_Env',
    max_episode_steps = 200,
)

#Acrobot environment with an adjustable height target
register(
    id='Var_Acro-v0',
    entry_point='Var_Games.Var_Acrobot.VarAcro:Var_Acro_Env',
    max_episode_steps = 500,
)

#---------------------------------------------#


#Carnot heat engine environment with an adjustable step size (dV)
register(
    id='Var_Carnot-v0',
    entry_point='Var_Games.Var_Carnot.carnot:CarnotEnv',
    max_episode_steps = 500,
)

