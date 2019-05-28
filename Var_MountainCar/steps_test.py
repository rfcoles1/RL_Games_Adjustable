import gym
import numpy as np
import matplotlib.pyplot as plt
from VarMC_BothSides import VMC_Env


StepsTaken = []

heights = np.arange(0,1,0.01)

N = 100
E = VMC_Env()

for i in heights:
    steps = 0
    for _ in range(N):
        E.reset(i)
        for j in range(200):
            s, reward, done, _ = E.step(2)

            if s[0] >= 0.5:
                break
        steps += j

    StepsTaken.append((1.0*steps)/N)

plt.plot(heights, StepsTaken)
plt.ylabel('Steps Taken')
plt.xlabel('Height multiplier')
plt.show()

