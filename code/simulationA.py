# Simulation A
#
# Settings:
# --------
# All taps = 5
# Hour = 12
# No PVs

# Outputs:
# -------
# ../results/SimulationA.pdf
# Voltage magnitudes plot. 

# Requirements:
# -------------
# 1. Make sure the simulation can be run from code folder:
#
#    $ cd code 
#    $ python simulationA.py
#
# 2. obey the relative path convention
#
# 3. check if ../results folder exists, create if not.
#
# 4. Generated PDF should be publication-ready.
#    there should be no post-processing required.
#
# 5. Try not to touch dss files manually unless it is absolutely necessary.
#    Use OpenDSS directives as much as possibles.
#
# 6. Keep relative path convention in DSS files.
#
print('Simulation A started!')
import opendssdirect as dss
import os
import numpy as np
import matplotlib.pyplot as plt

DIRECTORY_CODE = os.path.dirname(os.path.abspath(__file__))
DIRECTORY_CPEPOWERENG2025 = os.path.abspath(os.path.join(DIRECTORY_CODE, ".."))
DIRECTORY_DSS = os.path.join(DIRECTORY_CODE, "..", "data", "IEEE123Master_Loads.dss")
DIRECTORY_RESULTDIR = os.path.join(DIRECTORY_CPEPOWERENG2025, 'result')
DIRECTORY_RESULT = os.path.join(DIRECTORY_RESULTDIR, 'simulationA.pdf')

os.makedirs(DIRECTORY_RESULTDIR, exist_ok=True)

print(DIRECTORY_CODE)
dss.Text.Command('Clear')
dss.Text.Command(f'Compile {DIRECTORY_DSS}')
dss.Text.Command('set maxcontroliter = 10000')
dss.Text.Command('set mode = daily')
dss.Text.Command('set number = 1')
dss.Text.Command('set step = 1h')


taps = np.int32(np.ones(dss.RegControls.Count())*5)

dss.RegControls.First()
for i in range(dss.RegControls.Count()):
    dss.RegControls.MaxTapChange(0)
    dss.RegControls.TapNumber(taps[i])
    dss.RegControls.Next()



dss.Text.Command('set hour = 12')
dss.Solution.LoadMult(1)
dss.Text.Command('batchedit PVSystem..* enabled=false')
dss.Text.Command('batchedit InvControl..* enabled=false')

dss.Solution.Solve()

plt.figure(figsize=(12, 6))
plt.plot(dss.Circuit.AllNodeVmagPUByPhase(1), label='Phase A', marker='o', markersize=3, linewidth=2)
plt.plot(dss.Circuit.AllNodeVmagPUByPhase(2), label='Phase B', marker='o', markersize=3, linewidth=2)
plt.plot(dss.Circuit.AllNodeVmagPUByPhase(3), label='Phase C', marker='o', markersize=3, linewidth=2)

plt.ylabel('Voltage magnitude (p.u.)', fontsize=14)
plt.xlabel('Node Number', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14) 

plt.legend()
plt.grid()


plt.savefig(DIRECTORY_RESULT, format="pdf", dpi=300, bbox_inches="tight")
print('Simulation A ended!')