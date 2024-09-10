import numpy as np
import os
import shutil

allowed_paths = ['\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-09-30\\1\\AL031_2019-09-30_run1_g1\\AL031_2019-09-30_run1_g1_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-09-30\\1\\AL031_2019-09-30_run2_g0\\AL031_2019-09-30_run2_g0_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-01\\1\\AL031_2019-10-01_bank0_g0\\AL031_2019-10-01_bank0_g0_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-02\\1\\AL031_2019-10-02_bank0_g0\\AL031_2019-10-02_bank0_g0_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-07\\1\\ephys_bank0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-11\\1\\AL031_2019-10-11_bank01_connected_20mins_g0\\AL031_2019-10-11_bank01_connected_20mins_g0_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-11\\1\\AL031_2019-10-11_bank0_trial3_g0\\AL031_2019-10-11_bank0_trial3_g0_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-11\\1\\AL031_2019-10-11_bank1_trial3_g0\\AL031_2019-10-11_bank1_trial3_g0_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-14\\1\\AL031_2019-10-14_bank0_g0\\AL031_2019-10-14_bank0_g0_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-14\\1\\AL031_2019-10-14_connected_g0\\AL031_2019-10-14_connected_g0_imec0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-14\\1\\ephys_bank0\\PyKS\\output',
       '\\\\znas.cortexlab.net\\Subjects\\AL031\\2019-10-15\\1\\ephys_bank0\\AL031_2019-10-15_bank0_g0_imec0\\PyKS\\output']

used_paths = os.listdir(r"C:\Users\suyas\R_DATA_UnitMatch\AL031\19011116684\1")


mouse = "AL031"
for i, experiment_id in enumerate(allowed_paths):
    experiment_id = experiment_id[experiment_id.find(mouse):]
    experiment_id = experiment_id.replace(mouse, '')
    experiment_id = experiment_id.replace("\\", "_")
    allowed_paths[i] = experiment_id

print(len(set(allowed_paths)))
print(len(set(used_paths)))
c=(set(used_paths)-set(allowed_paths))

for dir in c:
    shutil.rmtree(os.path.join(r"C:\Users\suyas\R_DATA_UnitMatch\AL031\19011116684\1", dir))

