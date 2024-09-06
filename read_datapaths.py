import os
import numpy as np
import h5py
import mat73


# the mice we want to find the recordings for
mice = ["AL031", "AL032", "AL036", "AV008", "CB015", "CB016", "CB017", "CB018", "CB020", "EB019"]   

raw_waveforms = {}          # raw_waveforms[mouse name] = list of paths to raw waveforms for that mouse

rootdir = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"

# Find Unitmatch.mat for each recording
for mouse in mice:
    name_path = os.path.join(rootdir, mouse)
    probes = os.listdir(name_path)
    for probe in probes:
        name_probe = os.path.join(name_path, probe)
        locations = os.listdir(name_probe)
        for location in locations:
            name_probe_location = os.path.join(name_probe, location)
            if not os.path.exists(os.path.join(name_probe_location, "UnitMatch")):
                print(f"No UnitMatch folder where it was expected for mouse {mouse}")
            else:
                datapath = os.path.join(name_probe_location, "UnitMatch")
                f = h5py.File(os.path.join(datapath, "UnitMatch.mat"), 'r')
                f = mat73.loadmat(os.path.join(datapath, "UnitMatch.mat"))
                # find the directory to look for the raw waveforms
                paths = f["UMparam"]["KSDir"]
                raw_waveforms[mouse] = np.array(paths)


