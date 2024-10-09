import os
import numpy as np
import mat73


# the mice we want to find the recordings for
mice = ["AL031", "AL032", "AL036", "AV008", "CB015", "CB016", "CB017", "CB018", "CB020", "EB019"]   

def read_datapaths(mice):
    """
    Input should be a list of mouse names as strings, e.g. ["AL031", ...]
    Output is a dictionary uniquely identifying each (mouse, probe, location) and the relevant recordings.
    Structure of output given below...

    raw_waveforms_dict["mouse"] : a list of mouse names (can be repeats)
    raw_waveforms_dict["probe"] : the corresponding probe for each entry in the mouse list
    raw_waveforms_dict["loc"] : the corresponding locations on the probe recordings were taken from
    raw_waveforms_dict["recordings"] : corresponding list where each entry is a numpy array listing all the paths to the recordings folders.
    """
    if type(mice) == str:
        # Sanitise inputs so that a single string can be passed in rather than a list.
        mice = [mice]

    # Initialise output dictionary
    raw_waveforms_dict = {}
    raw_waveforms_dict["mouse"] = []
    raw_waveforms_dict["probe"] = []
    raw_waveforms_dict["loc"] = []
    raw_waveforms_dict["recordings"] = []

    base = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"

    # Find Unitmatch.mat for each recording
    for mouse in mice:
        name_path = os.path.join(base, mouse)
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
                    try:
                        f = mat73.loadmat(os.path.join(datapath, "UnitMatch.mat"), verbose=False)
                    except:
                        pass
                    # find the directory to look for the raw waveforms
                    paths = f["UMparam"]["KSDir"]

                    # build the dictionary containing all relevant information
                    raw_waveforms_dict["mouse"].append(mouse)
                    raw_waveforms_dict["probe"].append(probe)
                    raw_waveforms_dict["loc"].append(location)
                    raw_waveforms_dict["recordings"].append(np.array(paths))

    return raw_waveforms_dict