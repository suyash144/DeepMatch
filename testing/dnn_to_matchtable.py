import os
import pandas as pd
import mat73

server_base = r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap"
base = r"C:\Users\suyas\R_DATA_UnitMatch"

def dnn_to_matchtable():
    mice = os.listdir(base)
    for mouse in mice:
        if mouse=="AV008":
            continue
        name_path = os.path.join(base, mouse)
        probes = os.listdir(name_path)
        for probe in probes:
            name_probe = os.path.join(name_path, probe)
            locations = os.listdir(name_probe)
            for location in locations:
                name_probe_location = os.path.join(name_probe, location)
                mt_path = os.path.join(name_probe_location, "matchtable.csv")
                if not os.path.exists(mt_path):
                    print("No matchtable found for this combination of (mouse, probe, location): ")
                    print(f"Mouse: {mouse}, Probe: {probe}, Location: {location}")
                else:
                    matchtable = pd.read_csv(mt_path)   # shape: (no. of neurons found across all recs)^2 x 31
                    um_path = os.path.join(server_base,mouse,probe,location,"UnitMatch","UnitMatch.mat")
                    um = mat73.loadmat(um_path)
                    paths = um["UMparam"]["KSDir"]  # list of all the paths, order corresponds to RecSes numbers
                    exps = os.listdir(name_probe_location)
                    exps.remove("matchtable.csv")
                    for exp in exps:
                        # compare all experiments against each other and add to match table
                        pass
                

dnn_to_matchtable()