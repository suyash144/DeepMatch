import scipy.io
import os
import h5py

# mat = scipy.io.loadmat(r"\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap\AL032\19011111882\2\UnitMatch\UnitMatch")
f = h5py.File(r"\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap\AL032\19011111882\2\UnitMatch\UnitMatch.mat", 'r')

# print(list(f.keys()))
print(f["UMparam"]["KSDir"])