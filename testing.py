import UnitMatchPy.utils as util
import numpy as np

# util.load_good_waveforms(r"\\znas.cortexlab.net\Subjects\AL036\2020-06-03\ephys\AL036_2020-06-03_stripe240_natIm_g0\AL036_2020-06-03_stripe240_natIm_g0_imec0\pyKS\output", )
np.load(r"\\znas.cortexlab.net\Subjects\AL036\2020-06-03\ephys\AL036_2020-06-03_stripe240_natIm_g0\AL036_2020-06-03_stripe240_natIm_g0_imec0\pyKS\output\qMetrics\RawWaveforms\Unit0_RawSpikes.npy")
print("did first")
np.load(r"\\znas.cortexlab.net\Subjects\AL031\2019-09-30\1\AL031_2019-09-30_run3_g0\AL031_2019-09-30_run3_g0_imec0\PyKS\output\qMetrics\RawWaveforms\Unit0_RawSpikes.npy")
print("did second")