
import os,sys
import numpy as np

# record the session pair here

### train
mouse = 'AL031'
probe = '19011116684'
location = '1'
dates = ['2019-09-30','2019-10-01']
exps = ['AL031_2019-09-30_run1_g1_t0-imec0-ap','AL031_2019-10-01_bank0_g0_t0-imec0-ap']
session_pair = '1'

mouse = 'AL032'
probe = '19011111882'
location = '2'
dates = ['2020-03-05','2020-03-06']
exps = ['AL032_2020-03-05_stripe192_audioVis_g0_t0-imec0-ap','AL032_2020-03-06_stripe192_NatIm_g0_t0-imec0-ap']
session_pair = '3'

####### weird session pair
mouse = 'AL036'
probe = '19011116882'
location = '1'
dates = ['2020-02-12','2020-02-14']
exps = ['AL036_2020-02-12_stripe192_audioVis_g1_t0-imec0-ap','AL036_2020-02-14_stripe192_NatIm_g0_t0-imec0-ap']
session_pair = '3'

mouse = 'CB015'
probe = '19011110242'
location = '1'
dates = ['2021-09-10','2021-09-11']
exps = ['CB015_2021-09-10_NatImages_g0_t0-imec0-ap','CB015_2021-09-11_NatImages_g0_t0-imec0-ap']
session_pair = '1'

####### weird session pair
mouse = 'CB015'
probe = '19011110242'
location = '1'
dates = ['2021-09-12','2021-09-13']
exps = ['CB015_2021-09-12_NatImages_g0_t0-imec0-ap','CB015_2021-09-13_NatImages_g0_t0-imec0-ap']
session_pair = '2'

### Seen
mouse = 'AL032'
probe = 'Probe0'
location = '1'
dates = ['2019-11-21','2019-11-22']
exps = ['exp1','exp1']
session_pair = '1'

mouse = 'AL036'
probe = '19011116882'
location = '3'
dates = ['2020-02-24', '2020-02-25']
exps = ['AL036_2020-02-24_stripe240_NatIm_g0_t0-imec0-ap', 
        'AL036_2020-02-25_stripe240_NatIm_g0_t0-imec0-ap']
session_pair = '1'

mouse = 'CB016'
probe = '19011110242'
location = '2'
dates = ['2021-10-13', '2021-10-15']
exps = ['CB016_2021-10-13_NatImages_g0_t0-imec0-ap', 
        'CB016_2021-10-15_NatImages_g0_t0-imec0-ap']
session_pair = '1'

mouse = 'CB017'
probe = '19011110803'
location = '2'
dates = ['2021-10-11', '2021-10-13']
exps = ['CB017_2021-10-11_NatImages_g0_t0-imec0-ap',
        'CB017_2021-10-13_NatImages_g0_t0-imec0-ap']
session_pair = '1'

mouse = 'CB017'
probe = '19011110803'
location = '2'
dates = ['2021-10-27', '2021-10-29']
exps = ['CB017_2021-10-27_NatImages_g0_t0-imec0-ap',
        'CB017_2021-10-29_NatImages_g0_t0-imec0-ap']
session_pair = '2'

mouse = 'CB020'
probe = '19011110122'
location = '6'
dates = ['2021-11-25', '2021-11-26']
exps = ['CB020_2021-11-25_NatImagesShort_g0_t0-imec0-ap',
        'CB020_2021-11-26_NatImagesShort_g0_t0-imec0-ap']
session_pair = '1'

### Unseen
mouse = 'AL032'
probe = '19011111882'
location = '2'
dates = ['2019-12-03', '2019-12-04']
exps = ['AL032_2019-12-03_stripe192_natIm_g0_t0-imec0-ap', 
        'AL032_2019-12-04_stripe192_gratings_g0_t0-imec0-ap']
session_pair = '2'

mouse = 'AL036'
probe = '19011116882'
location = '3'
dates = ['2020-08-04', '2020-08-05']
exps = ['AL036_2020-08-04_stripe240r1_natIm_g0_t0-imec0-ap', 
        'AL036_2020-08-05_stripe240_natIm_g0_t0-imec0-ap']
session_pair = '2'

mouse = 'CB016'
probe = '19011110242'
location = '2'
dates = ['2021-10-05', '2021-10-06']
exps = ['CB016_2021-10-05_NatImages_g0_t0-imec0-ap', 
        'CB016_2021-10-06_NatImages_g0_t0-imec0-ap']
session_pair = '2'

# totaly new mouse
mouse = 'AV007'
probe = '19011110122'
location = '4'
dates = ['2022-03-09', '2022-03-10']
exps = ['AV007_2022-03-09_PassiveActive_g0_t0-imec0-ap', 
        'AV007_2022-03-10_Training_g0_t0-imec0-ap']
session_pair = '1'

mouse = 'AV007'
probe = '19011119461'
location = '11'
dates = ['2022-04-06', '2022-04-07']
exps = ['AV007_2022-04-06_ActivePassive_g0_t0-imec1-ap', 
        'AV007_2022-04-07_ActivePassive_g0_t0-imec1-ap']
session_pair = '2'

mouse = 'AV009'
probe = 'Probe1'
location = 'IMRO_10'
dates = ['2022-03-09', '2022-03-11']
exps = ['AV009_2022-03-09_PassiveActive_day2_g0_t0-imec0-ap', 
        'AV009_2022-03-11_PassiveActive_day2_g0_t0-imec0-ap']
session_pair = '1'

mouse = 'AV009'
probe = 'Probe1'
location = 'IMRO_10'
dates = ['2022-03-21', '2022-03-23']
exps = ['AV009_2022-03-21_ActivePassive_day2_g0_t0-imec0-ap', 
        'AV009_2022-03-23_ActivePassive_day2_g0_t0-imec0-ap']
session_pair = '2'

mouse = 'AV023'
probe = '19122519041'
location = '5'
dates = ['2022-12-11', '2022-12-12']
exps = ['AV023_2022-12-11_ActivePassive_g0_t0-imec0-ap', 
        'AV023_2022-12-12_ActivePassive_g0_t0-imec0-ap']
session_pair = '1'

mouse = 'AV023'
probe = '19122519041'
location = '5'
dates = ['2022-12-13', '2022-12-14']
exps = ['AV023_2022-12-13_ActivePassive_g0_t0-imec0-ap', 
        'AV023_2022-12-14_ActivePassive_g0_t0-imec0-ap']
session_pair = '2'

mouse = 'AV023'
probe = '19122519053'
location = '4'
dates = ['2022-11-30', '2022-12-01']
exps = ['AV023_2022-11-30_ActivePassive_g0_t0-imec1-ap', 
        'AV023_2022-12-01_ActivePassive_g0_t0-imec1-ap']
session_pair = '3'

mouse = 'AV023'
probe = '19122519053'
location = '5'
dates = ['2022-12-11', '2022-12-12']
exps = ['AV023_2022-12-11_ActivePassive_g0_t0-imec1-ap', 
        'AV023_2022-12-12_ActivePassive_g0_t0-imec1-ap']
session_pair = '4'

mouse = 'EB036'
probe = 'Probe1'
location = '1'
dates = ['2024-02-07', '2024-02-08']
exps = ['2024-02-07_exp1', 
        '2024-02-08_exp1']
session_pair = '1'

mouse = 'EB037'
probe = 'Probe0'
location = '1'
dates = ['2024-02-06', '2024-02-07']
exps = ['2024-02-06_exp1', 
        '2024-02-07_exp1']
session_pair = '1'

####### weird session pair
mouse = 'EB037'
probe = 'Probe1'
location = '1'
dates = ['2024-02-06', '2024-02-07']
exps = ['2024-02-06_exp1', 
        '2024-02-07_exp1']
session_pair = '2'

mouse = 'AV013'
probe = '19011119461'
location = '7'
dates = ['2022-06-28', '2022-06-29']
exps = ['AV013_2022-06-28_ActivePassive_g0_t0-imec0-ap', 
        'AV013_2022-06-29_ActivePassive_g0_t0-imec0-ap']
session_pair = '1'

mouse = 'AV013'
probe = '19011119461'
location = '8'
dates = ['2022-06-09', '2022-06-10']
exps = ['AV013_2022-06-09_ActivePassive_g0_t0-imec0-ap', 
        'AV013_2022-06-10_ActivePassive_g0_t0-imec0-ap']
session_pair = '2'