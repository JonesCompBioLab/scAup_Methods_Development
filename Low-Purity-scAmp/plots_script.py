from mix_match import mix_match_class
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

###############################################################

# Remaining are all Noamp
def create_mix_noamp(NO_ITER, max_ecDNA = 1000, total_cells = 4000) :
    mixes = []

    ecDNA_num_float = 0.0
    while ecDNA_num_float <= max_ecDNA :
        ecDNA_num = round(ecDNA_num_float)
        mix = {}
        Noamp_num = total_cells - ecDNA_num
        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 0)]
        
        for i in range(NO_ITER) :
            mixes.append(mix)

        ecDNA_num_float += total_cells/100 
    return mixes

# Remaining are all HSR
def create_mix_HSR(NO_ITER, max_ecDNA = 800, total_cells = 4000) :
    mixes = []

    ecDNA_num_float = 0.0
    while ecDNA_num_float <= max_ecDNA :
        ecDNA_num = round(ecDNA_num_float)
        mix = {}
        HSR_num = total_cells - ecDNA_num
        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
        
        for it in range(NO_ITER) :
            mixes.append(mix)
        ecDNA_num_float += total_cells/100 

    return mixes

# Half HSR, half Noamp
def create_mix_HSR_noamp(NO_ITER, max_ecDNA = 800, total_cells = 4000) :
    mixes = []
    
    ecDNA_num_float = 0.0
    while ecDNA_num_float <= max_ecDNA :
        ecDNA_num = round(ecDNA_num_float)
        mix = {}
        Noamp_num = int((total_cells - ecDNA_num)/2)
        HSR_num = int((total_cells - ecDNA_num)/2)

        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 0)]
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
        for it in range(NO_ITER) :
            mixes.append(mix)
        ecDNA_num_float += total_cells/100 

    return mixes

# HSR mimics ecDNA mean (distributions overlap)
def create_mix_HSR_p10(NO_ITER, max_ecDNA = 1600, total_cells = 4000) :
    mixes = []

    ecDNA_num_float = 0.0
    while ecDNA_num_float <= max_ecDNA :
        ecDNA_num = round(ecDNA_num_float)
        mix = {}
        HSR_num = int(total_cells - ecDNA_num)

        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
    
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 10)]
        for it in range(NO_ITER) :
            mixes.append(mix)
        ecDNA_num_float += total_cells/100 

    return mixes

# Just HSR and Noamp
def create_HSR_noamp(NO_ITER, max_HSR = 4000, total_cells = 4000) :
    mixes = []

    HSR_num_float = 0.0
    while HSR_num_float <= max_HSR :
        HSR_num = round(HSR_num_float)
        mix = {}
        Noamp_num = int(total_cells - HSR_num)

        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
    
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 0)]
        for it in range(NO_ITER) :
            mixes.append(mix)
        HSR_num_float += total_cells/100 

    return mixes


# Two HSR of increasing shift
def create_two_HSR(NO_ITER, HSR_1 = 2000, HSR_2 = 2000) :
    mixes = []

    for shift in range(0, 40) :
        mix = {'HSR' : [(HSR_1, 0), (HSR_2, shift)]}

        for it in range(NO_ITER) :
            mixes.append(mix)

    return mixes

#################################################################
# USER SETTINGS
NO_ITER = 20
methods = [('GMM (Unscaled)', 'GMM', {"var_scale" : 0}), ('GMM (Scaled)', 'GMM', {}), ('scAmp', 'NN', {}), ('GMM+KNN', 'KNN', {})]

ecDNA_file = "./vectors/COLO_MYC_ecDNA.tsv"
HSR_file = "./vectors/COLO_MYC_HSR.tsv"
Noamp_file = "./vectors/COLO_GAPDH_Noamp.tsv"
save_start = "paper_small_"

# {'Method' : (CREATE MIX FUNCTION, {ARGS}), 
# 'Name' : PLOT NAME, 
# 'Save' : PLOT SAVE LOCATION,
# 'ecDNA' : ecDNA LOCATION, 
# 'HSR' : HSR LOCATION, 
# 'Noamp' : NOAMP LOCATION}
settings_list = [
                #   {'Method' : (create_mix_HSR, {"total_cells" : 1000, "max_ecDNA" : 300}),
                #   'Name' : "ecDNA vs HSR (1000 Cells)",
                #   'Save' : f"./plots/{save_start}ecDNA_HSR_1000.pdf",
                #   'ecDNA' : ecDNA_file,
                #   'HSR' : HSR_file,
                #   'Noamp' : Noamp_file}
                #   ,
                #   {'Method' : (create_mix_HSR, {"total_cells" : 500, "max_ecDNA" : 200}),
                #   'Name' : "ecDNA vs HSR (500 Cells)",
                #   'Save' : f"./plots/{save_start}ecDNA_HSR_500.pdf",
                #   'ecDNA' : ecDNA_file,
                #   'HSR' : HSR_file,
                #   'Noamp' : Noamp_file}
                #   ,
                #   {'Method' : (create_mix_HSR, {"total_cells" : 800, "max_ecDNA" : 300}),
                #   'Name' : "ecDNA vs HSR (800 Cells)",
                #   'Save' : f"./plots/{save_start}ecDNA_HSR_800.pdf",
                #   'ecDNA' : ecDNA_file,
                #   'HSR' : HSR_file,
                #   'Noamp' : Noamp_file}
                #   ,
                #   {'Method' : (create_mix_HSR, {"total_cells" : 300, "max_ecDNA" : 120}),
                #   'Name' : "ecDNA vs HSR (300 Cells)",
                #   'Save' : f"./plots/{save_start}ecDNA_HSR_300.pdf",
                #   'ecDNA' : ecDNA_file,
                #   'HSR' : HSR_file,
                #   'Noamp' : Noamp_file}
                #   ,
                #   {'Method' : (create_mix_HSR, {"total_cells" : 8000, "max_ecDNA" : 2500}),
                #   'Name' : "ecDNA vs HSR (8000 Cells)",
                #   'Save' : f"./plots/{save_start}ecDNA_HSR_8000.pdf",
                #   'ecDNA' : ecDNA_file,
                #   'HSR' : HSR_file,
                #   'Noamp' : Noamp_file}
                #   ,
                #   {'Method' : (create_HSR_noamp, {"total_cells" : 500, "max_HSR" : 200}),
                #   'Name' : "HSR vs Noamp (500 Cells)",
                #   'Save' : f"./plots/{save_start}HSR_Noamp_500.pdf",
                #   'ecDNA' : ecDNA_file,
                #   'HSR' : HSR_file,
                #   'Noamp' : Noamp_file}
                #   ,
                  {'Method' : (create_HSR_noamp, {}),
                  'Name' : "HSR vs Noamp (4000 Cells)",
                  'Save' : f"./plots/{save_start}HSR_Noamp_4000.pdf",
                  'ecDNA' : ecDNA_file,
                  'HSR' : HSR_file,
                  'Noamp' : Noamp_file}
                  ,
                  {'Method' : (create_two_HSR, {}),
                  'Name' : "Two HSR Shifting (4000 Cells)",
                  'Save' : f"./plots/{save_start}two_HSR_4000.pdf",
                  'ecDNA' : ecDNA_file,
                  'HSR' : HSR_file,
                  'Noamp' : Noamp_file}
                  ,
                  {'Method' : (create_mix_noamp, {}),
                  'Name' : "ecDNA vs Noamp (4000 Cells)",
                  'Save' : f"./plots/{save_start}ecDNA_noAmp_4000.pdf",
                  'ecDNA' : ecDNA_file,
                  'HSR' : HSR_file,
                  'Noamp' : Noamp_file}
                  ,
                  {'Method' : (create_mix_HSR, {}),
                  'Name' : "ecDNA vs HSR (4000 Cells)",
                  'Save' : f"./plots/{save_start}ecDNA_HSR_4000.pdf",
                  'ecDNA' : ecDNA_file,
                  'HSR' : HSR_file,
                  'Noamp' : Noamp_file}
                  ,
                  {'Method' : (create_mix_HSR_noamp, {}),
                  'Name' : "ecDNA vs Half HSR, Half Noamp (4000 Cells)",
                  'Save' : f"./plots/{save_start}ecDNA_half_4000.pdf",
                  'ecDNA' : ecDNA_file,
                  'HSR' : HSR_file,
                  'Noamp' : Noamp_file}
                  ,
                  {'Method' : (create_mix_HSR, {}),
                  'Name' : "ecDNA vs HSR + 10 (4000 Cells)",
                  'Save' : f"./plots/{save_start}ecDNA_HSRp10_4000.pdf",
                  'ecDNA' : ecDNA_file,
                  'HSR' : HSR_file,
                  'Noamp' : Noamp_file}
                  ]
##################################################

for settings in settings_list :
    mm = mix_match_class(HSR_data=[(settings['HSR'], 'HSR')], ecDNA_data=[(settings['ecDNA'], 'ecDNA')], Noamp_data=[(settings['Noamp'],'Noamp')])

    mixes = settings['Method'][0](NO_ITER, **settings['Method'][1])
    mm.set_mixes(mixes)
    results = mm.calculate_mixes(methods = methods)

    plt.figure(figsize=(6.4, 3.5))


    for y, (name, vals) in enumerate(results.items()):
        vals = np.asarray(vals)
        vals = vals.reshape(-1, NO_ITER).mean(axis=1) * 100
        xs = range(0, len(vals))

        plt.plot(xs, vals, label = name, marker='o', alpha = 0.3)


    plt.title(settings['Name'])
    plt.legend()
    if settings['Method'][0].__name__ == "create_HSR_noamp" :
        plt.xlabel("HSR Percentage")
    elif settings['Method'][0].__name__ == "create_two_HSR" :
        plt.xlabel("Difference in Mean")
    else :
        plt.xlabel("ecDNA Percentage")

    plt.ylabel("Percent Predicted Correct")
    plt.tight_layout()
    plt.savefig(settings['Save'])
    plt.close()

