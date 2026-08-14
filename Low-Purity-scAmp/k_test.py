from mix_match import mix_match_class
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import pandas as pd
import sys

###############################################################

# Remaining are all Noamp
def create_mix_noamp(NO_ITER, ecDNA_percent, total_cells_l) :
    mixes = []

    for total_cells in total_cells_l :
        mix = {}
        ecDNA_num = round(total_cells * (ecDNA_percent/100))
        Noamp_num = total_cells - ecDNA_num
        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 0)]
            
        for i in range(NO_ITER) :
            mixes.append(mix)

    return mixes, "Noamp"

# Remaining are all HSR
def create_mix_HSR(NO_ITER, ecDNA_percent, total_cells_l) :
    mixes = []

    for total_cells in total_cells_l :
        mix = {}

        ecDNA_num = round(total_cells * (ecDNA_percent/100))
        HSR_num = total_cells - ecDNA_num
        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
            
        for i in range(NO_ITER) :
            mixes.append(mix)
    return mixes, "HSR"

# HSR vs Noamp
def create_mix_noamp_vHSR(NO_ITER, HSR_percent, total_cells_l) :
    mixes = []

    for total_cells in total_cells_l :
        mix = {}
        HSR_num = round(total_cells * (HSR_percent/100))
        Noamp_num = total_cells - HSR_num
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 0)]
            
        for i in range(NO_ITER) :
            mixes.append(mix)

    return mixes, "Noamp"

# Remaining are all HSR
def create_mix_noamp_aHSR(NO_ITER, ecDNA_percent, total_cells_l) :
    mixes = []

    for total_cells in total_cells_l :
        mix = {}

        ecDNA_num = round(total_cells * (ecDNA_percent/100))
        Noamp_num = round((total_cells - ecDNA_num) / 2)
        HSR_num = Noamp_num
        if ecDNA_num > 0 :
            mix['ecDNA'] = [(ecDNA_num, 0)]
        if Noamp_num > 0 :
            mix['Noamp'] = [(Noamp_num, 0)]
        if HSR_num > 0 :
            mix['HSR'] = [(HSR_num, 0)]
            
        for i in range(NO_ITER) :
            mixes.append(mix)
    return mixes, "Noamp_and_HSR"

#############################################
# Settings
ecDNA_percents = [0,2,5,10,20,30,50]
ks = range(20,301,20)
total_cells_l = range(400, 5201, 400)
no_iter = 30



ecDNA_vec = "./vectors/COLO_MYC_ecDNA.tsv"
HSR_vec = "./vectors/COLO_MYC_HSR.tsv"
Noamp_vec = "./vectors/COLO_GAPDH_Noamp.tsv"


mixfunc_key = sys.argv[1]
plot_out_name = sys.argv[2]
heatmap_name = sys.argv[3]


########################################################

mixfunc_dict = {
    "Noamp" : create_mix_noamp,
    "HSR" : create_mix_HSR,
    "Noamp_and_HSR" : create_mix_noamp_aHSR,
    "Noamp_v_HSR" : create_mix_noamp_vHSR
}

mixfunc = mixfunc_dict[mixfunc_key]

# Whether we are varying HSR or ecDNA
if mixfunc is create_mix_noamp_vHSR :
    vary_HSR = True
else :
    vary_HSR = False
    

# Helper function, plot the heatmap
def _plot_heatmap(heatmap, title, save_path, total_cells_l, ks) :
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=100,
    )
    
    # Add value labels to each heatmap cell
    for i in range(heatmap.shape[0]):      # rows (total_cells)
        for j in range(heatmap.shape[1]):  # columns (k)
            value = heatmap[i, j]
            ax.text(
                j, i,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value < 50 else "black",
                fontsize=8,
            )

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(list(ks))

    ax.set_yticks(range(len(total_cells_l)))
    ax.set_yticklabels(list(total_cells_l))

    ax.set_xlabel("k")
    ax.set_ylabel("Total cells")
    ax.set_title(title)

    cbar = plt.colorbar(im)
    cbar.set_label("% predicted ecDNA")

    plt.tight_layout()
    plt.savefig(save_path) 
    plt.close(fig)

# Create heatmap, plot, and return it
# is_HSR: set true if changing HSR rather than ecDNA
def make_and_plot_heatmap(results, plot_head, ecDNA_percent, total_cells_l, ks, is_HSR = False) :

    heatmap = np.zeros((len(total_cells_l), len(ks)))

    for x, k in enumerate(ks):
        vals = np.asarray(results[str(k)])

        # rows = total_cells, cols = iterations
        vals = vals.reshape(len(total_cells_l), no_iter)

        # average over iterations and convert to %
        heatmap[:, x] = vals.mean(axis=1) * 100

    if is_HSR :
        _plot_heatmap(heatmap, f"True HSR = {ecDNA_percent}%", f'./plots2/{plot_head}_{ecDNA_percent}.png', total_cells_l, ks)

    else :
        _plot_heatmap(heatmap, f"True ecDNA = {ecDNA_percent}%", f'./plots2/{plot_head}_{ecDNA_percent}.png', total_cells_l, ks)

    return heatmap

# Log the results
rows = []
sum_heatmap = np.zeros((len(total_cells_l), len(ks)))

num_nonzero = sum(p > 0 for p in ecDNA_percents)

for ecDNA_percent in ecDNA_percents :
    print(f"Processing ecDNA percent {ecDNA_percent}")

    num_ks = len(ks)
    num_total_cells = len(total_cells_l)

    mm = mix_match_class(HSR_data=[(HSR_vec, 'HSR')], ecDNA_data=[(ecDNA_vec, 'ecDNA')], Noamp_data=[(Noamp_vec,'Noamp')])

    methods = []
    for k in ks :
        methods.append((f'{k}', 'KNN', {'k' : k})) # ('KNN (k_mult 3)', 'KNN', {'k_mult' : 3})

    mix, null = mixfunc(no_iter, ecDNA_percent, total_cells_l)
    
    # HSR first
    mm.set_mixes(mix)
    results = mm.calculate_mixes(methods = methods)
    heatmap = make_and_plot_heatmap(results, plot_out_name, ecDNA_percent, total_cells_l, ks, vary_HSR)
    if ecDNA_percent > 0 :
        sum_heatmap += heatmap / num_nonzero

    if vary_HSR :
        perc_lab = "HSR_percent"
    else :
        perc_lab = "ecDNA_percent"

    # Add to log
    for i in range(heatmap.shape[0]):      # rows (total_cells)
        for j in range(heatmap.shape[1]):  # columns (k)
            new_row = {"k" : ks[j], "total_cells" : total_cells_l[i], perc_lab : ecDNA_percent, "pred_ecDNA_percent" : heatmap[i, j], "null" : null}
            rows.append(new_row)

  



_plot_heatmap(sum_heatmap, heatmap_name, f'./plots2/{plot_out_name}_all.png', total_cells_l, ks)
results_df = pd.DataFrame(rows)
results_df.to_csv(f"./k_results_{mixfunc.__name__}.tsv", sep = '\t', index = False)
