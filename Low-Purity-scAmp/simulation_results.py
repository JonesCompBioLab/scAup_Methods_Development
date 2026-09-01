########################################################################
# Runs low purity methods on simulated data and generates statistics
# Usage: cNMF_pipeline.py [run dir (should have the run info in name)] [out dir (main folder, automatically generates specific name)]
########################################################################

from low_purity_methods import GMM_ecDNA_predict, NN_ecDNA_predict, KNN_ecDNA_predict
from NN_model import SCAMP
import sys
import numpy as np
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import argparse
import time

###############################################################

parser = argparse.ArgumentParser(
    description="Pipeline for testing lps on simulation data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "run_dir",
    type=str,
    help="Main input dir"
)

parser.add_argument(
    "out_dir",
    type=str,
    help="Main output dir"
)


args = parser.parse_args()

###############################################################

# directory with the data of the run
run_dir = Path(args.run_dir)
# Full location of where we print things
out_dir = args.out_dir
model = SCAMP.load("./scamp_model_1.0/")

#################################################
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


run_results_folder = f"{out_dir}/{run_dir.name}/"


from collections import defaultdict

def parse_ecDNA_genes(filename):
    ecDNA_to_genes = defaultdict(list)

    in_ground_truth = False

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if line == "--GROUND TRUTH--":
                in_ground_truth = True
                continue

            if in_ground_truth and line.startswith("--"):
                break

            if in_ground_truth and line.startswith("gene_"):
                gene, *species = line.split()

                for s in species:
                    ecDNA_to_genes[s].append(gene.split(':')[0])

    return dict(ecDNA_to_genes)

NN_results_list = []
KNN_results_list = []
GMM_results_list = []
GMM_unscaled_results_list = []


for spec_dir in Path(run_dir).glob("*"):
    # Should be in the file names
    num_ecDNA_true = int(spec_dir.name.split('_')[0])
    comb_chance = float(spec_dir.name.split('_')[2])
    print("SPEC DIR")
    print(spec_dir)

    # Skip one species for now to make things easier
    if num_ecDNA_true < 1 :
        continue


    for cellbygene_path in Path(spec_dir).glob("*_cellxgene.tsv") :
        cellbygene = pd.read_csv(str(cellbygene_path), sep = '\t', index_col = 0)
        metadata_file = str(cellbygene_path).replace("cellxgene.tsv", "metadata.txt")
        cellbyspecies = pd.read_csv(str(cellbygene_path).replace("cellxgene.tsv", "cellxspecies.tsv"), sep = '\t', index_col = 0)
        out_name = cellbygene_path.name.split("_cellxgene.tsv")[0]
        print("OUT NAME")
        print(out_name)

        species_to_gene = parse_ecDNA_genes(metadata_file)
        print(species_to_gene)

        # Find the ecDNA percentage of each (not optimized but this should be fast anyways)
        gene_to_percentage = {}
        for species, gene_list in species_to_gene.items() :
            percentage_temp = (cellbyspecies[species] > 0).mean() * 100
            for gene in gene_list :
                gene_to_percentage[gene] = percentage_temp


        for gene_name in gene_to_percentage.keys() :
            NN_start = time.time()

            ecDNA_status = NN_ecDNA_predict(model, cellbygene[gene_name])
            NN_results_list.append({"Num_species" : num_ecDNA_true, "Comb_chance" : comb_chance, "ecDNA_Percentage" : gene_to_percentage[gene_name], "ecDNA" : ecDNA_status, "runtime" : time.time() - NN_start})

            KNN_start = time.time()
            
            ecDNA_status = KNN_ecDNA_predict(model, cellbygene[gene_name])
            KNN_results_list.append({"Num_species" : num_ecDNA_true, "Comb_chance" : comb_chance, "ecDNA_Percentage" : gene_to_percentage[gene_name], "ecDNA" : ecDNA_status, "runtime" : time.time() - KNN_start})

            GMM_start = time.time()
                        
            ecDNA_status = GMM_ecDNA_predict(model, cellbygene[gene_name])
            GMM_results_list.append({"Num_species" : num_ecDNA_true, "Comb_chance" : comb_chance, "ecDNA_Percentage" : gene_to_percentage[gene_name], "ecDNA" : ecDNA_status, "runtime" : time.time() - GMM_start})
            
            GMM_unscaled_start = time.time()
                        
            ecDNA_status = GMM_ecDNA_predict(model, cellbygene[gene_name], var_scale = 0)
            GMM_unscaled_results_list.append({"Num_species" : num_ecDNA_true, "Comb_chance" : comb_chance, "ecDNA_Percentage" : gene_to_percentage[gene_name], "ecDNA" : ecDNA_status, "runtime" : time.time() - GMM_unscaled_start})
            
os.makedirs(run_results_folder, exist_ok=True)                   
        
(pd.DataFrame(NN_results_list)).to_csv(f"{run_results_folder}/NN_results.tsv", index = None, sep = '\t')
(pd.DataFrame(KNN_results_list)).to_csv(f"{run_results_folder}/KNN_results.tsv", index = None, sep = '\t')
(pd.DataFrame(GMM_results_list)).to_csv(f"{run_results_folder}/GMM_scaled_results.tsv", index = None, sep = '\t')
(pd.DataFrame(GMM_unscaled_results_list)).to_csv(f"{run_results_folder}/GMM_results.tsv", index = None, sep = '\t')
