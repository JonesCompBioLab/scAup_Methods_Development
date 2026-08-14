from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from low_purity_methods import GMM_ecDNA_predict, NN_ecDNA_predict, KNN_ecDNA_predict
from NN_model import SCAMP


class mix_match_class():

    # List of tuples of files, the gene that is HSR/Noamp/ecDNA, and a name
    # Ex: [("./dataset/HSR_ecDNA_MYC/ecDNA_0.tsv", 'MYC', 'HSR COLO MYC')]
    def __init__(
        self,
        HSR_data = [("../dataset/HSR_ecDNA_MYC/ecDNA_0.tsv",'HSR COLO MYC')],
        Noamp_data = [("../dataset/noamp_ecDNA_MYC/ecDNA_0.tsv", 'Noamp COLO MYC')],
        ecDNA_data = [("../dataset/noamp_ecDNA_MYC/ecDNA_100.tsv", 'ecDNA COLO MYC')],
    ):
        print("Initiating MM...")
        self.HSR_data = HSR_data
        self.Noamp_data = Noamp_data
        self.ecDNA_data = ecDNA_data

        # Read the dataframes
        self.HSR_lists = []
        self.HSR_names = []
        for tup in HSR_data :
            input_df = pd.read_csv(tup[0], sep = '\t', index_col=0)
            self.HSR_lists.append(np.array(input_df.iloc[:, 0].values))
            self.HSR_names.append(tup[1])
        
        self.Noamp_lists = []
        self.Noamp_names = []
        for tup in Noamp_data :
            input_df = pd.read_csv(tup[0], sep = '\t', index_col=0)
            self.Noamp_lists.append(np.array(input_df.iloc[:, 0].values))
            self.Noamp_names.append(tup[1])

        self.ecDNA_lists = []
        self.ecDNA_names = []
        for tup in ecDNA_data :
            input_df = pd.read_csv(tup[0], sep = '\t', index_col=0)
            self.ecDNA_lists.append(np.array(input_df.iloc[:, 0].values))
            self.ecDNA_names.append(tup[1])

        print("Loading Model...")
        self.model = SCAMP.load("./scamp_model_1.0/")
        print("Initiation Complete")



    # Example mixes to test with
    def _sample_mixes(self) :

        # List of different ecDNA, HSR, Noamp counts, second number is shift. If three numbers, the last one indexes which ecDNA/HSR/Noamp
        self.mixes=[{'ecDNA' : [(400, 0)],'HSR' : [(3600, 10)]},
                    {'ecDNA' : [(400, 0)], 'Noamp' : [(3600, 0)]},
                    {'ecDNA' : [(4000, 0)]},
                    {'Noamp' : [(4000, 0)]},
                    {'ecDNA' : [(800, 20)],'HSR' : [(1200, 0)], 'Noamp' : [(2000, 0)]},
                    {'ecDNA' : [(800, 20)],'HSR' : [(1200, 0), (800, 20)], 'Noamp' : [(1200, 0)]},
                    {'ecDNA' : [(800, 20), (400, 10)],'HSR' : [(1200, 0), (800, 20)], 'Noamp' : [(1200, 0)]},
                    {'HSR' : [(1200, 0), (800, 20)], 'Noamp' : [(2000, 0)]},
                    {'HSR' : [(4000, 0)]},
                    {'HSR' : [(1200, 0)], 'Noamp' : [(2800, 0)]},
                    {'HSR' : [(1200, 0), (400, 30)], 'Noamp' : [(2400, 0)]},
                    ]

    # Note that this rewrites mixes   
    def example(self, methods = [('GMM Base', 'GMM', {}), ('NN Base', 'NN', {}), ('KNN Base', 'KNN', {})], show_hists = False) :
        self._sample_mixes()
        return self.calculate_mixes(methods = methods, show_hists = show_hists)

    # Set list of mixes
    # Format: list of dicts, {'ecDNA' : [(number, shift, ecDNA list index), ...], 'HSR' : [...], 'NoAmp' : [...]}
    # Example
    # mixes=[{'ecDNA' : [(400, 0)],'HSR' : [(3600, 10)]}, 
    #                 {'ecDNA' : [(400, 0, 0)], 'Noamp' : [(3600, 0, 0)]},
    #                 {'ecDNA' : [(4000, 0)]}]
    def set_mixes(self, mixes) :
        self.mixes = mixes

    def _get_sample(self, lst, number) :
        if number < len(lst) :
            return np.random.choice(lst, size=number, replace=False)
        else :
            print(f"Warning: list size {len(lst)} is smaller than requested number {number}")
            temp_list = []
            while number > len(lst):
                temp_list.append(lst)
                number -= len(lst)
            temp_list.append(np.random.choice(lst, size=number, replace=False))
            return np.concatenate(temp_list)
            
    # Actually calculate the mixes
    # Returns: results, a dict with {"[METHOD]" : [0,1,1,0]} detailing ecDNA detection (1 for ecDNA)
    # Specify method arguments in methods
    def calculate_mixes(self, methods = [('scAmp', 'NN', {}), ('GMM + NN', 'GMM', {}), ('KNN + NN', 'KNN', {})], show_hists = False) :

        result = {}
        for method in methods :
            result[method[0]] = []
            

        for mix in self.mixes :
            print(mix)
            input_lists = []
            num_dists = 0
            labels = []
            if 'ecDNA' in mix.keys() :
                for tup in mix['ecDNA'] :
                    # Assumes df_idx is zero unless specified
                    if len(tup) == 3 :
                        num, shift, df_idx = tup
                    else :
                        num, shift = tup
                        df_idx = 0

                    temp = self._get_sample(self.ecDNA_lists[df_idx], num) + shift
                    input_lists.append(temp)

                    num_dists += 1
                    labels.append(f"{self.ecDNA_names[df_idx]} {num}, Shift {shift}")

            if 'HSR' in mix.keys() :
                for tup in mix['HSR'] :
                    # Assumes df_idx is zero unless specified
                    if len(tup) == 3 :
                        num, shift, df_idx = tup
                    else :
                        num, shift = tup
                        df_idx = 0


                    temp = self._get_sample(self.HSR_lists[df_idx], num) + shift
                    input_lists.append(temp)

                    num_dists += 1
                    labels.append(f"{self.HSR_names[df_idx]} {num}, Shift {shift}")

            if 'Noamp' in mix.keys() :
                for tup in mix['Noamp'] :
                    # Assumes df_idx is zero unless specified
                    if len(tup) == 3 :
                        num, shift, df_idx = tup
                    else :
                        num, shift = tup
                        df_idx = 0


                    temp = self._get_sample(self.Noamp_lists[df_idx], num) + shift
                    input_lists.append(temp)

                    num_dists += 1
                    labels.append(f"{self.Noamp_names[df_idx]} {num}, Shift {shift}")

            # Combine them
            final_input_list = [item for sublist in input_lists for item in sublist]

            
            hist_text =  f"GT : {'ecDNA' in mix.keys()}\n"

            # Go through all methods
            for method_tup in methods :
                method_name, method, args = method_tup
                if method == "GMM" :
                    print("Runing GMM")
                    pred = GMM_ecDNA_predict(self.model, final_input_list, **args)
                elif method == "NN" :
                    print("Running NN")
                    pred = NN_ecDNA_predict(self.model, final_input_list, np.array(['TEST']), **args)
                elif method == "KNN" :
                    print("Running KNN")
                    pred = KNN_ecDNA_predict(self.model, final_input_list, **args)
                else :
                    print(f"Unknown Method: {method}")
                
                result[method_name].append(int(pred))
                hist_text += f"{method_name} : {pred}\n"


            
            if show_hists :
                # create shared bins
                temp_final_input = np.clip(final_input_list.copy(), None, 100)
                bins = np.histogram_bin_edges(temp_final_input, bins=100)
                for idx, lst in enumerate(input_lists) :
                    plt.hist(lst, bins=bins, alpha=0.5, label=labels[idx])
            
                plt.title(f"Copy Number Distributions")
                plt.text(
                    0.02, 0.98, f"{hist_text}",
                    transform=plt.gca().transAxes,
                    ha="left",
                    va="top"
                )
                plt.xlabel("Copy Number")
                plt.ylabel("Number of samples")
                plt.legend()
                plt.show()
                plt.close()

        return result