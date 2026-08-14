from mix_match import mix_match_class

# Initiate the class, include which vectors to use for HSR and ecDNA
mm = mix_match_class(HSR_data=[('./vectors/COLO_MYC_HSR.tsv', 'HSR')], ecDNA_data=[('./vectors/COLO_MYC_ecDNA.tsv', 'ecDNA')])


mixes = []
# 3 copies of this mix
for i in range(3) :
    mix = {}
    # 1000 ecDNA, no shift (mean stays the same)
    mix['ecDNA'] = [(1000, 0)]
    # 2000 HSR, shifted up by 5
    mix['HSR'] = [(2000, 5)]
    # 1000 Non-amplified cells
    mix['Noamp'] = [(1000, 0)]

    mixes.append(mix)

# Set the class' mixes
mm.set_mixes(mixes)

# Base GMM, KNN with k = 100 (tune parameters in the dicts)
result = mm.calculate_mixes(show_hists=False, methods = [("KNN (k = 200)", "KNN", {"k" : 200}), ("GMM", "GMM", {})])
print(result)


