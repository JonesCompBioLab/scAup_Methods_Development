from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import numpy as np
from NN_model import SCAMP
import torch
import math
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
'''
PARAMETERS
model : SCAMP model loaded from .pt file
input : copy number list
n_components : if known, the number of components. Set to None to let kneed find
one_thresh : if n_components is None, uses this to determine when the knee is at 1 
max_components : if n_components is None, checks up to these components
genes : gene name
decision_rule : scAmp NN threshold for classifying ecDNA status
kneedle_coeff : higher means kneedle tends to smaller distributions
var_scale : changes the estimated variance to var * weight ** (var_scale), to counteract small sample size

RETURNS
Boolean, the predicted ecDNA status
'''
def GMM_ecDNA_predict(model, input, 
                      n_components = None, 
                      one_thresh = 1.15, 
                      max_components = 15,
                      genes = np.array(['GENE']), 
                      decision_rule = 0.5,
                      kneedle_coeff = 3,
                      var_scale = 0.5
                      ) :
    
    input = np.array(input).reshape(-1, 1)

    gmm = _GMM_fit(input, n_components, one_thresh, max_components, kneedle_coeff)
    
    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        var = gmm.covariances_[i, 0, 0]
        weight = gmm.weights_[i]
        var_scaled = var * (weight ** var_scale)


        print(f"Mean: {mean}, Var: {var}, Var (scaled) : {var_scaled}, Weight: {weight}")

        if weight > 0.005 :
            samples = np.random.normal(mean, np.sqrt(var_scaled), size=1000).tolist()
            if NN_ecDNA_predict(model, samples, genes, decision_rule, max_percentile = 100) :
                print('ecDNA')
                return True

    return False
    

# Version that takes the real data and assigns them each to a distribution using 
# the posterior probabilities. Found not to work as well, despite properly
# sorting into the two distributions
# Here, labels should be a list of ecDNA or HSR or Noamp
def _DEV_GMM_ecDNA_predict_residual(
    model,
    input,
    labels = None,
    n_components=None,
    one_thresh=1.15,
    max_components=15,
    genes=np.array(['GENE']),
    decision_rule=0.5,
    kneedle_coeff=3,
    var_scale=0,
    plt_title = "Posterior Probabilities"
):

    input = np.array(input).reshape(-1, 1)

    gmm = _GMM_fit(
        input,
        n_components,
        one_thresh,
        max_components,
        kneedle_coeff
    )

    # Posterior probability that each input point belongs
    # to each GMM component.
    posteriors = gmm.predict_proba(input)

    component_samples = [
        [] for _ in range(gmm.n_components)
    ]

    _plot_gmm_posteriors_by_label(posteriors, labels, plt_title)

    # Assign every real point to a component by sampling
    # from its posterior probabilities.
    for i, probs in enumerate(posteriors):

        # Ignore components with <1% posterior probability.
        probs = probs.copy()
        probs[probs < 0.01] = 0
        if probs.sum() == 0:
            continue
        probs /= probs.sum()

        # Select component
        component = np.random.choice(
            gmm.n_components,
            p=probs
        )

        component_samples[component].append(input[i, 0])

    order = np.argsort([len(samples) for samples in component_samples])

    sorted_samples = [component_samples[i] for i in order]
    sorted_labels = [f"Component {i}" for i in order]

    plt.figure(figsize=(8, 6))

    plt.hist(
        sorted_samples,
        bins = np.linspace(0, 70, 31)
,
        stacked=True,
        label=sorted_labels,
        alpha=0.8
    )
    plt.legend()
    plt.title("COLO320DM MYC Posterior Predicted Deconvolution (2000 ecDNA + 2000 HSR)")
    plt.xlabel("Copy Number")
    plt.ylabel("No. Cells")
    plt.savefig("./plots/residual.png")
    plt.close()
    ecDNA = False
    # Run each predicted dist
    for i in range(gmm.n_components):
        weight = gmm.weights_[i]

        if weight > 0.005 and len(component_samples[i]) > 5:

            mean = gmm.means_[i, 0]
            var = gmm.covariances_[i, 0, 0]
            var_scaled = var * (weight ** var_scale)

            print(
                f"Mean: {mean}, "
                f"Var: {var}, "
                f"Var (scaled): {var_scaled}, "
                f"Weight: {weight}, "
                f"Real samples: {len(component_samples[i])}"
            )


            samples = component_samples[i]

            if len(samples) > 0:
                if NN_ecDNA_predict(
                    model,
                    samples,
                    genes,
                    decision_rule
                ):
                    print('ecDNA')
                    ecDNA =  True

    return ecDNA

# Testing the posterior probabilities
def _plot_gmm_posteriors_by_label(posteriors, labels, plt_title):
    posteriors = np.asarray(posteriors)
    labels = np.asarray(labels)

    if len(labels) != len(posteriors):
        raise ValueError(
            "labels and posteriors must contain the same number of samples."
        )

    unique_labels = list(dict.fromkeys(labels))
    n_components = posteriors.shape[1]

    fig, axes = plt.subplots(
        1,
        n_components,
        figsize=(4 * n_components, 5),
        squeeze=False
    )
    axes = axes[0]

    for component in range(n_components):

        ax = axes[component]

        data = []
        plot_labels = []

        for label in unique_labels:
            mask = labels == label

            data.extend(posteriors[mask, component])
            plot_labels.extend([label] * np.sum(mask))

        sns.violinplot(
            x=plot_labels,
            y=data,
            order=unique_labels,
            ax=ax,
            inner="box",
            cut=0
        )

        ax.set_title(f"GMM Component {component}")
        ax.set_xlabel("")
        ax.set_ylabel("Posterior probability")
        ax.set_ylim(0, 1)

    fig.suptitle(plt_title, fontsize=12)

    plt.tight_layout()
    plt.savefig('./plots/posterior.png')
    plt.close()

def _GMM_fit(input, n_components = None, one_thresh = 1.15, max_components = 15, kneedle_coeff = 3) :
    if n_components is None :
        bic_scores = []
        aic_scores = []
        ks = []
        for k in range(1, max_components):
            gmm = GaussianMixture(n_components=k, random_state=0)
            gmm.fit(input)

            bic_scores.append(gmm.bic(input))
            aic_scores.append(gmm.aic(input))
            ks.append(k)

        if bic_scores[0]/np.min(bic_scores) < one_thresh :
            n_components = 1
            print(f"Number of distributions (Kneed): {n_components}")

        else :

            bic_scaled = (bic_scores - (np.min(bic_scores))) / np.ptp(bic_scores)
            bic_scaled = bic_scaled ** kneedle_coeff
            # bic_scaled = np.log(np.array(bic_scores) - np.min(bic_scores) + 1e-8)
            
            kneedle = KneeLocator(
                ks,
                bic_scaled,
                curve="convex",      # or "concave" depending on your curve
                direction="decreasing"
            )

            n_components = kneedle.knee
            print(f"Number of distributions (Kneed): {n_components}")

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(input)

    return gmm


'''
PARAMETERS
model : SCAMP model loaded from .pt file
input : copy number list
genes : gene name
decision_rule : threshold for classifying ecDNA status
log_results : Whether or not to print results

RETURNS
Boolean, the predicted ecDNA status
'''
def NN_ecDNA_predict(model, input, genes = np.array(['GENE']), decision_rule = 0.5, log_results = False, min_copy_number=2.0, max_percentile=99.0) :
    input = np.array(input).reshape(-1, 1)

    X, genes_pass_filter = model.prepare_copy_numbers(
        input,
        genes,
        min_copy_number=min_copy_number,
        max_percentile=max_percentile,
        filter_copy_number=2.5,
    )


    if len(genes_pass_filter) == 0 :
        if log_results :
            print("No genes passed filter")
        return False

    # Just get the second to last layer
    _x = model.forward(torch.Tensor(X))
    _x = _x[0]
    ecDNA_pred = _x[1]/(_x[0] + _x[1])
    if log_results :
        print(f"Node 1: {_x[0]}, Node 2: {_x[1]}, prediction: {ecDNA_pred > decision_rule}")
    return (ecDNA_pred > decision_rule).item()


'''
PARAMETERS
model : SCAMP model loaded from .pt file
input : copy number list
genes : gene name
n_components : if known, the number of components. Set to None to let kneed find
one_thresh : if n_components is None, uses this to determine when the knee is at 1 
max_components : if n_components is None, checks up to these components
kneedle_coeff : higher means kneedle tends to smaller distributions
k_mult : multiplier for k. k = k_mult * sqrt(num cells)
ecDNA_percentage_thresh : threshold for ecDNA positive cells classifying ecDNA status
var_scale : changes the estimated variance to var * weight ** (var_scale), to counteract small sample size
decision_rule : scAmp NN threshold for classifying ecDNA status

RETURNS
Boolean, the predicted ecDNA status
''' 
def KNN_ecDNA_predict(model, input, 
                      genes = np.array(['GENE']), 
                      n_components = None, 
                      one_thresh = 1.15, 
                      max_components = 15,
                      kneedle_coeff = 3,
                      k_mult = 2.5, 
                      k = 100,
                      ecDNA_percentage_thresh = 0.001, 
                      var_scale = 0.5,
                      decision_rule = 0.5) :

    if len(input) > 4000:
        arr = np.random.choice(input, size=4000, replace=False)
    else:
        arr = np.asarray(input)

    arr = sorted(arr)
    if k is None :
        k = int(k_mult * math.sqrt(len(input)))
        # K should never be less than 10
        k = max(k, 10)
        # K is never greater than the input
        k = min(k, len(input))
    print(f"k = {k}")

    gmm = _GMM_fit(np.array(input).reshape(-1, 1), n_components, one_thresh, max_components, kneedle_coeff)
    x = np.array(arr).reshape(-1)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    # Adjust variance by weight
    adjusted_variances = variances * (weights ** var_scale)

    scores = []
    for mean, var, weight in zip(means, adjusted_variances, weights):
        scores.append(
            np.log(weight) +
            norm.logpdf(x, loc=mean, scale=np.sqrt(var))
        )

    scores = np.vstack(scores)

    # pick most likely component
    labels = np.argmax(scores, axis=0)
    arr = np.asarray(arr)
    means = {
        label: arr[labels == label].mean()
        for label in np.unique(labels)
    }

    n = len(arr)
    out = []

    window_size = k + 1  # include self


    for i in range(n):
        target = arr[i]
        target_label = labels[i]

        l, r = i - 1, i + 1
        count = 1

        while count < window_size:
            left_ok = (l >= 0 and labels[l] == target_label)
            right_ok = (r < n and labels[r] == target_label)

            if not left_ok and not right_ok:
                break

            if left_ok and right_ok:
                if abs(arr[l] - target) <= abs(arr[r] - target):
                    l -= 1
                else:
                    r += 1
            elif left_ok:
                l -= 1
            else:
                r += 1

            count += 1

        subset = np.array(arr[l + 1:r])
        mean_diff = np.mean(subset) - means[target_label]
        subset = subset - mean_diff

        if np.mean(subset) > 2.5 and np.var(subset) > 10:

            res = NN_ecDNA_predict(
                    model,
                    subset.tolist(),
                    genes,
                    decision_rule
                )

            out.append(res)
            

    print(f'Number of cells ecDNA positive: {sum(out)}')

    return sum(out) > (ecDNA_percentage_thresh * n)


def OLD_KNN_ecDNA_predict(model, input, 
                      genes = np.array(['GENE']), 
                      n_components = None, 
                      one_thresh = 1.15, 
                      max_components = 15,
                      kneedle_coeff = 3,
                      k_mult = 2.5, 
                      k = None,
                      ecDNA_percentage_thresh = 0.001, 
                      var_scale = 0.5,
                      decision_rule = 0.5) :
    arr = sorted(input)

    if k is None :
        k = int(k_mult * math.sqrt(len(input)))
        # K should never be less than 10
        k = max(k, 10)
        # K is never greater than the input
        k = min(k, len(input))
    print(f"k = {k}")

    gmm = _GMM_fit(np.array(input).reshape(-1, 1), n_components, one_thresh, max_components, kneedle_coeff)
    x = np.array(arr).reshape(-1)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_

    # Adjust variance by weight
    adjusted_variances = variances * (weights ** var_scale)

    scores = []
    for mean, var, weight in zip(means, adjusted_variances, weights):
        scores.append(
            np.log(weight) +
            norm.logpdf(x, loc=mean, scale=np.sqrt(var))
        )

    scores = np.vstack(scores)

    # pick most likely component
    labels = np.argmax(scores, axis=0)
    # print(sum(labels))

    n = len(arr)
    out = []

    window_size = k + 1  # include self


    for i in range(n):
        target = arr[i]
        target_label = labels[i]

        l, r = i - 1, i + 1
        count = 1

        while count < window_size:
            left_ok = (l >= 0 and labels[l] == target_label)
            right_ok = (r < n and labels[r] == target_label)

            if not left_ok and not right_ok:
                break

            if left_ok and right_ok:
                if abs(arr[l] - target) <= abs(arr[r] - target):
                    l -= 1
                else:
                    r += 1
            elif left_ok:
                l -= 1
            else:
                r += 1

            count += 1

        subset = np.array(arr[l + 1:r])

        if np.mean(subset) > 2.5 and np.var(subset) > 10:
            res = NN_ecDNA_predict(
                    model,
                    subset.tolist(),
                    genes,
                    decision_rule
                )

            out.append(res)
            

    print(f'Number of cells ecDNA positive: {sum(out)}')

    return sum(out) > (ecDNA_percentage_thresh * n)



'''
PARAMETERS
input : copy number list
k : number of neighbors to consider
mean_thresh : threshold needed to call a cell ecDNA positive
var_thresh : threshold needed to call a cell ecDNA positive
ecDNA_percentage_thresh : threshold for ecDNA positive cells classifying ecDNA status

RETURNS
Boolean, the predicted ecDNA status
''' 
# Just uses a cutoff, rather than running the scamp model
def _BASIC_KNN_ecDNA_predict(input, k = 100, mean_thresh = 5, var_thresh = 100, ecDNA_percentage_thresh = 0.01) :
    arr = sorted(input)
    
    n = len(arr)
    out = []

    window_size = k + 1  # include self

    for i in range(n):
        target = arr[i]

        l, r = i - 1, i + 1

        S = target
        Q = target * target
        count = 1

        while count < window_size:
            if l < 0:
                val = arr[r]
                r += 1
            elif r >= n:
                val = arr[l]
                l -= 1
            else:
                if abs(arr[l] - target) <= abs(arr[r] - target):
                    val = arr[l]
                    l -= 1
                else:
                    val = arr[r]
                    r += 1

            S += val
            Q += val * val
            count += 1

        mean = S / window_size
        var = (Q / window_size - mean * mean)**2

        out.append(int((mean > mean_thresh) and (var > var_thresh)))

    return sum(out) > (ecDNA_percentage_thresh * n)





'''
PARAMETERS
input : copy number list
n_components : if known, the number of components. Set to None to let kneed find
one_thresh : if n_components is None, uses this to determine when the knee is at 1 
max_components : if n_components is None, checks up to these components
var_thresh : threshold needed for variance to classify as ecDNA
mean_thresh : threshold needed for mean to classify as ecDNA

RETURNS
Boolean, the predicted ecDNA status
'''
def _BASIC_GMM_ecDNA_predict(input, 
                      n_components = None, 
                      one_thresh = 1.15, 
                      max_components = 15,
                      var_thresh = 100,
                      mean_thresh = 5
                      ) :
    
    input = np.array(input).reshape(-1, 1)
    if n_components is None :
        bic_scores = []
        aic_scores = []
        ks = []
        for k in range(1, max_components):
            gmm = GaussianMixture(n_components=k, random_state=0)
            gmm.fit(input)

            bic_scores.append(gmm.bic(input))
            aic_scores.append(gmm.aic(input))
            ks.append(k)

        # import matplotlib.pyplot as plt
        # fig, ax1 = plt.subplots(figsize=(7,5))

        # # BIC (left axis)
        # ax1.plot(ks, bic_scores, marker='o', color='tab:blue', label='BIC')
        # ax1.set_xlabel("Number of components (K)")
        # ax1.set_ylabel("BIC", color='tab:blue')
        # ax1.tick_params(axis='y', labelcolor='tab:blue')
        # plt.show()
        # plt.close()

        if bic_scores[0]/np.min(bic_scores) < one_thresh :
            n_components = 1
            print(f"Number of distributions (Kneed): {n_components}")

        else :

            bic_scaled = (bic_scores - np.min(bic_scores)) / np.ptp(bic_scores)
            # bic_scaled = np.log(np.array(bic_scores) - np.min(bic_scores) + 1e-8)
            
            kneedle = KneeLocator(
                ks,
                bic_scaled,
                curve="convex",      # or "concave" depending on your curve
                direction="decreasing"
            )

            n_components = kneedle.knee
            print(f"Number of distributions (Kneed): {n_components}")

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(input)


    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        var = gmm.covariances_[i, 0, 0]
        weight = gmm.weights_[i]

        # Super simple predictor
        if var >= var_thresh and mean >= mean_thresh :
            return True
    
    return False

# TODO: remove
def _DEV_GMM_ecDNA_predict(model, input, 
                      n_components = None, 
                      one_thresh = 1.15, 
                      max_components = 15,
                      genes = np.array(['GENE']), 
                      decision_rule = 0.5,
                      kneedle_coeff = 3
                      ) :
    
    input = np.array(input).reshape(-1, 1)
    if n_components is None :
        bic_scores = []
        aic_scores = []
        ks = []
        for k in range(1, max_components):
            gmm = GaussianMixture(n_components=k, random_state=0)
            gmm.fit(input)

            bic_scores.append(gmm.bic(input))
            aic_scores.append(gmm.aic(input))
            ks.append(k)

        if bic_scores[0]/np.min(bic_scores) < one_thresh :
            n_components = 1
            print(f"Number of distributions (Kneed): {n_components}")

        else :

            bic_scaled = (bic_scores - (np.min(bic_scores))) / np.ptp(bic_scores)
            bic_scaled = bic_scaled ** kneedle_coeff
            # bic_scaled = np.log(np.array(bic_scores) - np.min(bic_scores) + 1e-8)
            
            kneedle = KneeLocator(
                ks,
                bic_scaled,
                curve="convex",      # or "concave" depending on your curve
                direction="decreasing"
            )

            n_components = kneedle.knee
            print(f"Number of distributions (Kneed): {n_components}")

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(input)

    max_var = 0
    weight_return = 0
    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        var = gmm.covariances_[i, 0, 0]
        weight = gmm.weights_[i]

        print(f"Mean: {mean}, Var: {var}, Weight: {weight}")


        if weight > 0.005 :
            max_var = max(max_var, var)
            weight_return = weight
        #     samples = np.random.normal(mean, np.sqrt(var/2), size=1000).tolist()
        #     if NN_ecDNA_predict(model, samples, genes, decision_rule) :
        #         print('ecDNA')
        #         return True

    return max_var, weight_return, n_components

