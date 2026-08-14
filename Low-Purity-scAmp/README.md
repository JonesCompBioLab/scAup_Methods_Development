# Low-Purity-scAmp

Detecting ecDNA using single cell ATAC-seq data in cases where the fraction of ecDNA is low
This directory holds scripts used to test low purity methods. The actual methods for use are
implemented in https://github.com/JonesCompBioLab/scamp

## Files

```low_purity_methods.py``` : methods for detecting ecDNA with low purity

```NN_model.py``` : architecture for the original scAmp neural network

```scamp_model_1.0/model.pt``` : scAmp neural network

```mix_match.py``` : class used to easily test methods by mixing ecDNA, HSR, and No-amp data

```plots_script.py``` : generate plots from ```mix_match.py``` and ```low_purity_methods.py```

```k_test.py``` : generate plots and save data to test the value of k in k-nearest neighbors

```k_plot_with_line.ipynb``` : adds lines to value of k analysis

```./vectors/*``` : ecDNA cell line data

## Example

```run_mix_match.py``` contains an example for how to use the class

```python plots_script.py``` to make plots. Edit user settings in script.
