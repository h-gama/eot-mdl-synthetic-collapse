# Entropic Optimal Transport MDL (EOT-MDL) 
### Constraining the extent of latent collapse due to repeated cycles of auto-generated data

**Author:** Helena de Almeida Gama, M.S. Data Science, Felician University 
**Topics of interest:** auto-generated data, Information Theory, Data Mining 

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) 
[![Current Research Interest](https://img.shields.io/badge/Interest-Synthetic%20Data%20%26%20MDL-purple)]() 
[![GitHub](https://img.shields.io/badge/Other work-h--gama-black)](https://github.com/h-gama) 

--- 

## Background 

A large proportion of current machine learning practice will involve some degree of generation of synthetic data, whether for augmenting scarce real data sets, avoiding release of confidential information, reducing labelling effort, and so on. I am interested in the consequences of continuing to generate new training examples from previous AI-generated outputs. This is sometimes referred to as Model Autophagy Disorder (MAD) and results in hitting a "data wall".

### Context 
This work began as a wide-ranging theoretical problem set for a Data Mining class:
> "Part of building a theory of DM would be to postulate some overarching theory that would encompass most aspects of DM. You could choose an existing area of science, such as compression, and see what aspects of DM would fall under such a rubric. If others seem hard to accommodate, could you modify the basic theory to incorporate them?"

### Extension to synthetic data 
As noted above and elsewhere, many aspects of Data Mining may be viewed through the lens of MDL. However, this presupposes naturally-occurring data with some degree of irregularity; that is, points further out in the tails will require greater numbers of bits to represent, compared to points closer to areas of higher frequency. As AI-generated examples are used to re-train a model to generate further examples (over successive time periods, $t_0$ through $t_5$), it seems likely that more extreme points would be gradually lost in an attempt to most efficiently describe the remaining points. Thus the range of values represented would reduce over time, with various modes coalescing, and few or no examples remaining to define "normal" behaviour for outlier detection algorithms. It therefore seemed reasonable to attempt to constrain the amount of movement in feature space, measured by 1-Wasserstein distance, as well as impose a minimum level of Differential Entropy, allowing patterns to be detected in the data but retaining some examples from all regions of the original distributions.

--- 

## Method 

Adding terms to penalise movement away from original data distribution $P_{t_0}$ and loss of entropy:

$$\mathcal{L}_{\text{EOT-MDL}}(H, D_t) = L(H) + L(D_t \mid H) + \lambda W_1(P_{t_0}, P_t) - \gamma h(P_t)$$

where $L(H)$ is length of model description $H$ and $L(D_t \mid H)$ length required to describe the error terms after application of model to data set $D_t$ (together measured in no. bits); $W_1$ 1-Wasserstein distance between true data distribution and current generated examples; $h$ differential entropy of current data set, to ensure continued representation of extremes; $\lambda$ and $\gamma$ positive constants.

--- 

## Results 

I generated data consisting mainly of 3 modes, plus a further 5% of examples spread more widely. New examples were then generated 5 times in succession, both according to standard MDL criteria only, and with addition of the constraints outlined above.

### Summary statistics 

| No. Cycles | 1- WD to original (MDL) | 1- WD to original (EOT-MDL) | Diff. Entropy (MDL) | Diff. Entropy (EOT-MDL) | Best K-Means score, k=4 (MDL) | Best K-Means score, k=4 (EOT-MDL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | 
| **0** | 0.0000 | 0.0000 | 2.3846 | 2.3846 | 0.6612 | 0.6612 | 
| **1** | 0.4312 | 0.2584 | 2.1388 | 2.3458 | 0.6421 | 0.6590 | 
| **2** | 0.7854 | 0.3412 | 1.9304 | 2.3410 | 0.6185 | 0.6572 | 
| **3** | 1.0921 | 0.3985 | 1.7652 | 2.3381 | 0.5910 | 0.6558 | 
| **4** | 1.3518 | 0.4290 | 1.6389 | 2.3354 | 0.5682 | 0.6545 | 
| **5** | 1.5645 | 0.4432 | 1.5412 | 2.3336 | 0.5484 | 0.6536 | 

Thus the maximum distance from original data distribution was reduced by about 71.7% (to 0.4432); there was very little overall loss of information, as judged by Differential Entropy (97.9% remaining); and performance of a simple downstream Data Mining algorithm was only marginally affected, with reduction in best achieved Silhouette Coefficient of 1.1%, compared with 17.1% without imposition of constraints.

### Plots 

Changes in each measure over successive cycles: 

![Summary statistics](figure1_eot_mdl_comparison.png) 

Example 2-dimensional projections of original data and after 5 cycles, showing progressive coalescence of clusters:

![Sample points](figure2_latent_collapse.png) 

--- 

---

## Repository Files

* [`eot_mdl_simulation.ipynb`](eot_mdl_simulation.ipynb) — Google Colab / Jupyter Notebook implementing the recursive loop ($t_0 \to t_5$) and metric evaluation.
* [`simulation_results.csv`](simulation_results.csv) — Quantitative metrics comparing Standard MDL vs. EOT-MDL across cycles.
* [`figure1_eot_mdl_comparison.png`](figure1_eot_mdl_comparison.png) — Progression of Wasserstein distance, Differential Entropy, and K-Means Silhouette score.
* [`figure2_latent_collapse.png`](figure2_latent_collapse.png) — 2D projections illustrating cluster coalescence at $t_5$ and tail preservation under EOT-MDL.

### Reproducibility

All code can be executed directly in Google Colab after installing the required scientific packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

## Author

Helena de Almeida Gama

M.S. in Data Science Candidate — Felician University (Rutherford, NJ)

Passionate about synthetic data generation, distributional robustness, and theoretical foundations of machine learning.
