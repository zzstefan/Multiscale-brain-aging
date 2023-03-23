# The scirpts used to draw the figures in our manuscript

Figure 2 is the correlations between the chronological age and the predicted age obtained by prediction models built upon functional connectivity measures of individual scales and their combination. Specifically, we need to load the dataframe from python to R. We also append the example data here 'age_plot.pkl'.

In Figure 3, cal_correlation_network.py can generate Figure 3(A), the multi-scale organization of the brain networks from fine scales to coarse scales. plot_withinCircular.R is used to plot Figure 3(B), along with the required data file.

In Figure 4 is the functional connections showing significant correlations with chronological age. correlation_agedelta_scores.R is used to generate Figure 4.