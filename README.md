# FRA
This is a repository for Federated Aggregation of Mallows Rankings: A Comparitive Analysis of Borda and Lehmer Coding.

# Scripts

To access the functions used to implement the various algorithms, please see 'source/rankutils.py' - Each function in the file implements a different part of the algorithms. The function header explains the usage in detail.

The results on synthetic data generated using Mallows model can be replicated with 'source/synthetic_analyze.py'. Additionally, we present the analysis of three real-world datasets - Sushi prefrence, Jester and TCGA. All real-world data used is uploaded to data folder while the corresponding scripts can be found in source folder. All generated results get saved to the figures folder.

# Software requirements

The environemnt used to run the scripts can be recreaded with 'source/environment.yml' file.
