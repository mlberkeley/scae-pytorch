# SCAE Experiments
`google_stacked_capsule_autoencoders` is the clone of the Google repository

Reports: 
- [Main WandB report](https://wandb.ai/maximsmol/proj-google_stacked_capsule_autoencoders/reports/Semantic-Convolutions-ML-B---VmlldzozNTc3NTI)
- [Torch Port Report](https://wandb.ai/axquaris/StackedCapsuleAutoEncoders/reports/Part-Capsule-Autoencoder-in-Pytorch--VmlldzozNTczODM)

# Torch Port
See the `torch-imp` branch for the PyTorch port

# Monte-Carlo Optimization
- [`scripts/log_likelihood_opt.py`](scripts/log_likelihood_opt.py) - takes the entire likelihood function from the google SCAE repo and benchmarks it against a version of the function that just computes argmax k
- [`scirpts/mcmc_map_opt.py`](scripts/mcmc_map_opt.py) benchmarks MCMC runs against the evaluation of the naive log likelihood (adjust `NUM_PARAMS` at the top to tune)
