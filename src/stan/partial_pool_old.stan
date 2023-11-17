data {
    // We are doing partial pooling, so must include group-level data

    // Total number of observations
    int<lower=0> N;
    // Number of groups
    int<lower=0> n_groups;
    // Response
    vector[N] y;
    // Predictor
    vector[N] x;
    // Group ID
    int group_id[N];
}

parameters {
    // z variable for beta_0 reparam trick
    vector[n_groups] z_beta_0;
    // z variable for beta_1 reparam trick
    real z_beta_1;
    // Residual standard deviation
    real<lower=0> sigma_r;
    // Variance for group-level means
    real<lower=0> sigma_group_mean;
}

transformed parameters {
    vector[n_groups] beta_0;
    real beta_1;
    // Prior N(150, sigma_group_mean)
    beta_0 = 150.0 + z_beta_0;
    // Prior N(0, 3)
    beta_1 = 3.0 * z_beta_1;
}

model {
    // Priors
    sigma_r ~ normal(0, 5);
    sigma_group_mean ~ normal(0, 10);
    z_beta_0 ~ normal(0, sigma_group_mean);
    z_beta_1 ~ normal(0, 1);

    // Likelihood
    // individual intercepts & slopes (vectorized)
    y ~ normal(beta_0[group_id] + beta_1 * x, sigma_r);
}
