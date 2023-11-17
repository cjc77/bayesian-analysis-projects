data {
    // We are doing partial pooling, so must include group-level data

    // Total number of observations
    int<lower=0> N;
    // Number of individuals
    int<lower=0> n_indiv;
    // Number of groups
    int<lower=0> n_groups;
    // Response
    vector[N] y;
    // Predictor
    vector[N] x;
    // Individual ID for each observation
    int indiv_id[N];
    // Group ID for each individual
    int group_id_map[n_indiv];
}

parameters {
    vector[n_groups] u_0;
    vector[n_indiv] beta_0;
    real beta_1;
    // Residual standard deviation
    real<lower=0> sigma_r;
    // Variance for group-level means
    real<lower=0> sigma_group;
    real<lower=0> sigma_indiv;
}

// TODO: reparam trick

model {
    // Priors
    sigma_r ~ normal(0, 5);
    sigma_group ~ normal(0, 10);
    sigma_indiv ~ normal(0, 5);
    u_0 ~ normal(150, sigma_group);
    for (n in 1:n_indiv) {
        beta_0[n] ~ normal(u_0[group_id_map[n]], sigma_indiv);
    }
    beta_1 ~ normal(0, 3);

    // Likelihood
    // individual intercepts & slopes (vectorized)
    y ~ normal(beta_0[indiv_id] + beta_1 * x, sigma_r);
}
