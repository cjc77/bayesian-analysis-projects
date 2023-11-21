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

    // Mean for beta_0 prior
    vector[n_groups] beta_0_mu_group_prior;
    // SD for individual beta_0 term
    real<lower=0> beta_0_sigma_indiv_prior;
    // SD for group beta_0 term
    real<lower=0> beta_0_sigma_group_prior;
    // Mean for beta_1 prior
    real beta_1_mu_prior;
    // SD for beta_1 prior
    real<lower=0> beta_1_sigma_prior;
}

parameters {
    vector[n_indiv] z_beta_0;
    real z_beta_1;
    // Residual standard deviation
    real<lower=0> sigma_r;
}

transformed parameters {
    vector[n_indiv] beta_0;
    real beta_1;

    beta_0 = beta_0_mu_group_prior[group_id_map] + beta_0_sigma_indiv_prior * z_beta_0;
    beta_1 = beta_1_mu_prior + beta_1_sigma_prior * z_beta_1;
}

model {
    // Priors
    sigma_r ~ normal(0, 5);

    z_beta_0 ~ normal(0, 1);
    z_beta_1 ~ normal(0, 1);

    // Likelihood
    // individual intercepts & slopes (vectorized)
    y ~ normal(beta_0[indiv_id] + beta_1 * x, sigma_r);
}
