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
    // SD for beta_1 prior
    real<lower=0> beta_1_sigma_prior;
}

parameters {
    vector[n_groups] z_u_0;
    vector[n_indiv] z_beta_0;
    real z_beta_1;
    // Residual standard deviation
    real<lower=0> sigma_r;
    // // Variance for group-level means
    // real<lower=0> sigma_group;
    // // Variance for individual-level means
    // real<lower=0> sigma_indiv;
}

transformed parameters {
    vector[n_groups] u_0;
    vector[n_indiv] beta_0;
    real beta_1;

    // u_0 = beta_0_mu_group_prior + sigma_group * z_u_0;
    // beta_0 = u_0[group_id_map] + sigma_indiv * z_beta_0;
    u_0 = beta_0_mu_group_prior + beta_0_sigma_group_prior * z_u_0;
    beta_0 = u_0[group_id_map] + beta_0_sigma_indiv_prior * z_beta_0;

    beta_1 = beta_1_sigma_prior * z_beta_1;
}

model {
    // Priors
    sigma_r ~ normal(0, 5);
    // sigma_group ~ normal(0, beta_0_sigma_group_prior);
    // sigma_indiv ~ normal(0, beta_0_sigma_indiv_prior);

    z_u_0 ~ normal(0, 1);
    z_beta_0 ~ normal(0, 1);
    z_beta_1 ~ normal(0, 1);

    // Likelihood
    // individual intercepts & slopes (vectorized)
    y ~ normal(beta_0[indiv_id] + beta_1 * x, sigma_r);
}
