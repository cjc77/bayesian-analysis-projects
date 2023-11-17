data {
    // We are doing NO pooling of the intercept, so don't need to include group-level data
    // but do need individual-level data

    // Total number of observations
    int<lower=0> N;
    // Number of individuals
    int<lower=0> n_indiv;

    // Response
    vector[N] y;
    // Predictor
    vector[N] x;
    
    // Individual ID
    int indiv_id[N];

    // Mean for beta_0 prior
    vector[n_indiv] beta_0_mu_prior;
    // SD for beta_0 prior
    vector<lower=0>[n_indiv] beta_0_sigma_prior;
    // SD for beta_1 prior
    real<lower=0> beta_1_sigma_prior;

}

parameters {
    // z variable for beta_0 reparam trick
    vector[n_indiv] z_beta_0;
    // z variable for beta_1 reparam trick
    real z_beta_1;
    // Residual standard deviation
    real<lower=0> sigma_r;
}

transformed parameters {
    vector[n_indiv] beta_0;
    real beta_1;
    // Prior N(beta_0_mu_prior, beta_0_sigma_prior)
    beta_0 = beta_0_mu_prior + beta_0_sigma_prior .* z_beta_0;
    // Prior N(0, beta_1_sigma_prior)
    beta_1 = beta_1_sigma_prior * z_beta_1;
}

model {
    // Priors
    z_beta_0 ~ normal(0, 1);
    z_beta_1 ~ normal(0, 1);
    sigma_r ~ normal(0, 5);

    // Likelihood
    // individual intercepts & slopes (vectorized)
    y ~ normal(beta_0[indiv_id] + beta_1 * x, sigma_r);
}
