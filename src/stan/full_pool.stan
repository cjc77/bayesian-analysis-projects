data {
    // We are doing FULL pooling, so don't need to include group-level data

    // Total number of observations
    int<lower=0> N;

    // Response
    vector[N] y;
    // Predictor
    vector[N] x;
    
    // Mean for beta_0 prior
    real beta_0_mu_prior;
    // SD for beta_0 prior
    real<lower=0> beta_0_sigma_prior;
    // SD for beta_1 prior
    real<lower=0> beta_1_sigma_prior;
}

parameters {
    // z variable for beta_0 reparam trick
    real z_beta_0;
    // z variable for beta_1 reparam trick
    real z_beta_1;
    // Mean for beta_1 prior
    real beta_1_mu_prior;
    // Residual standard deviation
    real<lower=0> sigma_r;
}

transformed parameters {
    real beta_0;
    real beta_1;
    // Prior N(beta_0_mu_prior, beta_0_sigma_prior)
    beta_0 = beta_0_mu_prior + beta_0_sigma_prior * z_beta_0;
    // Prior N(0, beta_1_sigma_prior)
    beta_1 = beta_1_mu_prior + beta_1_sigma_prior * z_beta_1;
}

model {
    // Priors
    z_beta_0 ~ normal(0, 1);
    z_beta_1 ~ normal(0, 1);
    sigma_r ~ normal(0, 5);

    // Likelihood
    // shared intercept & shared slope
    y ~ normal(beta_0 + beta_1 * x, sigma_r);
}
