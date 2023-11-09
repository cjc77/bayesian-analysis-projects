data {
    // We are doing NO pooling, so don't need to include group-level data
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
}

parameters {
    // z variable for beta_0 reparam trick
    vector[n_indiv] z_beta_0;
    // z variable for beta_1 reparam trick
    vector[n_indiv] z_beta_1;
    // Residual standard deviation
    vector<lower=0>[n_indiv] sigma_r;
}

transformed parameters {
    vector[n_indiv] beta_0;
    vector[n_indiv] beta_1;
    // Prior N(150, 10)
    beta_0 = 150.0 + 10.0 * z_beta_0;
    // Prior N(0, 3)
    beta_1 = 3.0 * z_beta_1;
}

model {
    // Priors
    z_beta_0 ~ normal(0, 1);
    z_beta_1 ~ normal(0, 1);
    sigma_r ~ normal(0, 5);

    // Likelihood
    // individual intercepts & slopes (vectorized)
    y ~ normal(beta_0[indiv_id] + beta_1[indiv_id] .* x, sigma_r[indiv_id]);
}
