data {
    // We are doing FULL pooling, so don't need to include group-level data

    // Total number of observations
    int<lower=0> N;
    // Response
    vector[N] y;
    // Predictor
    vector[N] x;
}

parameters {
    // z variable for beta_0 reparam trick
    real z_beta_0;
    // z variable for beta_1 reparam trick
    real z_beta_1;
    // Residual standard deviation
    real<lower=0> sigma_r;
}

transformed parameters {
    real beta_0;
    real beta_1;
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
    // shared intercept & shared slope
    y ~ normal(beta_0 + beta_1 * x, sigma_r);
}
