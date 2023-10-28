data {
    // We are doing FULL pooling, so don't need to include class-level data

    // Total number of observations
    int<lower=0> N;
    // Exam scores
    vector[N] y;
    // Study hours
    vector[N] x;
}

parameters {
    // real beta_0;  // Intercept
    // real beta_1;  // Slope

    // Latent variable for beta_0
    real z_beta_0;
    // Latent variable for beta_1
    real z_beta_1;
    // Residual standard deviation
    real<lower=0> sigma_r;
}

transformed parameters {
    real beta_0;
    real beta_1;
    // Prior N(50, 10)
    beta_0 = 50.0 + 10.0 * z_beta_0;
    // Prior N(0, 10)
    beta_1 = 10.0 * z_beta_1;
}

model {
    // Priors
    z_beta_0 ~ normal(0, 1);
    z_beta_1 ~ normal(0, 1);
    sigma_r ~ normal(0, 5);

    // Likelihood
    y ~ normal(beta_0 + beta_1 * x, sigma_r);
}
