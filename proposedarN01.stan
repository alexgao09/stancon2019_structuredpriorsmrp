data {
  int<lower=0> N; // number of samples
  int<lower=0> N_groups_age; // the number of groups for age
  int<lower=0> N_groups_income; // the number of groups for income
  int<lower=1,upper=N_groups_age> age[N]; // the column vector of design matrix X for age
  int<lower=1,upper=N_groups_income> income[N]; // the column vector of design matrix X for income
  int y[N]; // the response vector
}
parameters {
  vector[N_groups_age] U_age; // the random effect for age, not multiplied by sigma_age
  vector[N_groups_income] U_income; // the random effect for income, not multiplied by sigma_income
  real<lower=0> sigma_age; // sd of U_age (hyperparam).
  real<lower=0> sigma_income; // sd of U_income (hyperparam).
  real intercept; // the intercept (global fixed effect)
  real<lower=0,upper=1> rho; // the autoregressive coefficient untransformed
}
transformed parameters { 
  vector[N_groups_age] U_age_transformed;
  vector[N_groups_income] U_income_transformed;
  vector[N] yhat;
  real<lower=-1,upper=1> rho_transformed;

  rho_transformed = (rho * 2) - 1; // the autoregressive coefficient

  U_age_transformed = sigma_age * U_age; // the random effect for age
  U_income_transformed = sigma_income * U_income; // the random effect for income

  for (i in 1:N) {
    yhat[i] = intercept + U_age_transformed[age[i]] + U_income_transformed[income[i]]; // the linear predictor at each point
  }
  
}
model {
  sigma_age ~ normal(0, 1); // sigma_A ~ halfnormal(0,1)
  sigma_income ~ normal(0, 1); // sigma_I ~ halfnormal(0,1)
  rho ~ beta(0.5, 0.5); // prior on autoregressive coefficient
  
  U_age[1] ~ normal(0, 1/sqrt(1-rho_transformed^2)); // before it was normal(0, 1) but this is wrong
  for (j in 2:N_groups_age) {
    U_age[j] ~normal(rho_transformed * U_age[j-1],1);
  }
  
  U_income ~ normal(0, 1); // random effect for income is normal
  intercept ~ normal(0, 1); 


  
  for (i in 1:N) {
    y[i] ~ bernoulli(inv_logit(yhat[i])); // the response
  }
  
}
