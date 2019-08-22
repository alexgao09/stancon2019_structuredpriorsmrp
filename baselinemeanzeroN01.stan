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
}
transformed parameters { 
  vector[N_groups_age] U_age_transformed;
  vector[N_groups_income] U_income_transformed;
  vector[N] yhat;

  U_age_transformed = sigma_age * U_age; // the random effect for age
  U_income_transformed = sigma_income * U_income; // the random effect for income

  for (i in 1:N) {
    yhat[i] = intercept + U_age_transformed[age[i]] + U_income_transformed[income[i]]; // the linear predictor at each point
  }
  
}
model {
  sigma_age ~ normal(0, 1);
  sigma_income ~ normal(0, 1);
  U_age ~ normal(0, 1); // random effect for age is normal 
  U_income ~ normal(0, 1); // random effect for income is normal
  intercept ~ normal(0, 1); 
  
  
  for (i in 1:N) {
    y[i] ~ bernoulli(inv_logit(yhat[i])); // the response
  }
  
}
