functions {
 // bijective softmax transformation
 vector softmax_id(vector alpha) {
   vector[num_elements(alpha) + 1] alphac1;
   for (k in 1:num_elements(alpha))
     alphac1[k] <- alpha[k];
   alphac1[num_elements(alphac1)] <- 0;
   return softmax(alphac1);
  }
}

data {
  int<lower=1> N_timepoints; // number of time points with measurements
  int<lower=0> N_timepoints_i; // number of time points without measurements
  int<lower=2> N_OTUs; // number of species

  // parameters of the priors
  real<lower=0> eta_sq_a;
  real<lower=0> eta_sq_b;
  real<lower=0> inv_rho_sq_a;
  real<lower=0> inv_rho_sq_b;
  real<lower=0> sigma_sq_a;
  real<lower=0> sigma_sq_b;
  real<lower=0> beta_a;
  real<lower=0> beta_b;

  real<lower=0> timepoints[N_timepoints]; // time points with measurements
  real<lower=0> timepoints_i[N_timepoints_i]; // time points without measurements
  int<lower=0> OTU_reads[N_timepoints,N_OTUs]; // observed read counts
}

transformed data {
  vector[N_timepoints+N_timepoints_i] timepoints_full;
  vector[N_timepoints+N_timepoints_i] mu_G;
  matrix[N_timepoints+N_timepoints_i,N_timepoints+N_timepoints_i] distances;
  matrix[N_timepoints+N_timepoints_i,N_timepoints+N_timepoints_i] diag_K;
  real jitter;

  jitter <- 1e-4;

  for (timepoint in 1:(N_timepoints+N_timepoints_i))
    mu_G[timepoint] <- 0;

  // concatenate time points vectors with and without measurements
  for (timepoint in 1:N_timepoints)
    timepoints_full[timepoint] <- timepoints[timepoint];
  for (timepoint in 1:N_timepoints_i)
    timepoints_full[N_timepoints+timepoint] <- timepoints_i[timepoint];

  // precalculate negative squared distances between time points
  for (t1 in 1:(N_timepoints+N_timepoints_i)) {
    distances[t1,t1] <- 0;
	diag_K[t1,t1] <- 1;
    for (t2 in (t1+1):(N_timepoints+N_timepoints_i)) {
      distances[t1,t2] <- -(timepoints_full[t1] - timepoints_full[t2])^2;
      distances[t2,t1] <- distances[t1,t2];
      diag_K[t1,t2] <- 0;
      diag_K[t2,t1] <- diag_K[t1,t2];
    }
  }
}

parameters {
  matrix[N_OTUs-1,N_timepoints] G_d; // with measurements
  matrix[N_OTUs-1,N_timepoints_i] G_i; // without measurements
  matrix[N_OTUs-1,N_timepoints] F;
  vector<lower=0,upper=1>[N_OTUs] Beta[N_timepoints];
  real<lower=0> eta_sq[N_OTUs-1];
  real<lower=0> inv_rho_sq[N_OTUs-1];
  real<lower=0> sigma_sq[N_OTUs-1];
}

transformed parameters {
  real<lower=0> rho_sq[N_OTUs-1];

  # rho_sq from inv_rho_sq, inv_rho_sq_a, and inv_rho_sq_b 
  for (otu in 1:(N_OTUs-1))
    rho_sq[otu] <- inv(inv_rho_sq_a+inv_rho_sq[otu]*inv_rho_sq_b); 
}

model {
  matrix[N_timepoints+N_timepoints_i,N_timepoints+N_timepoints_i] K[N_OTUs-1];
  matrix[N_OTUs-1,N_timepoints+N_timepoints_i] G;
  vector[N_OTUs] Theta[N_timepoints];

  # eta_sq and inv_rho_sq
  eta_sq ~ gamma(eta_sq_a,eta_sq_b);
  inv_rho_sq ~ normal(0,1);

  # F, sigma_sq, and G
  sigma_sq ~ normal(sigma_sq_a,sigma_sq_b);
  # concatenate G_d and G_i
  G <- append_col(G_d,G_i);
  for (otu in 1:(N_OTUs-1)) {
    // calculate covariance matrices given eta_sq, rho_sq
    K[otu] <- eta_sq[otu] * exp(distances * rho_sq[otu]);
    for (timepoint in 1:(N_timepoints+N_timepoints_i))
      K[otu,timepoint,timepoint] <- K[otu,timepoint,timepoint] + jitter;

    G[otu] ~ multi_normal(mu_G,K[otu]);
    F[otu] ~ multi_normal_cholesky(G_d[otu],diag_matrix(rep_vector(sigma_sq[otu],N_timepoints)));
  }

  # beta, Theta_G, and likelihood
  for (timepoint in 1:N_timepoints) {
    Beta[timepoint] ~ beta(beta_a,beta_b);
    Theta[timepoint] <- softmax_id(col(F,timepoint));
    OTU_reads[timepoint] ~ multinomial((Theta[timepoint] .* Beta[timepoint]) / dot_product(Theta[timepoint],Beta[timepoint]));
  }
}

generated quantities {
  simplex[N_OTUs] Theta_G[N_timepoints];
  simplex[N_OTUs] Theta_G_i[N_timepoints_i];
  // Theta_G samples at time points with and without measurements
  for (timepoint in 1:N_timepoints)
    Theta_G[timepoint] <- softmax_id(col(G_d,timepoint));
  for (timepoint in 1:N_timepoints_i)
    Theta_G_i[timepoint] <- softmax_id(col(G_i,timepoint));
}
