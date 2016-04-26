#!/usr/bin/env python
 
import argparse
import numpy

import pystan
import pickle
import logging
from hashlib import md5

def inv_softmax(Y):
  X = numpy.zeros((Y.shape[0],Y.shape[1]-1))
  for idx in range(0,Y.shape[0]):
    X[idx,:] = numpy.array([numpy.log(row/Y[idx,-1]) for row in Y[idx,0:-1]])
  return X

# adapted from https://pystan.readthedocs.org/en/latest/avoiding_recompilation.html
def stan_cache(model_name,optimization=False,**kwargs):
  f=open(model_name, 'rb')
  model_code=f.read()
  f.close()
  code_hash = md5(model_code.encode('ascii')).hexdigest()
  cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
  try:
    sm = pickle.load(open(cache_fn, 'rb'))
  except:
    sm = pystan.StanModel(file=model_name)
    with open(cache_fn, 'wb') as f:
      pickle.dump(sm, f)
  else:
    logging.info("Using cached StanModel")
  if not optimization:
    return sm.sampling(**kwargs)
  else:
    return sm.optimizing(**kwargs)

def get_samples(stan_code,data,init,iters=1000,chains=1,refresh=1):
  fit = stan_cache(stan_code,optimization=False,data=data,init=[init]*chains,iter=iters,chains=chains,verbose=True,refresh=refresh)
  print fit
  return fit.extract()

def stan_init_data(X,T,T_i=[]):
  # number of species and number of time points
  p, N_timepoints = X.shape[0], X.shape[1]

  # scale time points (given in days)
  T_scaled = T / 380.0
  if len(T_i) > 0:
    T_i_scaled = T_i / 380.0
  else:
    T_i_scaled = []

  # calculate ML estimates for initialization
  X_prop = (X.T.astype(float)+10)/numpy.tile((X.T+10).sum(1),[p,1]).T

  # init dictionary
  init = {'G_d':inv_softmax(X_prop).T,'G_i':numpy.zeros((p-1,len(T_i_scaled))),'F':inv_softmax(X_prop).T,'Beta':0.99*numpy.ones((N_timepoints,p)),'eta_sq':1.0*numpy.ones(p-1),'inv_rho_sq':1.0*numpy.ones(p-1),'sigma_sq':0.1*numpy.ones(p-1)}

  # data dictionary
  data = {'eta_sq_a':1.0,'eta_sq_b':0.5,'inv_rho_sq_a': 0.001,'inv_rho_sq_b':0.005,'sigma_sq_a':0.0,'sigma_sq_b':0.5,'beta_a':0.8,'beta_b': 0.4,'timepoints':T_scaled,'N_timepoints':N_timepoints,'N_OTUs':p,'OTU_reads':numpy.array(X.T).astype(int),'N_timepoints_i':len(T_i_scaled),'timepoints_i':T_i_scaled}

  return init, data

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GPMicrobiome')
  parser.add_argument('-t','--time',action='store',dest='time_points',type=str,required=True,help='file containing time points of measurements (required)')
  parser.add_argument('-p','--prediction',action='store',dest='time_points_i',type=str,required=False,help='file containing prediction time points (optional)')
  parser.add_argument('-d','--data',action='store',dest='count_data',type=str,required=True,help='file containing read counts (required)')
  parser.add_argument('-o','--output',action='store',dest='output_file',type=str,required=True,help='output file for pickling posterior samples (required)')
  parser.add_argument('-v','--version',action='version',version='%(prog)s 0.666')
  options = parser.parse_args()

  # read input
  X = numpy.loadtxt(options.count_data,skiprows=0,delimiter='\t')
  T = numpy.loadtxt(options.time_points,skiprows=0,delimiter='\t')
  if options.time_points_i is not None:
    T_i = numpy.loadtxt(options.time_points_i,skiprows=0,delimiter='\t')
    T_i = numpy.atleast_1d(T_i)
  else:
    T_i = []

  # prepare init and data dictionaries for stan
  init, data = stan_init_data(X,T,T_i)
 
  # get posterior samples using stan
  samples = get_samples('gpmicrobiome.stan',data,init)

  # write samples
  pickle.dump((T,T_i,samples),open(options.output_file,'w'))
