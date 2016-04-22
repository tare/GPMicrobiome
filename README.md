# GPMicrobiome

## Prerequisites 
* Python 2.7 (https://www.python.org/)
* PyStan (http://pystan.readthedocs.org/en/latest/)
* NumPy (http://www.numpy.org/)

For more information on Stan and PyStan, please see the documentation at http://mc-stan.org/interfaces/pystan.html.

## Command line interface

### Usage
The correct command line usage of the program is summarized by the following usage message  (```python gpmicrobiome.py --help ```)
```
usage: gpmicrobiome.py [-h] -t TIME_POINTS [-p TIME_POINTS_I] -d COUNT_DATA -o OUTPUT_FILE [-v]

GPMicrobiome

optional arguments: 
  -h, --help                                   show this help message and exit
  -t TIME_POINTS, --time TIME_POINTS           file containing time points of measurements (required)
  -p TIME_POINTS_I, --prediction TIME_POINTS_I file containing prediction time points (optional)
  -d COUNT_DATA, --data COUNT_DATA             file containing read counts (required)
  -o OUTPUT_FILE, --output OUTPUT_FILE         output file for pickling posterior samples (required)
  -v, --version                                show program's version number and exit
```

The user has to supply either two or three input data files and one output file. The two mandatory input data files have measurement time points (in days) and read counts for each species at every time point. The optional input data file contains time points for predictions (interpolation/extrapolation). The obtained posterior samples are written to the output file (existing file is overwritten).

The formats of the input files are explained below.

### Input data format
For demonstration purposes, let us assume that the names of the input files are **timepoints.tsv**, **prediction_timepoints.tsv**, and **data.tsv**.
The file containing measurement time points (*timepoints.tsv*) should have *T* lines where each line has one value representing measurement time point (in days). For instance, if there are seven measurements, which are taken daily, then
```
$ cat timepoints.tsv 
0
1
2
3
4
5
6
```

Additionally, for the sake of simplicity, let us assume that there are three (*M*=3) species. Then the file **data.tsv** containing read counts should have *M* lines and *T* tab-separated values per line
```
$ cat data.tsv 
9421  11123 10032 12132 76321 10923 8023
33134 31203 24103 26190 29893 35023 32310
62310 61032 57904 0 61203 60231 62031
```
Note that the order of columns in **data.tsv** should match the order of measurement time points specified in **timepoints.tsv**. 

The optional input file **prediction_timepoints.tsv** has the same format as **timepoints.tsv**. For instance, if the goal is to predict compositions at 4.5 and 9 days, then
```
$ cat prediction_timepoints.tsv 
4.5
9
```

### Sampling
If the goal is to estimate the underlying compositions at measurement time points without producing predictions, then the following command should be executed
```
python gpmicrobiome.py -t timepoints.tsv -d counts.tsv -o samples.p
```
Whereas, if the goal is also to produce predictions, then the following command should be executed
```
python gpmicrobiome.py -t timepoints.tsv -p prediction_timepoints.tsv -d counts.tsv -o samples.p
```
In both cases, **samples.p** will contain measurement time points, prediction time points, and posterior samples.

### Output handling
The output file **samples.p** can be read in Python as follows  
```python
import pickle
T,T_p,samples = pickle.load(open('samples.p','rb'))
```
The posterior means of Thetas can be printed as
```python
print samples['Theta_G'].mean(0).T
if samples.has_key('Theta_G_i'):
  print samples['Theta_G_i'].mean(0).T
```
Note that the *if* statement is used to check whether predictions were made. The orders of rows and columns correspond the orders of **timepoints.tsv**, **prediction_timepoints.tsv**, and **data.tsv**.

## Application programming interface
In addition to the command line interface, GPMicrobiome can be used directly from Python.

Assume that the user has data in numpy arrays **T** (1D array containing measurement time points), **T_p** (1D array containing prediction time points, empty array corresponds to the prediction-free case), and **counts** (2D array containing counts so that rows and columns represent species and time points, respectively). Then the sampling procedure can be done as follows
```python
from gpmicrobiome import stan_init_data, get_samples 
init, data = stan_init_data(X,T,T_p)
samples = get_samples('gpmicrobiome.stan',data,init)
```
