# GPMicrobiome

## Prerequisites 
* Python 2.7 (https://www.python.org/)
* PyStan (http://pystan.readthedocs.org/en/latest/)
* NumPy (http://www.numpy.org/)

For more information on Stan and PyStan, please see the documentation at http://mc-stan.org/interfaces/pystan.html.

## Command line interface

### Usage

```shell
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

For instance,

    python gpmicrobiome.py -t time

## Input data
Two examples from the manuscript (**foxp3_time.py** and **foxp3_ra.py**) are provided. The examples can be run as follows

```shell
python foxp3_time.py -c Data/control_data.txt -p Data/control_prior.txt -d Data/data.txt -m Data/design_matrix.txt
python foxp3_ra.py -c Data/control_data.txt -p Data/control_prior.txt -d Data/data.txt -m Data/design_matrix.txt
```

See also `python foxp3_time.py --help` and `python foxp3_ra.py --help`.

## Application programming interface

### Usage

```python
from gpmicrobiome import stan_init_data, get_samples 
init, data = stan_init_data(X,T,T_i)
```
