#!/bin/bash

grep -vF \
  -e "PyEMD/CEEMDAN.py:62: SyntaxWarning: invalid escape sequence '\e'" \
  -e "PyEMD/EEMD.py:40: SyntaxWarning: invalid escape sequence '\h'" \
  -e "Standard deviation of Gaussian noise" \
  -e "Scale for added noise" \
  -e 'UserWarning: Persisting input arguments took 0.61s to run.If this happens often in your code, it can cause performance problems (results will be correct in all cases). The reason for this is probably some large input arguments for a wrapped function.' \
  -e 'return self._cached_call(args, kwargs, shelving=False)[0]' \
  -e 'DUE TO PREEMPTION ***' \
  -e 'DUE TO TIME LIMIT ***' \
  -e 'DUE TO JOB REQUEUE ***' \
  -- *err

# grep -vF \
#   -e 'RuntimeWarning: invalid value encountered in divide' \
#   -e 'RuntimeWarning: divide by zero encountered in divide' \
#   -e 'return np.clip((X-X_med)/(X_mad/0.6745), -5, 5)' \
#   -e "PyEMD/CEEMDAN.py:62: SyntaxWarning: invalid escape sequence '\e'" \
#   -e "PyEMD/EEMD.py:40: SyntaxWarning: invalid escape sequence '\h'" \
#   -e "Standard deviation of Gaussian noise" \
#   -e "Scale for added noise" \
#   -e 'c /= stddev[None, :]' \
#   -e 'c /= stddev[:, None]' \
#   -- *err