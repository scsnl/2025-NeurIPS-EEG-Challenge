"""Run the preprocessing pipeline for challenge 1.

Original: scripts/challenges/challenge1/supervised_linear_preprocess.py

Note: the original imported preprocess from challenges.challenge2.supervised_linear,
where it is not defined -- it lives in supervised_linear_preprocess. That import
raised ImportError; corrected here to point at the right module.
"""

from preprocess import preprocess


if __name__=='__main__':
    preprocess(challenge=1)
