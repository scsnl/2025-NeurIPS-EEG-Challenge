"""supervised_linear_grid_search/{job_id} -> top K ->  supervised_linear_grid_search/{job_id}__top{K}
"""


from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from eegchallenge import config


K = 10
FROM = config.arrays_dir(1) / 'supervised_linear_grid_search'


def save_top_k(job_ids, re_search=True):
    to = FROM / f"{'_'.join(job_ids)}__top{K}"
    to.mkdir(exist_ok=True)
    print(f'Saving top {K} to:', to)
    estimators = []
    scores = []
    if re_search:
        file_paths = []
        for job_id in job_ids:
            for i in tqdm([i for i in (FROM/job_id).iterdir() if i.is_file()]):
                result = joblib.load(i)
                # try:
                #     result = joblib.load(i)
                # except (ValueError,EOFError):
                #     continue
                estimators.append(result.best_estimator_)
                scores.append(result.best_score_)
                file_paths.append(i)
        pd.DataFrame({'scores':scores, 'file_paths':file_paths}).to_csv(to/'summary_all.csv')
    else:
        df = pd.read_csv(to/'summary_all.csv',index_col=0)
        assert df.shape[0]==np.sum(
            [len([i for i in (FROM/job_id).iterdir() if i.is_file()]) for job_id in job_ids]
        )
        df = df.sort_values(by=['scores'], ascending=False)
        for i in df.index[:K]:
            result = joblib.load(df.loc[i,'file_paths'])
            estimators.append(result.best_estimator_)
            scores.append(result.best_score_)
            assert result.best_score_==df.loc[i,'scores']
    saved_names = []
    saved_scores = []
    saved_ranks = []
    for rank,model_idx in enumerate(np.argsort(scores)[-K:][::-1]):
        model = estimators[model_idx]
        score = scores[model_idx]
        name = f'challenge1_rank{rank+1}'
        joblib.dump(model, to/f"{name}.pkl")
        saved_ranks.append(rank+1)
        saved_names.append(name)
        saved_scores.append(score)
    pd.DataFrame({'saved_ranks':saved_ranks, 'saved_names':saved_names, 'saved_scores':saved_scores}).to_csv(to/'summary.csv')


def main():
    # save_top_k(['8865913','8867010'])
    save_top_k(['8865913'])
    save_top_k(['8867010',])


if __name__=='__main__':
    main()
