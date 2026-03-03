from pathlib import Path
from astra.analysis.loader import RunLoader
from astra.constants import PROJECT_ROOT
import pandas as pd


def main():
    entity = "lmse-university-of-toronto"
    project = "astra"
    tags = ['5fcv']
        
    loader = RunLoader(entity, project)
    runs = loader.get_runs(list(tags))

    """    run = next(iter(runs))
    hist = run.history(pandas=True)
    print(hist['_runtime'].unique())"""

    i = 0
    for run in runs:
        #conf = run.config
        #print(conf)

        summary = run.summary
        train_hist = run.history(pandas=True, keys=['train_loss_epoch']).drop(columns=['_step'])
        valid_hist = run.history(pandas=True, keys=['valid_loss_epoch']).drop(columns=['_step'])
        epoch_summary = train_hist.join(valid_hist)

        epoch_summary = epoch_summary.reset_index().rename(columns={'index': 'epoch'})

        arch, mode = loader._derive_metadata(run.config)
        epoch_summary['architecture'] = arch
        epoch_summary['experiment_mode'] = mode

        print(f"{summary}\n")
        print(f"{epoch_summary}\n\n")


        i += 1
        if i > 0:
            break
main()