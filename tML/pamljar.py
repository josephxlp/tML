# pipelines/amljar_pipeline.py
import os
import time
import json
import logging
import pandas as pd
from supervised.automl import AutoML
from utils.logging_utils import setup_logging

def amljar_pipeline(save_dir: str, modes: list, task: str, eval_metric: str,
                    total_time_limit: int, features_selection: bool = False,
                    train=None, test=None, fcols=None, tcol=None):

    setup_logging(os.path.join(save_dir, 'amljar'))
    ti = time.perf_counter()

    for mode in modes:
        logging.info(f'Starting AutoML with mode={mode}')

        sub_dir = os.path.join(save_dir, 'amljar', f'{mode}_{total_time_limit}')
        os.makedirs(sub_dir, exist_ok=True)

        automl = AutoML(
            results_path=sub_dir,
            mode=mode,
            ml_task=task,
            eval_metric=eval_metric,
            total_time_limit=total_time_limit,
            features_selection=features_selection
        )

        automl.fit(train[fcols], train[tcol])

        preds = automl.predict_proba(test[fcols])
        pred_labels = automl.predict(test[fcols])

        logging.info('Saving predictions...')
        pd.DataFrame({'Id': test.iloc[:, 0], 'Probability': preds[:, 1]}).to_csv(
            os.path.join(sub_dir, f'{mode}_{total_time_limit}_prob.csv'), index=False)

        pd.DataFrame({'Id': test.iloc[:, 0], 'Prediction': pred_labels}).to_csv(
            os.path.join(sub_dir, f'{mode}_{total_time_limit}_pred.csv'), index=False)

        try:
            with open(os.path.join(sub_dir, 'feature_importance.json'), 'w') as f:
                json.dump(automl.feature_importance(), f, indent=4)
        except Exception as e:
            logging.warning(f"Feature importance not available: {e}")

        try:
            leaderboard = automl.report(return_dataframe=True)
            leaderboard.to_csv(os.path.join(sub_dir, 'leaderboard.csv'), index=False)
        except Exception as e:
            logging.warning(f"Leaderboard not available: {e}")

        try:
            automl.save(os.path.join(sub_dir, 'model_export'))
        except Exception as e:
            logging.warning(f"Model export failed: {e}")

    tf = time.perf_counter() - ti
    logging.info(f"Total time taken: {int(tf // 86400)}d {int((tf % 86400) // 3600)}h {int((tf % 3600) // 60)}m {int(tf % 60)}s")
