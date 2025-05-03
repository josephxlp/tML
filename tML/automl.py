import os
import json
import time 
import logging
import pandas as pd
from autogluon.tabular import TabularPredictor
# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def agluon_pipeline(expname: str, agluon_dir: str, presents: list, problem_type: str,
                 eval_metric: str, verbosity: int, sample_weight: str,
                 train_data=None, valid_data=None, test=None, fcols=None, tcol=None,
                 time_limit=180, num_bag_folds=5, num_bag_sets=1, num_stack_levels=1,
                 calibrate_decision_threshold=False):

    ti = time.perf_counter()
    def supports_calibration(problem_type, eval_metric):
        return problem_type == 'binary' and eval_metric in ['roc_auc', 'log_loss', 'accuracy']

    for present in presents:
        sub_dir = os.path.join(agluon_dir, f'{str(present)}_{sample_weight}_{str(time_limit)}')
        os.makedirs(sub_dir, exist_ok=True)
        logging.info(f"Running experiment in {sub_dir} with preset: {present}")

        predictor = TabularPredictor(
            label=tcol,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=sub_dir,
            verbosity=verbosity,
            sample_weight=sample_weight
        )

        fit_args = {
            "train_data": train_data,
            "time_limit": time_limit,
            "presets": present
        }

        if expname == 'ag_tun':
            fit_args.update({
                "num_bag_folds": num_bag_folds,
                "num_bag_sets": num_bag_sets,
                "num_stack_levels": num_stack_levels
            })

        predictor.fit(**fit_args)
        # Save evaluation
        evaluation = predictor.evaluate(valid_data)
        with open(os.path.join(sub_dir, 'evaluation.json'), 'w') as f:
            json.dump(evaluation, f, indent=4)

        # Save leaderboard
        lboard = predictor.leaderboard(valid_data, silent=True)
        lboard.to_csv(os.path.join(sub_dir, 'leaderboard.csv'), index=False)

        # Save feature importance
        feat_imp = predictor.feature_importance(valid_data)
        feat_imp.to_csv(os.path.join(sub_dir, 'feature_importance.csv'))

        # Optional: Calibrate decision threshold
        calibration_data = {}
        if expname == 'ag_tun' and calibrate_decision_threshold and supports_calibration(problem_type, eval_metric):
            logging.info(f"Before calibration: threshold={predictor.decision_threshold}")
            scores_before = predictor.evaluate(valid_data)
            calibration_data['before'] = scores_before

            threshold = predictor.calibrate_decision_threshold()
            predictor.set_decision_threshold(threshold)

            logging.info(f"After calibration: threshold={predictor.decision_threshold}")
            scores_after = predictor.evaluate(valid_data)
            calibration_data['after'] = scores_after

            with open(os.path.join(sub_dir, 'calibration_scores.json'), 'w') as f:
                json.dump(calibration_data, f, indent=4)

        # Predict test set
        prob_output = predictor.predict_proba(test[fcols], as_multiclass=False)
        pred_output = predictor.predict(test[fcols])

        fname = f'{present[0]}_{expname}'
        pd.DataFrame({'Id': test.iloc[:, 0], 'Probability': prob_output}).to_csv(
            os.path.join(sub_dir, f'{fname}_prob.csv'), index=False)
        pd.DataFrame({'Id': test.iloc[:, 0], 'Prediction': pred_output}).to_csv(
            os.path.join(sub_dir, f'{fname}_pred.csv'), index=False)
    tf = time.perf_counter() - ti

    days = int(tf // (24 * 3600))
    tf = tf % (24 * 3600)
    hours = int(tf // 3600)
    tf %= 3600
    minutes = int(tf // 60)
    seconds = int(tf % 60)
    logging.info(f"Total time taken: {days}d {hours}h {minutes}m {seconds}s")

