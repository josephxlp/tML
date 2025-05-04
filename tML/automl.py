import os
import json
import time 
import logging
import pandas as pd
from autogluon.tabular import TabularPredictor
from supervised.automl import AutoML


def agluon_pipeline(expname: str, agluon_dir: str, presents: list, problem_type: str,
                 eval_metric: str, verbosity: int, sample_weight: str,
                 train_data=None, valid_data=None, test=None, fcols=None, tcol=None,
                 time_limit=180, num_bag_folds=5, num_bag_sets=1, num_stack_levels=1,
                 calibrate_decision_threshold=False):
    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


import os
import json
import time
import logging
import pandas as pd
from supervised.automl import AutoML

def amljar_pipeline(expname: str, save_dir: str, modes: list, task: str, eval_metric: str,
                    total_time_limit: int, train=None, test=None, fcols=None, tcol=None,
                    features_selection: bool = False):

    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ti = time.perf_counter()

    for mode in modes:
        logging.info(f'🔧 Starting AutoML experiment with mode={mode}')

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

        logging.info(f'⚙️  Fitting AutoML model...')
        automl.fit(train[fcols], train[tcol])

        fname = f'{mode}_{expname}'
        id_col = test.iloc[:, 0]

        # Save predictions
        if task in ['binary_classification', 'multiclass_classification']:
            logging.info('📈 Predicting probabilities...')
            preds_proba = automl.predict_proba(test[fcols])
            proba_path = os.path.join(sub_dir, f'{fname}_prob.csv')

            if task == 'binary_classification':
                submission_prob = pd.DataFrame({'Id': id_col, 'Probability': preds_proba[:, 1]})
            else:  # multiclass
                class_labels = automl.classes_
                submission_prob = pd.DataFrame(preds_proba, columns=class_labels)
                submission_prob.insert(0, 'Id', id_col)

            submission_prob.to_csv(proba_path, index=False)
            logging.info(f'✅ Probabilities saved to {proba_path}')

            logging.info('📌 Predicting labels...')
            preds = automl.predict(test[fcols])
            pred_path = os.path.join(sub_dir, f'{fname}_pred.csv')
            pd.DataFrame({'Id': id_col, 'Prediction': preds}).to_csv(pred_path, index=False)
            logging.info(f'✅ Predictions saved to {pred_path}')

        elif task == 'regression':
            logging.info('📌 Predicting regression values...')
            preds = automl.predict(test[fcols])
            pred_path = os.path.join(sub_dir, f'{fname}_pred.csv')
            pd.DataFrame({'Id': id_col, 'Prediction': preds}).to_csv(pred_path, index=False)
            logging.info(f'✅ Predictions saved to {pred_path}')
        else:
            raise ValueError(f"❌ Unsupported ml_task: {task}")

        # Save report
        try:
            report_path = os.path.join(sub_dir, f'{fname}_report.json')
            with open(report_path, 'w') as f:
                json.dump(automl.report(), f, indent=4)
            logging.info(f'📄 Report saved to {report_path}')
        except Exception as e:
            logging.warning(f"⚠️  Could not save report: {e}")

        # Save leaderboard
        try:
            leaderboard_df = automl.report()["models"]
            leaderboard_path = os.path.join(sub_dir, f'{fname}_leaderboard.csv')
            pd.DataFrame(leaderboard_df).to_csv(leaderboard_path, index=False)
            logging.info(f'🏅 Leaderboard saved to {leaderboard_path}')
        except Exception as e:
            logging.warning(f"⚠️  Could not save leaderboard: {e}")

        # Save feature importance if available
        try:
            feat_imp = automl.feature_importance()
            featimp_path = os.path.join(sub_dir, f'{fname}_feature_importance.csv')
            feat_imp.to_csv(featimp_path, index=False)
            logging.info(f'📊 Feature importance saved to {featimp_path}')
        except Exception as e:
            logging.warning(f"⚠️  Feature importance not available: {e}")

        # Save model
        try:
            model_dir = os.path.join(sub_dir, 'model')
            os.makedirs(model_dir, exist_ok=True)
            automl.save(model_dir)
            logging.info(f'💾 Model saved to {model_dir}')
        except Exception as e:
            logging.warning(f"⚠️  Could not save model: {e}")

    # Time summary
    tf = time.perf_counter() - ti
    days = int(tf // (24 * 3600))
    tf %= 24 * 3600
    hours = int(tf // 3600)
    tf %= 3600
    minutes = int(tf // 60)
    seconds = int(tf % 60)
    logging.info(f"⏱️  Total time taken: {days}d {hours}h {minutes}m {seconds}s")
