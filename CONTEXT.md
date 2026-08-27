# IEEE-CIS Fraud Detection

The deployable fraud-detection service around the LightGBM model, and the retraining / monitoring loop that keeps it current.

## Model lifecycle

**Champion**:
The model version currently served in production.
_Avoid_: production model, live model

**Challenger**:
A newly retrained model version pending evaluation against the champion.
_Avoid_: candidate model, new model

**Promotion**:
The act of making a challenger the production model; automatic when it beats the champion by a safe margin, otherwise gated for review.
_Avoid_: deployment

**Deployment**:
Making a promoted model runnable in the serving stack.
_Avoid_: promotion, shipping

**Retraining**:
Re-fitting a model from its registered hyperparameters when the retraining trigger fires.
_Avoid_: training, fine-tuning

**Operating threshold**:
The decision cut-off on the fraud score, chosen on the test set and stored with the model.
_Avoid_: cut-off, decision boundary

**Retraining corpus**:
The data a challenger is fit on: all historical training data plus the accumulated scored stream whose labels have been revealed.
_Avoid_: training data, dataset

**Retraining trigger**:
The event that starts a retraining pass: accumulated scored volume since the last retrain (default ~5,000) OR a drift alarm.
_Avoid_: retrain condition, retrain signal

**Served model**:
The model artifact serving surfaces actually load (`models/serving/champion_model`, published on promotion); falls back to the committed seed before the first promotion.
_Avoid_: live model, production artifact

**Registry stage**:
The MLflow lifecycle stage of a model version (`Staging` for a challenger awaiting review, `Production` for the champion, `Archived` for superseded versions).
_Avoid_: tag, status

## Serving

**Feature contract**:
The exact 218 columns and their dtypes (9 categoricals as `category`) a model was trained on; the API rejects any payload that deviates.
_Avoid_: schema, input format

**Real-time scoring**:
Scoring a single transaction through the API.
_Avoid_: online inference, live prediction

**Batch scoring**:
Scoring a CSV of transactions and writing scores that feed drift monitoring.
_Avoid_: offline prediction

**Stream simulator**:
The flow that replays the production stream through the API at accelerated cadence to drive the live demo.
_Avoid_: load test, replay job

## Data

**Test set**:
The 15% chronological slice used for offline evaluation: operating-threshold selection and champion-vs-challenger comparison.
_Avoid_: validation set, val

**Production stream**:
The 15% chronological slice held out of training and test, replayed to the serving stack to simulate live production traffic and to drive drift monitoring.
_Avoid_: stream, live data, online data

**Drift window**:
The (reference, current) pair compared by monitoring — the training distribution vs the scored production-stream distribution.
_Avoid_: drift

**Drift current-window store**:
The append-only log every batch-scored transaction lands in (`data/monitoring/current_window.csv`), the honest data source monitoring time-slices into the "current" half of the drift window.
_Avoid_: score log, monitoring store

**Label reveal**:
The point at which a scored production transaction's true outcome becomes known and it may join the retraining corpus.
_Avoid_: labeling, ground-truth arrival

**Reveal lag**:
The simulated delay between a production transaction being scored and its label being usable for retraining; transactions newer than the lag remain label-free.
_Avoid_: adjudication delay
