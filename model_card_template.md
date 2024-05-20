# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model developer: Magdalena Nowak
- Model version: 1.0.0
- Model type: Random Forest Classifier (n_estimators=50, max_depth=10, min_samples_leaf=5)

## Intended Use
Model predicts whether an individual's income is >50K (1) or <=50K (0).

## Training Data
Model data was publicly available Census Bureau data downloaded from the project starter provided at https://github.com/udacity/nd0821-c3-starter-code. Details about the data can be found here: https://archive.ics.uci.edu/dataset/20/census+income

Model was trained on 80% of data.

## Evaluation Data
Model was evaluated on 20% of data.

## Metrics
Test set precision: 0.7786259541984732
Test set recall: 0.5264516129032258
Test set fbeta: 0.6281755196304849

## Ethical Considerations
Model is trained on sensitive information regarding individual's race, sex, relationship status.  

## Caveats and Recommendations
Given the details on model performance for slices in sensitive categories like race and sex model should be further modified to limit bias. 
