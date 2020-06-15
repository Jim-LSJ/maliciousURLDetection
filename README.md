# Malicious URL Detection

### Dataset
https://drive.google.com/file/d/1qV8a4jyffEV-GYdensWTZ1jHa1HejAAU/view?usp=sharing

### Execution
- 44 features: `python3 train_ntust44_lightgbm.py`
- 60 features: `python3 train_mix60_lightgbm.py`


### Result
> <b>Result</b>
>> <b>ntust44.csv</b> -> Record the Precision, Recall, and F1 Score
>> <b>ntust44_feature_rank_gain.csv</b> -> Record the feature importance evaluated by information gain
>> <b>ntust44_feature_rank_split.csv</b> -> Record the feature importance evaluated by split node
>> <b>ntust44_loss.png</b> -> Record the training and validation loss
>> <b>mix60.csv</b>
>> <b>mix60_feature_rank_gain.csv</b>
>> <b>mix60_feature_rank_split.csv</b>
>> <b>mix60_loss.png</b>

> Feature_importance
>> <b>ntust44_gain_<valid_number>.png</b> -> Visualize the feature importance of the <valid_number> validation
>> <b>ntust44_split_<valid_number>.png</b> -> Visualize the feature importance of the <valid_number> validation
>> <b>mix60_gain_<valid_number>.png</b>
>> <b>mix60_split_<valid_number>.png</b>
