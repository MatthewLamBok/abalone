

import h2o
from h2o.automl import H2OAutoML
from info import normalization
import pandas as pd


# 1. Initialize H2O:
h2o.init()

# 2. Load data into H2O:
df = pd.read_csv('./data.csv')

normalization_type = 'MinMaxScaler'
df = normalization(df, normalization_type)
df_h2o = h2o.H2OFrame(df)

# 3. Split data into train, validation, and test set:
train, valid, test = df_h2o.split_frame(ratios=[0.7, 0.15], seed=42)

# 4. Specify predictors and response:
predictors = df_h2o.columns[1:]  # Assuming 'Sex' is the last column
response = "Sex"
print(predictors)

# 5. Run H2O AutoML:
aml = H2OAutoML(max_models=30, 
                max_runtime_secs=120,  
                seed=42)
aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)



# 6. View results and make predictions:
lb = aml.leaderboard
print(lb)

preds = aml.leader.predict(test)

# Display AUC and other metrics for the leader model on the test set
perf_test = aml.leader.model_performance(test)
print("AUC on test data:", perf_test.auc())
print("Confusion matrix on test data:\n", perf_test.confusion_matrix())

# 7. Convert predictions back to original format:
df_preds = h2o.as_list(preds)
print(df_preds.head())


# Save the model
model_path = h2o.save_model(model=aml.leader, path="./model_dir", force=True)
print(f"Model saved to: {model_path}")


# Save the model's performance metrics to a text file
with open("./model_dir/model_info.txt", "w") as file:
    # Save cross-validation metrics summary
    file.write("Cross Validation Metrics Summary:\n")
    file.write(str(aml.leader.cross_validation_metrics_summary()))
    file.write("\n\n")
    
    # Save performance on the validation set
    file.write("Performance on Validation Set:\n")
    file.write(str(aml.leader.model_performance(valid)))
    file.write("\n\n")
    
    # Save performance on the test set
    file.write("Performance on Test Set:\n")
    file.write(str(aml.leader.model_performance(test)))
    file.write("\n\n")


h2o.cluster().shutdown()


