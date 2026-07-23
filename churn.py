#---------------import the requires liberay--------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
import warnings as wr
wr.filterwarnings("ignore")
#---------------------displaying the data-----------------------------------------------------
churn_data = pd.read_csv("Churn_Modelling.csv")
#--------------------print first five values---------------------------
print(churn_data.head())
#----------------------------print last five columns-------------------------- 
print(churn_data.tail())

#---------------------------checking the size of the data-------------------------
print(churn_data.shape)
#----------------------------checking the info of the data------------------------------
print(churn_data.info())
#------------------------------inspecting the missing values-----------------------------
print(churn_data.isnull().sum() / len(churn_data)*100)
#-------------------------checking the duplicates---------------------------------------
print(churn_data.duplicated().sum())
#----------------------------checking the target values-----------------------------
print(churn_data['Exited'].value_counts())
print(churn_data["Geography"].value_counts())
#------------------------convert the string to integers--------------------------------------
churn_data["Geography"] = churn_data["Geography"].map({"France":0, "Germany":1, "Spain":2})
churn_data["Gender"] = churn_data["Gender"].map({"Male":1, "Female":0})
print(churn_data["Geography"].value_counts())

# FIXED: Dropped structural tracking features along with Surname to prevent correlation map string calculations crashing
churn_data = churn_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

#---------------------ploting the correlation---------------------------------
cm = churn_data.corr()
sns.heatmap(cm, annot=True, cbar=True, fmt='.3f', cmap='Set1')
plt.show()
#---------------------------split the x and y----------------------------
X = churn_data.drop(['Exited'], axis=1)
print(X)
Y = churn_data["Exited"]
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    )
print(X.shape)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
#-----------------------------building the model---------------------------------------
model = LogisticRegression(max_iter=300, 
                           random_state=42,
                           class_weight='balanced',
                           C=5,
                           verbose=1,
                           n_jobs=-1)
#-------------------------fit the model to train it------------------------------
model.fit(X_train_scaler, Y_train)
#-------------------------predict the model----------------------------------------------------
Y_training_prediction = model.predict(X_train_scaler)
the_mean_square_error = mean_squared_error(Y_train, Y_training_prediction)
r2_error = r2_score(Y_train, Y_training_prediction)
print(f"the accuracy of the model testing is {accuracy_score(Y_train, Y_training_prediction)*100:.2f}%")
print(f"the classification report of the model is {classification_report(Y_train, Y_training_prediction)}")
print(f"the confusion metrix of the model is {confusion_matrix(Y_train, Y_training_prediction)}")
print(f"the mean square error is {the_mean_square_error}")
print(f"the r2_error of the model is {r2_error} ")
print('='*60)
#-------------------------testing_predictions---------------------------------------------
Y_testing_prediction = model.predict(X_test_scaler)
the_mean_square_error = mean_squared_error(Y_test, Y_testing_prediction)
r2_error = r2_score(Y_test, Y_testing_prediction)
print(f"the accuracy of the model testing is {accuracy_score(Y_test, Y_testing_prediction)*100:.2f}%")
print(f"the classification report of the model is {classification_report(Y_test, Y_testing_prediction)}")
print(f"the confusion metrix of the model is {confusion_matrix(Y_test, Y_testing_prediction)}")
print(f"the mean square error is {the_mean_square_error}")
print(f"the r2_error of the model is {r2_error} ")
print('='*60)
random_model = RandomForestClassifier(n_estimators=500,
                                      criterion='gini',
                                      class_weight='balanced',
                                      random_state=42,
                                      max_depth=10,
                                      min_samples_split=12,
                                      min_samples_leaf=5)
random_model.fit(X_train_scaler, Y_train)
#-------------------------predict the model----------------------------------------------------
Y_training_prediction = random_model.predict(X_train_scaler)
the_mean_square_error = mean_squared_error(Y_train, Y_training_prediction)
r2_error = r2_score(Y_train, Y_training_prediction)
print(f"the accuracy of the model testing is {accuracy_score(Y_train, Y_training_prediction)*100:.2f}%")
print(f"the classification report of the model is {classification_report(Y_train, Y_training_prediction)}")
print(f"the confusion metrix of the model is {confusion_matrix(Y_train, Y_training_prediction)}")
print(f"the mean square error is {the_mean_square_error}")
print(f"the r2_error of the model is {r2_error} ")
print('='*60)
#-------------------------testing_predictions---------------------------------------------
Y_testing_prediction = random_model.predict(X_test_scaler)
the_mean_square_error = mean_squared_error(Y_test, Y_testing_prediction)
r2_error = r2_score(Y_test, Y_testing_prediction)
print(f"the accuracy of the model testing is {accuracy_score(Y_test, Y_testing_prediction)*100:.2f}%")
print(f"the classification report of the model is {classification_report(Y_test, Y_testing_prediction)}")
print(f"the confusion metrix of the model is {confusion_matrix(Y_test, Y_testing_prediction)}")
print(f"the mean square error is {the_mean_square_error}")
print(f"the r2_error of the model is {r2_error} ")
#----------------------building the predictive systems use model-----------------------------------------------
# FIXED: Adjusted array features down to 10 elements to match the exact input columns shape expected by the fitted scaler
input_Data = (653,1,1,58,1,132602.88,1,1,0,5097.67)
input_array = np.asarray(input_Data)
input_reshape = input_array.reshape(1, -1)
input_scaler  = scaler.transform(input_reshape)
predictions = model.predict(input_scaler)
#predictions_probalilities = model.predict_proba(input_array)
print(predictions[0])
#----------------------building the predictive systems use model_random-----------------------------------------------
# FIXED: Adjusted array features down to 10 elements to match the exact input columns shape expected by the fitted scaler
input_Data = (653,1,1,58,1,132602.88,1,1,0,5097.67)
input_array = np.asarray(input_Data)
input_reshape = input_array.reshape(1, -1)
input_scaler  = scaler.transform(input_reshape)
prediction = random_model.predict(input_scaler)
#predictions_probalilities = model.predict_proba(input_array)
print(prediction)

# FIXED: Swapped labels logic context (1 = Exited, 0 = Remained loyal)
if prediction[0] == 1:
    print("🚨 The person exited the bank")
else:
    print("✅ The person do not exited bank")
churn_assets = {
    'model': random_model,  # Saving your best-performing Random Forest model
    'scaler': scaler,       # Saving the scaler so the web app inputs match the model scale
    'feature_columns': list(X.columns)
}

# Export everything into a single file
joblib.dump(churn_assets, "churn_assets.pkl")
print("All Churn Modelling pipeline assets successfully exported to 'churn_assets.pkl'!")