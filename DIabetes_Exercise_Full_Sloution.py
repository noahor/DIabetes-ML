import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt  
import keras
from keras.models import Sequential
from keras.layers import *
from keras.models import load_model 
from keras import optimizers
from keras.optimizers import Adam
import time


# 1. Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(

    
    log_dir='C:\\logs\\Diabetes_dataset_Ex_logs',
    write_graph=True,
    histogram_freq=5
)

# 2. Load Diabetes dataset Diabetes_dataset.csv
df = pd.read_csv("Diabetes_dataset.csv")

# 3. Try To find Corelated   features 
corr = df.corr()    # data frame correlation function
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(corr)   # color code the rectangles by correlation value
plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
#plt.show() 

# 4. Delete  Corelated Colume 
del df['skin']
diabetes_map = {True : 1, False : 0}
df['Diabetes'] = df['Diabetes'].map(diabetes_map)


# 5. Create X (Features ) and Y (Colume to predict)
feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
predicted_class_names = ['Diabetes']
X = df[feature_col_names].values     # predictor feature columns (8 X m)
y = df[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)

# 6. Split your data using train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 


# 7. Scale your data using MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_X = scaler.fit_transform(X_train)
scaled_testing_X = scaler.transform(X_test)

#scaled_training_df = pd.DataFrame(scaled_training_X, columns=feature_col_names)
#scaled_training_df.to_csv("C:\\Users\\oron.noah\\Documents\\Oronnew.csv")

# 8. Create your Sequential Model and Add Dense layers 
model = Sequential()
model.add(Dense(12,input_dim =8 , activation ='relu'))
model.add(Dense(12,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

# 9. Compile your Model 
model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error', metrics=['accuracy'])  

# 10. Fit your model with your training X and Y
model.fit(scaled_training_X,y_train,epochs=100,verbose=2,callbacks=[logger])


# 11. evaluate your model with your testing X and Y
loss_and_accuracy = model.evaluate(scaled_testing_X, y_test, verbose=2)
print("The mean squared error (MSE) for the test data set is: {}".format(loss_and_accuracy))

loss,accuracy = loss_and_accuracy
print('loss: %.2f' % (loss))
print('Accuracy: %.2f' % (accuracy*100))

# 12. Load date to predict
df = pd.read_csv("DataToPredict.csv")

# 13. Scale your date to predict using the same MinMaxScaler from step 7.
scaled_X_predict = scaler.transform(df)

#scaled_training_df = pd.DataFrame(scaled_X_predict, columns=feature_col_names)
#scaled_training_df.to_csv("C:\\Users\\oron.noah\\Documents\\Oronnew_prediect.csv")

# 13. predict your model , you can use model.predict_classes to get 0 or 1 result. 
result = model.predict(scaled_X_predict)
print(result[0])

result = model.predict_classes(scaled_X_predict)
print(result[0])
time.sleep(10)
print("finish")