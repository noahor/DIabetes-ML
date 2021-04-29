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


# 1. Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(

    
    log_dir= "c:\logs",
    write_graph=True,
    histogram_freq=5
)

# 2. Load Diabetes dataset Diabetes_dataset.csv

df = pd.read_csv('\\Add Your File Here')
# 3. Try To find Corelated   features 
corr = df.corr()    # data frame correlation function
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(corr)   # color code the rectangles by correlation value
plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
plt.show() 

# 4. Delete  Corelated Colume 


# 5. Create X (Features ) and Y (Colume to predict)

# 6. Split your data using train_test_split function

# 7. Scale your data using MinMaxScaler 

# 8. Create your Sequential Model and Add Dense layers 

# 9. Compile your Model 

# 10. Fit your model with your training X and Y

# 11. evaluate your model with your testing X and Y

# 12. Load date to predict

# 13. Scale your date to predict using the same MinMaxScaler from step 7.


# 13. predict your model , you can use model.predict_classes to get 0 or 1 result. 
