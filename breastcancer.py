import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn import metrics
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def PreprocessData(df):
    #preprocessing
    X = df.drop('benign_0__mal_1',axis=1)
    y = df['benign_0__mal_1']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    scaler=MinMaxScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train,X_test,y_train,y_test

def BuildModel(X_train,y_train):
    # Building the model
    model=Sequential()
    model.add(Dense(30,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy')
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    model.fit(x=X_train,y=y_train,
              validation_data=(X_test,y_test),
              epochs=500,callbacks=[early_stop])
    pd.DataFrame(model.history.history).plot()
    plt.draw()
    model.save('breastcancer_15N_25EpPat_0.5D.h5')
    return model


def EvaluateModel(X_train, X_test, y_train, y_test,model):
    #Evaluating the model with new data
    # pred=model.predict_classes(X_test) Old method about to depreciate
    pred=pd.DataFrame((model.predict(X_test) > 0.5).astype("int32")) #new one for bin classification
    #np.argmax(model.predict(x), axis=-1) #new one for multiclass classification
    print(metrics.classification_report(y_test,pred))
    print(metrics.confusion_matrix(y_test,pred))

def main():
    df=pd.read_csv('./cancer_classification.csv')
    #Loading the model
    X_train,X_test,y_train,y_test=PreprocessData(df)
    # model=BuildModel(X_train,y_train)
    model=load_model('breastcancer_15N_25EpPat_0.5D.h5')
    EvaluateModel(X_train,X_test,y_train,y_test,model)
    plt.show()

if __name__ == "__main__":
    main()
