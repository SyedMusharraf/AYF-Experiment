import pandas as pd
import optuna
df = pd.read_excel('/Users/admin/Desktop/arsh-code/python.py/pandas/AYF_experiments (1) copy.xlsx')
df.drop(columns=['Exp','Formula','γ C2S','C12A7','C2AS','C$'],inplace = True)

targets = ['β C2S',"α' C2S",'C3S','C3A','C4A3$','C4AF','C']
features = []
for col in df.columns:
  if col not in targets:
    features.append(col)

X = df[features]
y = df[targets]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, PReLU,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import tensorflow.keras.backend as K
import tensorflow as tf


def custom(y_true,y_pred):
    weights = tf.constant([1.0,1.0,10.0,1.0,1.0,1.0,1.0])
    sq_diff = tf.square(y_true-y_pred)
    weight_sq = sq_diff*weights
    return tf.reduce_mean(weight_sq)

def objective(trial):
    model = Sequential()
    model.add(Dense(trial.suggest_int('units1',0,64,step=2),
                   activation='relu',input_shape=(X_train.shape[1],)))
   
    if trial.suggest_categorical('add_second_layer',[True,False]):
            model.add(Dense(trial.suggest_int('units2', 0, 64,step=2), activation='relu'))

    model.add(Dense(y_train.shape[1],activation='softplus'))
    # model.add(Lambda(lambda x:K.softplus(x)))

    learning_rate = trial.suggest_float('lr',0.0001,0.1,log= True)
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss = ['mse'], metrics=['mae'])
   

    lr_schedule = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=6,
        validation_data = (X_test,y_test),
        callbacks=[early_stop,lr_schedule],
        verbose=1
    )
    return min(history.history['val_loss'])
study = optuna.create_study(direction='minimize')
study.optimize(objective,n_trials = 30)

print("Best hyperparameters:", study.best_params)


