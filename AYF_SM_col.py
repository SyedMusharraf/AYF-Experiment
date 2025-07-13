import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("pandas/AYF_experiments (1).xlsx")
df.drop(columns=['Exp', 'Formula'],inplace=True)
df.rename(columns={"α' C2S":"AC2S"},inplace=True)
df.rename(columns={"γ C2S":"YC2S"},inplace=True)
# print(df.columns)
# exit()

targets = [ 'AC2S','C3S','β C2S','YC2S', 'C4AF', 'C12A7', 'C2AS', 'C3A',
        'C4A3$', 'C', 'C$']
# \print(targets)
features = [col for col in df.columns if col not in targets]
# print(features)
# exit()




# exit()
# train_columns = ['Temperature', 'Dwell', 'SO2 ppm', 'CaO', 'Al2O3', 'Fe2O3', 'SiO2',
#        'MgO', 'SO3', 'Na2O', 'K2O']
# target_columns = ['β C2S', "α' C2S", 'γ C2S', 'C3S', 'C3A',
#        'C4A3$', 'C4AF', 'C', 'C12A7', 'C$', 'C2AS']

X = df[features]
y = df[targets]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def tune_model(X_train,y_train):
    model = RandomForestRegressor(
        # n_estimators=172,
        # max_depth =2,
        # min_samples_split=5,
        #below  2s do nothing 
        random_state=42,
        n_jobs=-1
    )
    param_grid  = { "n_estimators": [100,150],
                        "max_depth": [2,4],
                        'min_samples_split': [2, 4]
                    }
    grid = GridSearchCV(model , param_grid,cv=5,n_jobs=-1)
    grid.fit(X_train,y_train)

    return grid.best_estimator_

best_model = tune_model(X_train,y_train)

y_pred = best_model.predict(X_test)

# feature importance

# feature_importance = best_model.feature_importances_
# feature = X.columns

# importance_df = pd.DataFrame({
#     'Feature':feature,
#     'Importance': feature_importance
# }).sort_values(by='Importance',ascending=False)

# plt.figure(figsize=(10,6))
# sns.barplot(x = 'Importance', y = 'Feature', data= importance_df,palette='viridis')
# plt.xlabel('Importance')
# plt.ylabel('features')
# plt.title(f"Features Importance graph of: All the columns")
# plt.tight_layout()
# plt.savefig(f'Single_Features.png')




#single model for each columns 
y_pred = np.array(y_pred)
for i, col in enumerate(targets):
    actual = y_test.iloc[:,i]
    prediction = y_pred[:,i]

    
    r2 = r2_score(actual,prediction)
    mse = mean_squared_error(actual, prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, prediction)

    print(f"\nthe metrics for: {col}")
    print(f"R2 value : {r2:.4f}")
    print(f"Mean_squared_error: {mse:.4f}")
    print(f"Root_Mean_sqrt_Error:{rmse:.4f}")
    print(f"Mean_absolute_Error:{mae:.4f}")


    plt.figure(figsize=[8,8])
    plt.scatter(actual,prediction,alpha=0.6,color="purple")
    plt.plot([actual.min(),actual.max()],[actual.min(),actual.max()],"k--")
    plt.xlabel("actual values")
    plt.ylabel("prediction values ")
    plt.title(f"One_model_{col} for actual v/s prediction values")
    plt.grid(True)
    plt.savefig(f'Single_Model_{col}')


exit()


# print(y_pred)
# exit()
r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 value : {r2:.2f}")
print(f"Mean_squared_error: {mse:.2f}")
print(f"Root_Mean_sqrt_Error:{rmse:.2f}")
print(f"Mean_absolute_Error:{mae:.2f}")






