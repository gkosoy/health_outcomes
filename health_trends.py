import hyperopt as hp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.impute import SimpleImputer
import time
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RepeatedKFold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

#input dataframes, "cancer_for_python" is joined df of geo+cancer dfs (joined in Rstudio)
geo_info = pd.read_csv("avg-household-size.csv")
cancer = pd.read_csv("cancer_reg.csv")
r_df = pd.read_csv("cancer_for_python_df.csv")
r_df_num = r_df.drop(columns=['geography', 'county', 'binnedinc', 'death_perc', 'Unnamed: 0', 'index.x', 'index.y'])
target_deathrate_col = cancer['target_deathrate']

#deal with missing values (delete sparse column, impute Median where needed)
mis_val_count = r_df_num.isnull().sum()
mis_val_len = len(mis_val_count[mis_val_count > 0])
r_df_num = r_df_num.drop(columns=["pctsomecol18_24"])
mis_val_count = r_df_num.isnull().sum()
mis_val_after_drop = mis_val_count[mis_val_count > 0]
mising_val_col_list = [i for i in mis_val_after_drop.index]
my_imputer = SimpleImputer(strategy="median")
r_df_num[mising_val_col_list] = pd.DataFrame(my_imputer.fit_transform(r_df_num[mising_val_col_list]))

#create scaled and non-scaled test/train split data
X = r_df_num.drop(columns=['target_deathrate'])
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
y = r_df_num.target_deathrate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

#first linear regression on unscaled data
lm_mod = linear_model.LinearRegression()
lm_mod.fit(X_train, y_train)
pred_lm = lm_mod.predict(X_test)
print("MAE Score is:", mean_absolute_error(y_test, pred_lm))
#linear regression on scaled data
lm_mod = linear_model.LinearRegression()
lm_mod.fit(X_train_scaled, y_train)
pred_lm = lm_mod.predict(X_test_scaled)
print("MAE Score is:", mean_absolute_error(y_test, pred_lm))
#ridge regression unscaled data
lm_ridge = linear_model.Ridge(alpha=.5)
lm_ridge.fit(X_train, y_train)
ridge_pred = lm_ridge.predict(X_test)
print("MAE Score is:", mean_absolute_error(y_test, ridge_pred))
#ridge regression scaled data
lm_ridge = linear_model.Ridge(alpha=.5)
lm_ridge.fit(X_train_scaled, y_train)
ridge_pred = lm_ridge.predict(X_test_scaled)
print("MAE Score is:", mean_absolute_error(y_test, ridge_pred))

#optimize alpha parameter for ridge regression and plot outcome
n_alphas = 200
alphas = np.logspace(-7, 5, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show()

#lasso model on scaled data (scaled data performing better until this point)
lm_lasso = linear_model.Lasso(alpha=.001)
lm_lasso.fit(X_train_scaled, y_train)
lasso_pred = lm_lasso.predict(X_test_scaled)
print("MAE Score is:", mean_absolute_error(y_test, lasso_pred))

#random forest regressor model on scaled data
rfc = RandomForestRegressor(random_state=0)
rfc.fit(X_train_scaled, y_train)
preds = rfc.predict(X_test_scaled)
print("MAE Score is:", mean_absolute_error(y_test, preds))
#looking at max depth tuning for model and plot
max_depths = np.arange(1,32)
train_results = []
test_results = []
for max_depth in max_depths:
   rf = RandomForestRegressor(max_depth=max_depth, n_jobs=-1)
   rf.fit(X_train_scaled, y_train)

   train_prediction = rf.predict(X_train_scaled)
   MAE_score = mean_absolute_error(y_train, train_prediction)
   train_results.append(MAE_score)

   test_prediction = rf.predict(X_test_scaled)
   MAE_score_2 = mean_absolute_error(y_test, test_prediction)
   test_results.append(MAE_score_2)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, label='train score')
line2, = plt.plot(max_depths, test_results, label='test score')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, fontsize=15)
plt.ylabel('MAE score', fontsize=15)
plt.xlabel('Tree depth', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
#look at minimum samples for a split, and plot results
min_samples_splits = np.arange(2, 10)
train_results = []
test_results = []
for min_samples in min_samples_splits:
   rf = RandomForestRegressor(min_samples_split=min_samples, n_jobs=-1)
   rf.fit(X_train_scaled, y_train)

   train_prediction = rf.predict(X_train_scaled)
   MAE_score = mean_absolute_error(y_train, train_prediction)
   train_results.append(MAE_score)

   test_prediction = rf.predict(X_test_scaled)
   MAE_score_2 = mean_absolute_error(y_test, test_prediction)
   test_results.append(MAE_score_2)

line1, = plt.plot(min_samples_splits, train_results, label='train score')
line2, = plt.plot(min_samples_splits, test_results, label='test score')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, fontsize=15)
plt.ylabel('MAE score', fontsize=15)
plt.xlabel('Min split', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

#boost regressor, use grid search to optimize 3 parameters
import xgboost as xgb
model = xgb.XGBRegressor(learning_rate=0.015, n_estimators=700, max_depth=6)
xgb_model = xgb.XGBRegressor()
from sklearn.model_selection import GridSearchCV
# set up our search grid
param_grid = {"max_depth":    [4, 5, 6],
              "n_estimators": [500, 600, 700],
              "learning_rate": [0.01, 0.015]}

# try out every combination of the above values
search = GridSearchCV(xgb_model, param_grid, cv=5).fit(X_train_scaled, y_train)

print("The best hyperparameters are ",search.best_params_)

#use K-folds to validate XGBregressor model
xgb_model_fix = xgb.XGBRegressor(learning_rate=0.015, max_depth=5, n_estimators=700)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(xgb_model_fix, X_scaled, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))


