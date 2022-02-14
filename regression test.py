import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\PC\Desktop\Projects\data\hamoye\energydata_complete.csv')
print(df.info())

#Question 12
x= reg[['T2']]
y= reg[['T6']]
from  sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test= train_test_split(x, y,test_size= 0.3 , random_state= 1 )
from sklearn.linear_model import LinearRegression
linear_model  = LinearRegression() 
#fit   the   model   to   the   training   datas
linear_model.fit(x_train, y_train) 
#obtain   predictions 
predicted_values   =   linear_model.predict(x_test) 
#RSQUARED
from    sklearn.metrics    import    r2_score 
r2_score   =   r2_score(y_test,   predicted_values) 
round(r2_score,2)

#Question 13
df = df.drop(['date'], axis = 1)

from    sklearn.preprocessing    import    MinMaxScaler 
scaler   =   MinMaxScaler() 
normalised_df =  pd.DataFrame(scaler.fit_transform(df),   columns=df.columns) 
features_df   =   normalised_df.drop(['lights'], axis = 1)
Appliances_target   =   normalised_df[ 'Appliances' ] 

from    sklearn.model_selection    import    train_test_split 
x_train,   x_test,   y_train,   y_test   =   train_test_split(features_df,   Appliances_target,test_size= 0.3 , random_state= 43 )

from sklearn.linear_model import LinearRegression
linear_model  = LinearRegression() 
#fit   the   model   to   the   training   datas
linear_model.fit(x_train,   y_train) 
#obtain   predictions 
predicted_values   =   linear_model.predict(x_test) 
#MAE 
from    sklearn.metrics    import    mean_absolute_error 
mae   =   mean_absolute_error(y_test,   predicted_values) 
round(mae, 2)

#Question 14
import    numpy    as    np 
rss   =   np.sum(np.square(y_test   -   predicted_values)) 
round(rss,    3 )

#Question 15
from    sklearn.metrics    import     mean_squared_error 
rmse   =   np.sqrt(mean_squared_error(y_test,   predicted_values)) 
round(rmse,    3 ) 

#Question 16
from    sklearn.metrics    import    r2_score 
r2_score   =   r2_score(y_test,   predicted_values) 
round(r2_score,    3 )

#Question 17
#RIDGE REGRESSION
from    sklearn.linear_model    import    Ridge 
ridge_reg   =   Ridge(alpha= 0.5 ) 
ridge_reg.fit(x_train,   y_train)

from    sklearn.linear_model    import    Lasso 
lasso_reg   =   Lasso(alpha= 0.001 ) 
lasso_reg.fit(x_train,   y_train) 


def get_weights_df(model,   feat,   col_name) : 
    
    #this   function   returns   the   weight   of   every   feature 
    weights= pd.Series(model.coef_,   feat.columns).sort_values() 
    weights_df=pd.DataFrame(weights).reset_index() 
    weights_df.columns = [ 'Features' ,   col_name] 
    weights_df[col_name]
    return weights_df

linear_model_weights = get_weights_df(linear_model,x_train,'Linear_Model_Weight' ) 
ridge_weights_df   =   get_weights_df(ridge_reg,x_train,'Ridge_Weight' ) 
lasso_weights_df   =   get_weights_df(lasso_reg,x_train,'Lasso_weight' ) 

final_weights   =   pd.merge(linear_model_weights,   ridge_weights_df,   on= 'Features' ) 
final_weights   =   pd.merge(final_weights,   lasso_weights_df,   on= 'Features' ) 
final_weights.sort_values('Linear_Model_Weight', ascending = True)

#question 20

from    sklearn.metrics    import     mean_squared_error 
rmse   =   np.sqrt(mean_squared_error(y_test,   predicted_values)) 
round(rmse,    3 ) 