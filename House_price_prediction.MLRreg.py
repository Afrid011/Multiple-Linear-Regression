import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class HPP:
    def __init__(self,path):
        try:
            self.df = pd.read_csv(path)
            #map city and country to numbers
            self.city_map = {city: idx for idx, city in enumerate(self.df['city'].unique())}
            self.df['city'] = self.df['city'].map(self.city_map)
            self.df['country'] = self.df['country'].map({'USA':0}) # Assuming only 'USA' is present
            self.X = self.df.iloc[:,1::] # except the first column all are x values independent
            self.y = self.df.iloc[:,0] # only the first column is y value dependent
            # Split into training and testing
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        except Exception as e:
            er_type,error_message,tb = sys.exc_info()
            print(f"Error in the line no : {tb.tb_lineno} and Error Message ; {error_message} ")

    def training(self):
        try:
            self.reg=LinearRegression()
            self.reg.fit(self.X_train,self.y_train)
            self.y_pred_train = self.reg.predict(self.X_train)

            # mean sqaured erroe
            mean_squared_error = [(yt - yp)**2 for yt,yp in zip(self.y_train,self.y_pred_train)]
            mse = sum(mean_squared_error)/len(mean_squared_error)

            # r2_score
            y_mean = sum(self.y_train)/len(self.y_train)
            ss_total = sum((yt - y_mean)**2 for yt in self.y_train)
            ss_res = sum((yt - yp)**2 for yt,yp in zip(self.y_train,self.y_pred_train))
            r2 = 1 - (ss_res/ss_total)

            print("trining complete")
            print(f"MSE of training data is : {mse}")
            print(f'R2_score of training data is : {r2}')
        except Exception as e:
            er_type,error_message,tb = sys.exc_info()
            print(f"Error in Lneno ;{tb.tb_lineno} and due to : {error_message}")


    def testing(self):
        try:
            self.y_pred_test = self.reg.predict(self.X_test)
            #mean square error
            mean_squared_error = [(yt - yp)**2 for yt,yp in zip(self.y_test,self.y_pred_test)]
            mse = sum(mean_squared_error)/len(mean_squared_error)
            # r2_score
            y_mean = sum(self.y_test)/len(self.y_test)
            ss_total = sum((yt - y_mean)**2 for yt in self.y_test)
            ss_res = sum(mean_squared_error)
            r2_score = 1 - (ss_res/ss_total)
            print("Testing Complete")
            print(f" MSE of testing data is : {mse}")
            print(f"R2_Score of testing data is ;{r2_score}")
        except Exception as e:
            er_type,error_message,tb = sys.exc_info()
            print(f"Error in Lneno ;{tb.tb_lineno} and due to : {error_message}")


if __name__ == "__main__":
    try:
        obj = HPP('newupdateddata.csv')
        obj.training()
        obj.testing()
    except Exception as e:
        er_type,error_message,tb = sys.exc_info()
        print(f"Error in Lneno ;{tb.tb_lineno} and due to : {error_message}")
