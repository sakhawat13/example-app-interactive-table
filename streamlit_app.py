import pickle
import xgboost as xgb


from ta import add_all_ta_features

# In[2]:


import pandas as pd 
import datetime
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#import matplotlib.pyplot as plt
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode


# In[3]:






model_xgb = xgb.XGBClassifier()
model1 = xgb.XGBClassifier()
model2 = xgb.XGBClassifier()
model1.load_model("model_non_profit.json")
model2.load_model("model_profit.json")
# model_xgb = model1
# model = st.checkbox('Specialized model')
# model_xgb.load_model("model_unspecialized.json")
# if model:
#     model_xgb = model2
# clf2 = pickle.load(open('classifier_w_indicator_model_reversed.sav', 'rb'))


# In[4]:

st.title("Stock Prediction")


today = datetime.date.today()
lastfive = today - datetime.timedelta(days=23)

day = today.strftime ("%d/%m/%Y")
five = lastfive.strftime ("%d/%m/%Y")






st.header("Upload a csv file downloaded from Investing.com")

st.caption("You can Drag and drop the file into the box")

from io import StringIO

stockdata = pd.DataFrame()

file = st.file_uploader("Please choose a csv file")

if file is not None:

    #To read file as bytes:

    bytes_data = file.getvalue()

#     st.write(bytes_data)
    
    df= pd.read_csv(file)
    stockdata = df
#     st.write(df)

# st.write(stockdata)


submit = st.button("Submit")
st.subheader("Green = Abnormal Profit,  Blue = Players detected,     Black = Normal,   Red = Players exiting ")
if submit:
    stockdata = stockdata[stockdata['Vol.'].notna()]
    stockdata["Vol."]=stockdata['Vol.'].replace({'K': '*1e3', 'M': '*1e6', '-':'-1'}, regex=True).map(pd.eval).astype(int)
    stockdata = stockdata[::-1]
    stockdata = add_all_ta_features(stockdata, open="Open", high="High", low="Low", close="Price", volume="Vol.", fillna=True)
    stockdata["VolAvgNDays"] = stockdata["Vol."].rolling(20).mean()  
    stockdata['Change %'] = stockdata['Change %'].str.rstrip('%').astype('float') / 100.0
    check = stockdata.drop(["Date"],axis=1)
    st.write(len(check.columns))
    pred = model1.predict(check)
    prof = model2.predict(check)
    stockdata["Prediction"] = pred
    stockdata["Profit"] = prof
    stockdata = stockdata[::-1]
    sts = stockdata[["Date","Price","Prediction","Profit"]]
#     st.write(stockdata)
    


    
    
    def aggrid_interactive_table(df: pd.DataFrame):

        """Creates an st-aggrid interactive table based on a dataframe.
        Args:
        df (pd.DataFrame]): Source dataframe
        Returns:
        dict: The selected row
        """
        options = GridOptionsBuilder.from_dataframe(
            df, enableRowGroup=True, enableValue=True, enablePivot=True
        )
        jscode = JsCode("""
                    function(params) {
                        
                        if (params.data.Profit === 1) {
                            return {
                                'color': 'white',
                                'backgroundColor': 'green'
                            }
                        }
                        if (params.data.Prediction === 0) {
                            return {
                                'color': 'white',
                                
                            }
                        }
                    };
                    """)  
        gridOptions=options.build()
        gridOptions['getRowStyle'] = jscode
        options.configure_side_bar()
        #options.configure_selection("single")

        selection = AgGrid(
            df,
            enable_enterprise_modules=True,
            gridOptions=gridOptions,
            height=500,
            width="100%",


            theme="alpine",
            #update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
        )
        return selection

    selection = aggrid_interactive_table(df=sts)
