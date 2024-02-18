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
model1 = pickle.load(open('model.pkl','rb'))

# model_xgb = model1
# model = st.checkbox('Specialized model')
# model_xgb.load_model("model_unspecialized.json")
# if model:
#     model_xgb = model2
# clf2 = pickle.load(open('classifier_w_indicator_model_reversed.sav', 'rb'))

def compute_moving_averages(df, windows=[20]):
    """
    Compute moving averages for all columns in the DataFrame for the given window sizes.

    Parameters:
    - df: pandas.DataFrame, the input DataFrame containing the data.
    - windows: list, a list of integers representing the window sizes for which to calculate the moving averages.

    Returns:
    - pandas.DataFrame, the original DataFrame with new columns for each moving average calculation.
    """
    # Loop through each column in the DataFrame
    for column in df.columns:
        # Check if the column data type is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Loop through each window size specified
            for window in windows:
                # Compute the moving average for the current column and window size
                ma_column_name = f"{column}_MA{window}"
                df[ma_column_name] = df[column].rolling(window=window).mean()
    return df


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
    stockdata1 = df
#     st.write(df)

# st.write(stockdata)


submit = st.button("Submit")
st.subheader("Green = Abnormal Profit,  Blue = Players detected,     Black = Normal,   Red = Players exiting ")
if submit:
    stockdata = stockdata1.iloc[::-1].reset_index(drop=True)
    stockdata = stockdata.fillna(0)
    stockdata = stockdata[stockdata['Vol.'].notna()]
    stockdata['Change %'] = stockdata['Change %'].str.replace('%', '', regex=True).astype(float)
    stockdata['Change %'] = stockdata['Change %']/100
    stockdata["Vol."]=stockdata['Vol.'].replace({'K': '*1e3', 'M': '*1e6', '-':'-1'}, regex=True).map(pd.eval).astype(int)
    # Ensure the correct data type for the 'Date' column
    # data_clean['Date'] = pd.to_datetime(data_clean['Date'])

    # Add all technical indicators available in the ta library
    stockdata = add_all_ta_features(
        stockdata,
        open="Open", high="High", low="Low", close="Price", volume="Vol."
    )
    stockdata = stockdata[['Date','Price',
     'Open',
     'High',
     'Low',
     'Vol.',
     'Change %',
     'volume_adi',
     'volume_obv',
     'volume_mfi',
     'volatility_bbm',
     'volatility_bbh',
     'volatility_bbl',
     'volatility_bbw',
     'volatility_bbp',
     'volatility_bbhi',
     'volatility_bbli',
     'trend_macd_signal',
     'trend_macd_diff',
     'trend_sma_fast',
     'trend_sma_slow',
     'trend_ema_fast',
     'trend_ema_slow',
     'trend_ichimoku_conv',
     'trend_ichimoku_base',
     'trend_ichimoku_a',
     'trend_ichimoku_b',
     'trend_stc',
     'trend_adx',
     'trend_adx_pos',
     'trend_adx_neg',
     'momentum_rsi']]
    stockdata= stockdata.replace([np.inf, -np.inf], 0)
    stockdata = stockdata.fillna(0)
    stockdata =  stockdata.drop(columns=['Date'])
    stockdata = compute_moving_averages(stockdata,[5])
    stockdata = stockdata.fillna(0)
    pred = model1.predict(stockdata)
    stockdata1["Prediction"] = pred
    stockdata1["Prediction"] = stockdata1["Prediction"]-1
    stockdata1['Prediction'] = stockdata1['Prediction'].replace(2, 1)
    sts = stockdata1[["Date","Price","Prediction"]]
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
                        
                        if (params.data.Prediction === 1) {
                            return {
                                'color': 'white',
                                'backgroundColor': 'green'
                            }
                        }
                        if (params.data.Prediction === -1) {
                            return {
                                'color': 'white',
                                'backgroundColor': 'red'
                                
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
