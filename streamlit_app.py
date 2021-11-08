# Import modules
import pandas as pd
import pickle
import base64
import os
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title('Steel prediction app') # setting title 
st.write('')

with st.expander('Focus Prediction'):
    form = st.form(key='my_form')
    capacity = form.number_input(label='Capacity', format="%.2f")
    door_height = form.number_input(label = 'Door Height', format="%.2f")
    door_width = form.number_input(label = 'Door Width', format="%.2f")
    metal_depth = form.number_input(label = 'Metal Depth', format="%.2f")
    bath_length = form.number_input(label = 'Bath Height', format="%.2f")
    submit_button = form.number_input(label = 'Submit')


    if submit_button:
        d = {'Capacity':[capacity], 'Door Height':[door_height], 'Door Width':[door_width], 'Metal Depth':[metal_depth],
        'Bath Length':[bath_length]}
        data = pd.DataFrame(data=d)
        st.write('')
        st.write('Data to be predicted')
        fig = ff.create_table(data)
        fig.update_layout(width=670)
        st.write(fig)

        st.write('')
        st.write('')
        st.write('Predicted Putput')