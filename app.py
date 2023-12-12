import pandas as pd
import streamlit as st
import os
import re
from typing import List

#
# app configuration
#
st.set_page_config(page_title='AMT Test' ,
                   page_icon='🚀')

##################################################################
#                             side bar 
##################################################################

#
# [view] header : system information
#
st.sidebar.markdown('## ☀️ AMT Test Automation ')
st.sidebar.caption("[NOTE] 'GPT-4 GENERATED TC' & 'SR' MT/ASR USED")
st.sidebar.markdown('-----')

#
# [control] test result selection component
#
st.sidebar.markdown('## 📅 Test Results ')
directory = 'testresults'
pattern = r'^testresult.*\.csv$'
files = [f for f in os.listdir(directory) if re.match(pattern, f)]
files = sorted(files, reverse=True)
file_to_idx = {file: idx for idx, file in enumerate(files)}
idx_to_file = {idx: file for idx, file in enumerate(files)}
selected_file = st.sidebar.selectbox('Choose a result : ', files, 0)

#
# [model] load data models to make this view by selection
#
pdf = None
idx = file_to_idx[selected_file]
if(idx < len(files)):
    pdf = pd.read_csv(os.path.join(directory, idx_to_file[idx+1]))
cdf = pd.read_csv(os.path.join(directory, selected_file))

#
# [view] display proper data in the loaded model
#
st.sidebar.markdown('#### *Automatic Speech Recognition* 🔊')
st.sidebar.metric(label = 'WER Score', 
                  value = round(cdf['wer'].mean(), 3), 
                  delta = 0 if pdf is None else round(cdf['wer'].mean() - pdf['wer'].mean(), 3))

st.sidebar.write('#### *Machine Translation* 📜')
st.sidebar.metric(label = 'BLEU Score',
                  value = round(cdf['bleu'].mean(), 3),
                  delta = 0 if pdf is None else round(cdf['bleu'].mean() - pdf['bleu'].mean(),3))

st.sidebar.write('#### *ASR & MT Integaration* 🔊 ➕ 📜 ')
st.sidebar.metric(label = 'BLEU Score',
                  value = round(cdf['i_bleu'].mean(), 3),
                  delta = 0 if pdf is None else round(cdf['i_bleu'].mean() - pdf['i_bleu'].mean(),3))

#
# [view] footer : display proper data in the loaded model
#
st.sidebar.markdown('-----')
st.sidebar.caption('v0.0.1 - 2023.12.18')
st.sidebar.caption('Samsung Research , S/W Innovation Center')


##################################################################
#                             main page
##################################################################
try:
    pass

except Exception as e:
    pass
finally:
    # [view] if all model data is laoded
    st.toast(f'{selected_file} is loaded.')
