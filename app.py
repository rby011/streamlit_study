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

st.write('### AMT Test Result & Analysis')
st.write('This is a test result summary and statistical analysis for each aspect')

tab_asr , tab_mt, tab_int = st.tabs(['ASR', 'MT', 'ASR & MT'])


# asr unit test result
with tab_asr:
    # result summary as data table
    with st.container():
        st.write('#### Result Summary')
        st.write('''This provides a test result table with which you can which types of utterances have
                relatively not-good result. For now, we only provide test result for korean language.''')
        # makke table to show
        with tab_asr:
            st.write('Table Here')
        with tab_mt:
            st.write('Table Here')
        with tab_int:
            st.write('Table Here')

    # statistical analysis title & list boxes
    with st.container():  
        st.write('#### Statistical Analysis')
        left_sa, right_sa = st.columns([0.4, 0.6])
        with left_sa:
            st.write('''This is a visualization of statistical analysisfor each aspect present at the above table. 
                    If you choose expander, you can see the detail of analysis result.''')
        with right_sa:
            left_lt, right_lt = st.columns([0.5, 0.5])
            with left_lt:
                st.selectbox('choose a language : ', ['korean', 'english', 'chinese', '...'])
            with right_lt:
                st.selectbox('choose an analysis target : ', ['utterance length', 'utterance style', 'speaker age', 'speaker gender'])

    # statistical analysis body
    with st.container():
        left_sab, right_sab = st.columns([0.6, 0.4])
        with left_sab:
            st.write(' chart here ')
            with st.expander('show analysis detail'):
                st.write("""
                - pearson correlation coefficient : 0.8 with p-value 0.002
                - regression coefficient : 0.8 with p-value 0.002
                """)
        with right_sab:
            st.selectbox('worst result @ 1 percentile : ', ['test', 'test', 'test', '...'])
            st.caption('script')
            st.write('가나다라ㅏㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹ')
            st.caption('transcript')
            st.write('가나다라ㅏㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹ')
            st.caption('audio * sr 10 hz, channel 1..')
            st.audio('test.wav')

# mt unit test result
with tab_mt:
    st.write('#### mt unit test result')

# asr & asr integration test result
with tab_int:
    st.write('#### asr & mt integration test result')


# try:
#     pass

# except Exception as e:
#     pass
# finally:
#     # [view] if all model data is laoded


st.toast(f'{selected_file} is loaded.')
