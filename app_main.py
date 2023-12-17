###############################################################################################
# streamlit latest 2.0.19?
# streamlit-ago
###############################################################################################

import pandas as pd
import streamlit as st
import os, re
from analysis_result import TestAnalyzer

from typing import List, Tuple, Any, Optional
from abc import ABC

import app_tab_asr as asr
import app_tab_mt as mt
import app_tab_it as it

##################################################################
#                        app configuration
##################################################################
st.set_page_config(page_title='AMT Test for POC' ,
                   layout="wide",
                   page_icon='🚀')

analyzer = TestAnalyzer('testresults')

##################################################################
#                             side bar 
# [role]
# - display latest version of test result 
#   . only kpi average and differences from previous test result
# - prepare result analysis for user's selection
##################################################################

# system information
st.sidebar.markdown('## ☀️ AMT Test Automation <sup>POC</sup> ', unsafe_allow_html=True)
st.sidebar.caption("[NOTE] 'GPT-4 GENERATED TC' & 'SR' MT/ASR USED")
st.sidebar.markdown('-----')

# test history selection 
st.sidebar.markdown('## 📅 Test Results ')

# user selection of a test result
selected_file = st.sidebar.selectbox('Choose a result : ', analyzer.result_files, 0)

# analysis set-up with the selected test result 
analyzer.configure(selected_file)

# display grand total and difference for the selected 
st.sidebar.markdown('#### *Automatic Speech Recognition* 🔊')

st.sidebar.metric(label = 'WER Score', 
                  value = analyzer.average('wer'), 
                  delta = analyzer.average('wer') - analyzer.average('wer',False))
st.sidebar.write('#### *Machine Translation* 📜')
st.sidebar.metric(label = 'BLEU Score',
                  value = analyzer.average('bleu'), 
                  delta = analyzer.average('bleu') - analyzer.average('bleu',False))
st.sidebar.write('#### *ASR & MT Integaration* 🔊 ➕ 📜 ')
st.sidebar.metric(label = 'BLEU Score',
                  value = analyzer.average('i_bleu'), 
                  delta = analyzer.average('i_bleu') - analyzer.average('i_bleu',False))

# footer 
st.sidebar.markdown('<hr/>', unsafe_allow_html=True)
st.sidebar.caption('v0.0.1 - 2023.12.19')
st.sidebar.caption('Samsung Research , S/W Innovation Center')

##################################################################
#                             main page
##################################################################
st.write('### Test Result & Analysis')
st.write('This is a test result summary and statistical analysis for each aspect')

tab_asr , tab_mt , tab_it = st.tabs(['ASR', 'MT', 'ASR & MT'])

# asr unit test result
with tab_asr:
    asr.show_page(analyzer)

# mt unit test result
with tab_mt:
    mt.show_page(analyzer)
    
# mt & asr integration test result
with tab_it:
    it.show_page(analyzer)