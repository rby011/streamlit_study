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

##################################################################
#                        app configuration
##################################################################
st.set_page_config(page_title='AMT Test' ,
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
st.sidebar.markdown('## ☀️ AMT Test Automation ')
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
st.sidebar.markdown('-----')
st.sidebar.caption('v0.0.1 - 2023.12.18')
st.sidebar.caption('Samsung Research , S/W Innovation Center')

##################################################################
#                             main page
##################################################################
st.write('### Test Result & Analysis')
st.write('This is a test result summary and statistical analysis for each aspect')

tab_asr , tab_mt , tab_int = st.tabs(['ASR', 'MT', 'ASR & MT'])

# asr unit test result
with tab_asr:
    print('tab_asr_root')
    
    # result summary as data table
    with st.container():

        print('tab_asr_root - result summary')

        st.write('#### Result Summary')
        st.write('''This provides a test result table with which you can which types of utterances have
                relatively not-good result. For now, we only provide test result for korean language.''')

        # make table to show
        st.write('<u> WER Score By Aspects : </u>', unsafe_allow_html=True)
        
        # make data frame for each aspect
        frames = analyzer.get_dataframes('wer', 'ASR')
        
        ldf = frames[0] # average wer for languages
        udf = frames[1] # average wer grouped by utterance length for languages
        adf = frames[2] # average wer grouped by age for languages
        gdf = frames[3] # average wer grouped by gender for languages
        sdf = frames[4] # average wer grouped by utterance style
                    
        # these MUST be located on model class
        for frame in frames:
            for col in frame.columns:
                if col == 'lang':
                    continue
                frame[col] = (frame[col] * 100).round(1)
            
        lcol, ucol, acol, gcol, scol = st.columns([4, 4, 5, 3, 3], gap='small')
        # Index(['lang', 'wer_c', 'wer_p', 'diff'], dtype='object')
        with lcol:
            st.write('* By Language')
            st.dataframe(
                ldf,
                column_config={
                    'lang'  : 'Lang.',
                    'wer_c' : st.column_config.NumberColumn(
                        'WER[C]',
                        help = '(%) Word error rate as current test result',
                        format = '%f'
                    ),
                    'wer_p' : st.column_config.NumberColumn(
                        'WER[P]',
                        help = '(%) Word error rate as previous test result',
                        format='%f'
                    ),
                    'diff'  : st.column_config.NumberColumn(
                        'DIFF',
                        help = '(%) WER[C] - WER[P]',
                        format='%f'
                    )
                },
                column_order=['lang', 'wer_c', 'wer_p', 'diff'],
                hide_index=True,
            )
        # Index(['lang', 'long', 'short', 'mid'], dtype='object')
        with ucol:
            st.write('* By Utterance Length')
            st.dataframe(
                udf,
                column_config={
                    'lang'  : 'Lang.',
                    'long' : st.column_config.NumberColumn(
                        'Long',
                        help = '(%)Utterance whose length is greater than the 3rd quanitle',
                        format = '%f'
                    ),
                    'short' : st.column_config.NumberColumn(
                        'Short',
                        help = '(%)Utterance whose length is smaller than the 1st quanitle',
                        format='%f'
                    ),
                    'mid'  : st.column_config.NumberColumn(
                        'MEDIUM',
                        help = '(%)Utterance whose length is greater than the 2nd & smaller than the 3rd quanitle',
                        format='%f'
                    )
                },
                column_order=['lang', 'long', 'mid', 'short'],
                hide_index=True,
            )
        # Index(['lang', 'thirties', 'twenties', 'teens', 'fourties'], dtype='object')
        with acol:
            st.write('* By Speaker Age')
            st.dataframe(
                adf,
                column_config={
                    'lang'  : 'Lang.',
                    'thirties' : st.column_config.NumberColumn(
                        '30s',
                        help = '(%) Speaker in 30s',
                        format = '%f'
                    ),
                    'twenties' : st.column_config.NumberColumn(
                        '20s',
                        help = '(%) Speaker in 20s',
                        format='%f'
                    ),
                    'teens'  : st.column_config.NumberColumn(
                        '10s',
                        help = '(%) Speaker in 10s',
                        format='%f'
                    ),
                    'fourties'  : st.column_config.NumberColumn(
                        '40s',
                        help = '(%) Speaker in 40s',
                        format='%f'
                    ),
                },
                column_order=['lang', 'teens', 'twenties', 'thirties', 'fourties'],
                hide_index=True,
            )
        # Index(['lang', 'male', 'female'], dtype='object')
        with gcol:
            st.write('* By Speaker Gender')
            st.dataframe(
                gdf,
                column_config={
                    'lang'  : 'Lang.',
                    'female' : st.column_config.NumberColumn(
                        'Female',
                        help = '(%) Female Speaker',
                        format = '%f'
                    ),
                    'male' : st.column_config.NumberColumn(
                        'Male',
                        help = '(%) Male Speaker',
                        format='%f'
                    ),
                },
                column_order=['lang', 'female', 'male'],
                hide_index=True,
            )
        # Index(['lang', 'spoken', 'written'], dtype='object')
        with scol:
            st.write('* By Utterance Style')
            st.dataframe(
                sdf,
                column_config={
                    'lang'  : 'Lang.',
                    'spoken' : st.column_config.NumberColumn(
                        'Spoken',
                        help = '(%) Spoken style utterance',
                        format = '%f'
                    ),
                    'written' : st.column_config.NumberColumn(
                        'Written',
                        help = '(%) Written style utterance',
                        format='%f'
                    ),
                },
                column_order=['lang', 'spoken', 'written'],
                hide_index=True,
            )

    # statistical analysis title & list boxes
    with st.container(): 
        print('tab_asr_root - detail analysis')

        st.write('#### Detail Analysis')
        left_sa, right_sa = st.columns([0.4, 0.6])
        with left_sa:
            st.write('''This is a visualization of statistical analysisfor each aspect present at the above table. 
                    If you choose expander, you can see the detail of analysis result.''')
        with right_sa:
            left_lt, right_lt = st.columns([0.5, 0.5])
            # language change
            with left_lt:
                asr_selected_language = st.selectbox('choose a language : ', analyzer.codes, disabled=True)
            # aspect value selection
            with right_lt:
                asr_selected_aspect_name = st.selectbox('choose an aspect : ', analyzer.aspects_names_dict.keys())
                if asr_selected_aspect_name == analyzer.aspects_names_list[0]:
                    asr_query_numeric_requested = True
                else:
                    asr_query_others_requested = True

    st.write(' ')

    # detail analysis body
    with st.container():
        print('tab_asr_root - detail analysis body')
        
        left_sab, right_sab = st.columns([0.3, 0.7])
        # chart and statistical analysis
        with left_sab:
            st.write('<u> Data distribution with wer score</u>',unsafe_allow_html=True)
            chart, info, frame = analyzer.get_analysis_result(asr_selected_language, 'wer', asr_selected_aspect_name)
            with st.container(border=True):
                st.pyplot(chart)
            with st.expander('show analysis detail'):
                st.write('\n'.join(info))
                
        # test-result query and display the query result
        with right_sab:
            st.write('<u> Retrieve raw test-results by conditions </u>',unsafe_allow_html=True)
            with st.container(border=True):
                st.write(' ')
                st.write(' ')
                
                # 'utterance length' aspect is selected
                if analyzer.aspects_names_dict[asr_selected_aspect_name] == 0: 
                    # slider bars for result query
                    right_sab_col1, right_sab_col2, right_sab_col3, right_sab_col4 = st.columns([1,1,1,1])
                    with right_sab_col1:
                        metric_min_query = st.slider('WER Min', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
                        print(metric_min_query)
                    with right_sab_col2:
                        metric_max_query = st.slider('WER Max', min_value=0.0, max_value=1.0, step=0.1, value=1.0)
                        print(metric_max_query)
                    with right_sab_col3:
                        aspect_min_query = st.slider('Sentence Length Min', 
                                               min_value=analyzer.aspects_min_values_dict[asr_selected_aspect_name], 
                                               max_value=analyzer.aspects_max_values_dict[asr_selected_aspect_name], 
                                               value=analyzer.aspects_max_values_dict[asr_selected_aspect_name]/2)
                        print(aspect_min_query)
                    with right_sab_col4:
                        aspect_max_query = st.slider('Sentence Length Max ', 
                                               min_value=analyzer.aspects_min_values_dict[asr_selected_aspect_name], 
                                               max_value=analyzer.aspects_max_values_dict[asr_selected_aspect_name], 
                                               value=analyzer.aspects_max_values_dict[asr_selected_aspect_name])
                        print(aspect_max_query)
                        
                    aspect_name = analyzer.aspects_columns_dict[asr_selected_aspect_name]
                    # set user query condition and query with the condition
                    asr_query_result = analyzer.get_testresults_by_numeric_asr(analyzer.aspects_columns_dict[asr_selected_aspect_name],
                                                                           aspect_max_query,
                                                                           aspect_min_query, 
                                                                           'wer',
                                                                           metric_max_query,
                                                                           metric_min_query, 
                                                                           ['wer', 'path', 'sentence', 'transcript'])
                    
                    # st.toast(f'{len(asr_query_result[0])} is gathered')

                    print('Retrieval after query :', len(asr_query_result[0]))
                else:
                    # slider bars for result query
                    right_sab_col6, right_sab_col7, right_sab_col8 = st.columns([1,1,1])
                    with right_sab_col6:
                        metric_min_query = st.slider('WER Min', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
                        print(metric_min_query)
                    with right_sab_col7:
                        metric_max_query = st.slider('WER Max', min_value=0.0, max_value=1.0, step=0.1, value=1.0)
                        print(metric_max_query)
                    with right_sab_col8:
                        aspect_val_query = st.selectbox(f'Choose {asr_selected_aspect_name}',analyzer.aspects_values_dict[asr_selected_aspect_name])
                        asr_query_others_requested = True
                        print(aspect_val_query)

                    asr_query_result = analyzer.get_testresults_by_categoric_asr(analyzer.aspects_columns_dict[asr_selected_aspect_name], 
                                                                             aspect_val_query, 
                                                                             'wer', 
                                                                             metric_max_query, 
                                                                             metric_min_query, 
                                                                             ['wer', 'path', 'sentence', 'transcript'])
                    # st.toast(f'{len(asr_query_result[0])} is gathered')
                    print('Retrieval after query:', len(asr_query_result[0]))

                if(len(asr_query_result[0]) > 0):
                    _sel_dict = {}
                    # ['wer', 'path', 'sentence', 'transcript']
                    for score, path, script, tscript in zip(asr_query_result[0], asr_query_result[1], asr_query_result[2], asr_query_result[3]):
                        _key = f'[{round(score, ndigits=2)}]  '
                        _key = _key + path
                        _val = [path, script, tscript]
                        _sel_dict[_key] = _val

                    # audio file list with addition info
                    asr_selected_clip_info = st.selectbox('audio clips by condition : ',  _sel_dict.keys())
                    if (asr_selected_clip_info != None) and os.path.exists( _sel_dict[asr_selected_clip_info][0]):
                        st.audio(_sel_dict[asr_selected_clip_info][0])
                    else:
                        with st.container(border=True):
                            st.write('Audio File Not Found')

                    # scipt and transcript for the selected audio clip                    
                    right_sab_col6  , right_sab_col7 = st.columns([0.07, 0.95])
                    with right_sab_col6:
                        st.caption('script')
                    with right_sab_col7:
                        st.write(_sel_dict[asr_selected_clip_info][1])
                    right_sab_col8  , right_sab_col9 = st.columns([0.07, 0.95])
                    with right_sab_col8:
                        st.caption('transcript')
                    with right_sab_col9:
                        st.write(_sel_dict[asr_selected_clip_info][2])
                else: # if there is no result
                    for _ in range(6):
                        st.write(' ')
                    st.write("Nothing 💨 💨 💨")
                    for _ in range(6):
                        st.write(' ')
               
                st.write(' ')
                print('asr page finished')
                print()

# mt unit test result
with tab_mt:
    print('tab_asr_root')
    
    # result summary as data table
    with st.container():

        print('tab_mt_root - result summary')

        st.write('#### Result Summary')
        st.write('''This provides a test result table with which you can which types of utterances have
                relatively not-good result. For now, we only provide test result for korean language.''')

        # make table to show
        st.write('<u> BLEU Score By Aspects : </u>', unsafe_allow_html=True)
        
        # make data frame for each aspect
        frames = analyzer.get_dataframes('bleu', 'MT')
        
        ldf = frames[0] # average wer for languages
        udf = frames[1] # average wer grouped by utterance length for languages
        sdf = frames[2] # average wer grouped by utterance style
                    
        # these MUST be located on model class
        for frame in frames:
            for col in frame.columns:
                if col == 'lang':
                    continue
                frame[col] = (frame[col] * 100).round(1)
            
        lcol, ucol, scol = st.columns([4, 4, 3], gap='small')
        # Index(['lang', 'wer_c', 'wer_p', 'diff'], dtype='object')
        with lcol:
            st.write('* By Language')
            st.dataframe(
                ldf,
                column_config={
                    'lang'  : 'Lang.',
                    'bleu_c' : st.column_config.NumberColumn(
                        'BLEU[C]',
                        help = '(%) Current BLEU score',
                        format = '%f'
                    ),
                    'bleu_p' : st.column_config.NumberColumn(
                        'BLEU[P]',
                        help = '(%) Previsou BLEU Score',
                        format='%f'
                    ),
                    'diff'  : st.column_config.NumberColumn(
                        'DIFF',
                        help = '(%) BLEU[C] - BLEU[P]',
                        format='%f'
                    )
                },
                column_order=['lang', 'bleu_c', 'bleu_p', 'diff'],
                hide_index=True,
            )
        # Index(['lang', 'long', 'short', 'mid'], dtype='object')
        with ucol:
            st.write('* By Utterance Length')
            st.dataframe(
                udf,
                column_config={
                    'lang'  : 'Lang.',
                    'long' : st.column_config.NumberColumn(
                        'Long',
                        help = '(%)Utterance whose length is greater than the 3rd quanitle',
                        format = '%f'
                    ),
                    'short' : st.column_config.NumberColumn(
                        'Short',
                        help = '(%)Utterance whose length is smaller than the 1st quanitle',
                        format='%f'
                    ),
                    'mid'  : st.column_config.NumberColumn(
                        'MEDIUM',
                        help = '(%)Utterance whose length is greater than the 2nd & smaller than the 3rd quanitle',
                        format='%f'
                    )
                },
                column_order=['lang', 'long', 'mid', 'short'],
                hide_index=True,
            )
        # Index(['lang', 'spoken', 'written'], dtype='object')
        with scol:
            st.write('* By Utterance Style')
            st.dataframe(
                sdf,
                column_config={
                    'lang'  : 'Lang.',
                    'spoken' : st.column_config.NumberColumn(
                        'Spoken',
                        help = '(%) Spoken style utterance',
                        format = '%f'
                    ),
                    'written' : st.column_config.NumberColumn(
                        'Written',
                        help = '(%) Written style utterance',
                        format='%f'
                    ),
                },
                column_order=['lang', 'spoken', 'written'],
                hide_index=True,
            )

    # statistical analysis title & list boxes
    with st.container(): 
        print('[MT] tab_mt_root - detail analysis')

        st.write('#### Detail Analysis')
        left_sa, right_sa = st.columns([0.4, 0.6])
        with left_sa:
            st.write('''This is a visualization of statistical analysisfor each aspect present at the above table. 
                    If you choose expander, you can see the detail of analysis result.''')
        with right_sa:
            left_lt, right_lt = st.columns([0.5, 0.5])
            # language change
            with left_lt:
                mt_selected_language = st.selectbox('choose a language : ', analyzer.codes, disabled=True , key='mt-sel-language')
            # aspect value selection
            with right_lt:
                mt_selected_aspect_name = st.selectbox('choose an aspect : ', analyzer.aspects_names_dict.keys(), key='mt-sel-aspect')
                if mt_selected_aspect_name == analyzer.aspects_names_list[0]:
                    mt_query_numeric_requested = True
                else:
                    mt_query_others_requested = True

    st.write(' ')

    # detail analysis body
    with st.container():
        print('[MT] tab_mt_root - detail analysis body')
        
        left_sab, right_sab = st.columns([0.3, 0.7])
        # chart and statistical analysis
        with left_sab:
            st.write('<u> Data distribution with bleu score</u>',unsafe_allow_html=True)
            chart, info, frame = analyzer.get_analysis_result(mt_selected_language, 'bleu', mt_selected_aspect_name)
            with st.container(border=True):
                st.pyplot(chart)
            with st.expander('show analysis detail'):
                st.write('\n'.join(info))
                
        # test-result query and display the query result
        with right_sab:
            st.write('<u> Retrieve raw test-results by conditions </u>',unsafe_allow_html=True)
            with st.container(border=True):
                st.write(' ')
                st.write(' ')
                
                # 'utterance length' aspect is selected
                if analyzer.aspects_names_dict[mt_selected_aspect_name] == 0: 
                    # slider bars for result query
                    right_sab_col1, right_sab_col2, right_sab_col3, right_sab_col4 = st.columns([1,1,1,1])
                    with right_sab_col1:
                        mt_metric_min_query = st.slider('BLEU Min', min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='mt-metric-min-numeric')
                        print(mt_metric_min_query)
                    with right_sab_col2:
                        mt_metric_max_query = st.slider('BLEU Max', min_value=0.0, max_value=1.0, step=0.1, value=1.0, key='mt-metric-max-numeric')
                        print(mt_metric_max_query)
                    with right_sab_col3:
                        mt_aspect_min_query = st.slider('Sentence Length Min', 
                                               min_value=analyzer.aspects_min_values_dict[mt_selected_aspect_name], 
                                               max_value=analyzer.aspects_max_values_dict[mt_selected_aspect_name], 
                                               value=analyzer.aspects_max_values_dict[mt_selected_aspect_name]/2, key = 'mt-min-aspect-numeric')
                        print(mt_aspect_min_query)
                    with right_sab_col4:
                        mt_aspect_max_query = st.slider('Sentence Length Max ', 
                                               min_value=analyzer.aspects_min_values_dict[mt_selected_aspect_name], 
                                               max_value=analyzer.aspects_max_values_dict[mt_selected_aspect_name], 
                                               value=analyzer.aspects_max_values_dict[mt_selected_aspect_name], key ='mt-max-aspect-numeric')
                        print(mt_aspect_max_query)
                        
                    mt_aspect_name = analyzer.aspects_columns_dict[mt_selected_aspect_name]
                    # set user query condition and query with the condition
                    mt_sel_item_list, mt_trans_dict, mt_grnd_dict = analyzer.get_testresults_by_numeric_mt(
                                                                    analyzer.aspects_columns_dict[mt_selected_aspect_name],
                                                                    mt_aspect_max_query, mt_aspect_min_query, 
                                                                    'bleu', mt_metric_max_query, mt_metric_min_query)
                    
                    print('[MT] Retrieval after query :', len(mt_sel_item_list[0]) if (mt_sel_item_list is not None and len(mt_sel_item_list) > 0) else None)
                else:
                    # slider bars for result query
                    right_sab_col6, right_sab_col7, right_sab_col8 = st.columns([1,1,1])
                    with right_sab_col6:
                        mt_metric_min_query = st.slider('BLEU Min', min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='mt-metric-min-others')
                        print(mt_metric_min_query)
                    with right_sab_col7:
                        mt_metric_max_query = st.slider('BLEU Max', min_value=0.0, max_value=1.0, step=0.1, value=1.0, key='mt-metric-max-others')
                        print(mt_metric_max_query)
                    with right_sab_col8:
                        mt_aspect_val_query = st.selectbox(f'Choose {mt_selected_aspect_name}',analyzer.aspects_values_dict[mt_selected_aspect_name], key='mt-aspect-others')
                        mt_query_others_requested = True
                        print(mt_aspect_val_query)

                    mt_sel_item_list, mt_trans_dict, mt_grnd_dict = analyzer.get_testresults_by_categoric_mt(
                                                                            analyzer.aspects_columns_dict[mt_selected_aspect_name], 
                                                                            mt_aspect_val_query, 'bleu', 
                                                                            mt_metric_max_query, mt_metric_min_query)
                    
                    print('[MT] Retrieval after query :', len(mt_sel_item_list[0]) if (mt_sel_item_list is not None and len(mt_sel_item_list) > 0) else None)

                if((mt_sel_item_list is not None) and (len(mt_sel_item_list) > 0)):
                    # audio file list with addition info
                    mt_sentence_info = st.selectbox('source setence : ',  mt_sel_item_list)
                    
                    mt_key_idx = 6
                    # scipt and transcript for the selected audio clip                    
                    right_sab_col6  , right_sab_col7 = st.columns([0.15, 0.85])
                    with right_sab_col6:
                        st.caption('translation')
                    with right_sab_col7:
                        st.write(mt_trans_dict[mt_sentence_info[mt_key_idx:].strip()])
                    right_sab_col8  , right_sab_col9 = st.columns([0.15, 0.85])
                    with right_sab_col8:
                        st.caption('refererences')
                    with right_sab_col9:
                        for line in mt_grnd_dict[mt_sentence_info[mt_key_idx:].strip()]:
                            st.write(line)
                else: # if there is no result
                    for _ in range(6):
                        st.write(' ')
                    st.write("Nothing 💨 💨 💨")
                    for _ in range(6):
                        st.write(' ')
               
                st.write(' ')
                print('mt page finished')
                print()

# asr & asr integration test result
with tab_int:
    st.write('#### asr & mt integration test result')


