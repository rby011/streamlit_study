###############################################################################################
# streamlit latest 2.0.19?
# streamlit-ago
###############################################################################################

import pandas as pd
import streamlit as st
import os, re
import analysis_result as ar

from typing import List, Tuple, Any, Optional
from abc import ABC

# TODO : to be modularized
class TestAnalyzer(ABC):
    def __init__(self, result_root_path:str):
        self.result_root_path = result_root_path
        
        # [assumption] result file name starts with testresult and its extesion is csv
        pattern = r'^testresult.*\.csv$'  
        self.result_files =  [f for f in os.listdir(result_root_path) if re.match(pattern, f)]
        self.result_files = sorted(self.result_files, reverse=True)
        
        # construct index tables with bi-directional ways
        self.file_to_idx = {file: idx for idx, file in enumerate(self.result_files)}
        self.idx_to_file = {idx: file for idx, file in enumerate(self.result_files)}

        # test result as dataframe
        self.cdf = None     # current  (the selected)
        self.pdf = None     # previsou (right before the selected, None if the selected is the first)
        
        # configuration flag
        self.configured = False
        
        # language codes , this order SHOULD BE KEPT
        self.codes = ['KR', 'EN', 'ES', 'FR', 'DE', 'IT', 'JP', 'CN', 'RU', 'PT', 
                      'AR', 'HI', 'SW', 'NL', 'SV', 'PL', 'TR', 'TH', 'HE', 'DA']

        # aspect identifiers at ui 
        self.aspects_names_list = ['Utterance Length', 'Utterance Style' 'Speaker Age', 'Speaker Gender']

        self.aspects_names_dict = {'Utterance Length':0, 
                              'Utterance Style':1, 
                              'Speaker Age':2, 
                              'Speaker Gender':3}

        # aspect identifiers at dataframe
        self.aspects_columns_dict = {'Utterance Length':'sentence_len', 
                                'Utterance Style':'style', 
                                'Speaker Age':'age', 
                                'Speaker Gender':'gender'}
        
        self.aspects_max_values_dict  = {'Utterance Length': None, 
                                    'Utterance Style': None, 
                                    'Speaker Age': None, 
                                    'Speaker Gender':None}

        self.aspects_min_values_dict  = {'Utterance Length': None, 
                                    'Utterance Style': None, 
                                    'Speaker Age': None, 
                                    'Speaker Gender':None}
        
        self.aspects_values_dict = {}
    '''
    this can be extended so as to set strategy (what to and how to anlyze) in the future
    
    : param selected_file : one of test result file
    
    : return : test
    ''' 
    def configure(self, selected_file:str) -> None:
        # preprocessed dataframe for the previous test result
        _idx = self.file_to_idx[selected_file]
        if(_idx < len(self.result_files)):
            self.pdf = ar.prepocess_for_vanlysis(os.path.join(self.result_root_path, self.idx_to_file[_idx+1]))
        
        # preprocessed dataframe from the selected test result
        self.cdf = ar.prepocess_for_vanlysis(os.path.join(self.result_root_path, selected_file))
        
        # super set that each aspect has
        _an_key_list = list(self.aspects_names_dict.keys()) 
        self.aspects_values_dict[_an_key_list[0]] = None
        for aspect in _an_key_list[1:]: # except for utterance length
            aspect_vals = list(self.cdf[self.aspects_columns_dict[aspect]].dropna().unique())
            self.aspects_values_dict[aspect] = aspect_vals 
        
        # max values
        self.aspects_max_values_dict[_an_key_list[0]] = float(self.cdf[self.aspects_columns_dict[self.aspects_names_list[0]]].max())
        self.aspects_min_values_dict[_an_key_list[0]] = float(self.cdf[self.aspects_columns_dict[self.aspects_names_list[0]]].min())
        
        # set configure flag
        self.configured = True
    
    '''
    provide average kpi score 
    '''
    def average(self, metric_name:str, is_current:bool = True, ndigits:int = 3) -> float:
        if(self.configured == False):
            raise Exception('## [ERROR] Analyzer has not been configured yet.')

        if(is_current == False):
            if(self.pdf is None): 
                return 0
            return round(self.pdf[metric_name].mean(),ndigits)

        return round(self.cdf[metric_name].mean(),ndigits)

    '''
    provides test reuslts with dataframes to display on the summary page.
    - language table : 
    - utternable table :
    - age table :
    - gender table :
    - style table : 
    
    : param metric_name : wer, bleu, i_bleu 
    : param type : ASR or MT 
    '''
    def get_dataframes(self, metric_name:str, type:str) -> List[pd.DataFrame]:
        # for abreviation
        cdf = self.cdf
        pdf = self.pdf
        codes = self.codes 
        
        frames = []
        
        # Langguage Dataframe for ASR, MT, Integration
        # - fill random values into the data frame
        ldf = ar.random_dataframe(fcol_name='lang', fcol_values = codes, 
                                  columns=[f'{metric_name}_c', f'{metric_name}_p'], 
                                  make_last=True, type_last='diff')
        # update with real test result only for KR
        ldf.loc[0, [f'{metric_name}_c']] = cdf[metric_name].mean()
        ldf.loc[0, [f'{metric_name}_p']] = pdf[metric_name].mean()
        ldf.loc[0, ['diff']] = cdf[metric_name].mean() - pdf[metric_name].mean()
        frames.append(ldf)
        
        # Utterance Dataframe for ASR, MT, Integration
        # - after dropping missing value
        # - fill random values into the data frame
        columns = cdf['sentence_len_type'].dropna().unique().tolist()
        udf = ar.random_dataframe(fcol_name='lang', fcol_values = codes, columns = columns)
        # update with real test result only for KR
        for col in columns[1:]:
            udf.loc[0, [col]] = cdf.groupby('sentence_len_type')[metric_name].mean()[col]
        frames.append(udf)
        
        if type == 'ASR':
            # Age Dataframe for ASR
            # - after dropping missing value
            # - fill random values into the data frame
            columns = cdf['age'].dropna().unique().tolist()
            adf = ar.random_dataframe(fcol_name='lang', fcol_values = codes, columns = columns)
            # update with real test result only for KR
            for col in columns[1:]:
                adf.loc[0, [col]] = cdf.groupby('age')[metric_name].mean()[col]
            frames.append(adf)
            
            # Gender Dataframe for ASR
            # - after dropping missing value
            # - fill random values into the data frame
            columns = cdf['gender'].dropna().unique().tolist()
            gdf = ar.random_dataframe(fcol_name='lang', fcol_values = codes, columns = columns)
            
            # update with real test result only for KR
            for col in columns[1:]:
                gdf.loc[0, [col]] = cdf.groupby('gender')[metric_name].mean()[col]
            frames.append(gdf)

        # Style Dataframe for ASR, MT, Integration
        # - after dropping missing value
        # - fill random values into the data frame
        columns = cdf['style'].dropna().unique().tolist()
        sdf = ar.random_dataframe(fcol_name='lang', fcol_values = codes, columns = columns)
        # update with real test result only for KR
        for col in columns[1:]:
                sdf.loc[0, [col]] = cdf.groupby('style')[metric_name].mean()[col]
        frames.append(sdf)
        
        return frames      
    
    def get_analysis_result(self, language:str, metric_name:str, aspect_name:str) -> Tuple[Any, List[str] , Optional[pd.DataFrame]]:
        cdf = self.cdf
        if(language != 'KR'):
            return None
        
        y_var = metric_name
        
        aspect_index = self.aspects_names_dict[aspect_name] 
        if aspect_index == 0:
            x_var = 'sentence_len'
            return ar.v_analyze_numerics(cdf, x_var, y_var)
        elif aspect_index == 1:
            x_var = 'style'
        elif aspect_index == 2:
            x_var = 'age'
        elif aspect_index == 3:    
            x_var = 'gender'
        return ar.v_analyze_categorics(cdf, x_var, y_var)
    
    def get_testresults_by_numeric(self, aspect_name:str, aspect_max:float, aspect_min:float, 
                                   metric_name:str, metric_max:float, metric_min:float,
                                   ret_columns:List[str]) -> List[List[str]]:
        # abbrivation
        cdf = self.cdf
        
        # conditional slicing for each given column in the ret_columns
        ret = []
        condition1 = (cdf[aspect_name] >= aspect_min) & (cdf[aspect_name] <= aspect_max)
        condition2 = (cdf[metric_name] <= metric_max) & (cdf[metric_name] >= metric_min)
        for i, ret_col in enumerate(ret_columns):
            ret.append(cdf[condition1 & condition2][ret_col].to_list())

        # exceptional case
        if len(ret) != len(ret_columns):
            ret = [[],[],[],[]]
        
        return ret

    def get_testresults_by_categoric(self, aspect_name:str, aspect_val:str, 
                                     metric_name:str, metric_max:float, metric_min:float,
                                     ret_columns:List[str]) -> List[List[str]]:
        # abbrivation
        cdf = self.cdf
        
        # conditional slicing for each given column in the ret_columns
        ret = []
        condition1 = (cdf[aspect_name] == aspect_val)
        condition2 = (cdf[metric_name] <= metric_max) & (cdf[metric_name] >= metric_min)
        for i, ret_col in enumerate(ret_columns):
            ret.append(cdf[condition1 & condition2][ret_col].to_list())

        # exceptional case
        if len(ret) != len(ret_columns):
            ret = [[],[],[],[]]

        return ret

##################################################################
#                        app configuration
##################################################################
st.set_page_config(page_title='AMT Test' ,
                   page_icon='🚀')

analyzer = TestAnalyzer('testresults')

# state variables
asr_query_numeric_requested = False
asr_query_others_requested = False
asr_query_result_key = 'asr_query_result'
asr_query_result_val = [[],[],[],[]]

if asr_query_result_key not in st.session_state:
    st.session_state[asr_query_result_key] = asr_query_result_val

# variables from control
asr_selected_language_key = "asr_selected_language"
asr_selected_language = analyzer.codes[0]
asr_selected_language_changed = False
if asr_selected_language_key not in st.session_state:
    st.session_state[asr_selected_language_key] = asr_selected_language

asr_selected_aspect_name_key = "asr_selected_aspect_name"
asr_selected_aspect_name = analyzer.aspects_names_list[0]
if asr_selected_aspect_name_key not in st.session_state:
    st.session_state[asr_selected_aspect_name_key] = asr_selected_aspect_name
asr_selected_aspect_name_changed = False

asr_selected_clip_info_key = "asr_selected_clip_info"
asr_selected_clip_info = None
if asr_selected_clip_info_key not in st.session_state:
    st.session_state[asr_selected_clip_info_key] = asr_selected_clip_info
asr_selected_clip_info_changed = False

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
                # st.session_state[asr_query_result_key] = [[],[],[],[]]
                # if st.session_state[asr_selected_language_key] != asr_selected_language:
                #     st.session_state[asr_selected_language_key] = asr_selected_language
                #     asr_selected_language_changed = True
                #     print('selected language changed : ', asr_selected_language)
            # aspect value selection
            with right_lt:
                asr_selected_aspect_name = st.selectbox('choose an aspect : ', analyzer.aspects_names_dict.keys())
                if asr_selected_aspect_name == analyzer.aspects_names_list[0]:
                    asr_query_numeric_requested = True
                else:
                    asr_query_others_requested = True
                # st.session_state[asr_query_result_key] = [[],[],[],[]]
                st.session_state[asr_selected_aspect_name_key] = asr_selected_aspect_name
                # if st.session_state[asr_selected_aspect_name_key] != asr_selected_aspect_name:
                #     st.session_state[asr_selected_aspect_name_key] = asr_selected_aspect_name
                #     asr_selected_aspect_name_changed = True
                #     print('selected aspect changed : ', asr_selected_aspect_name)

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
                    asr_query_result = analyzer.get_testresults_by_numeric(analyzer.aspects_columns_dict[asr_selected_aspect_name],
                                                                           aspect_max_query,
                                                                           aspect_min_query, 
                                                                           'wer',
                                                                           metric_max_query,
                                                                           metric_min_query, 
                                                                           ['wer', 'path', 'sentence', 'transcript'])
                    st.session_state[asr_query_result_key] = asr_query_result
                    
                    st.toast(f'{len(asr_query_result[0])} is gathered')

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

                    # aspect_val_query = analyzer.aspects_values_dict[asr_selected_aspect_name][0]
                    # aspect_name = analyzer.aspects_columns_dict[asr_selected_aspect_name]
                    asr_query_result = analyzer.get_testresults_by_categoric(analyzer.aspects_columns_dict[asr_selected_aspect_name], 
                                                                             aspect_val_query, 
                                                                             'wer', 
                                                                             metric_max_query, 
                                                                             metric_min_query, 
                                                                             ['wer', 'path', 'sentence', 'transcript'])
                    st.session_state[asr_query_result_key] = asr_query_result
                    
                    st.toast(f'{len(asr_query_result[0])} is gathered')
                    
                    print('Retrieval after query:', len(asr_query_result[0]))

                if(len(st.session_state[asr_query_result_key][0]) > 0):
                    asr_query_result = st.session_state[asr_query_result_key]
                    _sel_dict = {}
                    # ['wer', 'path', 'sentence', 'transcript']
                    for score, path, script, tscript in zip(asr_query_result[0], asr_query_result[1], asr_query_result[2], asr_query_result[3]):
                        _key = f'[{round(score, ndigits=2)}]  '
                        _key = _key + path
                        _val = [path, script, tscript]
                        _sel_dict[_key] = _val

                    # audio file list with addition info
                    asr_selected_clip_info = st.selectbox('audio clips by condition : ',  _sel_dict.keys())
                    # print(asr_selected_clip_info, ':' , _sel_dict[asr_selected_clip_info], ' -> ' , _sel_dict[asr_selected_clip_info][0])
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
                    st.write("No Test Result")
                    for _ in range(6):
                        st.write(' ')
               
                st.write(' ')
                
                print('page finished')
                print()
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
# st.toast(f'{selected_file} is loaded.')
