from analysis_result import TestAnalyzer

import streamlit as st

def show_page(analyzer:TestAnalyzer) -> None:
    # result summary as data table
    with st.container():

        print('# [UI][MT]tab_mt_root - result summary')

        st.write('#### Result Summary')
        st.write('''This provides a test result table with which you can which types of utterances have
                relatively not-good result. For now, we only provide test result for korean language.''')

        # make table to show
        st.write('<u> BLEU Score By Aspects : </u>', unsafe_allow_html=True)
        
        # make data frame for each aspect
        frames = analyzer.get_dataframes_ut('bleu', 'MT')
        
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
        print('# [UI][MT] dataframe is displayed')

    # statistical analysis title & list boxes
    with st.container(): 
        print('# [UI][MT] tab_mt_root - detail analysis')

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
                if mt_selected_aspect_name == analyzer.aspects_names_list[0]:  # TODO CHECK this is required?
                    mt_query_numeric_requested = True
                else:
                    mt_query_others_requested = True
        print('# [UI][MT] Query condition component are displayed')

    st.write(' ')

    # detail analysis body
    with st.container():
        print('# [UI][MT] tab_mt_root - detail analysis body')
        left_sab, right_sab = st.columns([0.3, 0.7])
        # chart and statistical analysis
        with left_sab:
            st.write('<u> Data distribution with bleu score</u>',unsafe_allow_html=True)
            chart, info, frame = analyzer.get_analysis_result_ut(mt_selected_language, 'bleu', mt_selected_aspect_name)
            with st.container(border=True):
                st.pyplot(chart)
            with st.expander('show analysis detail'):
                st.write(f'<b> Average scrore diff by groups in the {mt_selected_aspect_name}</b>', unsafe_allow_html=True)
                st.dataframe(frame)
                st.write(f'<b> Statistic anlaysis for the difference of average-scrore by groups</b>', unsafe_allow_html=True)
                st.write('\n'.join(info))
        print('# [UI][MT] data distribution and analysis expander is dispalyed')
                                
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
                    with right_sab_col2:
                        mt_metric_max_query = st.slider('BLEU Max', min_value=0.0, max_value=1.0, step=0.1, value=1.0, key='mt-metric-max-numeric')
                    with right_sab_col3:
                        mt_aspect_min_query = st.slider('Sentence Length Min', 
                                               min_value=analyzer.aspects_min_values_dict[mt_selected_aspect_name], 
                                               max_value=analyzer.aspects_max_values_dict[mt_selected_aspect_name], 
                                               value=analyzer.aspects_max_values_dict[mt_selected_aspect_name]/2, key = 'mt-min-aspect-numeric')
                    with right_sab_col4:
                        mt_aspect_max_query = st.slider('Sentence Length Max ', 
                                               min_value=analyzer.aspects_min_values_dict[mt_selected_aspect_name], 
                                               max_value=analyzer.aspects_max_values_dict[mt_selected_aspect_name], 
                                               value=analyzer.aspects_max_values_dict[mt_selected_aspect_name], key ='mt-max-aspect-numeric')
                        
                    mt_aspect_name = analyzer.aspects_columns_dict[mt_selected_aspect_name]
                    # set user query condition and query with the condition
                    mt_sel_item_list, mt_trans_dict, mt_grnd_dict = analyzer.get_testresults_by_numeric_mt(
                                                                    analyzer.aspects_columns_dict[mt_selected_aspect_name],
                                                                    mt_aspect_max_query, mt_aspect_min_query, 
                                                                    'bleu', mt_metric_max_query, mt_metric_min_query)
                    
                    print('# [UI][MT] Retrieval after query :', len(mt_sel_item_list[0]) if (mt_sel_item_list is not None and len(mt_sel_item_list) > 0) else None)
                else:
                    # slider bars for result query
                    right_sab_col6, right_sab_col7, right_sab_col8 = st.columns([1,1,1])
                    with right_sab_col6:
                        mt_metric_min_query = st.slider('BLEU Min', min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='mt-metric-min-others')
                    with right_sab_col7:
                        mt_metric_max_query = st.slider('BLEU Max', min_value=0.0, max_value=1.0, step=0.1, value=1.0, key='mt-metric-max-others')
                    with right_sab_col8:
                        mt_aspect_val_query = st.selectbox(f'Choose {mt_selected_aspect_name}',analyzer.aspects_values_dict[mt_selected_aspect_name], key='mt-aspect-others')
                        mt_query_others_requested = True

                    mt_sel_item_list, mt_trans_dict, mt_grnd_dict = analyzer.get_testresults_by_categoric_mt(
                                                                            analyzer.aspects_columns_dict[mt_selected_aspect_name], 
                                                                            mt_aspect_val_query, 'bleu', 
                                                                            mt_metric_max_query, mt_metric_min_query)
                    
                    print('# [UI][MT] Retrieval after query :', len(mt_sel_item_list[0]) if (mt_sel_item_list is not None and len(mt_sel_item_list) > 0) else None)

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
                    print('# [UI][MT] Retrieval result is displayed')
                else: # if there is no result
                    for _ in range(6):
                        st.write(' ')
                    st.write("Nothing 💨 💨 💨")
                    for _ in range(6):
                        st.write(' ')
                    print('# [UI][MT] Not retrieval result to display')

                st.write(' ')
                
                print('# [UI][MT] mt page finished')
                print()
