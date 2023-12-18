from analysis_result import TestAnalyzer

import streamlit as st

def show_page(analyzer:TestAnalyzer) -> None:
    # result summary as data table
    with st.container():

        print('# [UI][IT] tab_int_root - result summary')

        st.write('#### Result Summary')
        st.write('''This provides a test result table with which you can which types of utterances have
                relatively not-good result. For now, we only provide test result for korean language.''')

        # make table to show
        st.write('<u> BLEU Score and its Comparision By Aspects : </u>', unsafe_allow_html=True)
        
        # make data frame for each aspect
        frames = analyzer.get_dataframes_it('bleu')
        
        ldf = frames[0] # average bleu for languages
        udf = frames[1] # ut & it comparision grouped by utterance length for each language
        sdf = frames[2] # ut & it comparision grouped by utterance style for each language
                    
        # these MUST be located on model class
        ldf.iloc[:, 1:] = ldf.iloc[:, 1:].round(2)
        udf.iloc[:, 1:] = udf.iloc[:, 1:].round(2)
        sdf.iloc[:, 1:] = sdf.iloc[:, 1:].round(2)

        lcol, ucol, scol = st.columns([5, 7, 5], gap='small')
        
        # Index(['lang', 'i_blue_c', 'i_blue_p', 'diff'], dtype='object')
        with lcol:
            st.write('* By Language ')
            st.dataframe(
                ldf,
                column_config={
                    'lang'  : 'Lang.',
                    'i_bleu_c' : st.column_config.NumberColumn(
                        'BLEU[C]',
                        help = '(%) Current BLEU score',
                        format = '%f'
                    ),
                    'i_bleu_p' : st.column_config.NumberColumn(
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
                column_order=['lang', 'i_bleu_c', 'i_bleu_p', 'diff'],
                hide_index=True,
            )
        # Index(['lang', 'mid', 'short', 'long', 'ddif_0', 'diff_1', 'diff_2], dtype='object')
        with ucol:
            st.write('* By Utterance Length')
            st.dataframe(
                udf,
                column_config={
                    'lang'  : 'Lang.',
                    'mid' : st.column_config.NumberColumn(
                        'Long',
                        help = '(%)Utterance whose length is greater than the 3rd quanitle',
                        format = '%f'
                    ),
                    'diff_0' : st.column_config.NumberColumn(
                        'Diff',
                        help = '(%)Difference between the unit test BLEU score with the integration ',
                        format = '%f'
                    ),
                    'short' : st.column_config.NumberColumn(
                        'Short',
                        help = '(%)Utterance whose length is smaller than the 1st quanitle',
                        format='%f'
                    ),
                    'diff_1' : st.column_config.NumberColumn(
                        'Diff',
                        help = '(%)Difference between the unit test BLEU score with the integration ',
                        format = '%f'
                    ),
                    'long'  : st.column_config.NumberColumn(
                        'MEDIUM',
                        help = '(%)Utterance whose length is greater than the 2nd & smaller than the 3rd quanitle',
                        format='%f'
                    ),
                    'diff_2' : st.column_config.NumberColumn(
                        'Diff',
                        help = '(%)Difference between the unit test BLEU score with the integration ',
                        format = '%f'
                    ),
                    
                },
                column_order=['lang', 'long', 'diff_2', 'short', 'diff_1','mid', 'diff_0'],
                hide_index=True,
            )
        # Index(['lang', 'spoken', 'written', 'diff_0', 'diff_1'], dtype='object')
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
                    'diff_0' : st.column_config.NumberColumn(
                        'Diff',
                        help = '(%)Difference between the unit test BLEU score with the integration ',
                        format = '%f'
                    ),
                    'written' : st.column_config.NumberColumn(
                        'Written',
                        help = '(%) Written style utterance',
                        format='%f'
                    ),
                    'diff_1' : st.column_config.NumberColumn(
                        'Diff',
                        help = '(%)Difference between the unit test BLEU score with the integration ',
                        format = '%f'
                    ),
                },
                column_order=['lang', 'spoken', 'diff_0', 'written', 'diff_1'],
                hide_index=True,
            )
        print('# [UI][IT] dataframe is displayed')

    # statistical analysis title & list boxes
    with st.container(): 
        print('# [UI][IT] tab_it_root - detail analysis')

        st.write('#### Detail Analysis')
        left_sa, right_sa = st.columns([0.4, 0.6])
        with left_sa:
            st.write('''This is a visualization of statistical analysisfor each aspect present at the above table. 
                    If you choose expander, you can see the detail of analysis result.''')
        with right_sa:
            left_lt, right_lt = st.columns([0.5, 0.5])
            # language change
            with left_lt:
                it_selected_language = st.selectbox('choose a language : ', analyzer.codes, disabled=True , key='it-sel-language')
            # aspect value selection
            with right_lt:
                it_selected_aspect_name = st.selectbox('choose an aspect : ', analyzer.aspects_names_dict.keys(), key='it-sel-aspect')
                if it_selected_aspect_name == analyzer.aspects_names_list[0]:
                    it_query_numeric_requested = True
                else:
                    it_query_others_requested = True
        print('# [UI][IT] Query condition component are displayed')

    st.write(' ')
    
    # detail analysis body
    with st.container():
        print('# [UI][IT] tab_it_root - detail analysis body')
        left_sab, right_sab = st.columns([0.3, 0.7])
        # chart and statistical analysis
        with left_sab:
            st.write('<u> Data distribution with "bleu score difference"</u>',unsafe_allow_html=True)
            chart, info, frame = analyzer.get_analysis_result_it(it_selected_language, 'bleu', it_selected_aspect_name)
            with st.container(border=True):
                st.pyplot(chart)
            with st.expander('show analysis detail'):
                st.write(f'<b> Scrore diff by the {it_selected_aspect_name} for each quantile</b>', unsafe_allow_html=True)
                st.dataframe(frame)
                st.write(f'<b> Anlaysis the average-scrore difference by qunatile</b>', unsafe_allow_html=True)
                st.write('\n'.join(info))
        print('# [UI][IT] data distribution and analysis expander is dispalyed')
                
        # test-result query and display the query result
        with right_sab:
            st.write('<u> Retrieve raw test-results by conditions </u>',unsafe_allow_html=True)
            with st.container(border=True):
                st.write(' ')
                st.write(' ')
                # 'utterance length' aspect is selected
                if analyzer.aspects_names_dict[it_selected_aspect_name] == 0: 
                    # slider bars for result query
                    right_sab_col1, right_sab_col2, right_sab_col3, right_sab_col4 = st.columns([1,1,1,1])
                    with right_sab_col1:
                        
                        # TODO : get Diff Max and Min , value (Max+Min)/2
                        it_metric_min_query = st.slider('BLEU "Diff" Min (%)', 
                                                        min_value=analyzer.metric_diff_min * 100, 
                                                        max_value=analyzer.metric_diff_max * 100, 
                                                        step=0.1, 
                                                        value=(analyzer.metric_diff_max + analyzer.metric_diff_min)/2 * 100, 
                                                        key='it-metric-min-numeric')
                        it_metric_min_query = it_metric_min_query / 100
                    with right_sab_col2:
                        it_metric_max_query = st.slider('BLEU "Diff" Max (%)', 
                                                        min_value=analyzer.metric_diff_min * 100, 
                                                        max_value=analyzer.metric_diff_max * 100, 
                                                        step=0.1, 
                                                        value=analyzer.metric_diff_max * 100, 
                                                        key='it-metric-max-numeric')
                        it_metric_max_query = it_metric_max_query / 100
                    with right_sab_col3:
                        it_aspect_min_query = st.slider('Sentence Length Min', 
                                               min_value=analyzer.aspects_min_values_dict[it_selected_aspect_name], 
                                               max_value=analyzer.aspects_max_values_dict[it_selected_aspect_name], 
                                               value=analyzer.aspects_max_values_dict[it_selected_aspect_name]/2, key = 'it-min-aspect-numeric')
                    with right_sab_col4:
                        it_aspect_max_query = st.slider('Sentence Length Max ', 
                                               min_value=analyzer.aspects_min_values_dict[it_selected_aspect_name], 
                                               max_value=analyzer.aspects_max_values_dict[it_selected_aspect_name], 
                                               value=analyzer.aspects_max_values_dict[it_selected_aspect_name], key ='it-max-aspect-numeric')
                        
                    it_aspect_name = analyzer.aspects_columns_dict[it_selected_aspect_name]
                    
                    # set user query condition and query with the condition
                    it_sel_item_list, it_trans_dict, it_grnd_dict = analyzer.get_testresults_by_numeric_it(
                                                                    analyzer.aspects_columns_dict[it_selected_aspect_name],
                                                                    it_aspect_max_query, it_aspect_min_query, 
                                                                    'bleu', it_metric_max_query, it_metric_min_query)
                    
                    print('# [UI][IT] Retrieval after query :', len(it_sel_item_list[0]) if (it_sel_item_list is not None and len(it_sel_item_list) > 0) else None)
                    
                else:
                    # slider bars for result query
                    right_sab_col6, right_sab_col7, right_sab_col8 = st.columns([1,1,1])
                    with right_sab_col6:
                        it_metric_min_query = st.slider('BLEU "Diff" Min (%)', 
                                                        min_value=analyzer.metric_diff_min * 100, 
                                                        max_value=analyzer.metric_diff_max * 100, 
                                                        step=0.1, 
                                                        value=(analyzer.metric_diff_max + analyzer.metric_diff_min)/2 * 100, 
                                                        key='it-metric-min-others')
                        it_metric_min_query = it_metric_min_query / 100
                    with right_sab_col7:
                        it_metric_max_query = st.slider('BLEU "Diff" Max (%)', 
                                                        min_value=analyzer.metric_diff_min * 100, 
                                                        max_value=analyzer.metric_diff_max * 100, 
                                                        step=0.1, 
                                                        value=analyzer.metric_diff_max * 100, 
                                                        key='it-metric-max-others')
                        it_metric_max_query = it_metric_max_query /100
                    with right_sab_col8:
                        it_aspect_val_query = st.selectbox(f'Choose {it_selected_aspect_name}',
                                                           analyzer.aspects_values_dict[it_selected_aspect_name], key='it-aspect-others')

                    it_sel_item_list, it_trans_dict, it_grnd_dict = analyzer.get_testresults_by_categoric_it(
                                                                            analyzer.aspects_columns_dict[it_selected_aspect_name], 
                                                                            it_aspect_val_query, 'bleu', 
                                                                            it_metric_max_query, it_metric_min_query)
                    
                    print('# [UI][IT] Retrieval after query :', len(it_sel_item_list[0]) if (it_sel_item_list is not None and len(it_sel_item_list) > 0) else None)

                if((it_sel_item_list is not None) and (len(it_sel_item_list) > 0)):
                    it_sentence_info = st.selectbox('source setence : ',  it_sel_item_list)
                    
                    it_key_idx = it_sentence_info.find(']')  # TODO which one is better comparing previsous checking logic? 
                    if it_key_idx != -1:
                        it_key_idx = it_key_idx + 2
                        # scipt and transcript for the selected audio clip                    
                        right_sab_col6  , right_sab_col7 = st.columns([0.15, 0.85])
                        with right_sab_col6:
                            st.caption('translation')
                        with right_sab_col7:
                            key = it_sentence_info[it_key_idx:].strip()
                            st.write(it_trans_dict[key])
                        right_sab_col8  , right_sab_col9 = st.columns([0.15, 0.85])
                        with right_sab_col8:
                            st.caption('refererences')
                        with right_sab_col9:
                            key = it_sentence_info[it_key_idx:].strip()
                            for line in it_grnd_dict[key]:
                                st.write(line)
                    else:
                        st.write("Nothing 💨 💨 💨")
                        
                    print('# [UI][IT] Retrieval result is displayed')
                else: # if there is no result
                    for _ in range(6):
                        st.write(' ')
                    st.write("Nothing 💨 💨 💨")
                    for _ in range(6):
                        st.write(' ')
                    print('# [UI][IT] Not retrieval result to display')
               
                st.write(' ')
                print('# [UI][IT] it page finished')
                print()