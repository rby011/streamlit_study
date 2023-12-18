import os,re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from pprint import pprint
from statsmodels.formula.api import ols
from scipy.stats import pearsonr, shapiro, levene
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import MinMaxScaler

from typing import List, Tuple, Any,Optional, Dict
from faker import Faker 
from abc import ABC

'''
[cached]
chart creation and statstic analysis for two numerical variable
- scatter plot : display data in the space (x,y) 
- pearson correlation , regression anlaysis 

: param df    : panda dataframe including x_var and y_var
: param x_var : colunm name to analyze , esp., independent variable
: param y_var : colunm name to analyze , esp., depdent variable

: return : chart ojbect , string list to display 
'''
@st.cache_data
def v_analyze_numerics(df:pd.DataFrame , x_var:str, y_var:str) -> Tuple[Any, List[str], Any]:
    print(f'# [Engine] analysis begin for numeric x, y vars : {x_var} , {y_var}')
    ret = []

    print(f'statistical analysis between {x_var} with {y_var}')
    # pearson correlation
    correlation, p_value = pearsonr(df[x_var], df[y_var])
    print(f'- pearson analysis : {correlation} with p-val {p_value}')
    ret.append(f'* pearson analysis')
    ret.append(f'    * coefficient ({correlation}) with p-value({p_value})')

    statistics, p_value = shapiro(df[x_var])
    print(f'-- normality using shaprio-wilik - {x_var} : {statistics} with p-val {p_value}')
    ret.append(f"       * {x_var} normality (shapiro-wilk): {p_value}")

    statistics, p_value = shapiro(df[y_var])
    print(f'-- normality using shaprio-wilik - {x_var} : {statistics} with p-val {p_value}')
    ret.append(f"       * {y_var} normality (shapiro-wilk): {shapiro(df[y_var])[1]}")

    # regression analysis
    model = ols(f'{y_var} ~ {x_var}', data = df).fit()
    print(f'- regression analysis : coefficient{{model.params[0]}} with p-val {model.f_pvalue}')
    ret.append(f'* regression ')
    ret.append(f'  * coefficient ({model.params[0]}) with p-value({model.f_pvalue})')

    statistics, p_value = shapiro(model.resid)
    print(f'-- normality using shaprio-wilik - residuals : {statistics} with p-val {p_value}')
    ret.append(f"     * residual nomality (shaprio-wilk): {shapiro(model.resid)[1]}")

    statistics, p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f'-- homogeneity using breusch-pegan - residuals : {statistics} with p-val {p_value}')
    ret.append(f"     * residual homogeneity of variances (breusch-pagan) : {p_value}")
    
    # chart
    fig, ax = plt.subplots()
    sns.scatterplot(x = x_var, y = y_var, data = df, ax=ax)
    
    # data summary frame
    quantiles = df[x_var].quantile([0.25, 0.5, 0.75])
    q1 = df.loc[df[x_var] <= quantiles[0.25], y_var]
    q2 = df.loc[(df[x_var] > quantiles[0.25]) & (df[x_var] <= quantiles[0.5]), y_var]
    q3 = df.loc[(df[x_var] > quantiles[0.5]) & (df[x_var] <= quantiles[0.75]), y_var]
    q4 = df.loc[df[x_var] > quantiles[0.75], y_var]

    summary_df = pd.DataFrame({
        'quantile': ['Q1<', 'Q2<', 'Q3<', 'Q4<'],
        'mean': [q1.mean(), q2.mean(), q3.mean(), q4.mean()],
        'var': [q1.var(), q2.var(), q3.var(), q4.var()],
        'max': [q1.max(), q2.max(), q3.max(), q4.max()],
        'min': [q1.min(), q2.min(), q3.min(), q4.min()],
    })

    return fig , ret, summary_df


'''
[cached]
chart creation and statstic analysis for two cateogrical variable
- violin plot : display box plot and distribution for each category 
- regression anlaysis , one-way anova 

: param df    : panda dataframe including x_var and y_var
: param x_var : colunm name to analyze , esp., independent variable
: param y_var : colunm name to analyze , esp., depdent variable

: return : chart ojbect , average data frame for each category, string list to display 
'''
@st.cache_data
def v_analyze_categorics(df:pd.DataFrame , x_var:str, y_var:str) -> Tuple[Any, List[str], pd.DataFrame]:
    print(f'# [Engine] analysis begin for norminal x, y vars : {x_var} , {y_var}')

    ret = []

    # remove missing value
    df = df.dropna(subset=[x_var, y_var])

    # anonva
    model = ols(f'{y_var} ~ C({x_var})', data=df).fit()
    anova_table = anova_lm(model)
    print(anova_table)
    ret.append(f'* annova analysis (p-value) : {anova_table.iloc[0,-1]}')

    # nomality
    stat, p = shapiro(df[y_var])
    print(f'Shapiro-Wilk Test: statistic={stat}, p-value={p}')
    ret.append(f'  * {y_var} normality (p-value) : {p}')

    # homoscedasticity (variance)
    groups = [group[y_var].dropna() for _ , group in df.groupby(x_var)]
    stat, p = levene(*groups)
    print(f'Levene Test: statistic={stat}, p-value={p}')
    ret.append(f'  * {y_var} homogeneity (p-value) : {p}')

    # y_var average table grouped by x_var
    avg_df = df.groupby(x_var)[y_var].agg(['mean', 'var']).reset_index()
    print(avg_df)

    # viloin plot
    fig, ax = plt.subplots()
    sns.violinplot(x=x_var, y=y_var, data=df , 
                palette = sns.color_palette("hls", len(df[x_var].unique())),ax=ax)

    return fig, ret, avg_df 

'''
[cached]
preprocess data to analyze and visualize such as categorization, one-hot encoding

: param file_path : file path to test result csv file

: return : preprocessed pandas dataframe 
'''
@st.cache_data
def prepocess_for_vanlysis(file_path:str) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    print(f'# [Engine] preprocessing begin for a test result : {file_path}')
   
    # read test result file, remove 'ref' and duplications
    df = pd.read_csv(file_path)
    
    # trim for string values
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    gdict = df.groupby('sentence')['ref'].apply(list).to_dict()
    df = df.drop(columns=['ref']).drop_duplicates().reset_index(drop=True)

    # create 'style' var : written or spoken
    df['accents'] = df['accents'].replace({'사투리가 조금 있는':1, '서울':1, '경기도':1, 'reading book':0, '일반적인 성인 남성 ':1, 'Seoul':1})
    df['style'] = df['accents'].apply(lambda x: 'written' if x == 0 else 'spoken')
    print('- create style var with accents')
    
    # create 'sentence_len'
    df['sentence_len'] = df['sentence'].apply(lambda x: len(x))
    print('- create sentence_len var with accents')

    # create 'blue_diff' = i_bleu - bleu
    df['bleu_diff'] = df['i_bleu'] - df['bleu']
    print('- create bleu_diff var with accents')
    
    # one hot encoding for norminal vars for regression
    columns=['age', 'style', 'gender']
    for col in columns:
        df[col+'_org'] = df[col]
    df = pd.get_dummies(df, columns=['age', 'style', 'gender'], drop_first=False)
    for col in columns:
        df[col] = df[col+'_org']
    df = df.drop(columns=[col+'_org' for col in columns])
    print(f'- one hot encoding : {columns}')

    # categorize sentence length for table & create 'sentence_len_type'  
    # - long : larger than 3rd quantile 
    # - mid  : 1st  ~ 3rd quantile
    # - short : smaller than the 1st quantile
    q1 = df.sentence_len.quantile(q = 0.25) 
    q3 = df.sentence_len.quantile(q = 0.75)
    df['sentence_len_type'] = df['sentence_len'].apply(lambda x: 'long' if x > q3 else ('short' if x < q1 else 'mid'))
    print(f'- categorize sentence_len into short, mid, long')
    
    # return preprocessed dataframe
    return df, gdict


'''
[cached]
make a fake data frame with the below structure

[colum structure]
1st              : fcol_name with fcol_values
2nd ~ (last - 1) : columns ← random number generated
last             : some calculateion with the middle position columns, 
                   currently 'diff' (last - first) at the middle position column

: param fcol_name   : the name of first column
: param fcol_values : the values assigned to the first column
: param columns     : the name of middle position columns whose values are assgined with random
: param make_last   : if True, make a special column for a given calcuation type
: param type_last   : currentl diff only supported
                    : last column value - first colum value in the given columns
'''
@st.cache_data
def random_dataframe(fcol_name:str, fcol_values:List[str], 
                     columns:List[str], make_last:bool = False, type_last:str = '') -> pd.DataFrame:
    
    print(f'# [Engine] create dataframe to display ')
        
    columns.insert(0, fcol_name)
    if(make_last & (len(columns) > 1)):
        columns.append(type_last)
    
    rdf = pd.DataFrame(columns = columns)
    print(f'- make columns : {columns}')
        
    for r_idx, fcol_val in enumerate(fcol_values):
        # dictionary to fill into a raw
        tmp = {}
        # from the first to the last
        for c_idx, col in enumerate(columns):
            if c_idx == 0:
                tmp[col] = fcol_val
            else:
                tmp[col] = np.random.normal(0.5, 0.3)
        # last column
        if(make_last & (len(columns) > 1)):
            tmp[columns[-1]] = tmp[columns[1]] - tmp[columns[-1]] 
            
        # fill into this raw
        rdf.loc[r_idx] = tmp
        
    print(f'- construct dataframe  : ')
    print(rdf.head(2))
    
    return rdf

'''
generate dummy test result file to independent implemnetation of analyzer and ui
'''
def _generate_dummy_result(suite_file_path:str, result_file_root:str, result_file_name:str, n_result:int) -> None: 
    df_org = pd.read_csv(suite_file_path, sep='\t')

    # 'x' values
    # age : 10s = 0 , 20s = 1, 30s = 2, 40s = 3
    df_org.age.unique()     # nan, 'twenties', 'teens', 'thirties', 'fourties'
    df_org['n_age'] = df_org['age'].replace({'twenties':1, 'teens':0, 'thirties':2, 'fourties':3})

    # gender : male = 0 , female = 1
    df_org.gender.unique()  # nan, 'male', 'female'
    df_org['n_gender'] = df_org['gender'].replace({'male':0 , 'female':1})

    # sentence style : spoken = 0, written = 1 
    df_org.accents.unique() # nan, '사투리가 조금 있는', '서울', '경기도', 'reading book', '일반적인 성인 남성 ', 'Seoul'
    df_org['n_style'] = df_org['accents'].replace({'사투리가 조금 있는':1, '서울':1, '경기도':1, 'reading book':0, '일반적인 성인 남성 ':1, 'Seoul':1})
    df_org['style'] = df_org['n_style'].apply(lambda x: 'written' if x == 0 else 'spoken')

    # setence length
    df_org['sentence_len'] = df_org['sentence'].apply(lambda x: len(x))
    df_org['n_sentence_len'] = df_org['sentence_len']

    # min-max scaling for "^n_*" columns
    cols_to_scale = df_org.filter(regex='^n_').columns
    df_org[cols_to_scale] = MinMaxScaler().fit_transform(df_org[cols_to_scale])

    # translation
    df_org['translation'] = df_org['sentence'].apply(lambda x: '[Translated]' + x)
    
    # transcript
    df_org['transcript'] = df_org['sentence'].apply(lambda x: '[Transcripted]' + x)

    # path 
    df_org['path'].apply(lambda x: os.path.join('testsuites/cv-corpus-15.0-2023-09-08/ko/clips', 'path'))
    
    # 'y' values to generate depdending on 'x'values
    df_org['n_wer'] = None
    df_org['n_bleu'] = None
    df_org['n_i_bleu'] = None

    for i in range(n_result):
        df = df_org.copy()
        # generate random betas from normal distribution
        def generate_betas(locs:List[float] = [2.5, 1.5, 1.0, 0.7]) -> List[float]:
            beta = [
                np.random.normal(),                 # interception
                np.random.normal(2.5, 0.1),         # setence length
                np.random.normal(1.5, 0.06),        # setence style
                np.random.normal(1.0, 0.05),        # gender
                np.random.normal(0.7, 0.1),         # age
            ]
            return beta

        # wer = b0 + setence_length * b1 + style * b2 + gender * b3 + age * b4 +  error
        beta = generate_betas()
        df['n_wer'] = beta[0] + beta[1] * df['n_sentence_len'] \
                + beta[2] * df['n_style'] + beta[3] * df['n_gender'] + beta[4] * df['n_age']\
                + np.random.normal(10, 5)
        df['n_wer'] = MinMaxScaler().fit_transform(df[['n_wer']])
        df.loc[df['n_wer'].isna(),'n_wer'] = np.random.normal(0 , 1)
        df = df.rename(columns={'n_wer':'wer'})

        # bleu = b0 + setence_length * b1 + style * b2
        beta = generate_betas([3.0, 2.5, 1.5, 1.2])
        df['n_bleu'] = beta[0] + beta[1] * df['n_sentence_len'] + beta[2] * df['n_style'] + np.random.normal(10,5)
        df['n_bleu'] = MinMaxScaler().fit_transform(df[['n_bleu']])
        df.loc[df['n_bleu'].isna(),'n_bleu'] = np.random.normal(0 , 1)
        df = df.rename(columns={'n_bleu':'bleu'})

        # i_bleu = b0 + setence_length * b1 + style * b2
        beta = generate_betas([1.0, 1.5, 0.5, 0.2])
        df['n_i_bleu'] = beta[0] + beta[1] * df['n_sentence_len'] + beta[2] * df['n_style'] + np.random.normal(10,5)
        df['n_i_bleu'] = MinMaxScaler().fit_transform(df[['n_i_bleu']])
        df.loc[df['n_i_bleu'].isna(),'n_i_bleu'] = np.random.normal(0 , 1)
        df = df.rename(columns={'n_i_bleu':'i_bleu'})

        df = df.drop(columns = df.filter(regex = '^n_').columns)
        
        # force fomatting specific for the 'amt test' project's test results
        df = pd.concat([df] * 3)
        
        # fake reference text
        fake = Faker()
        df['ref'] = [fake.text() for _ in range(df.shape[0])]
        df = df.sort_values(by='path', ascending=1).reset_index(drop=True)

        # save test result
        result_file_path = os.path.join(result_file_root, result_file_name)
        df.to_csv(f'{result_file_path} - {i}.csv', index=False)
        

'''
this is mediator from the aspect of data to display bridging the raw test result and what to show in ui 
'''
class TestAnalyzer(ABC):
    def __init__(self, result_root_path:str):
        print('# [Analyzer] TestAnalyzer initialization')
        
        self.result_root_path = result_root_path
        
        # [assumption] result file name starts with testresult and its extesion is csv
        pattern = r'^testresult.*\.csv$'  
        self.result_files =  [f for f in os.listdir(result_root_path) if re.match(pattern, f)]
        self.result_files = sorted(self.result_files, reverse=True)
        print('- initialization with results files :')
        print(' \n'.join(self.result_files))
        
        # construct index tables with bi-directional ways
        self.file_to_idx = {file: idx for idx, file in enumerate(self.result_files)}
        self.idx_to_file = {idx: file for idx, file in enumerate(self.result_files)}

        # test result as dataframe
        self.cdf = None     # current test result (the selected)
        self.pdf = None     # previsou test result (right before the selected, None if the selected is the first)
        self.cgdict = None  # current sentence vs refs
        self.pgdict = None  # previous sentence vs refs

        # configuration flag
        self.configured = False
        
        # language codes , this order SHOULD BE KEPT
        self.codes = ['KR', 'EN', 'ES', 'FR', 'DE', 'IT', 'JP', 'CN', 'RU', 'PT', 
                      'AR', 'HI', 'SW', 'NL', 'SV', 'PL', 'TR', 'TH', 'HE', 'DA']

        # aspect identifiers at ui 
        self.aspects_names_list = [
            'Utterance Length', 
            'Utterance Style' 
            'Speaker Age', 
            'Speaker Gender'
        ]

        self.aspects_names_dict = {
            'Utterance Length':0, 
            'Utterance Style':1, 
            'Speaker Age':2, 
            'Speaker Gender':3
            }

        # aspect identifiers at dataframe
        self.aspects_columns_dict = {
            'Utterance Length':'sentence_len', 
            'Utterance Style':'style', 
            'Speaker Age':'age', 
            'Speaker Gender':'gender'
        }
        
        self.aspects_max_values_dict  = {
            'Utterance Length': None, 
            'Utterance Style': None, 
            'Speaker Age': None, 
            'Speaker Gender':None
        }

        self.aspects_min_values_dict  = {
            'Utterance Length': None, 
            'Utterance Style': None, 
            'Speaker Age': None, 
            'Speaker Gender':None
        }
        
        self.aspects_values_dict = {}
        self.metric_diff_min = None
        self.metric_diff_min = None

    '''
    this can be extended so as to set strategy (what to and how to anlyze) in the future
    
    : param selected_file : one of test result file
    
    : return : test
    ''' 
    def configure(self, selected_file:str) -> None:
        print(f'# [Analyzer] configure analyzer with test result files')
        
        # preprocessed dataframe for the previous test result
        print('selected file : ', selected_file)
        _idx = self.file_to_idx[selected_file]
        print('_idx : ', _idx, '/ ', (len(self.result_files) - 1))
        if(_idx < (len(self.result_files) - 1)):
            self.pdf , self.pgdict = prepocess_for_vanlysis(os.path.join(self.result_root_path, self.idx_to_file[_idx+1]))
            print(f'- preprocess previsou test result : {self.idx_to_file[_idx+1]}')
        
        # preprocessed dataframe from the selected test result
        self.cdf, self.cgdict = prepocess_for_vanlysis(os.path.join(self.result_root_path, selected_file))
        print(f'- preprocess current test result : {selected_file}')
        
        # super set that each aspect has
        _an_key_list = list(self.aspects_names_dict.keys()) 
        self.aspects_values_dict[_an_key_list[0]] = None
        for aspect in _an_key_list[1:]: # except for utterance length
            aspect_vals = list(self.cdf[self.aspects_columns_dict[aspect]].dropna().unique())
            self.aspects_values_dict[aspect] = aspect_vals 
        
        # max values of nornimal variables for each category, e.g., female   
        self.aspects_max_values_dict[_an_key_list[0]] = float(self.cdf[self.aspects_columns_dict[self.aspects_names_list[0]]].max())
        self.aspects_min_values_dict[_an_key_list[0]] = float(self.cdf[self.aspects_columns_dict[self.aspects_names_list[0]]].min())
        
        self.metric_diff_min = self.cdf['bleu_diff'].min()
        self.metric_diff_max = self.cdf['bleu_diff'].max()
        
        # set configure flag
        self.configured = True
        print(f'- extract simple statistics (max, min)')
        pprint(self.aspects_max_values_dict)
        pprint(self.aspects_min_values_dict)
        pprint(self.metric_diff_max)
        pprint(self.metric_diff_min)
    
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

        print(f'# [Analyzer] requested average for {metric_name} : {round(self.cdf[metric_name].mean(),ndigits)}')
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
    def get_dataframes_ut(self, metric_name:str, type:str) -> List[pd.DataFrame]:
        print(f'# [Analyzer] requested to make data frames for unit test with {metric_name} and {type}')
        
        # for abreviation
        cdf = self.cdf
        pdf = self.pdf
        codes = self.codes 
        
        frames = []
        
        # Langguage Dataframe for ASR, MT, Integration
        # - fill random values into the data frame
        print(f'- language dataframe created')
        ldf = random_dataframe(fcol_name='lang', fcol_values = codes, 
                               columns=[f'{metric_name}_c', f'{metric_name}_p'], 
                               make_last=True, type_last='diff')

        # update with real test result only for KR
        print(f'- language dataframe is updated with real test result for KR')
        ldf.loc[0, [f'{metric_name}_c']] = cdf[metric_name].mean()
        if pdf is not None:
            ldf.loc[0, [f'{metric_name}_p']] = pdf[metric_name].mean()
            ldf.loc[0, ['diff']] = cdf[metric_name].mean() - pdf[metric_name].mean()
        else:
            ldf.loc[0, [f'{metric_name}_p']] = None
            ldf.loc[0, ['diff']] = None
        frames.append(ldf)
        
        # Utterance Dataframe for ASR, MT
        # - after dropping missing value
        # - fill random values into the data frame
        print(f'- utterance length dataframe is created')
        columns = cdf['sentence_len_type'].dropna().unique().tolist()
        udf = random_dataframe(fcol_name='lang', fcol_values = codes, columns = columns)
        # update with real test result only for KR
        for col in columns[1:]:
            udf.loc[0, [col]] = cdf.groupby('sentence_len_type')[metric_name].mean()[col]
        frames.append(udf)
        
        if type == 'ASR':
            # Age Dataframe for ASR
            # - after dropping missing value
            # - fill random values into the data frame
            print(f'- age dataframe is created')
            columns = cdf['age'].dropna().unique().tolist()
            adf = random_dataframe(fcol_name='lang', fcol_values = codes, columns = columns)
            # update with real test result only for KR
            for col in columns[1:]:
                adf.loc[0, [col]] = cdf.groupby('age')[metric_name].mean()[col]
            frames.append(adf)
            
            # Gender Dataframe for ASR
            # - after dropping missing value
            # - fill random values into the data frame
            print(f'- gender dataframe is created')
            columns = cdf['gender'].dropna().unique().tolist()
            gdf = random_dataframe(fcol_name='lang', fcol_values = codes, columns = columns)
            
            # update with real test result only for KR
            for col in columns[1:]:
                gdf.loc[0, [col]] = cdf.groupby('gender')[metric_name].mean()[col]
            frames.append(gdf)

        # Style Dataframe for ASR, MT, Integration
        # - after dropping missing value
        # - fill random values into the data frame
        print(f'- style dataframe is created')
        columns = cdf['style'].dropna().unique().tolist()
        sdf = random_dataframe(fcol_name='lang', fcol_values = codes, columns = columns)
        # update with real test result only for KR
        for col in columns[1:]:
                sdf.loc[0, [col]] = cdf.groupby('style')[metric_name].mean()[col]
        frames.append(sdf)
        
        return frames     
    
    '''
    provides test reuslts with dataframes to display on the summary page.
    - language table   :  comparing ut and it  
    - utternable table :  comparing ut and it
    - style table      :  comparing ut and it
    
    : param metric_name:  metric name to compare. 
                          [assumptioin] integration metric result start with 'i_' in the given data frame 
    '''
    def get_dataframes_it(self, metric_name:str) -> List[pd.DataFrame]:
        print(f'# [Analyzer] requested to make data frames for integration test with {metric_name}')
        # for abreviation
        cdf = self.cdf
        pdf = self.pdf
        codes = self.codes 
        
        frames = []
        i_metric_name = 'i_' + metric_name
        u_metric_name = metric_name
        
        # Langguage Dataframe for Integration
        # - fill random values into the data frame
        print('- language data frame is creatd')
        ldf = random_dataframe(fcol_name='lang', fcol_values = codes, 
                               columns=[f'{i_metric_name}_c', f'{i_metric_name}_p'], 
                               make_last=True, type_last='diff')

        # update with real test result only for KR
        print(f'- language dataframe is updated with real test result for KR')
        ldf.loc[0, [f'{i_metric_name}_c']] = cdf[i_metric_name].mean()
        if pdf is not None:
            ldf.loc[0, [f'{i_metric_name}_p']] = pdf[i_metric_name].mean()
            ldf.loc[0, ['diff']] = cdf[i_metric_name].mean() - pdf[i_metric_name].mean()
        else:
            ldf.loc[0, [f'{i_metric_name}_p']] = None
            ldf.loc[0, ['diff']] = None
        frames.append(ldf)
        
        print(ldf.head(2))
        
        # Utterance Dataframe for Integration
        # - after dropping missing value
        # - fill random values into the data frame
        # define frame scheme
        print(f'- utterance length dataframe is created')
        columns = cdf['sentence_len_type'].dropna().unique().tolist()
        # This MUST BE located at random_dataframe, but... not enough time
        ext_cols = list(columns)
        for i in range(len(columns)):
            ext_cols.append(f'diff_{i}')
        udf = random_dataframe(fcol_name='lang', fcol_values = codes, columns = ext_cols)

        # update with real test result only for KR
        for i, col in enumerate(columns[1:]): 
            i_result = cdf.groupby('sentence_len_type')[i_metric_name].mean()[col]
            u_result = cdf.groupby('sentence_len_type')[u_metric_name].mean()[col]
            udf.loc[0, [f'diff_{i}']] = i_result
            udf.loc[0, [f'diff_{i}']] = i_result - u_result
        frames.append(udf)
        
        print(udf.head(2)) 
        
        # Style Dataframe for Integration
        # - after dropping missing value
        # - fill random values into the data frame
        print(f'- style dataframe is created')
        columns = cdf['style'].dropna().unique().tolist()
        ext_cols = list(columns)
        for i in range(len(columns)):
            ext_cols.append(f'diff_{i}')
        sdf = random_dataframe(fcol_name='lang', fcol_values = codes, columns = ext_cols)

        # update with real test result only for KR
        for  i, col in enumerate(columns[1:]): 
            i_result = cdf.groupby('style')[i_metric_name].mean()[col]
            u_result = cdf.groupby('style')[u_metric_name].mean()[col]
            sdf.loc[0, [f'diff_{i}']] = i_result
            sdf.loc[0, [f'diff_{i}']] = i_result - u_result
        frames.append(sdf)
        
        return frames     
    
    def get_analysis_result_ut(self, language:str, metric_name:str, aspect_name:str) -> Tuple[Any, List[str] , Optional[pd.DataFrame]]:
        cdf = self.cdf
        if(language != 'KR'):
            return None
        
        y_var = metric_name
        
        aspect_index = self.aspects_names_dict[aspect_name] 
        if aspect_index == 0:
            x_var = 'sentence_len'
            print(f'# [Analyzer] numeric variable anlysis for ut is requested : {x_var},{y_var}')
            return v_analyze_numerics(cdf, x_var, y_var)
        elif aspect_index == 1:
            x_var = 'style'
        elif aspect_index == 2:
            x_var = 'age'
        elif aspect_index == 3:    
            x_var = 'gender'
            print(f'# [Analyzer] norminal variable anlysis for ut is requested : {x_var},{y_var}')
        return v_analyze_categorics(cdf, x_var, y_var)


    def get_analysis_result_it(self, language:str, metric_name:str, aspect_name:str) -> Tuple[Any, List[str] , Optional[pd.DataFrame]]:
        cdf = self.cdf
        if(language != 'KR'):
            return None
        
        # for simplicity, test for metric diff between unit and integration
        # TODO : paired sample anlaysis or merge this into get_analysis_result  
        y_var = metric_name + '_diff'  
        
        aspect_index = self.aspects_names_dict[aspect_name] 
        if aspect_index == 0:
            x_var = 'sentence_len'
            print(f'# [Analyzer] numeric variable anlysis for it is requested : {x_var},{y_var}')
            return v_analyze_numerics(cdf, x_var, y_var)  
        elif aspect_index == 1:
            x_var = 'style'
        elif aspect_index == 2:
            x_var = 'age'
        elif aspect_index == 3:    
            x_var = 'gender'
        
        print(f'# [Analyzer] norminal variable anlysis for it is requested : {x_var},{y_var}')
        return v_analyze_categorics(cdf, x_var, y_var)

    
    ##
    # Query by condition
    ##
    def get_testresults_by_numeric_asr(self,  aspect_name:str, aspect_max:float, aspect_min:float, 
                                   metric_name:str, metric_max:float, metric_min:float,
                                   ret_columns:List[str]) -> List[List[str]]:
        # abbrivation
        cdf = self.cdf
        print('# [Analyzer] asr test result (numeric) query is requested')
        
        # conditional slicing for each given column in the ret_columns
        ret = []
        condition1 = (cdf[aspect_name] >= aspect_min) & (cdf[aspect_name] <= aspect_max)
        condition2 = (cdf[metric_name] <= metric_max) & (cdf[metric_name] >= metric_min)
        tdf = cdf[condition1 & condition2]
        for i, ret_col in enumerate(ret_columns):
            ret.append(tdf[ret_col].to_list())
            
        # exceptional case
        if len(ret) != len(ret_columns):
            ret = [[],[],[],[]]
        
        for i, ret_col in enumerate(ret_columns):
            print(f'- {ret_col}, {len(ret[i])}')
        
        return ret

    def get_testresults_by_categoric_asr(self,  aspect_name:str, aspect_val:str, 
                                     metric_name:str, metric_max:float, metric_min:float,
                                     ret_columns:List[str]) -> List[List[str]]:
        # abbrivation
        cdf = self.cdf
        print('# [Analyzer] asr test result(norminal) query is requested')

        # conditional slicing for each given column in the ret_columns
        ret = []
        condition1 = (cdf[aspect_name] == aspect_val)
        condition2 = (cdf[metric_name] <= metric_max) & (cdf[metric_name] >= metric_min)
        tdf = cdf[condition1 & condition2]
        for i, ret_col in enumerate(ret_columns):
            ret.append(tdf[ret_col].to_list())

        # exceptional case
        if len(ret) != len(ret_columns):
            ret = [[],[],[],[]]

        for i, ret_col in enumerate(ret_columns):
            print(f'- {ret_col}, {len(ret[i])}')

        return ret
    
    def get_testresults_by_numeric_mt(self,  aspect_name:str, aspect_max:float, aspect_min:float, 
                                      metric_name:str, metric_max:float, metric_min:float) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
        # abbrivation
        cdf = self.cdf.copy()
        print(f'# [Analyzer] mt test result (numeric) query is requested with {metric_min} ~ {metric_max} for {aspect_max} ~ {aspect_min}')
        
        # conditional slicing for each given column in the ret_columns
        condition1 = (cdf[aspect_name] >= aspect_min) & (cdf[aspect_name] <= aspect_max)
        condition2 = (cdf[metric_name] <= metric_max) & (cdf[metric_name] >= metric_min)
        cdf = cdf[condition1 & condition2]
        
        sentences = ('[' + cdf['bleu'].round(2).astype(str) + '] ' + cdf['sentence']).to_list()
        transcript_dict = cdf[['sentence','transcript']].set_index('sentence')['transcript'].to_dict()  # TODO CHECK transcript is correct ?
        
        print(f'- setences : {len(sentences)} , transcript : {len(transcript_dict)}, ground : {len(self.cgdict)}')
        return sentences, transcript_dict, self.cgdict
    
    def get_testresults_by_categoric_mt(self,  aspect_name:str, aspect_val:str, 
                                     metric_name:str, metric_max:float, metric_min:float)-> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
        # abbrivation
        cdf = self.cdf.copy()
        print(f'# [Analyzer] mt test result (numeric) query is requested with {metric_min} ~ {metric_max} for {aspect_val}')
        
        # conditional slicing for each given column in the ret_columns
        condition1 = (cdf[aspect_name] == aspect_val)
        condition2 = (cdf[metric_name] <= metric_max) & (cdf[metric_name] >= metric_min)
        cdf = cdf[condition1 & condition2]

        sentences = ('[' + cdf['bleu'].round(2).astype(str) + '] ' + cdf['sentence']).to_list()
        transcript_dict = cdf[['sentence','transcript']].set_index('sentence')['transcript'].to_dict()

        print(f'- setences : {len(sentences)} , transcript : {len(transcript_dict)}, ground : {len(self.cgdict)}')
        return sentences, transcript_dict, self.cgdict