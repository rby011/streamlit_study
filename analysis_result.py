import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy.stats import pearsonr, shapiro
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import MinMaxScaler

from typing import List, Tuple, Any
from faker import Faker 


'''
chart creation and statstic analysis for two numerical variable
- scatter plot : display data in the space (x,y) 
- pearson correlation , regression anlaysis 

: param df    : panda dataframe including x_var and y_var
: param x_var : colunm name to analyze , esp., independent variable
: param y_var : colunm name to analyze , esp., depdent variable

: return : chart ojbect , string list to display 
'''
def v_analyze_numerics(df:pd.DataFrame , x_var:str, y_var:str) -> Tuple[Any, List[str], Any]:
    # strings to display
    ret = []
    
    print(f'analysis between {x_var} with {y_var}')
    
    # pearson analysis TODO : view setting MUST be located at other module
    correlation, p_value = pearsonr(df[x_var], df[y_var])
    ret.append(f'* pearson analysis')
    ret.append(f'    * coefficient ({correlation}) with p-value({p_value})')
    ret.append(f"       * {x_var} normality (shapiro-wilk): {shapiro(df[x_var])[1]}")
    ret.append(f"       * {y_var} normality (shapiro-wilk): {shapiro(df[y_var])[1]}")

    # regression analysis
    model = ols(f'{y_var} ~ {x_var}', data = df).fit()
    ret.append(f'* regression ')
    ret.append(f'  * coefficient ({model.params[0]}) with p-value({model.f_pvalue})')
    ret.append(f"     * residual nomality (shaprio-wilk): {shapiro(model.resid)[1]}")
    ret.append(f"     * residual homogeneity of variances (breusch-pagan) : {het_breuschpagan(model.resid, model.model.exog)[1]}")
    
    # chart
    fig, ax = plt.subplots()
    sns.scatterplot(x = x_var, y = y_var, data = df, ax=ax)
    return fig , ret, []


'''
chart creation and statstic analysis for two cateogrical variable
- violin plot : display box plot and distribution for each category 
- regression anlaysis , one-way anova 

: param df    : panda dataframe including x_var and y_var
: param x_var : colunm name to analyze , esp., independent variable
: param y_var : colunm name to analyze , esp., depdent variable

: return : chart ojbect , average data frame for each category, string list to display 
'''
def v_analyze_categorics(df:pd.DataFrame , x_var:str, y_var:str) -> Tuple[Any, List[str], pd.DataFrame]:
    ret = []

    # display statistical analysis result
    # 1) average for each category
    avg_df = pd.DataFrame(df.groupby(by=x_var)[y_var].mean())

    # 2) analysis of variance
    # fit regression model
    formula = f'{y_var} ~ ' + ' + '.join(df.filter(regex=f'^{x_var}_').columns)
    model = ols(formula, data = df).fit()
    # one way anova and extract p-value from the result
    p_val = anova_lm(model).iloc[0,-1]

    ret.append(f'* anova analysis p-value : {p_val}')
    ret.append(f"  * residual nomality (shaprio-wilk) : {shapiro(model.resid)[1]}")
    ret.append(f"  * residual homogeneity of variances (breusch-pagan) : {het_breuschpagan(model.resid, model.model.exog)[1]}")

    # chart
    fig, ax = plt.subplots()
    sns.violinplot(x=x_var, y=y_var, data=df , palette = sns.color_palette("hls", len(df[x_var].unique())),ax=ax)
    
    return fig, ret, avg_df


'''
preprocess data to analyze and visualize such as categorization, one-hot encoding

: param file_path : file path to test result csv file

: return : preprocessed pandas dataframe 
'''
def prepocess_for_vanlysis(file_path:str) -> pd.DataFrame:
   
    # read test result file, remove 'ref' and duplications
    df = pd.read_csv(file_path)
    df = df.drop(columns=['ref']).drop_duplicates().reset_index(drop=True)

    # create 'style' var : written or spoken
    df['accents'] = df['accents'].replace({'사투리가 조금 있는':1, '서울':1, '경기도':1, 'reading book':0, '일반적인 성인 남성 ':1, 'Seoul':1})
    df['style'] = df['accents'].apply(lambda x: 'written' if x == 0 else 'spoken')

    # create 'sentence_len'
    df['sentence_len'] = df['sentence'].apply(lambda x: len(x))

    # one hot encoding for norminal vars for regression
    columns=['age', 'style', 'gender']
    for col in columns:
        df[col+'_org'] = df[col]
    df = pd.get_dummies(df, columns=['age', 'style', 'gender'], drop_first=False)
    for col in columns:
        df[col] = df[col+'_org']
    df = df.drop(columns=[col+'_org' for col in columns])
    
    # categorize sentence length for table & create 'sentence_len_type'  
    # - long : larger than 3rd quantile 
    # - mid  : 1st  ~ 3rd quantile
    # - short : smaller than the 1st quantile
    q1 = df.sentence_len.quantile(q = 0.25) 
    q3 = df.sentence_len.quantile(q = 0.75)
    df['sentence_len_type'] = df['sentence_len'].apply(lambda x: 'long' if x > q3 else ('short' if x < q1 else 'mid'))
    
    # return preprocessed dataframe
    return df


'''
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
def random_dataframe(fcol_name:str, fcol_values:List[str], 
                     columns:List[str], make_last:bool = False, type_last:str = '') -> pd.DataFrame:
    
    columns.insert(0, fcol_name)
    if(make_last & (len(columns) > 1)):
        columns.append(type_last)
    
    rdf = pd.DataFrame(columns = columns)
        
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