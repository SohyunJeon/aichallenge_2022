import pandas as pd
import numpy as np
import json

import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings



train = json.load(open('./data/train.json', 'r', encoding='utf8'))
submit = json.load(open('./data/sample_submission.json', 'r', encoding='utf8'))


#%%
data = train.copy()
result = {}
for k in data.keys():
    temp_df = pd.DataFrame.from_dict(data[k]).T
    temp_df = temp_df.reset_index().rename(columns={'index': 'date'})
    result[k] = temp_df


#%% 전처리 : 분석을 위한 데이터 형태 변환

def create_dataset(data:dict) -> dict:
    result = {}
    for k in data.keys():
        temp_df = pd.DataFrame.from_dict(data[k]).T
        temp_df = temp_df.reset_index().rename(columns={'index': 'date'})
        result[k] = temp_df
    return result

new_train = create_dataset(train)
new_submit = create_dataset(submit)

#%% Logic 1 :  trend 반영
range_train = 11 # (학습 데이터) 트렌드 학습 구간 길이
range_test = 20 # (테스트 데이터) 동일한 값을 적용할 길이
smooth = 0.91 # (테스트 데이터) 트렌드 적용 크기


for k in new_train.keys():
    print(k)
    # 학습 데이터에서 최근 평균
    temp_df = new_train[k].drop(['date'], axis=1)
    train_mean = np.nanmean(temp_df[::-1][:range_train], axis=0)
    train_mean = np.nan_to_num(train_mean, nan=0)

    # 테스트 데이터에 적용
    cycle = int(len(new_submit[k]) / range_test)
    for c in range(0, cycle):
        train_mean = train_mean * smooth
        apply_start = c * range_test
        apply_end = apply_start + range_test

        new_submit[k].iloc[apply_start:apply_end, 1:] = train_mean


#%% Logic 2 : 계절성 반영
def get_value(month: int, elem: str) -> float:
    result = month_mean.loc[month, elem]
    return result

threshold = 65
season_start = '20160201'
season_end = '20180201'


for k in new_train.keys():
    print(k)
    temp_target = new_train[k]
    analysis_period = temp_target.loc[(season_start <= temp_target['date']) & \
                                      (season_end > temp_target['date'])]

    check_temp = analysis_period.isnull().sum() # 모든값이 nan인 컬럼에 대해 0으로 대체
    check_cols = check_temp == len(analysis_period)
    analysis_period.loc[:, check_cols] = 0

    # month 추출
    analysis_period['month'] = pd.to_datetime(analysis_period['date']).dt.month

    # anova 검증을 통한 월별 차이의 유의미성 확인
    elems = check_cols.index.drop('date')
    aov_list = []
    for elem in elems:
        aov_model = ols(f'{elem} ~ C(month)', data=analysis_period).fit()
        aov = sm.stats.anova_lm(aov_model, typ=2).iloc[0, 2]
        if len(analysis_period[elem].unique()) == 1: # 모든값이 동일한 경우 예외처리
            aov = 0
        aov_list.append(aov)

    # 전체 기간의 월별 평균 계산
    temp_target['month'] = pd.to_datetime(temp_target['date']).dt.month
    month_mean = temp_target.groupby(['month']).mean()

    # 테스트 데이터에 적용
    submit_month = pd.to_datetime(new_submit[k]['date']).dt.month
    for elem, aov_val in zip(elems, aov_list):
        if aov_val > threshold:
            print(f'elem: {elem} / aov_val: {aov_val}')
            new_submit[k][elem] = submit_month.apply(lambda x: get_value(x, elem))



