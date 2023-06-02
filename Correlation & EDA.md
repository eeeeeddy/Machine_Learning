# Correlation & EDA

**23.05.23**

### Correlation

- 상관분석
    - 연속형 변수로 측정된 두 변수 간의 선형적 관계를 분석
    - A 변수가 증가함에 따라 B 변수도 증가/감소하는지를 분석
    - 상관분석에서 두 변수 사이의 선형적인 관계 정도를 나타내기 위해 상관계수를 사용

sklearn에서 제공하는 iris 데이터를 이용하였다.

```python
from sklearn.datasets import load_iris

df = load_iris()
```

iris 데이터의 iris.data와 iris.target을 이용하여 iris_df를 생성하였다.

```python
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

iris_df = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=cols)
iris_df.target = iris_df.target.map({0:'setosa', 1:'versicolor', 2:'virginica'})
```

생성된 iris_df는 다음과 같다.

|  | sepal_length | sepal_width | petal_length | petal_width | target |
| --- | --- | --- | --- | --- | --- |
| 0 | 5.1 | 3.5 | 1.4 | 0.2 | setosa |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 | setosa |
| 2 | 4.7 | 3.2 | 1.3 | 0.2 | setosa |
| 3 | 4.6 | 3.1 | 1.5 | 0.2 | setosa |
| 4 | 5.0 | 3.6 | 1.4 | 0.2 | setosa |
| ... | ... | ... | ... | ... | ... |
| 145 | 6.7 | 3.0 | 5.2 | 2.3 | virginica |
| 146 | 6.3 | 2.5 | 5.0 | 1.9 | virginica |
| 147 | 6.5 | 3.0 | 5.2 | 2.0 | virginica |
| 148 | 6.2 | 3.4 | 5.4 | 2.3 | virginica |
| 149 | 5.9 | 3.0 | 5.1 | 1.8 | virginica |

**corr( )** 메서드를 통해 각 컬럼간의 상관 계수를 확인할 수 있다.
상관계수 산정 방식에는 **피어슨 상관계수**, **켄달-타우 상관계수**, **스피어먼 상관계수**가 있다.

```python
iris_corr = iris_df.corr()
```

|  | sepal_length | sepal_width | petal_length | petal_width |
| --- | --- | --- | --- | --- |
| sepal_length | 1.000000 | -0.117570 | 0.871754 | 0.817941 |
| sepal_width | -0.117570 | 1.000000 | -0.428440 | -0.366126 |
| petal_length | 0.871754 | -0.428440 | 1.000000 | 0.962865 |
| petal_width | 0.817941 | -0.366126 | 0.962865 | 1.000000 |

### 상관계수 산정 방식

1. 피어슨 상관계수
    
    두 변수 간의 선형 상관 관계를 계량화한 수치. **코시-슈바르츠 부등식**에 의해 +1과 -1 사이의 값을 가진다.
    
    - +1 : 완벽한 양의 선형 상관 관계
    - -1 : 완벽한 음의 선형 상관 관계
    - 0 : 선형 상관 관계를 갖지 않는다.
2. 켄달-타우 상관계수
    
    두 변수들간의 순위를 비교해서 연관성을 계산하는 방식
    
3. 스피어먼 상관계수
    
    두 변수의 순위값 사이의 피어슨 상관계수와 같다.
    즉, 순서척도가 적용되는 경우에는 스피어먼 상관계수가, 간격척도가 적용되는 경우에는 피어슨 상관계수가 적용된다.
    두 변수가 선형관계가 아니어도 스피어먼 상관계수는 1이 될 수 있다. (순위간의 상관계수이기 때문에)
    

### EDA

데이터가 가지고있는 본연의 특징과 의미를 탐색하고, 다양한 각도에서 관찰하고 이해하며,
데이터를 분석하기 전에 **통계적인 방법이나 시각화 도구를 활용하여 데이터를 직관적으로 파악**하는 것

이를 통해, 데이터의 **패턴을 파악하고 잠재적인 변수 간 관계**를 이해하며,
이상치 또는 비정상적인 관측치와 같은 **예외적 현상(anomaly)를 발견** 및
정형화된 통계 방법을 사용하여 검정할 수 있는 **가설 수립을 위한 질문 도출**이 목적이다.

wine 데이터를 이용하여 EDA를 진행하였다.

- **데이터의 전반적인 정보 확인**
    
    행과 열의 크기, 컬럼명, 컬럼을 구성하는 데이터 유형, 결측치 등 정보 확인 가능
    
    ```python
    wine.info()
    ```
    
    ```python
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5320 entries, 1 to 6496
    Data columns (total 13 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   고정산     5320 non-null   float64
     1   휘발산     5320 non-null   float64
     2   구연산     5320 non-null   float64
     3   잔여당     5320 non-null   float64
     4   염화물     5320 non-null   float64
     5   무수아황산   5320 non-null   float64
     6   총이산화황   5320 non-null   float64
     7   밀도      5320 non-null   float64
     8   산성도     5320 non-null   float64
     9   황산염     5320 non-null   float64
     10  알콜도수    5320 non-null   float64
     11  와인품질    5320 non-null   int64  
     12  와인종류    5320 non-null   object 
    dtypes: float64(11), int64(1), object(1)
    memory usage: 581.9+ KB
    ```
    
- **데이터의 통계량 확인**
    
    컬럼별로 다양한 통계량(평균, 분산, 4분위수 등) 확인 가능
    
    ```python
    wine.describe()
    ```
    
    ![1.PNG](/img/상관관계_1.png)
    
- **컬럼별 상관계수 확인**
    
    ```python
    wine_corr = wine.corr()
    ```
    
    ![2.PNG](/img/상관관계_2.png)
    
- **각 변수별 분포 시각화**
    
    ```python
    plt.figure(figsize=(12, 12))
    
    for i in range(0, 11):
    	plt.subplot(3, 4, i+1)
    	sns.distplot(wine.iloc[:,i])
    
    plt.tight_layout()
    plt.show()
    ```
    
    ![06. 상관분석과EDA (1).png](/img/상관관계_3.png)
    
- **pandas data profiling을 이용한 EDA**
    
    ```python
    import pandas_profiling
    profile = wine.profile_report()
    
    profile.to_file('wine_profile.html')
    ```
    
    ![06. 상관분석과EDA.png](/img/상관관계_4.png)
    
- 와인 종류에 따른 알콜 도수의 차이가 있는지?
    - 귀무가설 : 와인 종류에 따라 알콜 도수에 차이가 없다.
    - 대립가설 : 와인 종류에 따라 알콜 도수에 차이가 있다.
    
    ```python
    red = wine[wine.와인종류 == 'red']['알콜도수']
    white = wine[wine.와인종류 == 'white']['알콜도수']
    
    tTestResult = stats.ttest_ind(red, wine)
    tTestResult
    ```
    
    ```python
    Ttest_indResult(statistic=-4.218888835968011, pvalue=2.4959339763303842e-05)
    ```
    
    **p-value = 2.49e-5 < 0.05** 이므로 95% 신뢰수준하에서 귀무가설 기각, 대립가설 채택
    와인 종류에 따라 알콜 도수는 통계적으로 유의미한 차이가 있음을 알 수 있다.