# Data Preprocessing

**23.05.22**

seaborn에서 제공하는 titanic 데이터를 이용하였다.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
```

### Missing Value

1. **결측치 탐색**
    
    **info( )** 를 통해 데이터프레임의 전반적인 정보를 컬럼별로 확인할 수 있다.
    **Non-Null Count**를 통해 결측치 현황을, **Dtype**을 통해 데이터 타입을 파악할 수 있다.
    
    ```python
    df.info()
    ```
    
    ```python
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 80.7+ KB
    ```
    
    **isnull( ).sum( )** 을 통해 컬럼별로 **Null값**의 수를 알 수 있다.
    
    ```python
    df.isnull().sum()
    ```
    
    ```python
    survived         0
    pclass           0
    sex              0
    age            177
    sibsp            0
    parch            0
    fare             0
    embarked         2
    class            0
    who              0
    adult_male       0
    deck           688
    embark_town      2
    alive            0
    alone            0
    dtype: int64
    ```
    
    그래프를 통해서 결측치 현황을 시각적으로 확인이 가능하다.
    
    ```python
    import missingno as msno
    
    msno.matrix(df, color=(0.1, 0.6, 0.6))
    plt.show()
    ```
    
    ![Untitled](/img/데이터전처리_1.png)
    
    ```python
    msno.bar(df, color=(0.1, 0.6, 0.6))
    plt.show()
    ```
    
    ![Untitled](/img/데이터전처리_2.png)
    
2. **결측치 처리**
    
    결측치는 보통 삭제하거나, 다른 값으로 대체한다. 
    (삭제는 이후 학습에 영향을 줄 수 있으므로 보통은 다른 값으로 대체한다.)
    다른 값으로 대체 시에는 평균, 중앙값등을 이용한다.
    
    **dropna( )** 를 통해 결측치를 제거할 수 있다. 
    thresh : 결측값이 아닌 값이 몇 개 미만일 경우에만 dropna 메서드를 적용시키는 인수
    
    결측값의 갯수가 df의 길이의 절반 미만일 경우 dropna 메서드를 수행하므로 결측치가 
    과반수 이상이었던 **deck** 컬럼이 삭제되었다.
    
    ```python
    df = df.dropna(thresh=int(len(df)/2), axis=1)
    df.isnull().sum()
    ```
    
    ```python
    survived         0
    pclass           0
    sex              0
    age            177
    sibsp            0
    parch            0
    fare             0
    embarked         2
    class            0
    who              0
    adult_male       0
    embark_town      2
    alive            0
    alone            0
    dtype: int64
    ```
    
    **notnull( )** 을 통해 Null값이 아닌 경우만 걸러낼 수 있다. <br>
    **embarked** 컬럼의 경우 결측값의 수가 2로 학습에 큰 영향을 주지 않기 때문에 삭제되었다.
    
    ```python
    df = df[df.embarked.notnull()]
    df.isnull().sum()
    ```
    
    ```python
    survived         0
    pclass           0
    sex              0
    age            177
    sibsp            0
    parch            0
    fare             0
    embarked         0
    class            0
    who              0
    adult_male       0
    embark_town      0
    alive            0
    alone            0
    dtype: int64
    ```
    
    age의 경우 중앙값으로 결측치를 대체하였다.
    
    ```python
    df.age.fillna(value = df.age.median(), inplace = True)
    df.isnull().sum()
    ```
    
    ```python
    survived       0
    pclass         0
    sex            0
    age            0
    sibsp          0
    parch          0
    fare           0
    embarked       0
    class          0
    who            0
    adult_male     0
    embark_town    0
    alive          0
    alone          0
    dtype: int64
    ```
    

### Outlier

이상치의 경우에는 보통 사분위수를 이용하여 전처리한다.
(이상치를 포함하여 학습하는 경우에 학습 정확도를 떨어뜨릴 수 있다.)
특정 값이 Boxplot의 최대/최소 값의 범위에서 벗어난 경우에 제거한다.

컬럼별로 반복문을 수행하면서 이상치를 제거한다.
아래 코드는 데이터프레임에 이상치가 아닌 값들을 저장하도록 작성했다.

```python
Q1 = df.quantile(0.25) # 1분위수
Q3 = df.quantile(0.75) # 3분위수
IQR = Q3 - Q1

for i in df.columns:
	df = df[ (df[i] <= Q3[i] + 1.5*IQR[i]) & (df[i] >= Q1[i] - 1.5*IQR[i]) ]
```

### Data Transform

- 왜도
    
    왜도(또는 비대칭도)는 실수 값 확률 변수의 확률분포 비대칭성을 나타내는 지표
    왜도의 값은 양수나 음수가 될 수 있으며, 정의되지 않을 수 있다.
    
    ```python
    round(df.fare.skew(), 2)
    ```
    
    ```python
    skew :  4.79
    ```
    
- 첨도
    
    확률분포의 꼬리가 두꺼운 정도를 나타내는 지표
    극단적인 편차 또는 이상치가 많을수록 큰 값을 나타낸다. 첨도값(K)이 3에 가까우면
    산포도가 정규분포에 가까우며, 3보다 작을 경우에는 정규분포보다 꼬리가 얇은 분포, 3보다 클 경우에는 정규분포보다 꼬리가 두꺼운 분포로 판단할 수 있다.
    
    ```python
    round(df.fare.kurt(), 2)
    ```
    
    ```python
    kurtosis :  33.4
    ```
    

### Scaling

- **Standard Scaler**
    
    개별 feature를 평균이 0, 분산이 1인 값으로 변환해주는 클래스
    
    ```python
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    
    ss.fit(df)
    ss.transform(df)
    ss.fit_transform(df)
    ```
    
- **Min-Max Scaler**
    
    데이터 값을 0과 1 사이의 범위 값으로 변환 (음수 값이 있다면, -1에서 1 값으로 변환)
    모든 특성이 값은 크기를 가지도록 한다.
    
    ```python
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    
    mms.fit(df)
    mms.transform(df)
    mms.fit_transform(df)
    ```
    
- **Robust Scaler**
    
    평균과 분산 대신에 중간값과 사분위값을 조정한다.
    특성들이 같은 스케일을 갖게 되고, 극단값에 영향을 받지 않는다.
    
    ```python
    from sklearn.preprocessing import RobustScaler
    rs = RobustScaler()
    
    rs.fit(df)
    rf.transform(df)
    rf.fit_transform(df)
    ```
    
- **Log Scale**
    
    ```python
    df['fare_log'] = np.log(df.fare)
    ```
    
- **fit_transform( )**
    
    **fit( )** 과 **transform( )** 을 순차적으로 수행하는 메서드
    학습데이터에서는 사용하여도 괜찮으나, 테스트 데이터에서는 사용 X
    
    ```python
    1. 가능하다면 전체 데이터의 스케일링 변환을 적용한 뒤 학습과 테스트 데이터로 분리
    
    2. 1이 여의치 않다면 테스트 데이터 변환 시에는 fit( )이나 fit_transform( )을 
    	적용하지않고, 학습 데이터로 이미 fit( )된 Scaler 객체를 이용해 transform( )으로
    	변환
    ```
    

### Featuring

- 더미화
    
    범주형 변수를 수치형 변수로 변경해주는 방법
    
    ```python
    # 1 : map() 이용
    df['cols'] = df['cols'].map({'A':0, 'B':1, 'C':2})
    ```
    
    ```python
    # 2 : get_dummies() 이용
    x_dummies = pd.get_dummies(x)
    ```
    

---

- 참고자료
    
    [sklearn으로 데이터 스케일링(Data Scaling)하는 5가지 방법🔥](https://dacon.io/codeshare/4526)