# 대구광역시의 취약장소 식별 및 안전도 향상

**대구광역시 공공데이터 활용 분석 프로젝트**

## 1. 분석 주제

**- 대구광역시의 취약장소 식별을 통한 안전도 향상**

## 1-1. 분석 배경
* 대구시의 범죄율이 꾸준히 증가하고 있으며, 특히 2025년부터 검거율도 감소하고 있어서 이를 해결할 방안의 필요성을 느낌

**<분석내용>**
* 다양한 변수를 활용하여 **범죄 발생이 집중되는 위험지역**을 식별 후, 이를 바탕으로 안전도 향상 방안 도출

## 2. 데이터 설명

### 2-1. 대구광역시 CCTV 정보 
* **출처:** local data
* **데이터 내용:** 대구광역시 CCTV 설치 현황
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 위도 | CCTV 설치 위치 위도 좌표 | Float |
| 경도 | CCTV 설치 위치 경도 좌표 | Float |
| 설치목적구분 | CCTV 설치 목적(차량방범, 재난재해, 교통단속 제외) | Float |


### 2-2. 대구광역시 안전비상벨 위치정보
* **출처:** local data
* **데이터 내용:** 대구광역시 안전비상벨 설치 현황
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 위도 | 안전비상벨 설치 위치 위도 좌표 | Float |
| 경도 | 안전비상벨 설치 위치 경도 좌표 | Float |
| 설치목적구분 | 안전비상벨 설치 목적 구분 | Float |


### 2-3. 대구광역시 유흥주점 현황
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 유흥주점 현황
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 업소명 | 유흥주점 업소명 | Float |
| 업소주소 | 유흥주점 주소 정보 | Float |


### 2-4. 대구광역시 가로등 현황
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 가로등 설치 현황
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 위도 | 가로등 설치 위치 위도 좌표 | Float |
| 경도 | 가로등 설치 위치 경도 좌표 | Float |


### 2-5. 대구광역시 보안등 현황
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 보안등 설치 현황
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 위도 | 보안등 설치 위치 위도 좌표 | Float |
| 경도 | 보안등 설치 위치 경도 좌표 | Float |


### 2-6. 대구광역시 상가정보
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 상가정보
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 상권업종소분류명 | 입시.교과학원, 요리주점, 일반 유흥 주점 | Float |
| 위도 | 상가 위치 위도 좌표 | Float |
| 경도 | 상가 위치 경도 좌표 | Float |

### 2-7. 대구광역시 관서별 경찰서 위치
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 경찰서(경찰서,파출소,지구대) 현황
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 경찰서명 | 경찰서명 정보 | Float |
| 주소 | 경찰서 주소 정보 | Float |


### 2-8. 대구광역시 치안센터 위치
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 치안센터 현황
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 치안센터명 | 치안센터명 정보 | Float |
| 주소 | 치안센터 주소 정보 | Float |


### 2-9. 대구광역시 학교 위치
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 초.중.고등학교 위치
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 이름 | 초.중.고등학교명 정보 | Float |
| 위도 | 초.중.고등학교 위치 위도 좌표 | Float |
| 경도 | 초.중.고등학교 위치 경도 좌표 | Float |

### 2-10. 대구광역시 대학교 위치
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 대학교 현황
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 이름 | 대학교명 정보 | Float |
| 위도 | 대학교 위치 위도 좌표 | Float |
| 경도 | 대학교 위치 경도 좌표 | Float |


### 2-11. 대구광역시 동.읍.면별 세대 및 인구
* **출처:** 공공데이터포털
* **데이터 내용:** 대구광역시 동.읍.면별 세대 및 인구 정보
* **주요 컬럼:**

| 컬럼명 | 설명 | 데이터 타입 |
|--------|------|------------|
| 세대수 | 대구광역시 동.읍.면별 세대수 | Float |
| 등록인구(남) | 대구광역시 동.읍.면별 등록인구(남) | Float |
| 등록인구(여) | 대구광역시 동.읍.면별 등록인구(여) | Float |
| 한국인(남) | 대구광역시 동.읍.면별 한국인(남) | Float |
| 한국인(여) | 대구광역시 동.읍.면별 한국인(여) | Float |
| 외국인(남) | 대구광역시 동.읍.면별 외국인(남) | Float |
| 외국인(여) | 대구광역시 동.읍.면별 외국인(여) | Float |
| 세대당 인구 | 대구광역시 동.읍.면별 세대당 인구수 | Float |
| 인구밀도 | 대구광역시 동.읍.면별 인구밀도 | Float |
| 면적(km^2) | 대구광역시 동.읍.면별 면적 | Float |
| 평균연령 | 대구광역시 동.읍.면별 평균연령 | Float |

### 2-12. 전국 범죄 발생 지역별 통계 데이터 (2023년)
* **출처:** kosis
* **데이터 내용:** 대구광역시 구별 범죄발생수
* **주요 컬럼:**
| 범죄별 | 강력범죄, 폭력범죄 발생수 정보 | Float |


### 2-13. 전국 범죄 발생 지역별 통계 데이터 (2023년)
* **출처:** kosis
* **데이터 내용:** 범죄율 증감률
* **주요 컬럼:**


## 3. 데이터 통합 및 전처리 과정
### 전처리 단계별 과정 

#### 1단계: 위도/경도 좌표로 변환
**목적:** 주소 정보를 위도/경도 좌표로 변환

```python
import pandas as pd
import requests
import time
from tqdm import tqdm

# 카카오 API를 사용한 주소 → 좌표 변환
def get_coords_from_address(address, api_key):
    if not isinstance(address, str) or not address.strip():
        return None, None, "주소 없음"
    
    cleaned_address = re.sub(r'\s*\([^)]*\)', '', address).strip()
    headers = {'Authorization': f'KakaoAK {api_key}'}
    params = {'query': cleaned_address}
    
    try:
        response = requests.get(API_URL, headers=headers, params=params)
        if response.status_code == 401:
            return None, None, "API 키 인증 실패 (401)"
        response.raise_for_status()
        data = response.json()
        
        if data['documents']:
            lon = data['documents'][0]['x']
            lat = data['documents'][0]['y']
            return lat, lon, "성공"
        else:
            return None, None, "결과 없음"
    except Exception as e:
        return None, None, f"오류: {e}"
```



#### 2단계: 행정동 정보 추가 
**목적:** 위도/경도 좌표를 행정동 정보로 변환

#### 2-1. 카카오 API 리버스 지오코딩

```python
def get_dong_from_coords(latitude, longitude, api_key):
    """좌표를 행정동으로 변환"""
    if pd.isna(latitude) or pd.isna(longitude):
        return None
        
    headers = {'Authorization': f'KakaoAK {api_key}'}
    params = {'x': longitude, 'y': latitude}
    
    try:
        response = requests.get(API_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['documents']:
            address_info = data['documents'][0]['address']
            # 행정동 우선, 없으면 법정동 사용
            dong_name = address_info.get('region_3depth_h_name') or \
                       address_info.get('region_3depth_name')
            return dong_name
        else:
            return None
    except Exception:
        return None
```

#### 2-2. VWorld API로 세부 행정동 보완

```python
def get_dong_from_vworld(latitude, longitude, api_key):
    """VWorld API로 세부 행정동 정보 획득"""
    params = {
        'service': 'address',
        'request': 'getaddress',
        'version': '2.0',
        'crs': 'epsg:4326',
        'point': f'{longitude},{latitude}',
        'format': 'json',
        'type': 'both',
        'key': api_key
    }
    
    try:
        response = requests.get(API_URL, params=params)
        data = response.json()
        
        if data['response']['status'] == 'OK':
            dong_name = data['response']['result'][0]['structure']['level4A']
            return dong_name
        else:
            return None
    except Exception:
        return None
```



#### 3단계: 데이터 필터링 및 정제
**목적:** 분석에 불필요한 데이터 제거

```python
# CCTV 데이터에서 분석 대상 외 항목 제거
condition1 = df['구분'] == 'CCTV'
purpose_to_remove = ['차량방범', '재난재해', '교통단속']
condition2 = df['설치목적'].isin(purpose_to_remove)

# 조건에 해당하는 행 삭제
df_filtered = df.drop(df[condition1 & condition2].index)
```



#### 4단계: 결측 행정동 예측 
**목적:** 좌표는 있지만 행정동 정보가 없는 데이터 보완

```python
from sklearn.neighbors import KNeighborsClassifier

# 학습용 데이터와 예측용 데이터 분리
df_train = df.dropna(subset=['행정동'])
df_predict = df[df['행정동'].isnull()]

# KNN 모델로 행정동 예측
X_train = df_train[['위도', '경도']]
y_train = df_train['행정동']

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)

# 예측 실행
predicted_dongs = knn.predict(df_predict[['위도', '경도']])
df.loc[df['행정동'].isnull(), '행정동'] = predicted_dongs
```



#### 5단계: 행정동별 데이터 집계 
**목적:** 각 시설을 행정동별로 집계하여 통합 데이터셋 구성

```python
# 어린이용 CCTV 집계
kid_cctv = safe.loc[(safe['구분']=='CCTV')&(safe['설치목적']=='어린이보호'),:]
kid_cctv = kid_cctv.groupby('행정동')['위도'].count().reset_index()
main = pd.merge(main, kid_cctv, on='행정동', how='left')

# 방범용 CCTV 집계
other_cctv = safe.loc[(safe['구분']=='CCTV')&(safe['설치목적']!='어린이보호'),:]
other_cctv = other_cctv.groupby('행정동')['경도'].count().reset_index()
main = pd.merge(main, other_cctv, on='행정동', how='left')

# 안전비상벨 집계
bell = safe.loc[(safe['구분']=='안전비상벨'),:]
bell = bell.groupby('행정동')['설치목적'].count().reset_index()
main = pd.merge(main, bell, on='행정동', how='left')

# 경찰서 관련 시설 집계
police = danger.loc[(danger['구분']=='경찰서')|
                    (danger['구분']=='지구대')|
                    (danger['구분']=='파출소')|
                    (danger['구분']=='치안센터'),:]
police = police.groupby('행정동')['이름'].count().reset_index()
main = pd.merge(main, police, on='행정동', how='left')

# 유흥업소 집계
uheong = danger.loc[(danger['구분']=='유흥주점영업'),:]
uheong = uheong.groupby('행정동')['도로명'].count().reset_index()
main = pd.merge(main, uheong, on='행정동', how='left')

# 교육시설 집계
ele = danger.loc[(danger['구분']=='초등학교'),:]
ele = ele.groupby('행정동')['이름'].count().reset_index()
main = pd.merge(main, ele, on='행정동', how='left')

m_h = danger.loc[(danger['구분']=='중학교')|(danger['구분']=='고등학교'),:]
m_h = m_h.groupby('행정동')['도로명'].count().reset_index()
main = pd.merge(main, m_h, on='행정동', how='left')

uni = danger.loc[(danger['구분']=='대학교'),:]
uni = uni.groupby('행정동')['지번'].count().reset_index()
main = pd.merge(main, uni, on='행정동', how='left')

# 상가 정보 집계
com = com.groupby('행정동')['경도'].count().reset_index()
main = pd.merge(main, com, on='행정동', how='left')

# 결측값 처리
main = main.fillna(0)
```

## 4.최종 통합 데이터 구조 
**총 141행 × 29개 컬럼**

# 최종 데이터셋 구성 {#sec-final-dataset}

## 인구통계 변수 (11개) {#sec-demo-vars}

| 변수명 | 설명 |
|--------|------|
| 세대수 (세대) | 행정동별 총 세대수 |
| 등록인구 (명) | 총 등록인구 (남/여 구분) |
| 한국인 (명) | 한국인 인구 (남/여 구분) |
| 외국인 (명) | 외국인 인구 (남/여 구분) |
| 세대당 인구 (명) | 평균 세대당 인구수 |
| 65세이상고령자 (명) | 고령자 인구 |
| 평균연령 (세) | 행정동 평균연령 |
| 인구밀도 (명/km^2) | 단위면적당 인구밀도 |
| 면적 (㎢) | 행정동 면적 |

## 안전 인프라 변수 (6개) {#sec-safety-vars}

| 변수명 | 설명 |
|--------|------|
| 경찰서 수 | 경찰서, 파출소, 지구대, 치안센터 총계 |
| 가로등 수 | 행정동 내 가로등 설치 개수 |
| 보안등 수 | 행정동 내 보안등 설치 개수 |
| 어린이용 CCTV 수 | 어린이 보호구역 CCTV 개수 |
| 방범용 CCTV수 | 방범목적 CCTV 설치 개수 |
| 안전비상벨 수 | 안전비상벨 설치 개수 |

## 지역 특성 변수 (5개) {#sec-region-vars}

| 변수명 | 설명 |
|--------|------|
| 유흥업소 수 | 행정동 내 유흥주점 개수 |
| 초등학교 수 | 초등학교 개수 |
| 중,고등학교 수 | 중학교, 고등학교 개수 |
| 대학교 수 | 대학교 개수 |
| 상가 수 | 상업시설 개수 |

## 종속 변수 (1개)

| 변수명 | 설명 |
|--------|------|
| 범죄발생 수 | 총 범죄수 * 총 인구수 / 구 인구수  |


# 향후 분석 계획
## 위험도 모델링 계획

### 위험도 산출 공식
**순위험도 = Σ(위험요소 × 가중치) - Σ(안전요소 × 가중치)**

### 초기 변수 분류 및 가중치 설정
문헌 및 통계 분석, 전문가 의견, 상관분석으로 설정하고, 이후 데이터를 기반으로 변수를 다시 재분류할 예정입니다.

#### 위험요소 (Risk Factors, + 가중치) {#sec-risk-factors-detail}

| 변수 | 정의 | 초기 가중치 | 설정 근거 |
|------|------|-------------|-----------|
| **인구밀도** | 등록인구 ÷ 면적(㎢) | 0.25 | 범죄 기회 증가와 강한 양의 상관관계 예상. 경찰청 통계에 따르면 인구밀도 상위 10% 지역은 범죄율 평균 1.6배 높음 |
| **외국인 비율** | 외국인 수 ÷ 등록인구 | 0.10 | 언어·문화 장벽으로 지역 커뮤니케이션·방범 네트워크 약화 가능성. 법무부 범죄분석에서 외국인 밀집지 일부 범죄 유형이 높게 나타남 |
| **고령자 비율** | 65세 이상 인구 ÷ 등록인구 | 0.15 | 신체적 취약성과 응급 대응 지연으로 피해 확률 증가. 통계청 자료에서 고령화율 상위 지역은 절도·보이스피싱 피해율↑ |
| **여성 비율** | 여성 인구 ÷ 등록인구 | 0.10 | 신체적 취약성으로 인한 범죄 피해 가능성 증가 |
| **평균연령** | 주민 평균 나이 | 0.10 | 지역 인구 구조 파악. 평균연령이 너무 높거나 낮으면 특정 범죄 유형 집중 가능 |
| **세대당 인구수** | 등록인구 ÷ 세대수 | 0.10 | 1.0~1.2는 독거·소형가구 많음 → 방범 취약. 2.0 이상은 다가구 주택 가능성↑ → 밀도 상승 |
| **상가밀도** | 상가 수 ÷ 면적(㎢) | 0.15 | 상권 활성은 유동인구↑, 기회범죄 발생↑. 서울시 자료에서 상가밀도 상위 25% 지역은 소매치기·절도율 약 1.4배 |
| **상가비율** | 상가 수 ÷ 인구 × 1,000 | 0.10 | 인구 대비 상권 접근성. 낮은 인구에 상가 집중시 야간 취약성↑ |
| **유흥업소 밀도** | 유흥업소 수 ÷ 면적(㎢) | 0.15 | 야간 폭력·성범죄와 통계적으로 유의한 양의 상관 발견 사례 다수 |
| **중고등학교 밀도** | (중+고) 수 ÷ 면적(㎢) | 0.10 | 청소년 유동과 폭력 사건 가능성. 청소년 비중 높은 지역 범죄유형 변화 가능 |
| **대학교 밀도** | 대학교 수 ÷ 면적(㎢) | 0.10 | 야간활동 증가, 음주 관련 사건 등 상권과 결합 위험 |

#### 안전요소 (Protective Factors, - 가중치) {#sec-protective-factors-detail}

| 변수 | 정의 | 초기 가중치 | 설정 근거 |
|------|------|-------------|-----------|
| **경찰시설 밀도** | 경찰서·파출소·지구대 수 ÷ 면적 | -0.20 | 치안 대응력↑, 범죄 억제 효과. 한국형 치안서비스 연구에서 경찰밀도 상위 지역의 범죄율 약 25%↓ |
| **조명 밀도** | (가로등+보안등) ÷ 면적 | -0.15 | 야간 가시성↑, 범죄 기회 감소. 국토연구원 연구에서 조명 밀도와 야간범죄율은 역상관 |
| **CCTV 밀도** | (어린이용+방범용 CCTV) ÷ 면적 | -0.20 | 범죄 억제·증거 확보 효과. 설치 지역의 절도율 평균 27% 감소 보고 |
| **비상벨 밀도** | 비상벨 수 ÷ 면적 | -0.10 | 긴급 대응 가능성↑. 범죄 피해자 신고 반응속도 향상 |
| **초등학교 밀도** | 초등학교 수 ÷ 면적 | -0.10 | 아동 보호구역 지정·단속 강화 → 안전요소 |
| **아파트 비율** | 아파트 세대수 ÷ 전체 세대수 | -0.15 | 경비·출입통제·공용 CCTV 등 방범 인프라 효과 |

**위험요소 가중치 분포**:
- 높음(0.20~0.25): 인구밀도, 상가밀도, 유흥업소밀도
- 중간(0.10~0.15): 외국인비율, 고령자비율, 상가비율
- 낮음(0.10): 여성비율, 평균연령, 세대당인구수, 중고등학교밀도, 대학교밀도

**안전요소 가중치 분포**:
- 높음(-0.20): 경찰시설밀도, CCTV밀도
- 중간(-0.15): 조명밀도, 아파트비율
- 낮음(-0.10): 비상벨밀도, 초등학교밀도

### 모델 검증 계획 {#sec-model-validation-plan}

#### 1단계: 순위험도 계산

```python
# 위험요소 점수 계산
risk_score = (df['인구밀도'] * 0.25 + 
              df['외국인비율'] * 0.10 + 
              df['고령자비율'] * 0.15 + 
              df['여성비율'] * 0.10 +
              df['평균연령'] * 0.10 +
              df['세대당인구수'] * 0.10 +
              df['상가밀도'] * 0.15 +
              df['상가비율'] * 0.10 +
              df['유흥업소밀도'] * 0.15 +
              df['중고등학교밀도'] * 0.10 +
              df['대학교밀도'] * 0.10)

# 안전요소 점수 계산
safety_score = (df['경찰시설밀도'] * 0.20 + 
                df['조명밀도'] * 0.15 + 
                df['CCTV밀도'] * 0.20 + 
                df['비상벨밀도'] * 0.10 + 
                df['초등학교밀도'] * 0.10 + 
                df['아파트비율'] * 0.15)

# 순위험도 = 위험요소 - 안전요소
df['순위험도'] = risk_score - safety_score
```

#### 2단계: 모델 유효성 검증

```python
from scipy.stats import spearmanr, pearsonr

# 순위험도와 실제 범죄발생률 간 상관관계 분석
corr_s, pval_s = spearmanr(df['순위험도'], df['범죄발생률'])
corr_p, pval_p = pearsonr(df['순위험도'], df['범죄발생률'])

print(f"Spearman 상관계수: {corr_s:.4f} (p-value: {pval_s:.4f})")
print(f"Pearson 상관계수: {corr_p:.4f} (p-value: {pval_p:.4f})")
```

#### 3단계: 개별 변수 검증

각 변수와 범죄발생률 간의 개별 상관관계를 분석하여 초기 가설을 검증합니다.

```python
# 개별 변수별 상관분석 결과 저장
correlation_results = {}

# 위험요소 검증
risk_variables = ['인구밀도', '외국인비율', '고령자비율', '여성비율', 
                  '평균연령', '세대당인구수', '상가밀도', '상가비율', 
                  '유흥업소밀도', '중고등학교밀도', '대학교밀도']

# 안전요소 검증
safety_variables = ['경찰시설밀도', '조명밀도', 'CCTV밀도', 
                    '비상벨밀도', '초등학교밀도', '아파트비율']

for var in risk_variables + safety_variables:
    corr, pval = spearmanr(df[var], df['범죄발생률'])
    correlation_results[var] = {'correlation': corr, 'p_value': pval}
    print(f"{var}: r = {corr:.4f}, p = {pval:.4f}")
```

### 시각화 계획 {#sec-visualization-plan-detail}

#### 범죄 현황 분석 시각화
1. **시간대별 범죄 분석**: 야간 범죄 비율 및 시간대별 분포
2. **장소별 범죄 분석**: 범죄 발생 장소의 특성 분석  
3. **공간적 분포**: 대구시 행정동별 범죄 발생률 지도 시각화

#### 변수 간 관계 분석 시각화


::: {.callout-note}
## 다음 단계

1. **상관관계 분석** 실시
2. 분석 결과를 바탕으로 **변수 분류 및 가중치 재조정 예정**

:::

