# 근접전계-원전계 변환 프로그램

프로브를 이용한 근접전계 스캔 데이터를 기반으로 원전계 패턴을 계산하는 전자기장 변환 프로그램입니다.

## 기능

- CSV 파일에서 근접전계 측정 데이터 로드
- 2차원 평면 스캔 데이터 (좌표별 크기/위상) 처리
- Plane Wave Spectrum 방법을 이용한 원전계 변환
- 원전계 패턴 시각화 (2D/3D/극좌표)
- Back Projection (원전계 → 근접전계 역변환)
- 원본 데이터 vs Back Projection 비교 분석

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 방법 1: Python 스크립트 (자동 실행)

#### 1. 예제 데이터 생성 (테스트용)

실제 측정 데이터가 없는 경우, 시뮬레이션된 예제 데이터를 생성할 수 있습니다:

```bash
python generate_sample_data.py
```

이 스크립트는 다음과 같은 소스의 근접전계 데이터를 생성합니다:
- 전기 쌍극자 (Dipole)
- 패치 안테나 (Patch)
- 사각 개구 (Aperture)

#### 2. 파라미터 설정

`config.txt` 파일을 편집하여 변환 파라미터를 설정합니다:

```
# 주파수 (Hz)
frequency = 10e9

# 근접전계 측정 평면의 Z 좌표 (m)
z_distance = 0.01

# 원전계 계산 거리 (m)
far_field_distance = 1.0

# Theta 각도 범위 (시작, 끝, 스텝) 단위: 도
theta_range = 0, 90, 1

# Phi 각도 범위 (시작, 끝, 스텝) 단위: 도
phi_range = 0, 360, 5
```

#### 3. 근접전계-원전계 변환 실행

```bash
python near_to_far_field.py
```


---

### 방법 2: 주피터 노트북 (인터랙티브) ⭐ 추천!

더 직관적이고 단계별로 실행하며 결과를 확인하고 싶다면 주피터 노트북 사용을 권장합니다.

```bash
jupyter notebook near_to_far_field_transform.ipynb
```

**주피터 노트북의 장점:**
- ✅ 셀 단위로 단계별 실행 가능
- ✅ 인라인 시각화로 즉시 결과 확인
- ✅ 파라미터를 쉽게 조정하며 실험 가능
- ✅ 각 단계마다 상세한 설명 포함
- ✅ 원하는 부분만 다시 실행 가능

**노트북 구성:**
1. 소개 및 개요
2. 라이브러리 임포트
3. 파라미터 설정
4. 예제 데이터 생성 (선택사항)
5. CSV 데이터 로드
6. 원본 근접전계 시각화
7. 원전계 변환 수행
8. 원전계 패턴 시각화
9. Back Projection
10. 비교 분석 및 오차 통계

각 셀을 순서대로 실행하면 모든 과정을 단계별로 확인할 수 있습니다.

---

## 파일 설명

- `near_to_far_field.py` - 메인 변환 프로그램 (스크립트 버전)
- `near_to_far_field_transform.ipynb` - 주피터 노트북 버전
- `generate_sample_data.py` - 예제 데이터 생성 스크립트
- `config.txt` - 파라미터 설정 파일
- `requirements.txt` - 필요한 Python 패키지
- `README.md` - 이 파일
1. CSV 데이터 로드
2. 원본 근접전계 데이터 시각화
3. 원전계 변환 및 패턴 시각화
4. Back Projection 및 비교 분석

## CSV 파일 형식

측정 데이터는 다음 형식의 CSV 파일이어야 합니다:

```csv
x,y,magnitude,phase
-0.1,-0.1,1.234,45.6
-0.1,-0.095,1.456,47.8
...
```

**필수 컬럼:**
- `x`: X 좌표 (미터)
- `y`: Y 좌표 (미터)
- `magnitude`: 전기장 크기 (V/m)
- `phase`: 위상 (도)

## 출력 파일

프로그램은 다음 이미지 파일들을 생성합니다:

1. `near_field_original.png` - 원본 근접전계 데이터 (크기/위상)
2. `far_field_pattern.png` - 원전계 패턴 (2D, 극좌표, 3D)
3. `comparison.png` - 원본 vs Back Projection 비교

## 이론 배경

### Plane Wave Spectrum 방법

근접전계를 평면파의 중첩으로 표현하여 원전계로 변환하는 방법입니다:

1. **2D FFT**: 공간 도메인 → 스펙트럼 도메인 변환
2. **전파**: 스펙트럼을 원전계 거리까지 전파
3. **샘플링**: 각 관찰 각도에서 스펝트럼 값 추출

### Back Projection

원전계에서 근접전계로 역변환하여 알고리즘의 정확도를 검증합니다.

## 파라미터 설명

- **frequency**: 측정 주파수 (Hz)
  - 파장과 전파 특성 결정
  
- **z_distance**: 근접전계 측정 평면의 Z 좌표 (m)
  - 안테나/소스로부터 프로브까지의 거리
  
- **far_field_distance**: 원전계 계산 거리 (m)
  - 일반적으로 2D²/λ 이상 (D: 안테나 크기, λ: 파장)
  
- **theta_range**: 천정각 범위
  - (시작각, 끝각, 각도 간격) 단위: 도
  - 0° = Z축 방향, 90° = XY 평면
  
- **phi_range**: 방위각 범위
  - (시작각, 끝각, 각도 간격) 단위: 도
  - 0° = X축 방향, 90° = Y축 방향

## 예제

### 10 GHz 패치 안테나 측정

```python
# config.txt
frequency = 10e9
z_distance = 0.01  # 10mm 높이에서 스캔
far_field_distance = 1.0  # 1m 원전계
theta_range = 0, 90, 1
phi_range = 0, 360, 5
```

### 고해상도 원전계 패턴

더 세밀한 원전계 패턴이 필요한 경우:

```python
theta_range = 0, 90, 0.5  # 0.5도 간격
phi_range = 0, 360, 2     # 2도 간격
```

## 주의사항

1. **샘플링 간격**: 근접전계 측정 시 λ/2 이하의 간격으로 샘플링 필요
2. **스캔 영역**: 충분히 큰 영역을 스캔해야 정확한 원전계 계산 가능
3. **Evanescent Wave**: 전파하지 않는 파동은 자동으로 필터링됨
4. **메모리**: 고해상도 변환 시 많은 메모리 필요

## 문제 해결

### "CSV 파일에 필수 컬럼이 없습니다" 오류
- CSV 파일에 `x`, `y`, `magnitude`, `phase` 컬럼이 모두 있는지 확인

### Back Projection 오차가 큰 경우
- 근접전계 스캔 영역이 충분히 큰지 확인
- 샘플링 간격이 λ/2 이하인지 확인
- Evanescent wave가 많은 경우 (매우 근접한 측정) 오차 증가 가능

### 메모리 부족
- `theta_range`, `phi_range`의 간격을 늘려서 해상도 낮춤
- 근접전계 데이터의 샘플링 수를 줄임

## 라이선스

MIT License

## 참고 문헌

- J. Appel-Hansen, "Accurate determination of gain and radiation patterns by radar cross-section measurements," IEEE Trans. Antennas Propag., 1979.
- D. M. Kerns, "Plane-wave scattering-matrix theory of antennas and antenna-antenna interactions," NBS Monograph, 1981.
