# 안테나 시뮬레이션

다이폴 안테나의 복사 패턴(Radiation Pattern)을 계산하고 시각화하는 Python 프로그램입니다.

## 기능

- **반파장 다이폴 안테나 시뮬레이션**
  - 전자기파 복사 패턴 계산
  - 2D 극좌표 플롯 (선형 및 dB 스케일)
  - 3D 복사 패턴 시각화
  - 안테나 주요 파라미터 계산

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 실행

```bash
python antenna_simulation.py
```

### Python 코드에서 사용

```python
from antenna_simulation import DipoleAntenna

# 2.4 GHz 안테나 생성
antenna = DipoleAntenna(frequency=2.4e9)

# 파라미터 확인
params = antenna.calculate_parameters()
print(params)

# 2D 복사 패턴 플롯
fig_2d = antenna.plot_2d_pattern()

# 3D 복사 패턴 플롯
fig_3d = antenna.plot_3d_pattern()

# 플롯 표시
import matplotlib.pyplot as plt
plt.show()
```

### 다른 주파수로 시뮬레이션

```python
# 5 GHz 안테나 (WiFi 5GHz 대역)
antenna_5ghz = DipoleAntenna(frequency=5e9)

# 900 MHz 안테나 (LoRa 대역)
antenna_900mhz = DipoleAntenna(frequency=900e6)

# 커스텀 길이의 안테나
antenna_custom = DipoleAntenna(frequency=2.4e9, length=0.05)  # 5cm
```

## 출력 파일

실행 시 다음 이미지 파일이 생성됩니다:

- `antenna_pattern_2d.png` - 2D 극좌표 복사 패턴 (선형 및 dB)
- `antenna_pattern_3d.png` - 3D 복사 패턴

## 안테나 이론

### 반파장 다이폴 안테나

반파장 다이폴 안테나는 가장 기본적이고 널리 사용되는 안테나입니다.

**주요 특성:**
- 길이: λ/2 (반파장)
- 입력 임피던스: 약 73Ω
- 지향성: 2.15 dBi
- 반전력 빔폭: 약 78°
- 복사 패턴: 도넛 모양 (omnidirectional in azimuth)

**복사 패턴 수식:**

```
E(θ) = cos((π/2)cos(θ)) / sin(θ)
```

여기서 θ는 안테나 축으로부터의 각도입니다.

## 응용 분야

- RF 안테나 설계
- 무선 통신 시스템 분석
- WiFi, Bluetooth, LoRa 등의 안테나 특성 연구
- 전자기학 교육 및 학습

## 라이센스

MIT License
