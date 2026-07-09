#!/usr/bin/env python3
"""
근접전계 측정 데이터 생성 스크립트 (예제용)
실제 측정 데이터 대신 시뮬레이션된 데이터를 생성합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_dipole_field(x, y, z, frequency=10e9, dipole_moment=1.0, dipole_pos=(0, 0, 0)):
    """
    전기 쌍극자(Electric Dipole)의 근접전계 계산
    
    Parameters:
    -----------
    x, y, z : array-like
        관측점 좌표
    frequency : float
        주파수 (Hz)
    dipole_moment : float
        쌍극자 모멘트
    dipole_pos : tuple
        쌍극자 위치 (x0, y0, z0)
    
    Returns:
    --------
    Ez : complex array
        Z 방향 전기장 성분
    """
    c = 3e8  # 빛의 속도
    wavelength = c / frequency
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency
    mu0 = 4 * np.pi * 1e-7
    epsilon0 = 8.854e-12
    eta = np.sqrt(mu0 / epsilon0)
    
    # 쌍극자로부터의 거리 벡터
    x0, y0, z0 = dipole_pos
    dx = x - x0
    dy = y - y0
    dz = z - z0
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # 근접전계 근사식 (간략화)
    # 실제로는 더 복잡한 식이지만, 예제용으로 단순화
    Ez = (dipole_moment / (4 * np.pi * epsilon0)) * (
        (3 * dz * dz / r**5 - 1 / r**3) +
        1j * k * (3 * dz * dz / r**4 - 1 / r**2) -
        k**2 * dz * dz / r**3
    ) * np.exp(-1j * k * r)
    
    return Ez


def generate_patch_antenna_field(x, y, z, frequency=10e9):
    """
    패치 안테나의 근접전계 근사 모델
    """
    c = 3e8
    wavelength = c / frequency
    k = 2 * np.pi / wavelength
    
    # 간단한 코사인 분포
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    
    # 방사 패턴 (코사인 분포)
    pattern = np.cos(theta)**2
    
    # 위상 (구면파)
    phase = -k * r
    
    # 거리에 따른 감쇠
    amplitude = pattern / (r + 0.001)  # 특이점 방지
    
    field = amplitude * np.exp(1j * phase)
    
    return field


def generate_aperture_field(x, y, z, frequency=10e9, aperture_size=0.05):
    """
    사각 개구(Aperture)의 근접전계
    """
    c = 3e8
    wavelength = c / frequency
    k = 2 * np.pi / wavelength
    
    # 개구 내부의 균일한 분포
    aperture_field = np.zeros_like(x, dtype=complex)
    
    # 개구 영역 정의
    mask = (np.abs(x) < aperture_size/2) & (np.abs(y) < aperture_size/2)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # 개구 내부: 균일한 장
    aperture_field[mask] = 1.0 * np.exp(-1j * k * z)
    
    # 개구 외부: 회절 효과 (Fresnel 근사)
    aperture_field[~mask] = (aperture_size**2 / (r[~mask] + 0.001)) * \
                            np.exp(-1j * k * r[~mask]) * \
                            np.sinc(k * x[~mask] * aperture_size / (2 * np.pi)) * \
                            np.sinc(k * y[~mask] * aperture_size / (2 * np.pi))
    
    return aperture_field


def add_noise(field, snr_db=40):
    """
    신호에 노이즈 추가
    """
    signal_power = np.mean(np.abs(field)**2)
    noise_power = signal_power / (10**(snr_db/10))
    
    noise_real = np.random.normal(0, np.sqrt(noise_power/2), field.shape)
    noise_imag = np.random.normal(0, np.sqrt(noise_power/2), field.shape)
    noise = noise_real + 1j * noise_imag
    
    return field + noise


def generate_near_field_data(
    x_range=(-0.1, 0.1, 0.005),  # (시작, 끝, 스텝) 단위: 미터
    y_range=(-0.1, 0.1, 0.005),
    z_scan=0.01,  # 스캔 평면의 z 좌표
    frequency=10e9,
    source_type='dipole',  # 'dipole', 'patch', 'aperture'
    add_noise_flag=True,
    snr_db=40,
    output_file='near_field_data.csv'
):
    """
    근접전계 측정 데이터 생성 및 CSV 저장
    """
    print("=" * 60)
    print("근접전계 데이터 생성")
    print("=" * 60)
    
    # 좌표 그리드 생성
    x = np.arange(x_range[0], x_range[1], x_range[2])
    y = np.arange(y_range[0], y_range[1], y_range[2])
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * z_scan
    
    nx = len(x)
    ny = len(y)
    
    print(f"그리드 크기: {nx} x {ny}")
    print(f"X 범위: {x_range[0]*1000:.1f} ~ {x_range[1]*1000:.1f} mm")
    print(f"Y 범위: {y_range[0]*1000:.1f} ~ {y_range[1]*1000:.1f} mm")
    print(f"Z 스캔 높이: {z_scan*1000:.1f} mm")
    print(f"주파수: {frequency/1e9:.2f} GHz")
    print(f"소스 타입: {source_type}")
    
    # 전기장 계산
    if source_type == 'dipole':
        field = generate_dipole_field(X, Y, Z, frequency)
    elif source_type == 'patch':
        field = generate_patch_antenna_field(X, Y, Z, frequency)
    elif source_type == 'aperture':
        field = generate_aperture_field(X, Y, Z, frequency)
    else:
        raise ValueError(f"알 수 없는 소스 타입: {source_type}")
    
    # 노이즈 추가
    if add_noise_flag:
        field = add_noise(field, snr_db)
        print(f"노이즈 추가됨 (SNR: {snr_db} dB)")
    
    # 크기와 위상 추출
    magnitude = np.abs(field)
    phase_rad = np.angle(field)
    phase_deg = np.rad2deg(phase_rad)
    
    # 데이터프레임 생성
    data = {
        'x': X.flatten(),
        'y': Y.flatten(),
        'magnitude': magnitude.flatten(),
        'phase': phase_deg.flatten()
    }
    
    df = pd.DataFrame(data)
    
    # CSV 저장
    df.to_csv(output_file, index=False)
    print(f"\nCSV 파일 저장됨: {output_file}")
    print(f"총 데이터 포인트: {len(df)}")
    
    # 데이터 미리보기
    print("\n데이터 미리보기:")
    print(df.head(10))
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 크기
    im1 = axes[0].contourf(X * 1000, Y * 1000, magnitude, levels=50, cmap='hot')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    axes[0].set_title('근접전계 크기')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label='크기 (V/m)')
    
    # 위상
    im2 = axes[1].contourf(X * 1000, Y * 1000, phase_deg, levels=50, cmap='hsv')
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Y (mm)')
    axes[1].set_title('근접전계 위상')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], label='위상 (도)')
    
    plt.tight_layout()
    plt.savefig('generated_near_field.png', dpi=150, bbox_inches='tight')
    print("시각화 저장됨: generated_near_field.png")
    plt.show()
    
    return df


if __name__ == "__main__":
    # 예제 1: 쌍극자 안테나
    print("\n예제 1: 전기 쌍극자 근접전계 생성")
    df1 = generate_near_field_data(
        x_range=(-0.08, 0.08, 0.004),
        y_range=(-0.08, 0.08, 0.004),
        z_scan=0.01,
        frequency=10e9,
        source_type='dipole',
        add_noise_flag=True,
        snr_db=50,
        output_file='near_field_data.csv'
    )
    
    # 추가 예제를 생성하려면 아래 주석 해제
    """
    # 예제 2: 패치 안테나
    print("\n예제 2: 패치 안테나 근접전계 생성")
    df2 = generate_near_field_data(
        x_range=(-0.1, 0.1, 0.005),
        y_range=(-0.1, 0.1, 0.005),
        z_scan=0.015,
        frequency=5e9,
        source_type='patch',
        output_file='near_field_patch.csv'
    )
    
    # 예제 3: 사각 개구
    print("\n예제 3: 사각 개구 근접전계 생성")
    df3 = generate_near_field_data(
        x_range=(-0.12, 0.12, 0.006),
        y_range=(-0.12, 0.12, 0.006),
        z_scan=0.02,
        frequency=15e9,
        source_type='aperture',
        output_file='near_field_aperture.csv'
    )
    """
    
    print("\n" + "=" * 60)
    print("데이터 생성 완료!")
    print("=" * 60)
