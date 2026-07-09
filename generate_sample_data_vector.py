#!/usr/bin/env python3
"""
벡터 근접전계 측정 데이터 생성 스크립트
Ex, Ey, Ez 성분을 모두 포함하는 완전한 벡터 전계 생성
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_vector_dipole_field(x, y, z, freq, orientation='z', dipole_moment=1.0):
    """
    벡터 전기 쌍극자의 근접전계 계산
    
    Parameters:
    -----------
    orientation : str
        쌍극자 방향 ('x', 'y', 'z')
    """
    c = 3e8
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    mu0 = 4 * np.pi * 1e-7
    epsilon0 = 8.854187817e-12
    
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-6)  # 특이점 방지
    
    # 단위 벡터
    r_hat_x = x / r
    r_hat_y = y / r
    r_hat_z = z / r
    
    # 쌍극자 방향 벡터
    if orientation == 'x':
        p_x, p_y, p_z = dipole_moment, 0, 0
    elif orientation == 'y':
        p_x, p_y, p_z = 0, dipole_moment, 0
    else:  # 'z'
        p_x, p_y, p_z = 0, 0, dipole_moment
    
    # p · r̂
    p_dot_r = p_x * r_hat_x + p_y * r_hat_y + p_z * r_hat_z
    
    # 전기장 계산 (near-field approximation)
    # E = (k²/(4πε₀)) * {(r̂ × p) × r̂ / r + [3r̂(r̂·p) - p] * (1/r³ - ik/r²)} * e^(-ikr)
    
    factor1 = k**2 / (4 * np.pi * epsilon0)
    factor2 = 1 / r**3 - 1j * k / r**2
    
    # 방사 항
    rad_term_x = (p_x - p_dot_r * r_hat_x)
    rad_term_y = (p_y - p_dot_r * r_hat_y)
    rad_term_z = (p_z - p_dot_r * r_hat_z)
    
    # 근접장 항
    near_term_x = (3 * p_dot_r * r_hat_x - p_x)
    near_term_y = (3 * p_dot_r * r_hat_y - p_y)
    near_term_z = (3 * p_dot_r * r_hat_z - p_z)
    
    # 전체 전기장
    Ex = factor1 * (k**2 * rad_term_x / r + near_term_x * factor2) * np.exp(-1j * k * r)
    Ey = factor1 * (k**2 * rad_term_y / r + near_term_y * factor2) * np.exp(-1j * k * r)
    Ez = factor1 * (k**2 * rad_term_z / r + near_term_z * factor2) * np.exp(-1j * k * r)
    
    return Ex, Ey, Ez


def generate_patch_antenna_field(x, y, z, freq, patch_width=0.015):
    """
    패치 안테나의 벡터 근접전계 근사
    """
    c = 3e8
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-6)
    
    # 패치 안테나는 주로 y 방향 전계
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    
    # 방사 패턴
    pattern_theta = np.cos(theta)**2
    pattern_phi = np.sin(2 * phi) * np.sin(theta)
    
    # 거리에 따른 감쇠
    amplitude = 1.0 / (r + 0.001)
    phase = -k * r
    field_factor = amplitude * np.exp(1j * phase)
    
    # 전계 성분
    Ex = 0.3 * pattern_theta * np.cos(phi) * field_factor
    Ey = pattern_theta * np.sin(phi) * field_factor
    Ez = 0.5 * pattern_phi * field_factor
    
    return Ex, Ey, Ez


def generate_slot_antenna_field(x, y, z, freq, slot_length=0.02):
    """
    슬롯 안테나의 벡터 근접전계
    """
    c = 3e8
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-6)
    
    # 슬롯은 주로 x 방향 전계
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    
    # 슬롯 패턴
    pattern = np.sinc(k * slot_length * np.sin(theta) * np.cos(phi) / (2 * np.pi))
    
    amplitude = 1.0 / (r + 0.001)
    phase = -k * r
    field_factor = amplitude * pattern * np.exp(1j * phase)
    
    Ex = field_factor * np.cos(theta) * np.cos(phi)
    Ey = field_factor * np.cos(theta) * np.sin(phi)
    Ez = -field_factor * np.sin(theta)
    
    return Ex, Ey, Ez


def add_noise(field_x, field_y, field_z, snr_db=40):
    """
    벡터 전계에 노이즈 추가
    """
    # 총 신호 파워
    signal_power = (np.mean(np.abs(field_x)**2) + 
                   np.mean(np.abs(field_y)**2) + 
                   np.mean(np.abs(field_z)**2))
    
    noise_power = signal_power / (10**(snr_db/10)) / 3  # 3개 성분으로 나눔
    
    # 각 성분에 노이즈 추가
    noise_x = (np.random.normal(0, np.sqrt(noise_power/2), field_x.shape) + 
              1j * np.random.normal(0, np.sqrt(noise_power/2), field_x.shape))
    noise_y = (np.random.normal(0, np.sqrt(noise_power/2), field_y.shape) + 
              1j * np.random.normal(0, np.sqrt(noise_power/2), field_y.shape))
    noise_z = (np.random.normal(0, np.sqrt(noise_power/2), field_z.shape) + 
              1j * np.random.normal(0, np.sqrt(noise_power/2), field_z.shape))
    
    return field_x + noise_x, field_y + noise_y, field_z + noise_z


def generate_vector_near_field_data(
    x_range=(-0.08, 0.08, 0.004),
    y_range=(-0.08, 0.08, 0.004),
    z_scan=0.01,
    frequency=10e9,
    source_type='dipole_z',  # 'dipole_x', 'dipole_y', 'dipole_z', 'patch', 'slot'
    add_noise_flag=True,
    snr_db=40,
    output_file='near_field_data_vector.csv'
):
    """
    벡터 근접전계 측정 데이터 생성 및 CSV 저장
    """
    print("=" * 60)
    print("벡터 근접전계 데이터 생성")
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
    
    # 벡터 전기장 계산
    if source_type == 'dipole_x':
        Ex, Ey, Ez = generate_vector_dipole_field(X, Y, Z, frequency, 'x')
    elif source_type == 'dipole_y':
        Ex, Ey, Ez = generate_vector_dipole_field(X, Y, Z, frequency, 'y')
    elif source_type == 'dipole_z':
        Ex, Ey, Ez = generate_vector_dipole_field(X, Y, Z, frequency, 'z')
    elif source_type == 'patch':
        Ex, Ey, Ez = generate_patch_antenna_field(X, Y, Z, frequency)
    elif source_type == 'slot':
        Ex, Ey, Ez = generate_slot_antenna_field(X, Y, Z, frequency)
    else:
        raise ValueError(f"알 수 없는 소스 타입: {source_type}")
    
    # 노이즈 추가
    if add_noise_flag:
        Ex, Ey, Ez = add_noise(Ex, Ey, Ez, snr_db)
        print(f"노이즈 추가됨 (SNR: {snr_db} dB)")
    
    # 크기와 위상 추출
    Ex_mag = np.abs(Ex)
    Ex_phase = np.rad2deg(np.angle(Ex))
    
    Ey_mag = np.abs(Ey)
    Ey_phase = np.rad2deg(np.angle(Ey))
    
    Ez_mag = np.abs(Ez)
    Ez_phase = np.rad2deg(np.angle(Ez))
    
    # 데이터프레임 생성
    data = {
        'x': X.flatten(),
        'y': Y.flatten(),
        'Ex_mag': Ex_mag.flatten(),
        'Ex_phase': Ex_phase.flatten(),
        'Ey_mag': Ey_mag.flatten(),
        'Ey_phase': Ey_phase.flatten(),
        'Ez_mag': Ez_mag.flatten(),
        'Ez_phase': Ez_phase.flatten()
    }
    
    df = pd.DataFrame(data)
    
    # CSV 저장
    df.to_csv(output_file, index=False)
    print(f"\nCSV 파일 저장됨: {output_file}")
    print(f"총 데이터 포인트: {len(df)}")
    
    # 데이터 미리보기
    print("\n데이터 미리보기:")
    print(df.head(10))
    
    # 통계
    print(f"\n전계 성분 통계:")
    print(f"  |Ex| 평균: {np.mean(Ex_mag):.2e} V/m")
    print(f"  |Ey| 평균: {np.mean(Ey_mag):.2e} V/m")
    print(f"  |Ez| 평균: {np.mean(Ez_mag):.2e} V/m")
    
    E_total = np.sqrt(Ex_mag**2 + Ey_mag**2 + Ez_mag**2)
    print(f"  |E_total| 평균: {np.mean(E_total):.2e} V/m")
    
    # 시각화
    fig = plt.figure(figsize=(18, 10))
    
    # 전체 크기
    ax1 = fig.add_subplot(2, 4, 1)
    im1 = ax1.contourf(X * 1000, Y * 1000, E_total, levels=50, cmap='hot')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('Total |E|')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='V/m')
    
    # 각 성분
    components = [('Ex', Ex_mag), ('Ey', Ey_mag), ('Ez', Ez_mag)]
    for idx, (name, field) in enumerate(components):
        ax = fig.add_subplot(2, 4, idx + 2)
        im = ax.contourf(X * 1000, Y * 1000, field, levels=50, cmap='hot')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(f'|{name}|')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='V/m')
    
    # 벡터 필드 (quiver)
    ax5 = fig.add_subplot(2, 4, 5)
    skip = 3  # 화살표 간격
    ax5.quiver(X[::skip, ::skip] * 1000, Y[::skip, ::skip] * 1000,
              np.real(Ex[::skip, ::skip]), np.real(Ey[::skip, ::skip]),
              E_total[::skip, ::skip], cmap='hot')
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Y (mm)')
    ax5.set_title('Vector Field (Ex, Ey)')
    ax5.set_aspect('equal')
    
    # 위상
    phases = [('Ex', Ex_phase), ('Ey', Ey_phase), ('Ez', Ez_phase)]
    for idx, (name, phase) in enumerate(phases):
        ax = fig.add_subplot(2, 4, idx + 6)
        im = ax.contourf(X * 1000, Y * 1000, phase, levels=50, cmap='hsv')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(f'{name} Phase')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='deg')
    
    plt.tight_layout()
    fig_name = output_file.replace('.csv', '.png')
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    print(f"시각화 저장됨: {fig_name}")
    plt.show()
    
    return df


if __name__ == "__main__":
    # 예제 1: Z 방향 쌍극자
    print("\n예제 1: Z 방향 전기 쌍극자")
    df1 = generate_vector_near_field_data(
        x_range=(-0.08, 0.08, 0.004),
        y_range=(-0.08, 0.08, 0.004),
        z_scan=0.01,
        frequency=10e9,
        source_type='dipole_z',
        add_noise_flag=True,
        snr_db=50,
        output_file='near_field_data_vector.csv'
    )
    
    # 추가 예제
    """
    # 예제 2: X 방향 쌍극자
    print("\n예제 2: X 방향 전기 쌍극자")
    df2 = generate_vector_near_field_data(
        x_range=(-0.08, 0.08, 0.004),
        y_range=(-0.08, 0.08, 0.004),
        z_scan=0.01,
        frequency=10e9,
        source_type='dipole_x',
        output_file='near_field_dipole_x.csv'
    )
    
    # 예제 3: 패치 안테나
    print("\n예제 3: 패치 안테나")
    df3 = generate_vector_near_field_data(
        x_range=(-0.1, 0.1, 0.005),
        y_range=(-0.1, 0.1, 0.005),
        z_scan=0.015,
        frequency=5e9,
        source_type='patch',
        output_file='near_field_patch_vector.csv'
    )
    
    # 예제 4: 슬롯 안테나
    print("\n예제 4: 슬롯 안테나")
    df4 = generate_vector_near_field_data(
        x_range=(-0.1, 0.1, 0.005),
        y_range=(-0.1, 0.1, 0.005),
        z_scan=0.02,
        frequency=8e9,
        source_type='slot',
        output_file='near_field_slot_vector.csv'
    )
    """
    
    print("\n" + "=" * 60)
    print("벡터 데이터 생성 완료!")
    print("=" * 60)
