#!/usr/bin/env python3
"""
안테나 시뮬레이션 사용 예제
다양한 주파수 대역의 안테나를 시뮬레이션합니다.
"""

from antenna_simulation import DipoleAntenna
import matplotlib.pyplot as plt


def compare_frequencies():
    """다양한 주파수의 안테나 비교"""
    frequencies = {
        'LoRa 900MHz': 900e6,
        'WiFi 2.4GHz': 2.4e9,
        'WiFi 5GHz': 5e9,
    }
    
    print("=" * 70)
    print("다양한 주파수 대역 안테나 비교")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), 
                            subplot_kw=dict(projection='polar'))
    
    for idx, (name, freq) in enumerate(frequencies.items()):
        antenna = DipoleAntenna(frequency=freq)
        params = antenna.calculate_parameters()
        
        print(f"\n[{name}]")
        print(f"  파장: {params['파장 (cm)']:.2f} cm")
        print(f"  안테나 길이: {params['안테나 길이 (cm)']:.2f} cm")
        
        # 복사 패턴 계산
        import numpy as np
        theta = np.linspace(0, 2 * np.pi, 360)
        theta_half = np.linspace(0, np.pi, 180)
        E = antenna.radiation_pattern(theta_half)
        E_full = np.concatenate([E, E[::-1]])
        E_normalized = E_full / np.max(E_full)
        
        # 플롯
        axes[idx].plot(theta, E_normalized, linewidth=2)
        axes[idx].set_title(f'{name}\n({params["파장 (cm)"]:.2f} cm)', 
                           fontsize=11, pad=15)
        axes[idx].set_theta_zero_location('N')
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('frequency_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 비교 플롯 저장: frequency_comparison.png")
    print("=" * 70)
    plt.show()


def analyze_single_antenna():
    """단일 안테나 상세 분석"""
    print("\n" + "=" * 70)
    print("단일 안테나 상세 분석 (2.4 GHz)")
    print("=" * 70)
    
    antenna = DipoleAntenna(frequency=2.4e9)
    
    # 파라미터 출력
    params = antenna.calculate_parameters()
    print("\n[안테나 사양]")
    for key, value in params.items():
        if isinstance(value, (int, float)):
            print(f"  {key:25s}: {value:.4f}")
        else:
            print(f"  {key:25s}: {value}")
    
    # 특정 각도에서의 전계 강도 계산
    import numpy as np
    
    print("\n[특정 각도에서의 복사 강도]")
    angles = [0, 30, 45, 60, 90]
    for angle in angles:
        theta_rad = np.radians(angle)
        E = antenna.radiation_pattern(theta_rad)
        E_normalized = E / np.max(antenna.radiation_pattern(np.pi/2))
        E_dB = 20 * np.log10(E_normalized + 1e-10)
        print(f"  {angle:3d}° : {E_normalized:.4f} ({E_dB:6.2f} dB)")
    
    print("=" * 70)


if __name__ == "__main__":
    # 예제 1: 단일 안테나 분석
    analyze_single_antenna()
    
    # 예제 2: 주파수 비교
    compare_frequencies()
