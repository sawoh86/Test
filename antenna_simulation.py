#!/usr/bin/env python3
"""
안테나 시뮬레이션 코드
다이폴 안테나의 복사 패턴(Radiation Pattern)을 계산하고 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class DipoleAntenna:
    """반파장 다이폴 안테나 시뮬레이터"""
    
    def __init__(self, frequency=2.4e9, length=None):
        """
        Parameters:
        -----------
        frequency : float
            동작 주파수 (Hz), 기본값: 2.4 GHz
        length : float
            안테나 길이 (m), None인 경우 반파장으로 자동 설정
        """
        self.frequency = frequency
        self.wavelength = 3e8 / frequency  # 파장 계산 (c = λf)
        self.length = length if length else self.wavelength / 2
        
    def radiation_pattern(self, theta, phi=0):
        """
        복사 패턴 계산
        
        Parameters:
        -----------
        theta : array-like
            천정각 (0 ~ π)
        phi : array-like
            방위각 (0 ~ 2π)
            
        Returns:
        --------
        E_theta : array-like
            전계 강도
        """
        # 반파장 다이폴 안테나의 복사 패턴
        # E(θ) = cos((π/2)cos(θ)) / sin(θ)
        
        theta = np.asarray(theta)
        
        # 특이점 처리 (θ = 0, π)
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = np.cos((np.pi / 2) * np.cos(theta))
            denominator = np.sin(theta)
            E_theta = np.where(np.abs(denominator) > 1e-10, 
                             numerator / denominator, 
                             0)
        
        return np.abs(E_theta)
    
    def plot_2d_pattern(self, num_points=360):
        """2D 복사 패턴 플롯 (극좌표)"""
        theta = np.linspace(0, 2 * np.pi, num_points)
        
        # 0 ~ π 범위의 패턴을 계산하고 대칭으로 확장
        theta_half = np.linspace(0, np.pi, num_points // 2)
        E = self.radiation_pattern(theta_half)
        
        # 전체 원을 위해 대칭 복사
        E_full = np.concatenate([E, E[::-1]])
        
        # 정규화
        E_normalized = E_full / np.max(E_full)
        
        # 플롯
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), 
                                       subplot_kw=dict(projection='polar'))
        
        # 선형 스케일
        ax1.plot(theta, E_normalized, 'b-', linewidth=2)
        ax1.set_title(f'복사 패턴 (선형 스케일)\n주파수: {self.frequency/1e9:.2f} GHz', 
                     fontsize=12, pad=20)
        ax1.set_theta_zero_location('N')
        ax1.grid(True)
        
        # dB 스케일
        E_dB = 20 * np.log10(E_normalized + 1e-10)
        E_dB = np.maximum(E_dB, -40)  # -40 dB 이하 제한
        
        ax2.plot(theta, E_dB, 'r-', linewidth=2)
        ax2.set_title(f'복사 패턴 (dB 스케일)\n주파수: {self.frequency/1e9:.2f} GHz', 
                     fontsize=12, pad=20)
        ax2.set_theta_zero_location('N')
        ax2.set_ylim([-40, 0])
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_pattern(self, num_points=50):
        """3D 복사 패턴 플롯"""
        theta = np.linspace(0, np.pi, num_points)
        phi = np.linspace(0, 2 * np.pi, num_points)
        
        THETA, PHI = np.meshgrid(theta, phi)
        
        # 복사 패턴 계산
        E = self.radiation_pattern(THETA)
        E_normalized = E / np.max(E)
        
        # 구면 좌표를 직교 좌표로 변환
        X = E_normalized * np.sin(THETA) * np.cos(PHI)
        Y = E_normalized * np.sin(THETA) * np.sin(PHI)
        Z = E_normalized * np.cos(THETA)
        
        # 3D 플롯
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 표면 플롯
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, 
                              linewidth=0, antialiased=True, 
                              alpha=0.8)
        
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title(f'3D 복사 패턴\n다이폴 안테나 (주파수: {self.frequency/1e9:.2f} GHz)', 
                    fontsize=14, pad=20)
        
        # 컬러바 추가
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='정규화된 전계 강도')
        
        # 시야각 설정
        ax.view_init(elev=20, azim=45)
        
        return fig
    
    def calculate_parameters(self):
        """안테나 주요 파라미터 계산"""
        params = {
            '주파수 (GHz)': self.frequency / 1e9,
            '파장 (m)': self.wavelength,
            '파장 (cm)': self.wavelength * 100,
            '안테나 길이 (m)': self.length,
            '안테나 길이 (cm)': self.length * 100,
            '길이/파장 비': self.length / self.wavelength,
            '입력 임피던스 (Ω)': 73,  # 반파장 다이폴의 이론값
            '지향성 (dBi)': 2.15,  # 반파장 다이폴의 이론값
            '반전력 빔폭 (도)': 78,  # 반파장 다이폴의 이론값
        }
        return params


def main():
    """메인 함수 - 예제 실행"""
    print("=" * 60)
    print("안테나 시뮬레이션 프로그램")
    print("=" * 60)
    
    # 2.4 GHz 다이폴 안테나 생성 (WiFi 대역)
    antenna = DipoleAntenna(frequency=2.4e9)
    
    # 안테나 파라미터 출력
    print("\n[안테나 파라미터]")
    params = antenna.calculate_parameters()
    for key, value in params.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    
    print("\n복사 패턴 생성 중...")
    
    # 2D 복사 패턴 플롯
    fig_2d = antenna.plot_2d_pattern()
    fig_2d.savefig('antenna_pattern_2d.png', dpi=150, bbox_inches='tight')
    print("✓ 2D 복사 패턴 저장: antenna_pattern_2d.png")
    
    # 3D 복사 패턴 플롯
    fig_3d = antenna.plot_3d_pattern()
    fig_3d.savefig('antenna_pattern_3d.png', dpi=150, bbox_inches='tight')
    print("✓ 3D 복사 패턴 저장: antenna_pattern_3d.png")
    
    print("\n시뮬레이션 완료!")
    print("=" * 60)
    
    # 플롯 표시
    plt.show()


if __name__ == "__main__":
    main()
