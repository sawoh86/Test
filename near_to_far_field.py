#!/usr/bin/env python3
"""
근접전계(Near-Field) → 원전계(Far-Field) 변환 프로그램
Plane Wave Spectrum 방법을 이용한 전자기장 변환
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os


class NearToFarFieldTransform:
    def __init__(self, config):
        """
        Parameters:
        -----------
        config : dict
            frequency : float - 주파수 (Hz)
            z_distance : float - 근접전계 스캔 평면과 원점 사이의 거리 (m)
            far_field_distance : float - 원전계 계산 거리 (m)
            theta_range : tuple - 원전계 관찰 각도 범위 (deg), (시작, 끝, 스텝)
            phi_range : tuple - 원전계 관찰 각도 범위 (deg), (시작, 끝, 스텝)
        """
        self.frequency = config['frequency']
        self.z_distance = config['z_distance']
        self.far_field_distance = config['far_field_distance']
        self.theta_range = config['theta_range']
        self.phi_range = config['phi_range']
        
        self.c = 3e8  # 빛의 속도
        self.wavelength = self.c / self.frequency
        self.k0 = 2 * np.pi / self.wavelength  # 파수
        
        self.near_field_data = None
        self.x_coords = None
        self.y_coords = None
        self.X = None
        self.Y = None
        self.far_field_pattern = None
        self.back_projected = None
        
    def load_csv(self, filename):
        """
        CSV 파일에서 근접전계 데이터 로드
        CSV 형식: x, y, magnitude, phase
        """
        print(f"CSV 파일 로드 중: {filename}")
        df = pd.read_csv(filename)
        
        required_columns = ['x', 'y', 'magnitude', 'phase']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV 파일에 필수 컬럼이 없습니다: {required_columns}")
        
        self.x_coords = np.unique(df['x'].values)
        self.y_coords = np.unique(df['y'].values)
        
        nx = len(self.x_coords)
        ny = len(self.y_coords)
        
        magnitude = df['magnitude'].values.reshape(ny, nx)
        phase_deg = df['phase'].values.reshape(ny, nx)
        phase_rad = np.deg2rad(phase_deg)
        
        self.near_field_data = magnitude * np.exp(1j * phase_rad)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        
        print(f"데이터 로드 완료: {nx} x {ny} 포인트")
        print(f"X 범위: {self.x_coords[0]:.4f} ~ {self.x_coords[-1]:.4f} m")
        print(f"Y 범위: {self.y_coords[0]:.4f} ~ {self.y_coords[-1]:.4f} m")
        
        return self.near_field_data
    
    def compute_far_field(self):
        """
        Plane Wave Spectrum 방법을 이용한 원전계 변환
        """
        print("\n원전계 변환 수행 중...")
        
        dx = self.x_coords[1] - self.x_coords[0]
        dy = self.y_coords[1] - self.y_coords[0]
        
        nx = len(self.x_coords)
        ny = len(self.y_coords)
        
        # 2D FFT를 이용한 스펙트럼 계산
        spectrum = np.fft.fft2(self.near_field_data)
        spectrum = np.fft.fftshift(spectrum)
        
        # 공간 주파수
        kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        KX, KY = np.meshgrid(kx, ky)
        
        # 전파 상수
        kz_squared = self.k0**2 - KX**2 - KY**2
        KZ = np.sqrt(kz_squared.astype(complex))
        
        # Evanescent wave 필터링 (전파 불가능한 파동 제거)
        propagating_mask = (kz_squared > 0)
        KZ[~propagating_mask] = 0
        
        # 원전계 거리로 전파
        propagation_distance = self.far_field_distance - self.z_distance
        propagation_factor = np.exp(1j * KZ * propagation_distance)
        far_spectrum = spectrum * propagation_factor
        
        # 원전계 각도별 패턴 계산
        theta = np.deg2rad(np.arange(self.theta_range[0], 
                                     self.theta_range[1], 
                                     self.theta_range[2]))
        phi = np.deg2rad(np.arange(self.phi_range[0], 
                                   self.phi_range[1], 
                                   self.phi_range[2]))
        
        THETA, PHI = np.meshgrid(theta, phi)
        
        self.far_field_pattern = np.zeros_like(THETA, dtype=complex)
        
        # 각 방향에 대해 스펙트럼 샘플링
        for i, th in enumerate(theta):
            for j, ph in enumerate(phi):
                kx_sample = self.k0 * np.sin(th) * np.cos(ph)
                ky_sample = self.k0 * np.sin(th) * np.sin(ph)
                
                # 가장 가까운 스펙트럼 값 찾기 (보간)
                idx_x = np.argmin(np.abs(kx - kx_sample))
                idx_y = np.argmin(np.abs(ky - ky_sample))
                
                if propagating_mask[idx_y, idx_x]:
                    self.far_field_pattern[j, i] = far_spectrum[idx_y, idx_x]
        
        self.theta_grid = THETA
        self.phi_grid = PHI
        self.spectrum = far_spectrum
        self.KX = KX
        self.KY = KY
        self.KZ = KZ
        
        print("원전계 변환 완료!")
        return self.far_field_pattern
    
    def back_projection(self):
        """
        Back Projection: 원전계 → 근접전계 역변환
        """
        print("\nBack Projection 수행 중...")
        
        # 스펙트럼에서 역전파
        propagation_distance = self.far_field_distance - self.z_distance
        back_propagation_factor = np.exp(-1j * self.KZ * propagation_distance)
        back_spectrum = self.spectrum * back_propagation_factor
        
        # 역 FFT
        back_spectrum = np.fft.ifftshift(back_spectrum)
        self.back_projected = np.fft.ifft2(back_spectrum)
        
        print("Back Projection 완료!")
        return self.back_projected
    
    def plot_near_field(self, field_data=None, title="근접전계 데이터", filename=None):
        """
        근접전계 데이터 시각화
        """
        if field_data is None:
            field_data = self.near_field_data
        
        magnitude = np.abs(field_data)
        phase = np.angle(field_data, deg=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 크기 플롯
        im1 = axes[0].contourf(self.X * 1000, self.Y * 1000, magnitude, 
                               levels=50, cmap='hot')
        axes[0].set_xlabel('X (mm)')
        axes[0].set_ylabel('Y (mm)')
        axes[0].set_title(f'{title} - 크기')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0], label='크기 (V/m)')
        
        # 위상 플롯
        im2 = axes[1].contourf(self.X * 1000, self.Y * 1000, phase, 
                               levels=50, cmap='hsv')
        axes[1].set_xlabel('X (mm)')
        axes[1].set_ylabel('Y (mm)')
        axes[1].set_title(f'{title} - 위상')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1], label='위상 (도)')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"그림 저장됨: {filename}")
        
        plt.show()
    
    def plot_far_field(self, filename=None):
        """
        원전계 패턴 시각화 (극좌표 및 직교좌표)
        """
        magnitude_db = 20 * np.log10(np.abs(self.far_field_pattern) + 1e-10)
        magnitude_db -= np.max(magnitude_db)  # 정규화
        
        fig = plt.figure(figsize=(16, 5))
        
        # 1. 2D 히트맵
        ax1 = fig.add_subplot(131)
        im = ax1.contourf(np.rad2deg(self.theta_grid), 
                         np.rad2deg(self.phi_grid), 
                         magnitude_db, 
                         levels=50, cmap='jet')
        ax1.set_xlabel('Theta (도)')
        ax1.set_ylabel('Phi (도)')
        ax1.set_title('원전계 패턴 (dB)')
        plt.colorbar(im, ax=ax1, label='상대 크기 (dB)')
        
        # 2. Phi=0 평면 극좌표 플롯
        ax2 = fig.add_subplot(132, projection='polar')
        phi_0_idx = len(self.phi_grid) // 2
        theta_cut = self.theta_grid[phi_0_idx, :]
        pattern_cut = magnitude_db[phi_0_idx, :]
        ax2.plot(theta_cut, pattern_cut)
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_title('원전계 패턴 (Phi=0°)')
        ax2.set_ylim([-40, 0])
        ax2.grid(True)
        
        # 3. 3D 표면 플롯
        ax3 = fig.add_subplot(133, projection='3d')
        X_3d = np.abs(self.far_field_pattern) * np.sin(self.theta_grid) * np.cos(self.phi_grid)
        Y_3d = np.abs(self.far_field_pattern) * np.sin(self.theta_grid) * np.sin(self.phi_grid)
        Z_3d = np.abs(self.far_field_pattern) * np.cos(self.theta_grid)
        
        surf = ax3.plot_surface(X_3d, Y_3d, Z_3d, cmap='jet', alpha=0.8)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('원전계 3D 패턴')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"그림 저장됨: {filename}")
        
        plt.show()
    
    def plot_comparison(self, filename=None):
        """
        원본 근접전계 vs Back Projection 비교
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        original_mag = np.abs(self.near_field_data)
        original_phase = np.angle(self.near_field_data, deg=True)
        
        back_mag = np.abs(self.back_projected)
        back_phase = np.angle(self.back_projected, deg=True)
        
        error_mag = np.abs(original_mag - back_mag)
        error_phase = np.abs(original_phase - back_phase)
        
        # 원본 크기
        im1 = axes[0, 0].contourf(self.X * 1000, self.Y * 1000, original_mag, 
                                  levels=50, cmap='hot')
        axes[0, 0].set_title('원본 근접전계 - 크기')
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 원본 위상
        im2 = axes[0, 1].contourf(self.X * 1000, self.Y * 1000, original_phase, 
                                  levels=50, cmap='hsv')
        axes[0, 1].set_title('원본 근접전계 - 위상')
        axes[0, 1].set_xlabel('X (mm)')
        axes[0, 1].set_ylabel('Y (mm)')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 크기 오차
        im3 = axes[0, 2].contourf(self.X * 1000, self.Y * 1000, error_mag, 
                                  levels=50, cmap='viridis')
        axes[0, 2].set_title('크기 오차')
        axes[0, 2].set_xlabel('X (mm)')
        axes[0, 2].set_ylabel('Y (mm)')
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Back projection 크기
        im4 = axes[1, 0].contourf(self.X * 1000, self.Y * 1000, back_mag, 
                                  levels=50, cmap='hot')
        axes[1, 0].set_title('Back Projection - 크기')
        axes[1, 0].set_xlabel('X (mm)')
        axes[1, 0].set_ylabel('Y (mm)')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Back projection 위상
        im5 = axes[1, 1].contourf(self.X * 1000, self.Y * 1000, back_phase, 
                                  levels=50, cmap='hsv')
        axes[1, 1].set_title('Back Projection - 위상')
        axes[1, 1].set_xlabel('X (mm)')
        axes[1, 1].set_ylabel('Y (mm)')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # 위상 오차
        im6 = axes[1, 2].contourf(self.X * 1000, self.Y * 1000, error_phase, 
                                  levels=50, cmap='viridis')
        axes[1, 2].set_title('위상 오차')
        axes[1, 2].set_xlabel('X (mm)')
        axes[1, 2].set_ylabel('Y (mm)')
        axes[1, 2].set_aspect('equal')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"그림 저장됨: {filename}")
        
        plt.show()
        
        # 오차 통계
        print("\n=== Back Projection 오차 분석 ===")
        print(f"크기 평균 오차: {np.mean(error_mag):.6f}")
        print(f"크기 최대 오차: {np.max(error_mag):.6f}")
        print(f"크기 상대 오차: {np.mean(error_mag/original_mag)*100:.2f}%")
        print(f"위상 평균 오차: {np.mean(error_phase):.2f}°")
        print(f"위상 최대 오차: {np.max(error_phase):.2f}°")


def load_config(config_file='config.txt'):
    """
    설정 파일에서 파라미터 로드
    """
    config = {}
    
    if os.path.exists(config_file):
        print(f"설정 파일 로드 중: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            if ',' in value:
                                config[key] = tuple(map(float, value.split(',')))
                            else:
                                config[key] = float(value)
                        except ValueError:
                            config[key] = value
    else:
        print("설정 파일이 없습니다. 기본값 사용")
        config = {
            'frequency': 10e9,  # 10 GHz
            'z_distance': 0.01,  # 10 mm
            'far_field_distance': 1.0,  # 1 m
            'theta_range': (0, 90, 1),
            'phi_range': (0, 360, 5)
        }
    
    return config


def main():
    """
    메인 실행 함수
    """
    print("=" * 60)
    print("근접전계 → 원전계 변환 프로그램")
    print("=" * 60)
    
    # 설정 로드
    config = load_config('config.txt')
    
    print("\n=== 설정 파라미터 ===")
    print(f"주파수: {config['frequency']/1e9:.2f} GHz")
    print(f"근접전계 스캔 거리: {config['z_distance']*1000:.1f} mm")
    print(f"원전계 계산 거리: {config['far_field_distance']:.2f} m")
    print(f"Theta 범위: {config['theta_range']}°")
    print(f"Phi 범위: {config['phi_range']}°")
    
    # 변환 객체 생성
    transformer = NearToFarFieldTransform(config)
    
    # CSV 데이터 로드
    csv_file = 'near_field_data.csv'
    if not os.path.exists(csv_file):
        print(f"\n경고: {csv_file} 파일이 없습니다!")
        print("예제 데이터를 생성하려면 generate_sample_data.py를 실행하세요.")
        return
    
    transformer.load_csv(csv_file)
    
    # 1. 원본 근접전계 데이터 시각화
    print("\n" + "=" * 60)
    print("1. 원본 근접전계 데이터 시각화")
    print("=" * 60)
    transformer.plot_near_field(title="원본 근접전계 데이터", 
                                filename="near_field_original.png")
    
    # 2. 원전계 변환
    print("\n" + "=" * 60)
    print("2. 원전계 변환 수행")
    print("=" * 60)
    transformer.compute_far_field()
    transformer.plot_far_field(filename="far_field_pattern.png")
    
    # 3. Back Projection
    print("\n" + "=" * 60)
    print("3. Back Projection 수행")
    print("=" * 60)
    transformer.back_projection()
    transformer.plot_comparison(filename="comparison.png")
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
