#!/usr/bin/env python3
"""
벡터 근접전계(Near-Field) → 원전계(Far-Field) 변환 프로그램
전계 방향(Ex, Ey, Ez)을 고려한 완전한 벡터 전자기장 변환
Plane Wave Spectrum 방법 + 편파 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os


class VectorNearToFarFieldTransform:
    """
    벡터 전계를 고려한 근접전계-원전계 변환 클래스
    Ex, Ey, Ez 성분을 모두 처리하고 편파 분석 포함
    """
    
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
            field_components : list - 사용할 전계 성분 ['Ex', 'Ey', 'Ez']
        """
        self.frequency = config['frequency']
        self.z_distance = config['z_distance']
        self.far_field_distance = config['far_field_distance']
        self.theta_range = config['theta_range']
        self.phi_range = config['phi_range']
        self.field_components = config.get('field_components', ['Ex', 'Ey', 'Ez'])
        
        # 물리 상수
        self.c = 3e8  # 빛의 속도 (m/s)
        self.mu0 = 4 * np.pi * 1e-7  # 진공 투자율 (H/m)
        self.epsilon0 = 8.854187817e-12  # 진공 유전율 (F/m)
        self.eta0 = np.sqrt(self.mu0 / self.epsilon0)  # 진공 임피던스 (Ω)
        
        self.wavelength = self.c / self.frequency
        self.k0 = 2 * np.pi / self.wavelength  # 파수
        
        # 데이터 저장
        self.near_field_data = {}  # {'Ex': array, 'Ey': array, 'Ez': array}
        self.x_coords = None
        self.y_coords = None
        self.X = None
        self.Y = None
        
        # 변환 결과
        self.far_field_E = {}  # {'Ex': array, 'Ey': array, 'Ez': array}
        self.far_field_H = {}  # {'Hx': array, 'Hy': array, 'Hz': array}
        self.far_field_theta = None  # E_theta 성분
        self.far_field_phi = None    # E_phi 성분
        self.poynting_vector = None
        self.back_projected = {}
        
    def load_csv(self, filename):
        """
        CSV 파일에서 벡터 근접전계 데이터 로드
        
        CSV 형식: x, y, Ex_mag, Ex_phase, Ey_mag, Ey_phase, Ez_mag, Ez_phase
        또는: x, y, magnitude, phase (단일 성분, 이전 버전과 호환)
        """
        print(f"CSV 파일 로드 중: {filename}")
        df = pd.read_csv(filename)
        
        # 좌표 추출
        self.x_coords = np.unique(df['x'].values)
        self.y_coords = np.unique(df['y'].values)
        
        nx = len(self.x_coords)
        ny = len(self.y_coords)
        
        # 벡터 성분 로드
        if 'Ex_mag' in df.columns:
            # 벡터 필드 형식
            for component in ['Ex', 'Ey', 'Ez']:
                if f'{component}_mag' in df.columns:
                    mag_col = f'{component}_mag'
                    phase_col = f'{component}_phase'
                    
                    magnitude = df[mag_col].values.reshape(ny, nx)
                    phase_deg = df[phase_col].values.reshape(ny, nx)
                    phase_rad = np.deg2rad(phase_deg)
                    
                    self.near_field_data[component] = magnitude * np.exp(1j * phase_rad)
                    print(f"  {component} 성분 로드 완료")
                else:
                    # 해당 성분이 없으면 0으로 초기화
                    self.near_field_data[component] = np.zeros((ny, nx), dtype=complex)
                    print(f"  {component} 성분 없음 (0으로 설정)")
        
        elif 'magnitude' in df.columns:
            # 스칼라 필드 형식 (이전 버전과 호환)
            magnitude = df['magnitude'].values.reshape(ny, nx)
            phase_deg = df['phase'].values.reshape(ny, nx)
            phase_rad = np.deg2rad(phase_deg)
            
            # 기본값으로 Ez에 할당
            self.near_field_data['Ez'] = magnitude * np.exp(1j * phase_rad)
            self.near_field_data['Ex'] = np.zeros((ny, nx), dtype=complex)
            self.near_field_data['Ey'] = np.zeros((ny, nx), dtype=complex)
            print("  스칼라 필드 → Ez 성분으로 로드")
            print("  Ex, Ey는 0으로 설정")
        
        else:
            raise ValueError("CSV 파일 형식이 올바르지 않습니다.")
        
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        
        print(f"\n데이터 로드 완료:")
        print(f"  그리드 크기: {nx} x {ny} 포인트")
        print(f"  X 범위: {self.x_coords[0]:.4f} ~ {self.x_coords[-1]:.4f} m")
        print(f"  Y 범위: {self.y_coords[0]:.4f} ~ {self.y_coords[-1]:.4f} m")
        print(f"  전계 성분: {list(self.near_field_data.keys())}")
        
        return self.near_field_data
    
    def compute_far_field(self):
        """
        벡터 Plane Wave Spectrum 방법을 이용한 원전계 변환
        모든 전계 성분(Ex, Ey, Ez)을 고려
        """
        print("\n원전계 변환 수행 중...")
        
        dx = self.x_coords[1] - self.x_coords[0]
        dy = self.y_coords[1] - self.y_coords[0]
        
        nx = len(self.x_coords)
        ny = len(self.y_coords)
        
        # 공간 주파수
        kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, dy) * 2 * np.pi
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        KX, KY = np.meshgrid(kx, ky)
        
        # 전파 상수
        kz_squared = self.k0**2 - KX**2 - KY**2
        KZ = np.sqrt(kz_squared.astype(complex))
        
        # Evanescent wave 필터링
        propagating_mask = (kz_squared > 0)
        KZ[~propagating_mask] = 0
        
        # 전파 거리
        propagation_distance = self.far_field_distance - self.z_distance
        propagation_factor = np.exp(1j * KZ * propagation_distance)
        
        # 각 전계 성분에 대해 스펙트럼 변환
        spectra = {}
        for component in ['Ex', 'Ey', 'Ez']:
            if component in self.near_field_data:
                spectrum = np.fft.fft2(self.near_field_data[component])
                spectrum = np.fft.fftshift(spectrum)
                spectra[component] = spectrum * propagation_factor
            else:
                spectra[component] = np.zeros((ny, nx), dtype=complex)
        
        # 원전계 각도
        theta = np.deg2rad(np.arange(self.theta_range[0], 
                                     self.theta_range[1], 
                                     self.theta_range[2]))
        phi = np.deg2rad(np.arange(self.phi_range[0], 
                                   self.phi_range[1], 
                                   self.phi_range[2]))
        
        THETA, PHI = np.meshgrid(theta, phi)
        
        # 원전계 패턴 초기화
        for component in ['Ex', 'Ey', 'Ez']:
            self.far_field_E[component] = np.zeros_like(THETA, dtype=complex)
        
        # 각 방향에 대해 스펙트럼 샘플링
        print("  스펙트럼 샘플링 중...")
        for i, th in enumerate(theta):
            for j, ph in enumerate(phi):
                kx_sample = self.k0 * np.sin(th) * np.cos(ph)
                ky_sample = self.k0 * np.sin(th) * np.sin(ph)
                
                # 가장 가까운 스펙트럼 값 찾기
                idx_x = np.argmin(np.abs(kx - kx_sample))
                idx_y = np.argmin(np.abs(ky - ky_sample))
                
                if propagating_mask[idx_y, idx_x]:
                    for component in ['Ex', 'Ey', 'Ez']:
                        self.far_field_E[component][j, i] = spectra[component][idx_y, idx_x]
        
        # 구면 좌표계로 변환 (E_theta, E_phi)
        self._compute_spherical_components(THETA, PHI)
        
        # 자기장 계산 (H = k × E / (ω*μ0))
        self._compute_magnetic_field(THETA, PHI)
        
        # Poynting 벡터 계산
        self._compute_poynting_vector()
        
        self.theta_grid = THETA
        self.phi_grid = PHI
        self.spectra = spectra
        self.KX = KX
        self.KY = KY
        self.KZ = KZ
        
        print("원전계 변환 완료!")
        return self.far_field_E
    
    def _compute_spherical_components(self, THETA, PHI):
        """
        직교 좌표계(Ex, Ey, Ez)에서 구면 좌표계(E_theta, E_phi)로 변환
        """
        Ex = self.far_field_E['Ex']
        Ey = self.far_field_E['Ey']
        Ez = self.far_field_E['Ez']
        
        # E_theta = E_x * cos(theta) * cos(phi) + E_y * cos(theta) * sin(phi) - E_z * sin(theta)
        self.far_field_theta = (Ex * np.cos(THETA) * np.cos(PHI) + 
                                Ey * np.cos(THETA) * np.sin(PHI) - 
                                Ez * np.sin(THETA))
        
        # E_phi = -E_x * sin(phi) + E_y * cos(phi)
        self.far_field_phi = -Ex * np.sin(PHI) + Ey * np.cos(PHI)
        
        print("  구면 좌표 성분 계산 완료")
    
    def _compute_magnetic_field(self, THETA, PHI):
        """
        자기장 계산: H = (k × E) / (ω * μ0)
        원전계에서: H = (r̂ × E) / η0
        """
        Ex = self.far_field_E['Ex']
        Ey = self.far_field_E['Ey']
        Ez = self.far_field_E['Ez']
        
        # 방향 단위 벡터
        r_hat_x = np.sin(THETA) * np.cos(PHI)
        r_hat_y = np.sin(THETA) * np.sin(PHI)
        r_hat_z = np.cos(THETA)
        
        # H = (r̂ × E) / η0
        self.far_field_H['Hx'] = (r_hat_y * Ez - r_hat_z * Ey) / self.eta0
        self.far_field_H['Hy'] = (r_hat_z * Ex - r_hat_x * Ez) / self.eta0
        self.far_field_H['Hz'] = (r_hat_x * Ey - r_hat_y * Ex) / self.eta0
        
        print("  자기장 계산 완료")
    
    def _compute_poynting_vector(self):
        """
        Poynting 벡터 계산: P = Re(E* × H)
        """
        Ex = self.far_field_E['Ex']
        Ey = self.far_field_E['Ey']
        Ez = self.far_field_E['Ez']
        
        Hx = self.far_field_H['Hx']
        Hy = self.far_field_H['Hy']
        Hz = self.far_field_H['Hz']
        
        # P = E* × H
        Px = np.real(np.conj(Ey) * Hz - np.conj(Ez) * Hy)
        Py = np.real(np.conj(Ez) * Hx - np.conj(Ex) * Hz)
        Pz = np.real(np.conj(Ex) * Hy - np.conj(Ey) * Hx)
        
        self.poynting_vector = {
            'Px': Px,
            'Py': Py,
            'Pz': Pz,
            'magnitude': np.sqrt(Px**2 + Py**2 + Pz**2)
        }
        
        print("  Poynting 벡터 계산 완료")
    
    def back_projection(self):
        """
        Back Projection: 원전계 → 근접전계 역변환
        """
        print("\nBack Projection 수행 중...")
        
        propagation_distance = self.far_field_distance - self.z_distance
        back_propagation_factor = np.exp(-1j * self.KZ * propagation_distance)
        
        for component in ['Ex', 'Ey', 'Ez']:
            back_spectrum = self.spectra[component] * back_propagation_factor
            back_spectrum = np.fft.ifftshift(back_spectrum)
            self.back_projected[component] = np.fft.ifft2(back_spectrum)
        
        print("Back Projection 완료!")
        return self.back_projected
    
    def plot_near_field(self, field_data=None, title="근접전계 데이터", filename=None):
        """
        벡터 근접전계 데이터 시각화
        """
        if field_data is None:
            field_data = self.near_field_data
        
        # 전체 전계 크기 계산
        E_total = np.zeros_like(list(field_data.values())[0])
        for component in ['Ex', 'Ey', 'Ez']:
            if component in field_data:
                E_total += np.abs(field_data[component])**2
        E_total = np.sqrt(E_total)
        
        fig = plt.figure(figsize=(18, 12))
        
        # 전체 크기
        ax1 = fig.add_subplot(2, 4, 1)
        im1 = ax1.contourf(self.X * 1000, self.Y * 1000, E_total, 
                          levels=50, cmap='hot')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f'{title} - Total Magnitude')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='|E| (V/m)')
        
        # 각 성분 크기
        for idx, component in enumerate(['Ex', 'Ey', 'Ez']):
            if component in field_data:
                ax = fig.add_subplot(2, 4, idx + 2)
                magnitude = np.abs(field_data[component])
                im = ax.contourf(self.X * 1000, self.Y * 1000, magnitude, 
                               levels=50, cmap='hot')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_title(f'{component} Magnitude')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax, label='Magnitude (V/m)')
        
        # 각 성분 위상
        for idx, component in enumerate(['Ex', 'Ey', 'Ez']):
            if component in field_data:
                ax = fig.add_subplot(2, 4, idx + 5)
                phase = np.angle(field_data[component], deg=True)
                im = ax.contourf(self.X * 1000, self.Y * 1000, phase, 
                               levels=50, cmap='hsv')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_title(f'{component} Phase')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax, label='Phase (deg)')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"그림 저장됨: {filename}")
        
        plt.show()
    
    def plot_far_field(self, filename=None):
        """
        원전계 패턴 시각화 (편파 성분 포함)
        """
        # 전체 전계 크기
        E_total = np.sqrt(np.abs(self.far_field_theta)**2 + 
                         np.abs(self.far_field_phi)**2)
        magnitude_db = 20 * np.log10(E_total + 1e-10)
        magnitude_db -= np.max(magnitude_db)
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. 전체 패턴 (dB)
        ax1 = fig.add_subplot(2, 4, 1)
        im = ax1.contourf(np.rad2deg(self.theta_grid), 
                         np.rad2deg(self.phi_grid), 
                         magnitude_db, 
                         levels=50, cmap='jet')
        ax1.set_xlabel('Theta (deg)')
        ax1.set_ylabel('Phi (deg)')
        ax1.set_title('Total Far-Field Pattern (dB)')
        plt.colorbar(im, ax=ax1, label='Relative Magnitude (dB)')
        
        # 2. E_theta 성분
        ax2 = fig.add_subplot(2, 4, 2)
        E_theta_db = 20 * np.log10(np.abs(self.far_field_theta) + 1e-10)
        E_theta_db -= np.max(E_theta_db)
        im = ax2.contourf(np.rad2deg(self.theta_grid), 
                         np.rad2deg(self.phi_grid), 
                         E_theta_db, 
                         levels=50, cmap='jet')
        ax2.set_xlabel('Theta (deg)')
        ax2.set_ylabel('Phi (deg)')
        ax2.set_title('E_theta Component (dB)')
        plt.colorbar(im, ax=ax2, label='dB')
        
        # 3. E_phi 성분
        ax3 = fig.add_subplot(2, 4, 3)
        E_phi_db = 20 * np.log10(np.abs(self.far_field_phi) + 1e-10)
        E_phi_db -= np.max(E_phi_db)
        im = ax3.contourf(np.rad2deg(self.theta_grid), 
                         np.rad2deg(self.phi_grid), 
                         E_phi_db, 
                         levels=50, cmap='jet')
        ax3.set_xlabel('Theta (deg)')
        ax3.set_ylabel('Phi (deg)')
        ax3.set_title('E_phi Component (dB)')
        plt.colorbar(im, ax=ax3, label='dB')
        
        # 4. Poynting 벡터
        ax4 = fig.add_subplot(2, 4, 4)
        P_mag = self.poynting_vector['magnitude']
        P_db = 10 * np.log10(P_mag / np.max(P_mag) + 1e-10)
        im = ax4.contourf(np.rad2deg(self.theta_grid), 
                         np.rad2deg(self.phi_grid), 
                         P_db, 
                         levels=50, cmap='hot')
        ax4.set_xlabel('Theta (deg)')
        ax4.set_ylabel('Phi (deg)')
        ax4.set_title('Poynting Vector (dB)')
        plt.colorbar(im, ax=ax4, label='dB')
        
        # 5-6. 극좌표 플롯 (Phi=0, Phi=90)
        for idx, phi_idx in enumerate([0, len(self.phi_grid)//4]):
            ax = fig.add_subplot(2, 4, 5 + idx, projection='polar')
            theta_cut = self.theta_grid[phi_idx, :]
            pattern_cut = magnitude_db[phi_idx, :]
            ax.plot(theta_cut, pattern_cut, 'b-', linewidth=2, label='Total')
            
            E_theta_cut = E_theta_db[phi_idx, :]
            E_phi_cut = E_phi_db[phi_idx, :]
            ax.plot(theta_cut, E_theta_cut, 'r--', linewidth=1, label='E_theta')
            ax.plot(theta_cut, E_phi_cut, 'g--', linewidth=1, label='E_phi')
            
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            phi_deg = np.rad2deg(self.phi_grid[phi_idx, 0])
            ax.set_title(f'Pattern (Phi={phi_deg:.0f}°)')
            ax.set_ylim([-40, 0])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True)
        
        # 7. 3D 패턴
        ax7 = fig.add_subplot(2, 4, 7, projection='3d')
        r_3d = np.abs(E_total)
        X_3d = r_3d * np.sin(self.theta_grid) * np.cos(self.phi_grid)
        Y_3d = r_3d * np.sin(self.theta_grid) * np.sin(self.phi_grid)
        Z_3d = r_3d * np.cos(self.theta_grid)
        
        surf = ax7.plot_surface(X_3d, Y_3d, Z_3d, cmap='jet', alpha=0.8)
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        ax7.set_zlabel('Z')
        ax7.set_title('3D Far-Field Pattern')
        
        # 8. 편파 비율
        ax8 = fig.add_subplot(2, 4, 8)
        polarization_ratio = np.abs(self.far_field_theta) / (np.abs(self.far_field_phi) + 1e-10)
        polarization_ratio_db = 10 * np.log10(polarization_ratio)
        im = ax8.contourf(np.rad2deg(self.theta_grid), 
                         np.rad2deg(self.phi_grid), 
                         polarization_ratio_db, 
                         levels=50, cmap='RdBu_r', vmin=-20, vmax=20)
        ax8.set_xlabel('Theta (deg)')
        ax8.set_ylabel('Phi (deg)')
        ax8.set_title('Polarization Ratio (E_theta/E_phi in dB)')
        plt.colorbar(im, ax=ax8, label='dB')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"그림 저장됨: {filename}")
        
        plt.show()
    
    def plot_comparison(self, filename=None):
        """
        원본 근접전계 vs Back Projection 비교 (벡터)
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        
        for idx, component in enumerate(['Ex', 'Ey', 'Ez']):
            original_mag = np.abs(self.near_field_data[component])
            back_mag = np.abs(self.back_projected[component])
            error_mag = np.abs(original_mag - back_mag)
            
            # 원본
            im1 = axes[idx, 0].contourf(self.X * 1000, self.Y * 1000, original_mag, 
                                       levels=50, cmap='hot')
            axes[idx, 0].set_title(f'Original {component}')
            axes[idx, 0].set_xlabel('X (mm)')
            axes[idx, 0].set_ylabel('Y (mm)')
            axes[idx, 0].set_aspect('equal')
            plt.colorbar(im1, ax=axes[idx, 0])
            
            # Back projection
            im2 = axes[idx, 1].contourf(self.X * 1000, self.Y * 1000, back_mag, 
                                       levels=50, cmap='hot')
            axes[idx, 1].set_title(f'Back Projected {component}')
            axes[idx, 1].set_xlabel('X (mm)')
            axes[idx, 1].set_ylabel('Y (mm)')
            axes[idx, 1].set_aspect('equal')
            plt.colorbar(im2, ax=axes[idx, 1])
            
            # 오차
            im3 = axes[idx, 2].contourf(self.X * 1000, self.Y * 1000, error_mag, 
                                       levels=50, cmap='viridis')
            axes[idx, 2].set_title(f'{component} Error')
            axes[idx, 2].set_xlabel('X (mm)')
            axes[idx, 2].set_ylabel('Y (mm)')
            axes[idx, 2].set_aspect('equal')
            plt.colorbar(im3, ax=axes[idx, 2])
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"그림 저장됨: {filename}")
        
        plt.show()
        
        # 오차 통계
        print("\n=== Back Projection 오차 분석 ===")
        for component in ['Ex', 'Ey', 'Ez']:
            original_mag = np.abs(self.near_field_data[component])
            back_mag = np.abs(self.back_projected[component])
            error_mag = np.abs(original_mag - back_mag)
            
            if np.max(original_mag) > 1e-10:
                print(f"\n{component}:")
                print(f"  크기 평균 오차: {np.mean(error_mag):.6f}")
                print(f"  크기 최대 오차: {np.max(error_mag):.6f}")
                print(f"  크기 상대 오차: {np.mean(error_mag/(original_mag+1e-10))*100:.2f}%")


def load_config(config_file='config_vector.txt'):
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
            'frequency': 10e9,
            'z_distance': 0.01,
            'far_field_distance': 1.0,
            'theta_range': (0, 90, 1),
            'phi_range': (0, 360, 5),
            'field_components': ['Ex', 'Ey', 'Ez']
        }
    
    return config


def main():
    """
    메인 실행 함수
    """
    print("=" * 60)
    print("벡터 근접전계 → 원전계 변환 프로그램")
    print("전계 방향(Ex, Ey, Ez) 고려")
    print("=" * 60)
    
    # 설정 로드
    config = load_config('config_vector.txt')
    
    print("\n=== 설정 파라미터 ===")
    print(f"주파수: {config['frequency']/1e9:.2f} GHz")
    print(f"근접전계 스캔 거리: {config['z_distance']*1000:.1f} mm")
    print(f"원전계 계산 거리: {config['far_field_distance']:.2f} m")
    print(f"Theta 범위: {config['theta_range']}°")
    print(f"Phi 범위: {config['phi_range']}°")
    
    # 변환 객체 생성
    transformer = VectorNearToFarFieldTransform(config)
    
    # CSV 데이터 로드
    csv_file = 'near_field_data_vector.csv'
    if not os.path.exists(csv_file):
        print(f"\n경고: {csv_file} 파일이 없습니다!")
        print("예제 데이터를 생성하려면 generate_sample_data_vector.py를 실행하세요.")
        return
    
    transformer.load_csv(csv_file)
    
    # 1. 원본 근접전계 데이터 시각화
    print("\n" + "=" * 60)
    print("1. 원본 근접전계 데이터 시각화")
    print("=" * 60)
    transformer.plot_near_field(title="원본 벡터 근접전계", 
                                filename="near_field_vector_original.png")
    
    # 2. 원전계 변환
    print("\n" + "=" * 60)
    print("2. 원전계 변환 수행")
    print("=" * 60)
    transformer.compute_far_field()
    transformer.plot_far_field(filename="far_field_vector_pattern.png")
    
    # 3. Back Projection
    print("\n" + "=" * 60)
    print("3. Back Projection 수행")
    print("=" * 60)
    transformer.back_projection()
    transformer.plot_comparison(filename="comparison_vector.png")
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
