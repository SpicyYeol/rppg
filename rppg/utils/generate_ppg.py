import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian, butter, filtfilt
import pandas as pd
import os
from tqdm import tqdm  # 진행 상태 표시를 위해 사용


def generate_dynamic_synthetic_ppg(duration_sec=60, fs=100,
                                   base_hr=60, hrv_std=1.0,
                                   respiratory_rate=15, respiratory_amplitude=0.05,
                                   phases=None):
    """
    동적 변화를 반영한 합성 PPG 신호를 생성합니다.

    Args:
        duration_sec (float): 신호의 지속 시간 (초).
        fs (int): 샘플링 주파수 (Hz).
        base_hr (float): 기본 심박수 (BPM).
        hrv_std (float): HRV의 표준편차 (ms).
        respiratory_rate (float): 호흡수 (호흡/분).
        respiratory_amplitude (float): 호흡 신호의 진폭.
        phases (list of dict, optional): 각 시기별 파라미터 설정.
            각 dict는 다음과 같은 키를 가져야 합니다:
                - 'start_time': 시기의 시작 시간 (초)
                - 'end_time': 시기의 종료 시간 (초)
                - 'hr': 시기의 심박수 (BPM)
                - 'bp': 시기의 혈압 (tuple, systolic/diastolic)
                - 'ae': 시기의 동맥 탄력성 (str, 'elastic', 'stiff', 'moderate')

    Returns:
        t (np.ndarray): 시간 벡터.
        ppg (np.ndarray): 합성 PPG 신호.
        signal_params (dict): 신호 생성에 사용된 파라미터.
    """
    if phases is None:
        # 기본 시기 설정: 전체 기간을 하나의 시기로 간주
        phases = [{
            'start_time': 0,
            'end_time': duration_sec,
            'hr': base_hr,
            'bp': (120, 80),
            'ae': 'moderate'
        }]

    # 시간 벡터 생성
    t = np.linspace(0, duration_sec, int(duration_sec * fs))
    ppg = np.zeros_like(t)

    # 호흡 신호 생성 (저주파 신호)
    respiratory_freq = respiratory_rate / 60  # Hz
    respiratory_signal = respiratory_amplitude * np.sin(2 * np.pi * respiratory_freq * t)

    # 시기별 파라미터 설정
    # 각 시기에 따라 심박수, 혈압, 동맥 탄력성 정의
    beat_times = []
    signal_params = {
        'respiratory_rate_bpm': respiratory_rate,
        'respiratory_amplitude': respiratory_amplitude,
        'hrv_std_ms': hrv_std,
        'phases': []
    }

    for phase in phases:
        phase_duration = phase['end_time'] - phase['start_time']
        phase_hr = phase['hr']
        phase_bp = phase['bp']
        phase_ae = phase['ae']

        # 기본 심박 간격
        phase_ibi = 60 / phase_hr  # 초

        # 심박수에 따른 심박 수 계산
        num_beats = int(phase_duration / phase_ibi) + 1

        # HRV: 심박 간격에 변동 추가 (초 단위)
        hrv_variation = np.random.normal(0, hrv_std / 1000, size=num_beats)  # 초 단위
        ibis = phase_ibi + hrv_variation
        ibis = np.clip(ibis, phase_ibi - 0.3, phase_ibi + 0.3)  # 심박 간격의 범위 제한

        # 누적 심박 타이밍
        phase_beat_times = np.cumsum(ibis) + phase['start_time']
        phase_beat_times = phase_beat_times[phase_beat_times < phase['end_time']]
        beat_times.extend(phase_beat_times)

        # 시기별 파라미터 기록
        signal_params['phases'].append({
            'hr_bpm': phase_hr,
            'bp_systolic': phase_bp[0],
            'bp_diastolic': phase_bp[1],
            'ae': phase_ae
        })

    beat_times = np.array(beat_times)

    # 시기별 동맥 탄력성 매핑
    ae_mapping = {
        'elastic': {'dicrotic_notch_depth': -0.6, 'dicrotic_notch_shift': 0.0},
        'moderate': {'dicrotic_notch_depth': -0.5, 'dicrotic_notch_shift': 0.0},
        'stiff': {'dicrotic_notch_depth': -0.4, 'dicrotic_notch_shift': 0.0},
    }

    # 각 심박에 대해 파형 생성
    for idx, beat_time in enumerate(beat_times):
        # 현재 시기의 파라미터 찾기
        phase_idx = np.searchsorted([p['end_time'] for p in phases], beat_time)
        if phase_idx >= len(phases):
            phase_idx = len(phases) - 1
        phase = phases[phase_idx]
        phase_param = signal_params['phases'][phase_idx]

        phase_hr = phase_param['hr_bpm']
        phase_bp_systolic = phase_param['bp_systolic']
        phase_bp_diastolic = phase_param['bp_diastolic']
        phase_ae = phase_param['ae']

        # 혈압에 따른 진폭 조정 (예: systolic BP에 비례)
        bp_amplitude = phase_bp_systolic / 120  # 예시 스케일링

        # 동맥 탄력성에 따른 이중 고동 조정
        ae_params = ae_mapping.get(phase_ae, ae_mapping['moderate'])
        dicrotic_notch_depth = ae_params['dicrotic_notch_depth']
        # dicrotic_notch_shift = ae_params['dicrotic_notch_shift']  # 현재는 사용하지 않음

        # 파형 구성 요소
        # 수축기(upstroke)
        upstroke_duration = 0.1  # 100ms
        upstroke_std = 0.02  # 20ms
        upstroke_length = int(upstroke_duration * fs)
        upstroke = gaussian(upstroke_length, std=upstroke_std * fs)
        upstroke = upstroke / np.max(upstroke)  # 정규화

        # 수축기 피크(systolic peak)
        systolic_peak_duration = 0.05  # 50ms
        systolic_peak_std = 0.01  # 10ms
        systolic_peak_length = int(systolic_peak_duration * fs)
        systolic_peak = gaussian(systolic_peak_length, std=systolic_peak_std * fs)
        systolic_peak = systolic_peak / np.max(systolic_peak)  # 정규화

        # 이중 고동(dicrotic notch)
        dicrotic_notch_duration = 0.05  # 50ms
        dicrotic_notch_std = 0.01  # 10ms
        dicrotic_notch_length = int(dicrotic_notch_duration * fs)
        dicrotic_notch = dicrotic_notch_depth * gaussian(dicrotic_notch_length, std=dicrotic_notch_std * fs)

        # 이완기(diastolic peak)
        diastolic_peak_duration = 0.05  # 50ms
        diastolic_peak_std = 0.015  # 15ms
        diastolic_peak_length = int(diastolic_peak_duration * fs)
        diastolic_peak = 0.7 * gaussian(diastolic_peak_length, std=diastolic_peak_std * fs)

        # 이완기 하강(diastolic downstroke)
        diastolic_downstroke_duration = 0.1  # 100ms
        diastolic_downstroke_std = 0.02  # 20ms
        diastolic_downstroke_length = int(diastolic_downstroke_duration * fs)
        diastolic_downstroke = -gaussian(diastolic_downstroke_length, std=diastolic_downstroke_std * fs)

        # 합성 PPG 파형
        ppg_pulse = np.concatenate([
            upstroke * bp_amplitude,
            systolic_peak * bp_amplitude,
            dicrotic_notch * bp_amplitude,
            diastolic_peak * bp_amplitude,
            diastolic_downstroke * bp_amplitude
        ])

        # 파형의 위치 계산
        pulse_start_time = beat_time - upstroke_duration
        if pulse_start_time < 0:
            # 파형이 신호 시작 전에 위치할 경우, 잘라냄
            ppg_pulse = ppg_pulse[-int(pulse_start_time * fs):]
            pulse_start_idx = 0
        else:
            pulse_start_idx = int(pulse_start_time * fs)

        pulse_end_idx = pulse_start_idx + len(ppg_pulse)
        if pulse_end_idx > len(ppg):
            # 파형이 신호 끝을 넘을 경우, 잘라냄
            ppg_pulse = ppg_pulse[:len(ppg) - pulse_start_idx]
            pulse_end_idx = len(ppg)

        ppg[pulse_start_idx:pulse_end_idx] += ppg_pulse

    # 호흡 신호 반영
    ppg += respiratory_signal

    # 노이즈 추가
    noise = 0.02 * np.random.randn(len(ppg))
    ppg += noise

    # 필터링 (저주파 잡음 제거)
    b, a = butter(3, 5 / (fs / 2), btype='low')
    ppg = filtfilt(b, a, ppg)

    # 정규화
    ppg = ppg / np.max(ppg)

    return t, ppg, signal_params

def generate_1000_ppg_signals(output_dir='synthetic_ppg_signals',
                              num_signals=1000,
                              duration_sec=60, fs=100):
    """
    1분 길이의 랜덤한 심박 신호 1,000개를 생성하고 저장합니다.

    Args:
        output_dir (str): 생성된 PPG 신호를 저장할 디렉토리.
        num_signals (int): 생성할 PPG 신호의 개수.
        duration_sec (float): 각 신호의 지속 시간 (초).
        fs (int): 샘플링 주파수 (Hz).

    Returns:
        None
    """
    import os
    from tqdm import tqdm  # 진행 상태 표시를 위해 사용

    # 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 합성 PPG 신호 저장을 위한 배열
    all_ppg_signals = np.zeros((num_signals, int(duration_sec * fs)))

    for i in tqdm(range(num_signals), desc="Generating PPG Signals"):
        # 랜덤하게 시기별 파라미터 설정
        # 예시로 3개의 시기로 나눔
        phases = []
        phase_duration = duration_sec / 3
        for p in range(3):
            start_time = p * phase_duration
            end_time = (p + 1) * phase_duration
            # 심박수: 60~100 BPM
            hr = np.random.uniform(60, 100)
            # 혈압: 수축기 110~140, 이완기 70~90
            bp_systolic = np.random.uniform(110, 140)
            bp_diastolic = np.random.uniform(70, 90)
            # 동맥 탄력성: 'elastic', 'moderate', 'stiff'
            ae = np.random.choice(['elastic', 'moderate', 'stiff'])

            phase = {
                'start_time': start_time,
                'end_time': end_time,
                'hr': hr,
                'bp': (bp_systolic, bp_diastolic),
                'ae': ae
            }
            phases.append(phase)

        # 랜덤하게 호흡수 설정: 12~20 호흡/분
        respiratory_rate = np.random.uniform(12, 20)
        # 호흡 진폭 설정: 0.03~0.07
        respiratory_amplitude = np.random.uniform(0.03, 0.07)

        # HRV 설정: 표준편차 0.5~2.0ms
        hrv_std = np.random.uniform(0.5, 2.0)

        # 합성 PPG 신호 생성
        t, ppg = generate_dynamic_synthetic_ppg(
            duration_sec=duration_sec,
            fs=fs,
            base_hr=60,  # 기본 심박수는 phases에 의해 조정됨
            hrv_std=hrv_std,
            respiratory_rate=respiratory_rate,
            respiratory_amplitude=respiratory_amplitude,
            phases=phases
        )

        # 정규화된 PPG 신호 저장
        all_ppg_signals[i] = ppg

    # NumPy 파일로 저장
    np.save(os.path.join(output_dir, 'synthetic_ppg_signals.npy'), all_ppg_signals)
    print(f"Saved {num_signals} synthetic PPG signals to '{output_dir}/synthetic_ppg_signals.npy'")


def visualize_ppg_signal(ppg_signal, fs=100, signal_index=0, output_dir='synthetic_ppg_signals'):
    """
    생성된 PPG 신호를 시각화합니다.

    Args:
        ppg_signal (np.ndarray): 합성 PPG 신호.
        fs (int): 샘플링 주파수 (Hz).
        signal_index (int): 시각화할 신호의 인덱스.
        output_dir (str): PPG 신호가 저장된 디렉토리.

    Returns:
        None
    """
    t = np.linspace(0, len(ppg_signal) / fs, len(ppg_signal))
    plt.figure(figsize=(15, 5))
    plt.plot(t, ppg_signal, label=f'Synthetic PPG Signal #{signal_index}')
    plt.title(f'Synthetic PPG Signal #{signal_index}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_ppg_signals(output_dir='synthetic_ppg_signals',
                              num_signals=1000,
                              duration_sec=60, fs=100):
    """
    1분 길이의 랜덤한 심박 신호 1,000개를 생성하고 저장합니다.

    Args:
        output_dir (str): 생성된 PPG 신호를 저장할 디렉토리.
        num_signals (int): 생성할 PPG 신호의 개수.
        duration_sec (float): 각 신호의 지속 시간 (초).
        fs (int): 샘플링 주파수 (Hz).

    Returns:
        None
    """
    # 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 합성 PPG 신호 저장을 위한 배열
    all_ppg_signals = np.zeros((num_signals, int(duration_sec * fs)))

    # 파라미터 저장을 위한 리스트
    all_params = []

    for i in tqdm(range(num_signals), desc="Generating PPG Signals"):
        # 랜덤하게 시기별 파라미터 설정
        # 예시로 3개의 시기로 나눔
        phases = []
        phase_duration = duration_sec / 3
        for p in range(3):
            start_time = p * phase_duration
            end_time = (p + 1) * phase_duration
            # 심박수: 60~100 BPM
            hr = np.random.uniform(60, 100)
            # 혈압: 수축기 110~140, 이완기 70~90
            bp_systolic = np.random.uniform(110, 140)
            bp_diastolic = np.random.uniform(70, 90)
            # 동맥 탄력성: 'elastic', 'moderate', 'stiff'
            ae = np.random.choice(['elastic', 'moderate', 'stiff'])

            phase = {
                'start_time': start_time,
                'end_time': end_time,
                'hr': hr,
                'bp': (bp_systolic, bp_diastolic),
                'ae': ae
            }
            phases.append(phase)

        # 랜덤하게 호흡수 설정: 12~20 호흡/분
        respiratory_rate = np.random.uniform(12, 20)
        # 호흡 진폭 설정: 0.03~0.07
        respiratory_amplitude = np.random.uniform(0.03, 0.07)

        # HRV 설정: 표준편차 0.5~2.0ms
        hrv_std = np.random.uniform(0.5, 2.0)

        # 합성 PPG 신호 생성
        t, ppg, signal_params = generate_dynamic_synthetic_ppg(
            duration_sec=duration_sec,
            fs=fs,
            base_hr=60,  # 기본 심박수는 phases에 의해 조정됨
            hrv_std=hrv_std,
            respiratory_rate=respiratory_rate,
            respiratory_amplitude=respiratory_amplitude,
            phases=phases
        )

        # 정규화된 PPG 신호 저장
        all_ppg_signals[i] = ppg

        # 파라미터 저장
        params = {
            'signal_index': i,
            'respiratory_rate_bpm': respiratory_rate,
            'respiratory_amplitude': respiratory_amplitude,
            'hrv_std_ms': hrv_std
        }
        for phase_idx, phase in enumerate(phases):
            params[f'phase{phase_idx + 1}_hr_bpm'] = phase['hr']
            params[f'phase{phase_idx + 1}_bp_systolic'] = phase['bp'][0]
            params[f'phase{phase_idx + 1}_bp_diastolic'] = phase['bp'][1]
            params[f'phase{phase_idx + 1}_ae'] = phase['ae']

        all_params.append(params)

    # NumPy 파일로 저장
    np.save(os.path.join(output_dir, 'synthetic_ppg_signals.npy'), all_ppg_signals)
    print(f"Saved {num_signals} synthetic PPG signals to '{output_dir}/synthetic_ppg_signals.npy'")

    # 파라미터를 DataFrame으로 변환 후 CSV로 저장
    df_params = pd.DataFrame(all_params)
    df_params.to_csv(os.path.join(output_dir, 'synthetic_ppg_params.csv'), index=False)
    print(f"Saved PPG signal parameters to '{output_dir}/synthetic_ppg_params.csv'")


def visualize_ppg_signal(ppg_signal, fs=100, signal_index=0, output_dir='synthetic_ppg_signals'):
    """
    생성된 PPG 신호를 시각화합니다.

    Args:
        ppg_signal (np.ndarray): 합성 PPG 신호.
        fs (int): 샘플링 주파수 (Hz).
        signal_index (int): 시각화할 신호의 인덱스.
        output_dir (str): PPG 신호가 저장된 디렉토리.

    Returns:
        None
    """
    t = np.linspace(0, len(ppg_signal) / fs, len(ppg_signal))
    plt.figure(figsize=(15, 5))
    plt.plot(t, ppg_signal, label=f'Synthetic PPG Signal #{signal_index}')
    plt.title(f'Synthetic PPG Signal #{signal_index}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


# 실행 예시
if __name__ == "__main__":
    # 1. 1000개의 PPG 신호 생성 및 저장
    generate_ppg_signals(
        output_dir='synthetic_ppg_signals',
        num_signals=1000,
        duration_sec=60,
        fs=100
    )

    # 2. 생성된 PPG 신호 중 일부 시각화
    # 저장된 PPG 신호 불러오기
    ppg_data = np.load('synthetic_ppg_signals/synthetic_ppg_signals.npy')

    # 파라미터 불러오기
    df_params = pd.read_csv('synthetic_ppg_signals/synthetic_ppg_params.csv')

    # 예시로 첫 번째 신호 시각화
    visualize_ppg_signal(ppg_data[0], fs=100, signal_index=0, output_dir='synthetic_ppg_signals')

    # 예시로 몇 개의 신호 시각화
    for idx in range(5):
        visualize_ppg_signal(ppg_data[idx], fs=100, signal_index=idx, output_dir='synthetic_ppg_signals')