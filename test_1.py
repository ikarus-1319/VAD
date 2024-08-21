import numpy as np
import scipy.signal as signal
import soundfile as sf

# Hàm lọc tần số
def frequency_filter(data, sample_rate, cutoff_freq, filter_type='low'):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    if filter_type == 'low':
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'high':
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    elif filter_type == 'band':
        b, a = signal.butter(4, normal_cutoff, btype='bandpass', analog=False)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

# Hàm lọc trung bình động
def moving_average_filter(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Đọc file âm thanh
input_file = '/home/pc-trunghieu-20/code/silero-vad/audio.wav'
output_file = '/home/pc-trunghieu-20/code/silero-vad/handle_audio.wav'
data, sample_rate = sf.read(input_file)

# Áp dụng lọc tần số (Low-pass filter với tần số cắt 1000 Hz)
cutoff_freq = 1000  # Tần số cắt (Hz)
filtered_data = frequency_filter(data, sample_rate, cutoff_freq, filter_type='low')

# Áp dụng lọc trung bình động (với kích thước cửa sổ là 5)
window_size = 5
smoothed_data = moving_average_filter(filtered_data, window_size)

# Lưu file âm thanh đã được xử lý
sf.write(output_file, smoothed_data, sample_rate)

print(f"File âm thanh đã được lọc tiếng ồn và lưu tại: {output_file}")
