import numpy as np
import scipy.signal
import librosa

class NoiseReducer:
    def __init__(self, n_grad_freq=2, n_grad_time=4, n_fft=512, win_length=512, hop_length=512 // 4, n_std_thresh=0.5, prop_decrease=0.8):
        """
        Initialize the NoiseReducer class with parameters for noise reduction.

        Parameters:
        n_grad_freq (int): Number of frequency channels to smooth over with the mask. Default is 2.
        n_grad_time (int): Number of time channels to smooth over with the mask. Default is 4.
        n_fft (int): Number of audio samples between STFT columns. Default is 512.
        win_length (int): Length of each frame of audio. Default is 512.
        hop_length (int): Number of audio samples between STFT columns. Default is 128 (512 // 4).
        n_std_thresh (float): How many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal. Default is 0.5.
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none). Default is 0.8.

        Returns:
        None
        """
        self.n_grad_freq = n_grad_freq
        self.n_grad_time = n_grad_time
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_std_thresh = n_std_thresh
        self.prop_decrease = prop_decrease

    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def _istft(self, y):
        return librosa.istft(y, hop_length=self.hop_length, win_length=self.win_length)

    def _amp_to_db(self, x):
        return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

    def _db_to_amp(self, x):
        return librosa.core.db_to_amplitude(x, ref=1.0)

    def remove_noise(self, audio_clip, noise_clip):
        noise_stft = self._stft(noise_clip)
        noise_stft_db = self._amp_to_db(np.abs(noise_stft))

        mean_freq_noise = np.mean(noise_stft_db, axis=1)
        std_freq_noise = np.std(noise_stft_db, axis=1)
        noise_thresh = mean_freq_noise + std_freq_noise * self.n_std_thresh

        sig_stft = self._stft(audio_clip)
        sig_stft_db = self._amp_to_db(np.abs(sig_stft))

        mask_gain_dB = np.min(self._amp_to_db(np.abs(sig_stft)))

        smoothing_filter = np.outer(
            np.concatenate(
                [np.linspace(0, 1, self.n_grad_freq + 1, endpoint=False), np.linspace(1, 0, self.n_grad_freq + 2)]
            )[1:-1],
            np.concatenate(
                [np.linspace(0, 1, self.n_grad_time + 1, endpoint=False), np.linspace(1, 0, self.n_grad_time + 2)]
            )[1:-1],
        )
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

        db_thresh = np.repeat(
            np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
            np.shape(sig_stft_db)[1],
            axis=0,
        ).T

        sig_mask = sig_stft_db < db_thresh
        sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
        sig_mask = sig_mask * self.prop_decrease

        sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
        )

        sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
        sig_stft_amp = (self._db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)

        recovered_signal = self._istft(sig_stft_amp)
        return recovered_signal