import librosa
import numpy as np
from scipy.signal import lfilter, get_window
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def func_format(x, pos):
    return "%d" % (1000 * x)

class RhythmFeatures:
    """韵律学特征"""
    def __init__(self, wave_data, sr=None, frame_len=512, n_fft=None, win_step=2 / 3, window="hamming"):
        """
        初始化
        :param input_file: 输入音频文件
        :param sr: 所输入音频文件的采样率，默认为None
        :param frame_len: 帧长，默认512个采样点(32ms,16kHz),与窗长相同
        :param n_fft: FFT窗口的长度，默认与窗长相同
        :param win_step: 窗移，默认移动2/3，512*2/3=341个采样点(21ms,16kHz)
        :param window: 窗类型，默认汉明窗
        """
        self.wave_data = wave_data
        self.sr = sr

        self.frame_len = frame_len  # 帧长，单位采样点数
        # self.wave_data, self.sr = librosa.load(self.input_file, sr=sr)
        self.window_len = frame_len  # 窗长512
        if n_fft is None:
            self.fft_num = self.window_len  # 设置NFFT点数与窗长相等
        else:
            self.fft_num = n_fft
        self.win_step = win_step
        self.hop_length = 160 #round(self.window_len * win_step)  # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
        self.window = window

    def energy(self):
        """
        每帧内所有采样点的幅值平方和作为能量值
        :return: 每帧能量值，np.ndarray[shape=(1，n_frames), dtype=float64]
        """
        mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.fft_num, hop_length=self.hop_length,
                                       win_length=self.frame_len, window=self.window, center=True))
        # breakpoint()
        pow_spec = np.square(mag_spec)
        energy = np.sum(pow_spec, axis=0)
        energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # 避免能量值为0，防止后续取log出错(eps是取非负的最小值)
        return energy


class QualityFeatures:
    """声音质量特征（音质）"""

    def __init__(self, sr=16000, frame_len=512, n_fft=None, win_step=2 / 3, window="hamming"):
        """
        初始化
        :param input_file: 输入音频文件
        :param sr: 所输入音频文件的采样率，默认为None
        :param frame_len: 帧长，默认512个采样点(32ms,16kHz),与窗长相同
        :param n_fft: FFT窗口的长度，默认与窗长相同
        :param win_step: 窗移，默认移动2/3，512*2/3=341个采样点(21ms,16kHz)
        :param window: 窗类型，默认汉明窗
        """
        # self.input_file = input_file
        self.sr = sr

        self.frame_len = frame_len  # 帧长，单位采样点数
        # self.wave_data, self.sr = librosa.load(self.input_file, sr=sr)
        self.n_fft = n_fft
        self.window_len = frame_len  # 窗长512
        self.win_step = win_step
        # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
        self.hop_length = 160 #round(self.window_len * win_step)
        self.window = window

    def formant(self, wave_data_np, ts_e=0.01, ts_f_d=200, ts_b_u=2000):
        """
        LPC求根法估计每帧前三个共振峰的中心频率及其带宽
        :param ts_e: 能量阈值：默认当能量超过0.01时认为可能会出现共振峰
        :param ts_f_d: 共振峰中心频率下阈值：默认当中心频率超过200，小于采样频率一半时认为可能会出现共振峰
        :param ts_b_u: 共振峰带宽上阈值：默认低于2000时认为可能会出现共振峰
        :return: F1/F2/F3、B1/B2/B3,每一列为一帧 F1/F2/F3或 B1/B2/B3，np.ndarray[shape=(3, n_frames), dtype=float64]
        """
        # breakpoint()
        wave_data = wave_data_np
        _data = lfilter([1., 0.83], [1], wave_data)  # 预加重0.83：高通滤波器
        inc_frame = self.hop_length  # 窗移
        # n_frame = int(np.ceil(len(_data) / inc_frame))  # 分帧数
        # n_pad = n_frame * self.window_len - len(_data)  # 末端补零数
        # breakpoint()
        # 使用与STFT相同的分帧逻辑
        _data = np.pad(_data, (self.window_len//2, self.window_len//2), mode='constant')
        n_frame = 1 + (len(_data) - self.window_len) // inc_frame
        n_pad = (n_frame-1) * inc_frame + self.window_len - len(_data)

        _data = np.append(_data, np.zeros(n_pad))  # 无法整除则末端补零
        win = get_window(self.window, self.window_len, fftbins=False)  # 获取窗函数
        formant_frq = []  # 所有帧组成的第1/2/3共振峰中心频率
        formant_bw = []  # 所有帧组成的第1/2/3共振峰带宽
        rym = RhythmFeatures(wave_data, self.sr,
                             self.frame_len, self.n_fft, self.win_step, self.window)
        e = rym.energy()  # 获取每帧能量值
        e = e / np.max(e)  # 归一化
        for i in range(n_frame):
            f_i = _data[i * inc_frame:i * inc_frame + self.window_len]  # 分帧
            if np.all(f_i == 0):  # 避免上面的末端补零导致值全为0，防止后续求LPC线性预测误差系数出错(eps是取非负的最小值)
                f_i[0] = np.finfo(np.float64).eps
            f_i_win = f_i * win  # 加窗
            # 获取LPC线性预测误差系数，即滤波器分母多项式，阶数为 预期共振峰数3 *2+2，即想要得到F1-3
            a = librosa.lpc(f_i_win, order=8)
            rts = np.roots(a)  # 求LPC返回的预测多项式的根,为共轭复数对
            # 只保留共轭复数对一半，即虚数部分为+或-的根
            rts = np.array([r for r in rts if np.imag(r) >= 0])
            rts = np.where(rts == 0, np.finfo(np.float64).eps,
                           rts)  # 避免值为0，防止后续取log出错(eps是取非负的最小值)
            ang = np.arctan2(np.imag(rts), np.real(rts))  # 确定根对应的角(相位）
            # F(i) = ang(i)/(2*pi*T) = ang(i)*f/(2*pi)
            # 将以角度表示的rad/sample中的角频率转换为赫兹sample/s
            frq = ang * (self.sr / (2 * np.pi))
            indices = np.argsort(frq)  # 获取frq从小到大排序索引
            frequencies = frq[indices]  # frq从小到大排序
            # 共振峰的带宽由预测多项式零点到单位圆的距离表示: B(i) = -ln(r(i))/(pi*T) = -ln(abs(rts[i]))*f/pi
            bandwidths = -(self.sr / np.pi) * np.log(np.abs(rts[indices]))
            formant_f = []  # F1/F2/F3
            formant_b = []  # B1/B2/B3
            if e[i] > ts_e:  # 当能量超过ts_e时认为可能会出现共振峰
                # 采用共振峰频率大于ts_f_d小于self.sr/2赫兹，带宽小于ts_b_u赫兹的标准来确定共振峰
                for j in range(len(frequencies)):
                    if (ts_f_d < frequencies[j] < self.sr/2) and (bandwidths[j] < ts_b_u):
                        formant_f.append(frequencies[j])
                        formant_b.append(bandwidths[j])
                # 只取前三个共振峰
                if len(formant_f) < 3:  # 小于3个，则补nan
                    formant_f += ([np.nan] * (3 - len(formant_f)))
                else:  # 否则只取前三个
                    formant_f = formant_f[0:3]
                formant_frq.append(np.array(formant_f))  # 加入帧列表
                if len(formant_b) < 3:
                    formant_b += ([np.nan] * (3 - len(formant_b)))
                else:
                    formant_b = formant_b[0:3]
                formant_bw.append(np.array(formant_b))
            else:  # 能量过小，认为不会出现共振峰，此时赋值为nan
                formant_frq.append(np.array([np.nan, np.nan, np.nan]))
                formant_bw.append(np.array([np.nan, np.nan, np.nan]))
        formant_frq = np.array(formant_frq).T
        formant_bw = np.array(formant_bw).T
        # print(formant_frq.shape, np.nanmean(formant_frq, axis=1))
        # print(formant_bw.shape, np.nanmean(formant_bw, axis=1))
        return formant_frq, formant_bw

def create_formant_bandwidth_mask(fmt_frq, fmt_bw, sr, n_fft):
    """
    创建考虑带宽的共振峰掩码矩阵
    
    参数:
        magnitude_spectrogram: 幅度谱 [256, 1024]
        fmt_frq: 共振峰频率 [3, 1024]
        fmt_bw: 共振峰带宽 [3, 1024]
        sr: 采样率
        n_fft: FFT点数
    
    返回:
        formant_mask: 二值矩阵 [256, 1024]
    """
    formant_mask = np.zeros((256, 1024), dtype=np.int32)
    freq_resolution = sr / n_fft
    
    for t in range(fmt_frq.shape[1]):
        for f_idx in range(3):
            freq = fmt_frq[f_idx, t]
            bandwidth = fmt_bw[f_idx, t]
            
            if np.isnan(freq) or np.isnan(bandwidth):
                continue
            
            # 中心频率对应的bin
            center_bin = int(round(freq / freq_resolution)) - 1
            
            # 带宽对应的bin范围
            bw_bins = int(round(bandwidth / freq_resolution))
            lower_bin = max(0, center_bin - bw_bins // 2)
            upper_bin = min(256 - 1, center_bin + bw_bins // 2)
            
            # 标记整个带宽范围内的bin
            formant_mask[lower_bin:upper_bin+1, t] = 1
    
    return formant_mask

if __name__ == '__main__':

    wave_data_np, sr = librosa.load("/home3/sppro/cqchen5/MAE-emovec2-d24-TAP-formant/example_wav/ouput.wav", sr=16000)
    quality_features = QualityFeatures()

    fmt_frq, fmt_bw = quality_features.formant(wave_data_np)  # 3个共振峰中心频率及其带宽
    formant_mask = create_formant_bandwidth_mask(fmt_frq, fmt_bw, 16000, 512)
    print(f"formant_mask.shape:{formant_mask.shape}")
    print(f"formant_mask:{formant_mask.sum().sum()}")
