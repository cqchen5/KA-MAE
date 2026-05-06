# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# AST: https://github.com/YuanGongND/ast
# --------------------------------------------------------
import csv, os, sys
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
import torch.distributed as dist
import random
import math

class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
        
class DistributedWeightedSampler(Sampler):
    #dataset_train, samples_weight,  num_replicas=num_tasks, rank=global_rank
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.weights = torch.from_numpy(weights)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # targets = self.dataset.targets
        # # select only the wanted targets for this subsample
        # targets = torch.tensor(targets)[indices]
        # assert len(targets) == self.num_samples
        # # randomly sample this subset, producing balanced classes
        # weights = self.calculate_weights(targets)
        weights = self.weights[indices]

        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]
        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, use_fbank=False, fbank_dir=None, roll_mag_aug=False, load_video=False, mode='train'):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        # with open(dataset_json_file, 'r') as fp:
        #     data_json = json.load(fp)
        self._load_txt_file(self.datapath)
         # 创建情感标签映射
        self.emotion_labels = sorted(list(set([item for item in self.label])))
        self.label_num = len(self.emotion_labels)
        print("总共有{}个情感标签".format(self.label_num))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.emotion_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        # self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        if 'multilabel' in self.audio_conf.keys():
            self.multilabel = self.audio_conf['multilabel']
        else:
            self.multilabel = False
        print(f'multilabel: {self.multilabel}')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        # self.norm_mean = self.audio_conf.get('mean')
        # self.norm_std = self.audio_conf.get('std')
        # print('Dataset: {}, mean {:.3f} and std {:.3f}'.format(self.dataset, self.norm_mean, self.norm_std))
        print('Dataset: {}'.format(self.dataset))
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')
        # self.index_dict = make_index_dict(label_csv)
        # self.label_num = len(self.index_dict)
        self.roll_mag_aug=roll_mag_aug
        # print(f'number of classes: {self.label_num}')
        print(f'size of dataset {self.__len__()}')

        self.target_length = self.audio_conf.get('target_length')
        self.shift_size = 160
        self.win_length = 512
        self.fft_size = 512
        self.pad_size = int(self.win_length / self.shift_size) + 2 # 计算窗长内分帧个数
        if self.pad_size % 2 != 0:
            self.pad_size = self.pad_size + 1
            assert self.pad_size % 2 == 0

        self.window = getattr(torch, "hann_window")(self.win_length)
        self.center = True
        self.onesided = True


    def _load_txt_file(self, txt_file):
        """加载txt文件，返回[(wav_path, emotion)]列表"""
        self.data = []
        self.label = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    wav_path, emotion = line.split('\t', 1)
                    if os.path.exists(wav_path):
                        self.data.append(wav_path)
                        self.label.append(emotion)
                    else:
                        print(f"警告: 文件不存在 {wav_path}")
    def _roll_mag_aug(self, waveform):
        waveform=waveform.numpy()
        idx=np.random.randint(len(waveform))
        rolled_waveform=np.roll(waveform,idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform*mag)
    
    # def stft(self, x, fft_size, hop_size, win_length, window, center, onesided):
    #     """Perform STFT and convert to magnitude spectrogram.
    #     Args:
    #         x (Tensor): Input signal tensor (B, T).
    #         fft_size (int): FFT size.
    #         hop_size (int): Hop size.
    #         win_length (int): Window length.
    #         window (str): Window function type.
    #     Returns:
    #         Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    #     """
    #     x_stft = torch.stft(x,
    #                         fft_size,
    #                         hop_size,
    #                         win_length,
    #                         window,
    #                         center=center,
    #                         onesided=onesided,
    #                         return_complex=True)
    #     # real = x_stft.real
    #     # imag = x_stft.imag

    #     # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    #     # return real, imag, x_stft
    #     return x_stft
    def complex_nan_to_zero(self, x: torch.Tensor) -> torch.Tensor:
        """
        将复数 tensor 中的 NaN (实部或虚部) 逐个替换为 0。
        保留原 device 和 dtype（complex64/complex128）。
        """
        assert x.is_complex(), "输入必须是 complex dtype"
        real = x.real
        imag = x.imag
        # 如果当前 torch 支持 complex nan_to_num，这里也没问题；即便不支持也能工作
        real = torch.nan_to_num(real, nan=0.0) if hasattr(torch, "nan_to_num") else torch.where(torch.isnan(real), torch.zeros_like(real), real)
        imag = torch.nan_to_num(imag, nan=0.0) if hasattr(torch, "nan_to_num") else torch.where(torch.isnan(imag), torch.zeros_like(imag), imag)
        return torch.complex(real, imag)
    def stft(self, x, fft_size, hop_size, win_length, window, center, onesided):
        """Perform STFT and convert to magnitude spectrogram.
        Args:
            x (Tensor): Input signal tensor (B, T).
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (str): Window function type.
        Returns:
            Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
        """
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, center=center, onesided=onesided, return_complex=True)
        if torch.isnan(x_stft).any():
            x_stft = self.complex_nan_to_zero(x_stft)
            print("x_stft has nan!!!")
        # breakpoint()
        # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
        # return real, imag, x_stft
        return x_stft
    
    def compute_time_length(self, target_length_cur):
        # calculate method
        # total_frames = (T + 2 * (n_fft // 2) - win_length) // hop_length + 1
        # total_frames = (T - win_length) // hop_length + 1
        if self.center:
            y = (target_length_cur - 1) * self.shift_size + self.win_length - 2 * (self.fft_size // 2)
            # y = (target_length_cur - 1) * self.shift_size
            # y = y + self.fft_size - 2 * (self.fft_size // 2)
        else:
            assert False
            y = (target_length_cur - 1) * self.shift_size
            y = y + self.win_length
        
        return y
    def center_crop_data(self, stft_data, audio_data):
        assert stft_data.shape[-1] == (self.target_length + 2*self.pad_size)
        multi_channel_audios_stft = stft_data[..., self.pad_size:-self.pad_size] # center crop
        assert multi_channel_audios_stft.shape[-1] == self.target_length

        audio_length = self.compute_time_length(self.target_length)
        assert ((audio_data.shape[-1] - audio_length) >= 0) and ((audio_data.shape[-1] - audio_length) % 2 == 0)

        audio_pad = int((audio_data.shape[-1] - audio_length) / 2)
        assert audio_pad == (self.pad_size * self.shift_size)
        audio_data = audio_data[..., audio_pad:-audio_pad] # center crop
        assert self.compute_time_length(self.target_length) == audio_data.shape[-1]

        padding = torch.zeros(multi_channel_audios_stft.shape, dtype=torch.bool)
        
        return multi_channel_audios_stft, audio_data, padding
    
    def random_crop_audio_length(self, x):
        time_var = self.compute_time_length(self.target_length + 2*self.pad_size)
        p = time_var - x.shape[-1]
        # breakpoint()
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, p, 0, 0))
            x = m(x)
        elif p < 0:
            start = random.randint(0, x.shape[-1] - time_var)
            x = x[:, start:(start + time_var)]
        
        return x

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            # breakpoint()
            waveform, sr = torchaudio.load(filename)
            # waveform = waveform.mean(dim=0, keepdim=True)   #多声道
            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        # 498 128, 998, 128
        multi_channel_audios = self.random_crop_audio_length(waveform)
        
        multi_channel_audios_stft = self.stft(multi_channel_audios,
                                              self.fft_size,
                                              self.shift_size,
                                              self.win_length,
                                              self.window,
                                              self.center,
                                              onesided=self.onesided)

        multi_channel_audios_stft, multi_channel_audios_crop, padding = self.center_crop_data(multi_channel_audios_stft,
                                                                                         multi_channel_audios)
        # torchaudio.save("/train20/sppro/permanent/cqchen5/AudioMAE/AudioMAE-TAP/waveform.wav", multi_channel_audios_crop, 16000)
        # print(filename)
        # breakpoint()
        multi_channel_audios_stft = multi_channel_audios_stft[:, 1:, :]
        dc_component = multi_channel_audios_stft[:, :1, :]

        multi_channel_audios_mag = torch.abs(multi_channel_audios_stft)

        # 防止取log后出现Nan
        multi_channel_audios_mag = torch.clamp(multi_channel_audios_mag, min=1e-12).transpose(2, 1)
        multi_channel_audios_mag = torch.log(multi_channel_audios_mag)

        multi_channel_audios_phase = torch.angle(multi_channel_audios_stft)

        if filename2 == None:
            return multi_channel_audios_mag.squeeze(0), 0, multi_channel_audios_crop.squeeze(0)
        else:
            return multi_channel_audios_mag.squeeze(0), mix_lambda, multi_channel_audios_crop.squeeze(0)


    def _fbank(self, filename, filename2=None):
        if filename2 == None:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fbank = np.load(fn1)
            return torch.from_numpy(fbank), 0
        else:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fn2 = os.path.join(self.fbank_dir, os.path.basename(filename2).replace('.wav','.npy'))
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            fbank = mix_lambda * np.load(fn1) + (1-mix_lambda) * np.load(fn2)  
            return torch.from_numpy(fbank), mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup: # for audio_exp, when using mixup, assume multilabel
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]

            # get the mixed fbank
            if not self.use_fbank:
                fbank, mix_lambda, audio = self._wav2fbank(datum, mix_datum)
            else:
                fbank, mix_lambda, audio = self._fbank(datum, mix_datum)
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # # add sample 1 labels
            # for label_str in datum['labels'].split(','):
            label_indices[int(self.label_to_idx[self.label[index]])] += mix_lambda
            # add sample 2 labels
            # for label_str in mix_datum['labels'].split(','):
            label_indices[int(self.label_to_idx[self.label[mix_sample_idx]])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            if not self.use_fbank:
                fbank, mix_lambda, audio = self._wav2fbank(datum)
            else:
                fbank, mix_lambda, audio = self._fbank(datum)
            # for label_str in datum['labels'].split(','):
            label_indices[int(self.label_to_idx[self.label[index]])] = 1.0

            if self.multilabel:
                label_indices = torch.FloatTensor(label_indices)
            else:
                # remark : for ft cross-ent
                label_indices = int(self.label_to_idx[self.label[index]])
                # print("label_indices:", label_indices)
        # SpecAug for training (not for eval)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank) # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq
        # fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        if self.noise == True: # default is false, true for spc
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # return fbank.unsqueeze(0), label_indices, datum
        # breakpoint()
        # print("label_indices", label_indices)
        # breakpoint()
        return fbank.unsqueeze(0), label_indices, audio

    def __len__(self):
        return len(self.data)


