import torch
import torch.nn.functional as F

# 提取共振峰 - 批次处理版本
def find_formants_batch(magnitude_batch, frequencies, top_k=3):
    """
    参数:
        magnitude_batch: 幅度谱张量 [B, F, T]
        frequencies: 频率数组 [F]
        top_k: 提取的峰值数量
    
    返回:
        formants: 共振峰频率 [B, top_k, T]
    """
    B, F, T = magnitude_batch.shape
    
    # 在频率维度上找到top_k个峰值
    values, indices = torch.topk(magnitude_batch, top_k, dim=1)  # [B, top_k, T]
    
    # 将索引转换为频率值
    # 扩展frequencies以匹配批次维度
    freq_expanded = frequencies.unsqueeze(0).unsqueeze(-1).expand(B, F, T)  # [B, F, T]
    
    # 使用gather收集对应的频率值
    formants = torch.gather(freq_expanded, 1, indices)  # [B, top_k, T]
    
    return formants


# 更高效的向量化掩码创建版本
def create_formant_bandwidth_mask_vectorized(fmt_freq_batch, sr, n_fft):
    """
    向量化版本的共振峰掩码创建
    """
    B, top_k, T = fmt_freq_batch.shape
    F = n_fft // 2
    
    freq_resolution = sr / n_fft
    
    # 计算对应的bin索引
    bin_indices = torch.round(fmt_freq_batch / freq_resolution).long()  # [B, top_k, T]
    
    # 创建有效的索引掩码
    valid_mask = (bin_indices >= 0) & (bin_indices < F) & (~torch.isnan(fmt_freq_batch))
    
    # 初始化掩码
    formant_mask = torch.zeros((B, F, T), 
                              dtype=torch.int32, 
                              device=fmt_freq_batch.device)
    
    # # 使用scatter_填充掩码
    # for b in range(B):
    #     for k in range(top_k):
    #         # 获取当前共振峰的有效索引
    #         valid_indices = valid_mask[b, k, :]
    #         time_indices = torch.arange(T, device=fmt_freq_batch.device)[valid_indices]
    #         freq_indices = bin_indices[b, k, :][valid_indices]
            
    #         # 设置掩码
    #         formant_mask[b, freq_indices, time_indices] = 1
    
    # return formant_mask
    
    # 向量化方法：使用高级索引
    # 扩展维度以匹配所有可能的组合
    batch_indices = torch.arange(B, device=fmt_freq_batch.device).view(B, 1, 1).expand(B, top_k, T)
    topk_indices = torch.arange(top_k, device=fmt_freq_batch.device).view(1, top_k, 1).expand(B, top_k, T)
    time_indices = torch.arange(T, device=fmt_freq_batch.device).view(1, 1, T).expand(B, top_k, T)
    
    # 获取所有有效位置的索引
    valid_batch = batch_indices[valid_mask]
    valid_freq = bin_indices[valid_mask]
    valid_time = time_indices[valid_mask]
    
    # 一次性设置所有有效位置为1
    formant_mask[valid_batch, valid_freq, valid_time] = 1

    return formant_mask

# 主处理函数 - 批次处理版本
def process_batch_magnitude(magnitude_batch, sr=16000, n_fft=512, device='cuda'):
    """
    处理批次幅度谱张量
    
    参数:
        magnitude_batch: 幅度谱张量 [B, F, T]
        sr: 采样率
        n_fft: FFT窗口大小
        device: 计算设备
    
    返回:
        formant_mask: 共振峰掩码 [B, F, T]
        formants: 共振峰频率 [B, top_k, T]
    """
    # 确保张量在正确的设备上
    magnitude_batch = magnitude_batch.to(device)
    
    # 计算频率数组
    frequencies = torch.fft.fftfreq(n_fft, 1/sr)[1:n_fft//2 + 1].to(device)
    
    # 提取共振峰
    formants = find_formants_batch(magnitude_batch, frequencies, top_k=20)
    
    # 创建共振峰掩码
    formant_mask = create_formant_bandwidth_mask_vectorized(formants, sr, n_fft)
    
    return formant_mask, formants

# 使用示例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 模拟批次幅度谱数据 [B, F, T]
    batch_size = 4
    freq_bins = 256
    time_frames = 1024
    
    # 创建随机幅度谱数据
    magnitude_batch = torch.randn(batch_size, freq_bins, time_frames).abs().to(device)
    
    formant_mask, formants = process_batch_magnitude(magnitude_batch, device=device)
    # breakpoint()
    print(f"共振峰掩码形状: {formant_mask.shape}")