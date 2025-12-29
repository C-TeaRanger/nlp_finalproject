import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_attention(input_sentence, output_words, attentions, save_path=None):
    """
    绘制 Attention 权重热力图
    
    参数:
    input_sentence: 源句子字符串 (例如 "i love you") 或 token 列表
    output_words: 生成的单词列表 (例如 ["wo", "ai", "ni", "<eos>"])
    attentions: Attention 矩阵，形状应为 (output_len, input_len)
                类型通常为 Tensor 或 Numpy Array
    save_path: 如果不为None，则保存图片到该路径
    """
    
    # 1. 数据转换：确保 attentions 是 numpy 数组
    if hasattr(attentions, 'cpu'):
        attentions = attentions.cpu().numpy()
    
    # 2. 创建画布
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # 3. 绘制热力图 (使用 'viridis' 或 'bone' 色系)
    cax = ax.matshow(attentions, cmap='viridis')
    fig.colorbar(cax)

    # 4. 设置坐标轴标签
    # 注意：matplotlib 的 tick 从 0 开始，所以我们需要偏移一点或者手动设置
    
    # 处理输入句子：如果是字符串则分割，如果是列表则直接用
    if isinstance(input_sentence, str):
        input_list = input_sentence.strip().split(' ')
    else:
        input_list = input_sentence

    # 加上 <EOS> 标记以便对齐
    input_list = input_list + ['<EOS>']
    
    ax.set_xticklabels([''] + input_list, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # 强制在每个刻度显示标签
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    
    if save_path:
        plt.savefig(save_path)