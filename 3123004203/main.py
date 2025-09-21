import os
import sys
import re
import jieba
from collections import Counter
from Levenshtein import distance as levenshtein_distance  # 用于计算编辑距离
from line_profiler_pycharm import profile

@profile
def read_file(file_path):
    """读取指定路径的文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")

@profile
def preprocess_text(text):
    """文本预处理：去除标点、多余空白和长文本中的常见停用词"""
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 去除多余的空白符
    text = re.sub(r'\s+', ' ', text).strip()
    # 长文本去除常见停用词（不计入查重）
    if len(text) > 800:
        text = re.sub(r'[的了是很我有和也吧啊你他她]', '', text)
    return text

@profile
def segment_text(text):
    """使用Jieba对文本进行分词处理"""
    words = jieba.lcut(text, cut_all=True)
    # 过滤空字符串
    processed_words = [word for word in words if word.strip()]
    return processed_words

@profile
def calculate_term_frequency(words):
    """计算词频统计"""
    return Counter(words)

@profile
def text_to_vector(words, vocabulary):
    """将分词结果转换为基于词汇表的词频向量"""
    vector = [0] * len(vocabulary)
    word_count = calculate_term_frequency(words)

    for word, count in word_count.items():
        if word in vocabulary:
            idx = vocabulary.index(word)
            vector[idx] = count
    return vector

@profile
def calculate_edit_distance_similarity(text1, text2):
    """计算编辑距离相似度（更适合短文本）"""
    edit_distance = levenshtein_distance(text1, text2)
    max_length = max(len(text1), len(text2))
    if max_length == 0:
        return 1.0
    return 1 - (edit_distance / max_length)

@profile
def calculate_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    vec1 = [float(i) for i in vec1]
    vec2 = [float(i) for i in vec2]

    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude1 = sum(x **2 for x in vec1)** 0.5
    magnitude2 = sum(x **2 for x in vec2)** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)

@profile
def calculate_similarity(file_path1, file_path2, cosine_weight=0.7, edit_distance_weight=0.3):
    """综合计算两个文本文件的相似度（余弦相似度+编辑距离相似度加权）"""
    # 读取文件内容
    raw_text1 = read_file(file_path1)
    raw_text2 = read_file(file_path2)

    # 文本预处理
    processed_text1 = preprocess_text(raw_text1)
    processed_text2 = preprocess_text(raw_text2)

    if not processed_text1 or not processed_text2:
        raise ValueError("错误：存在空文件，无法进行查重！请检查输入文件。")

    # 文本分词
    text1_words = segment_text(processed_text1)
    text2_words = segment_text(processed_text2)

    # 构建联合词汇表
    vocabulary = list(set(text1_words + text2_words))
    
    # 转换为词频向量
    vector1 = text_to_vector(text1_words, vocabulary)
    vector2 = text_to_vector(text2_words, vocabulary)

    # 计算余弦相似度
    cosine_similarity = calculate_cosine_similarity(vector1, vector2)
    print(f"余弦相似度：{cosine_similarity:.4f}")
    
    # 计算编辑距离相似度
    edit_similarity = calculate_edit_distance_similarity(processed_text1, processed_text2)
    print(f"编辑距离相似度：{edit_similarity:.4f}")
    
    # 加权计算最终相似度
    final_similarity = (cosine_weight * cosine_similarity) + (edit_distance_weight * edit_similarity)
    return final_similarity

@profile
def main():
    """主函数：处理命令行参数并执行查重计算"""
    if len(sys.argv) != 4:
        print("参数输入错误！正确用法: python text_plagiarism_checker.py <源文件路径> <待检测文件路径> <结果输出文件路径>")
        sys.exit(1)

    # 解析命令行参数（文件路径）
    source_file_path = sys.argv[1]
    check_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    # 计算相似度
    try:
        similarity_score = calculate_similarity(
            source_file_path, 
            check_file_path,
            cosine_weight=0.7, 
            edit_distance_weight=0.3
        )
        print(f"文本查重结果：{similarity_score:.2f}")
        
        # 将结果写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(f"源文件: {source_file_path}\n")
            output_file.write(f"待检测文件: {check_file_path}\n")
            output_file.write(f"查重相似度: {similarity_score:.4f}\n")
    except Exception as e:
        print(f"处理过程出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 示例用法（注释部分）
    # source = "../examples/orig.txt"
    # target = "../examples/orig_0.8_dis_10.txt"
    # score = calculate_similarity(source, target)
    # print(f"文本查重结果：{score:.2f}")
    main()
