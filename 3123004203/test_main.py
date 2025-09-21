import unittest
import os
import tempfile
from plagiarism_checker import (  # 假设原代码文件名为plagiarism_checker.py
    read_file,
    preprocess_text,
    segment_text,
    calculate_term_frequency,
    text_to_vector,
    calculate_edit_distance_similarity,
    calculate_cosine_similarity,
    calculate_similarity
)

class TestPlagiarismChecker(unittest.TestCase):
    """文本查重工具的单元测试类"""
    
    # 测试用例1：测试正常读取文件
    def test_read_file_success(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            f.write("测试文件内容")
            file_path = f.name
        
        content = read_file(file_path)
        self.assertEqual(content, "测试文件内容")
        os.unlink(file_path)
    
    # 测试用例2：测试读取不存在的文件
    def test_read_file_not_found(self):
        non_existent_path = "non_existent_file.txt"
        with self.assertRaises(FileNotFoundError):
            read_file(non_existent_path)
    
    # 测试用例3：测试文本预处理（标点和空白符处理）
    def test_preprocess_text_basic(self):
        text = "  这是一段测试文本！包含，标点符号。 还有 多余的空格  \n"
        processed = preprocess_text(text)
        self.assertEqual(processed, "这是一段测试文本包含标点符号还有多余的空格")
    
    # 测试用例4：测试长文本的停用词去除
    def test_preprocess_text_stopwords(self):
        # 长文本（超过800字符）应该去除停用词
        long_text = "的" * 801  # 构造超过800字符的文本
        processed = preprocess_text(long_text)
        self.assertEqual(processed, "")  # 所有内容都是停用词，处理后应为空
        
        # 短文本不应去除停用词
        short_text = "这是我的测试文本，很简单的"
        processed_short = preprocess_text(short_text)
        self.assertEqual(processed_short, "这是我的测试文本很简单的")
    
    # 测试用例5：测试文本分词功能
    def test_segment_text(self):
        text = "这是一段测试文本"
        words = segment_text(text)
        # 检查分词结果不为空
        self.assertTrue(len(words) > 0)
        # 检查分词结果包含预期词语
        self.assertIn("测试", words)
        self.assertIn("文本", words)
    
    # 测试用例6：测试空文本分词
    def test_segment_empty_text(self):
        words = segment_text("")
        self.assertEqual(words, [])
    
    # 测试用例7：测试词频计算
    def test_calculate_term_frequency(self):
        words = ["测试", "文本", "测试", "单元"]
        freq = calculate_term_frequency(words)
        self.assertEqual(freq["测试"], 2)
        self.assertEqual(freq["文本"], 1)
        self.assertEqual(freq["单元"], 1)
        self.assertEqual(freq.get("不存在", 0), 0)
    
    # 测试用例8：测试向量生成
    def test_text_to_vector(self):
        words = ["测试", "文本", "测试"]
        vocabulary = ["测试", "文本", "单元"]
        vector = text_to_vector(words, vocabulary)
        self.assertEqual(vector, [2, 1, 0])  # 测试:2次，文本:1次，单元:0次
    
    # 测试用例9：测试编辑距离相似度
    def test_edit_distance_similarity(self):
        # 相同文本
        self.assertEqual(calculate_edit_distance_similarity("测试文本", "测试文本"), 1.0)
        # 完全不同的文本
        self.assertAlmostEqual(
            calculate_edit_distance_similarity("abc", "def"), 
            0.0, 
            places=4
        )
        # 部分相似的文本
        self.assertAlmostEqual(
            calculate_edit_distance_similarity("测试文本", "测试文档"), 
            0.75, 
            places=4
        )
        # 空文本
        self.assertEqual(calculate_edit_distance_similarity("", ""), 1.0)
        self.assertEqual(calculate_edit_distance_similarity("", "测试"), 0.0)
    
    # 测试用例10：测试余弦相似度
    def test_cosine_similarity(self):
        # 相同向量
        self.assertEqual(calculate_cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)
        # 正交向量
        self.assertEqual(calculate_cosine_similarity([1, 0], [0, 1]), 0.0)
        # 部分相似向量
        self.assertAlmostEqual(
            calculate_cosine_similarity([1, 1, 0], [1, 2, 0]), 
            0.94868, 
            places=4
        )
        # 零向量
        self.assertEqual(calculate_cosine_similarity([0, 0], [1, 2]), 0.0)
    
    # 测试用例11：测试综合相似度计算
    def test_calculate_similarity(self):
        # 创建两个临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f1, \
             tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f2:
            
            f1.write("这是一段用于测试的文本内容")
            f2.write("这是一段用于测试的文本内容")  # 与f1内容相同
            path1, path2 = f1.name, f2.name
        
        # 相同文本的相似度应该接近1.0
        similarity = calculate_similarity(path1, path2)
        self.assertGreater(similarity, 0.95)
        
        # 修改第二个文件内容
        with open(path2, 'w', encoding='utf-8') as f2:
            f2.write("这是另一段完全不同的内容，与原始文本没有任何相似之处")
        
        # 不同文本的相似度应该较低
        similarity = calculate_similarity(path1, path2)
        self.assertLess(similarity, 0.3)
        
        # 清理临时文件
        os.unlink(path1)
        os.unlink(path2)
    
    # 测试用例12：测试空文件处理
    def test_empty_file_handling(self):
        # 创建一个正常文件和一个空文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f1, \
             tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f2:
            
            f1.write("正常的文本内容")
            # f2保持为空
            path1, path2 = f1.name, f2.name
        
        # 空文件应该抛出异常
        with self.assertRaises(ValueError):
            calculate_similarity(path1, path2)
        
        # 清理临时文件
        os.unlink(path1)
        os.unlink(path2)

if __name__ == '__main__':
    unittest.main()
    