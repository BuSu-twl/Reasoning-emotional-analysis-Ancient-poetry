"""
改进版 baseline.py
使用 vLLM 部署的 Qwen2.5-7B-Instruct 模型进行古诗词分析
修复了原代码的问题并添加了最佳实践
"""
import os
import json
import re
import argparse
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoemAnalyzerLLM:
    """基于大语言模型的古诗词分析器"""
    
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_retries: int = 3
    ):
        """
        初始化分析器
        
        Args:
            api_base: OpenAI 兼容 API 的基础 URL
            api_key: API 密钥（本地服务通常为 "EMPTY"）
            max_retries: 最大重试次数
        """
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.max_retries = max_retries
        self.model = self._get_model_name()
        
    def _get_model_name(self) -> str:
        """获取可用的模型名称"""
        try:
            models = self.client.models.list()
            if models.data:
                model_name = models.data[0].id
                logger.info(f"使用模型: {model_name}")
                return model_name
            else:
                raise ValueError("未找到可用模型")
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            raise
    
    @staticmethod
    def _validate_input_data(data: Dict[str, Any]) -> bool:
        """
        验证输入数据格式
        
        Args:
            data: 输入数据字典
            
        Returns:
            验证通过返回 True，否则返回 False
        """
        required_fields = ['index', 'content', 'qa_words', 'qa_sents', 'choose']
        for field in required_fields:
            if field not in data:
                logger.error(f"缺少必要字段: {field}")
                return False
        return True
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中提取 JSON 格式的内容
        
        Args:
            text: 可能包含 JSON 的文本
            
        Returns:
            解析后的 JSON 字典，如果失败返回 None
        """
        # 方法1: 尝试直接解析整个文本
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # 方法2: 使用正则表达式提取 JSON 对象
        # 匹配最外层的大括号对（支持嵌套）
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 简单嵌套
            r'\{.*?\}',  # 非贪婪匹配
            r'```json\s*(\{.*?\})\s*```',  # 代码块中的 JSON
            r'```\s*(\{.*?\})\s*```',  # 普通代码块
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                json_str = match.group(1) if match.lastindex else match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        logger.warning("无法从文本中提取有效的 JSON")
        return None
    
    def _build_prompt(self, data: Dict[str, Any]) -> str:
        """
        构建发送给模型的 Prompt
        
        Args:
            data: 输入数据
            
        Returns:
            格式化的 Prompt 字符串
        """
        return f"""
你是一个古诗词专家，现在有一些古诗词需要你的帮助。
我会给你提供一个 JSON 数据，格式如下：
- **"index"**：古诗词的序号  
- **"title"**：古诗词的标题  
- **"author"**：古诗词的作者  
- **"content"**：古诗词的内容  
- **"qa_words"**：古诗词中需要翻译的词语  
- **"qa_sents"**：古诗词中需要提供白话文译文的句子  
- **"choose"**：一个包含多个选项的字典，每个选项代表该诗词可能表达的情感  

这是我的数据：{json.dumps(data, ensure_ascii=False, indent=2)}

### 你的任务：
请你根据提供的数据，生成如下 JSON 格式的结果：
- **"idx"**：古诗词的序号（与输入的 index 保持一致）
- **"ans_qa_words"**：对 "qa_words" 词语的含义进行解释，格式为字典
- **"ans_qa_sents"**：对 "qa_sents" 句子提供白话文译文，格式为字典
- **"choose_id"**：从 "choose" 选项中选择最符合该古诗词情感的标号，仅输出对应的字母（如 "A", "B" 等）

### **json输出格式示例：**

{{
    "idx": {data.get('index', '')},
    "ans_qa_words": {{
        "词语1": "词语1的含义",
        "词语2": "词语2的含义"
    }},
    "ans_qa_sents": {{
        "句子1": "句子1的白话文翻译",
        "句子2": "句子2的白话文翻译"
    }},
    "choose_id": "A"
}}

### **注意事项：**
1. **仅返回 JSON 结果，不需要额外解释或输出其他内容。**  
2. **请确保 "choose_id" 只输出选项的字母，不要附加其他文本。**  
3. **请保持输出格式整洁，符合 JSON 规范。**
4. **不要给出任何注释和注意，只需要给出答案，不需要给出其他无关内容**
5. **必须用中文解答。**
6. **确保所有输出的 JSON 格式正确，可以被 json.loads() 解析。**
"""
    
    def analyze(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        分析单条古诗词数据
        
        Args:
            data: 输入数据字典
            
        Returns:
            分析结果字典，如果失败返回 None
        """
        # 验证输入数据
        if not self._validate_input_data(data):
            return None
        
        prompt = self._build_prompt(data)
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"尝试第 {attempt + 1} 次调用 API...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    temperature=0.3  # 降低温度以获得更稳定的输出
                )
                
                content = response.choices[0].message.content.strip()
                logger.debug(f"模型返回内容（前100字符）: {content[:100]}")
                
                # 提取 JSON
                result = self._extract_json_from_text(content)
                
                if result:
                    # 验证结果格式
                    if 'idx' in result:
                        return result
                    else:
                        logger.warning(f"返回结果缺少 'idx' 字段: {result}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败（尝试 {attempt + 1}/{self.max_retries}）: {e}")
                if attempt < self.max_retries - 1:
                    continue
            except Exception as e:
                logger.error(f"API 调用失败（尝试 {attempt + 1}/{self.max_retries}）: {e}")
                if attempt < self.max_retries - 1:
                    continue
        
        # 所有重试都失败，返回默认结果
        logger.error(f"处理数据失败（index: {data.get('index', 'unknown')}），返回默认结果")
        return {
            "idx": data.get("index", ""),
            "ans_qa_words": {},
            "ans_qa_sents": {},
            "choose_id": ""
        }
    
    def batch_analyze(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量分析数据
        
        Args:
            input_data: 输入数据列表
            
        Returns:
            分析结果列表
        """
        results = []
        
        for data in tqdm(input_data, desc="处理中"):
            result = self.analyze(data)
            if result:
                results.append(result)
                logger.debug(f"成功处理: index={data.get('index')}")
            else:
                logger.warning(f"处理失败: index={data.get('index')}")
                # 添加失败标记的结果
                results.append({
                    "idx": data.get("index", ""),
                    "ans_qa_words": {},
                    "ans_qa_sents": {},
                    "choose_id": "",
                    "_error": True
                })
        
        return results


def load_json_file(file_path: str) -> Any:
    """
    加载 JSON 文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        JSON 数据
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Any, file_path: str, indent: int = 4):
    """
    保存数据到 JSON 文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: JSON 缩进
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    logger.info(f"结果已保存到: {file_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='使用大语言模型分析古诗词数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python baseline_improved.py -i input.json -o output.json
  python baseline_improved.py -i input.json -o output.json --api-base http://localhost:8000/v1 --max-retries 5
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入 JSON 文件路径'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出 JSON 文件路径'
    )
    parser.add_argument(
        '--api-base',
        type=str,
        default='http://localhost:8000/v1',
        help='API 基础 URL (默认: http://localhost:8000/v1)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default='EMPTY',
        help='API 密钥 (默认: EMPTY)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='最大重试次数 (默认: 3)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理的数据条数（用于测试）'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))
    
    try:
        # 加载输入数据
        logger.info(f"加载输入文件: {args.input}")
        input_data = load_json_file(args.input)
        
        if not isinstance(input_data, list):
            raise ValueError("输入数据必须是 JSON 数组格式")
        
        # 限制处理数量（用于测试）
        if args.limit:
            input_data = input_data[:args.limit]
            logger.info(f"限制处理数量为: {args.limit}")
        
        logger.info(f"共加载 {len(input_data)} 条数据")
        
        # 初始化分析器
        logger.info(f"初始化分析器 (API: {args.api_base})")
        analyzer = PoemAnalyzerLLM(
            api_base=args.api_base,
            api_key=args.api_key,
            max_retries=args.max_retries
        )
        
        # 批量分析
        logger.info("开始分析...")
        output_data = analyzer.batch_analyze(input_data)
        
        # 保存结果
        save_json_file(output_data, args.output)
        
        # 统计信息
        success_count = sum(1 for r in output_data if not r.get('_error', False))
        logger.info(f"处理完成！")
        logger.info(f"  总数据量: {len(input_data)}")
        logger.info(f"  成功处理: {success_count}")
        logger.info(f"  失败数量: {len(output_data) - success_count}")
        logger.info(f"  输出文件: {args.output}")
        
    except FileNotFoundError as e:
        logger.error(f"文件错误: {e}")
        return 1
    except ValueError as e:
        logger.error(f"数据格式错误: {e}")
        return 1
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

