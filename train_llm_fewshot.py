"""
基于大模型的Few-Shot Learning训练系统
适用于小数据集（164条）和分散标签（227个唯一标签）的情况
使用大模型的推理能力，无需微调
"""
import json
import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_llm_fewshot.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


class FewShotLLMTrainer:
    """基于Few-Shot Learning的大模型训练器"""
    
    def __init__(self, model_type: str = "qwen", api_key: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            model_type: 模型类型 ("openai", "qwen", "local")，默认: qwen
            api_key: API密钥
        """
        self.model_type = model_type
        self.api_key = api_key
        self.client = None
        self.examples = []  # Few-shot示例
        
        # 初始化同义词词典
        self._init_synonym_dict()
        
        self._init_client()
    
    def _init_synonym_dict(self):
        """初始化情感词同义词词典"""
        # 常见情感词的同义词组
        self.synonym_groups = [
            # 思念类
            {"思念", "想念", "怀念", "怀旧", "思乡", "思恋", "眷恋"},
            # 孤独类
            {"孤独", "孤寂", "寂寞", "孤单", "孤苦", "寂寥"},
            # 悲伤类
            {"悲伤", "悲哀", "悲痛", "悲凉", "悲苦", "悲愤", "悲愁"},
            # 忧愁类
            {"忧愁", "忧虑", "担忧", "愁苦", "惆怅", "愁闷", "愁思"},
            # 震撼类
            {"震撼", "震惊", "惊叹", "惊异", "惊诧", "震撼人心"},
            # 愤怒类
            {"愤怒", "愤慨", "愤懑", "愤恨", "气愤", "恼怒"},
            # 无奈类
            {"无奈", "无助", "无力", "无望", "无可奈何"},
            # 期待类
            {"期待", "期望", "盼望", "渴望", "向往", "希冀"},
            # 伤感类
            {"伤感", "感伤", "伤怀", "伤离", "伤别", "伤情"},
            # 豪情类
            {"豪情", "豪迈", "豪放", "豪壮", "豪气", "壮志"},
            # 陶醉类
            {"陶醉", "沉醉", "迷恋", "痴迷", "着迷"},
            # 闲适类
            {"闲适", "悠闲", "安逸", "舒适", "惬意", "自在"},
            # 赞美类
            {"赞美", "称赞", "赞赏", "赞叹", "颂扬", "歌颂"},
            # 遗憾类
            {"遗憾", "惋惜", "可惜", "懊悔", "悔恨"},
            # 思乡类（单独列出，因为可能和思念有区别）
            {"思乡", "乡愁", "乡思", "怀乡"},
        ]
        
        # 构建同义词映射表（每个词映射到其同义词组）
        self.synonym_dict = {}
        for group in self.synonym_groups:
            for word in group:
                self.synonym_dict[word] = group

    @staticmethod
    def _extract_emotion_keywords(text: str) -> List[str]:
        """
        从情感描述中提取关键词用于模糊匹配。

        说明：
        - 将中文标点统一为逗号分隔
        - 按常见连接词/顿号分割
        - 去掉空白和常见虚词
        """
        if not text:
            return []

        # 统一全角/半角标点并替换为逗号
        normalized = str(text)
        for ch in ['，', '。', '；', ';', '、', '：', ':', '！', '？', ' ']:
            normalized = normalized.replace(ch, ',')

        # 处理常见连接词（把“思念与孤独”这类拆成“思念,孤独”）
        for ch in ['与', '和', '及', '并', '跟', '及其']:
            normalized = normalized.replace(ch, ',')

        parts = [p.strip() for p in normalized.split(',') if p.strip()]

        # 常见无意义或过宽泛的词，避免影响匹配
        stop_words = {
            "之情", "之感", "之意", "情感", "情绪",
            "情", "感", "心情", "情怀", "心境",
            "等", "以及"
        }

        keywords_set = set()
        for p in parts:
            # 提取汉字连续片段
            words = re.findall(r'[\u4e00-\u9fff]+', p)
            for w in words:
                w = w.strip()
                if not w:
                    continue
                # 去掉停用词和过短的单字（如“之”“与”“和”）
                if len(w) == 1:
                    continue
                if w in stop_words:
                    continue
                keywords_set.add(w)

        return list(keywords_set)
    
    def _get_synonyms(self, word: str) -> set:
        """
        获取一个词的所有同义词（包括自身）
        
        Args:
            word: 输入词
            
        Returns:
            同义词集合
        """
        if word in self.synonym_dict:
            return self.synonym_dict[word]
        return {word}
    
    def _expand_keywords_with_synonyms(self, keywords: set) -> set:
        """
        扩展关键词集合，包含所有同义词
        
        Args:
            keywords: 原始关键词集合
            
        Returns:
            扩展后的关键词集合（包含同义词）
        """
        expanded = set(keywords)
        for keyword in keywords:
            synonyms = self._get_synonyms(keyword)
            expanded.update(synonyms)
        return expanded
    
    def _calculate_semantic_similarity(self, word1: str, word2: str) -> float:
        """
        计算两个词的语义相似度（基于字符重叠和同义词关系）
        
        Args:
            word1: 第一个词
            word2: 第二个词
            
        Returns:
            相似度分数（0.0-1.0）
        """
        # 1. 完全匹配
        if word1 == word2:
            return 1.0
        
        # 2. 同义词匹配
        synonyms1 = self._get_synonyms(word1)
        synonyms2 = self._get_synonyms(word2)
        if synonyms1 & synonyms2:  # 有交集说明是同义词
            return 0.9
        
        # 3. 字符重叠度（Jaccard相似度）
        chars1 = set(word1)
        chars2 = set(word2)
        if chars1 and chars2:
            intersection = chars1 & chars2
            union = chars1 | chars2
            char_similarity = len(intersection) / len(union) if union else 0.0
            
            # 如果字符重叠度很高（>0.6），认为语义相似
            if char_similarity > 0.6:
                return 0.7
        
        # 4. 包含关系
        if word1 in word2 or word2 in word1:
            return 0.6
        
        return 0.0
    
    def _match_keywords_with_similarity(
        self, 
        true_keywords: set, 
        pred_keywords: set,
        similarity_threshold: float = 0.7
    ) -> Tuple[set, List[Tuple[str, str, float]]]:
        """
        使用语义相似度匹配关键词
        
        Args:
            true_keywords: 真实标签关键词集合
            pred_keywords: 预测关键词集合
            similarity_threshold: 相似度阈值
            
        Returns:
            (匹配的关键词集合, [(true_word, pred_word, similarity), ...])
        """
        matched = set()  # 存储匹配的真实关键词
        matches = []
        
        # 先尝试同义词扩展匹配
        expanded_true = self._expand_keywords_with_synonyms(true_keywords)
        expanded_pred = self._expand_keywords_with_synonyms(pred_keywords)
        
        # 找出通过同义词匹配的原始关键词对
        for true_word in true_keywords:
            true_synonyms = self._get_synonyms(true_word)
            for pred_word in pred_keywords:
                pred_synonyms = self._get_synonyms(pred_word)
                # 如果两个词的同义词组有交集，说明是同义词
                if true_synonyms & pred_synonyms:
                    if true_word not in matched:
                        matched.add(true_word)
                        matches.append((true_word, pred_word, 1.0))
        
        # 对未匹配的词进行相似度计算
        unmatched_true = true_keywords - matched
        unmatched_pred = pred_keywords - {m[1] for m in matches}
        
        for true_word in unmatched_true:
            best_match = None
            best_similarity = 0.0
            
            for pred_word in unmatched_pred:
                similarity = self._calculate_semantic_similarity(true_word, pred_word)
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_match = pred_word
            
            if best_match:
                matched.add(true_word)
                matches.append((true_word, best_match, best_similarity))
                # 从待匹配集合中移除已匹配的预测词
                unmatched_pred.discard(best_match)
        
        return matched, matches
    
    def _init_client(self):
        """初始化客户端"""
        if self.model_type == "openai":
            try:
                from openai import OpenAI
                import httpx
                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error("=" * 60)
                    logger.error("❌ OpenAI API Key 未设置！")
                    logger.error("请使用以下方式之一设置API Key：")
                    logger.error("1. 命令行参数: --api-key sk-xxx")
                    logger.error("2. 环境变量: $env:OPENAI_API_KEY=\"sk-xxx\" (PowerShell)")
                    logger.error("3. 环境变量: export OPENAI_API_KEY=\"sk-xxx\" (Linux/Mac)")
                    logger.error("=" * 60)
                    self.client = None
                    return
                # 设置更长的超时时间：连接超时10秒，读取超时180秒
                timeout = httpx.Timeout(180.0, connect=10.0)
                self.client = OpenAI(api_key=api_key, timeout=timeout)
                self.model_name = "gpt-3.5-turbo"
                logger.info(f"✅ OpenAI客户端初始化成功，使用模型: {self.model_name}")
                logger.info(f"⏱️  超时设置: 连接10秒，请求180秒")
            except ImportError:
                logger.error("❌ OpenAI库未安装！请运行: pip install openai")
                self.client = None
            except Exception as e:
                logger.error(f"❌ 初始化OpenAI客户端失败: {e}")
                self.client = None
        
        elif self.model_type == "qwen":
            try:
                import dashscope
                api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY")
                if not api_key:
                    logger.error("=" * 60)
                    logger.error("❌ 通义千问 API Key 未设置！")
                    logger.error("请使用以下方式之一设置API Key：")
                    logger.error("1. 命令行参数: --api-key sk-xxx")
                    logger.error("2. 环境变量: $env:DASHSCOPE_API_KEY=\"sk-xxx\" (PowerShell)")
                    logger.error("3. 环境变量: export DASHSCOPE_API_KEY=\"sk-xxx\" (Linux/Mac)")
                    logger.error("=" * 60)
                    self.client = None
                    return
                dashscope.api_key = api_key
                self.client = dashscope
                self.model_name = "qwen-turbo"
                logger.info(f"✅ 通义千问客户端初始化成功，使用模型: {self.model_name}")
                logger.info(f"⏱️  超时设置: 180秒（通过Generation.call的timeout参数）")
            except ImportError as e:
                logger.error("=" * 60)
                logger.error("❌ dashscope库未安装！")
                logger.error("请运行: pip install dashscope")
                logger.error(f"导入错误详情: {e}")
                logger.error("=" * 60)
                self.client = None
            except Exception as e:
                logger.error(f"❌ 初始化通义千问客户端失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.client = None
        
        elif self.model_type == "local":
            try:
                from openai import OpenAI
                import httpx
                # 假设本地部署了兼容OpenAI API的LLM服务
                local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")
                logger.info(f"正在连接本地LLM服务: {local_url}")
                # 本地服务可能需要更长的超时时间
                timeout = httpx.Timeout(300.0, connect=10.0)
                self.client = OpenAI(
                    base_url=local_url,
                    api_key="EMPTY",
                    timeout=timeout
                )
                self.model_name = "local-llm"
                logger.info(f"✅ 本地LLM客户端初始化成功，使用模型: {self.model_name}")
                logger.info(f"⏱️  超时设置: 连接10秒，请求300秒")
            except ImportError:
                logger.error("=" * 60)
                logger.error("❌ OpenAI库未安装！请运行: pip install openai")
                logger.error("=" * 60)
                self.client = None
            except Exception as e:
                logger.error("=" * 60)
                logger.error(f"❌ 初始化本地LLM客户端失败: {e}")
                logger.error("请确保本地LLM服务正在运行")
                logger.error("=" * 60)
                self.client = None
    
    def build_few_shot_prompt(
        self,
        test_poem: Dict[str, Any],
        examples: List[Dict[str, Any]],
        max_examples: int = 5
    ) -> str:
        """
        构建Few-Shot提示词
        
        Args:
            test_poem: 待分析的诗词
            examples: 示例诗词列表
            max_examples: 最大示例数量
            
        Returns:
            格式化后的提示词
        """
        prompt = """你是一个古诗词情感分析专家。请根据给定的示例，分析古诗词的情感。

示例：

"""
        
        # 添加Few-Shot示例
        for i, example in enumerate(examples[:max_examples], 1):
            content = example.get('content', '')
            emotion = example.get('emotion', '')
            title = example.get('title', '')
            
            prompt += f"""示例{i}：
标题：{title}
诗词：{content}
情感：{emotion}

"""
        
        # 添加待分析的诗词
        test_content = test_poem.get('content', '')
        test_title = test_poem.get('title', '')
        
        prompt += f"""请分析以下古诗词的情感：

标题：{test_title}
诗词：{test_content}

请以JSON格式返回分析结果：
{{
    "sentiment": "主要情感（从示例中选择最接近的，或使用新的描述）",
    "confidence": 0.0-1.0之间的浮点数,
    "explanation": "简要解释为什么是这种情感"
}}

只返回JSON，不要其他文字。"""
        
        return prompt
    
    def predict(self, poem: Dict[str, Any], examples: List[Dict[str, Any]], max_retries: int = 3) -> Dict[str, Any]:
        """
        使用Few-Shot Learning进行预测
        
        Args:
            poem: 待分析的诗词
            examples: Few-Shot示例
            max_retries: 最大重试次数
            
        Returns:
            预测结果
        """
        if not self.client:
            logger.error("客户端未初始化")
            return {"sentiment": "unknown", "confidence": 0.0, "explanation": "客户端未初始化"}
        
        prompt = self.build_few_shot_prompt(poem, examples)
        
        # 重试逻辑
        last_error = None
        result_text = None
        for attempt in range(max_retries):
            try:
                if self.model_type == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=300
                    )
                    result_text = response.choices[0].message.content.strip()
                    break  # 成功则跳出重试循环
                
                elif self.model_type == "qwen":
                    from dashscope import Generation
                    # 使用更长的超时时间（180秒），以适应网络延迟和API响应时间
                    response = Generation.call(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=300,
                        timeout=180  # 增加到180秒超时
                    )
                    # 检查响应状态
                    if response.status_code != 200:
                        error_msg = f"通义千问API调用失败: {response.message}"
                        raise RuntimeError(error_msg)
                    # 获取响应文本
                    if hasattr(response, 'output') and hasattr(response.output, 'text'):
                        result_text = response.output.text.strip()
                    elif hasattr(response, 'output') and hasattr(response.output, 'choices'):
                        # 兼容choices格式的响应
                        if response.output.choices and len(response.output.choices) > 0:
                            result_text = response.output.choices[0].message.content.strip()
                        else:
                            raise ValueError("响应中没有有效的文本内容")
                    else:
                        # 兼容不同的响应格式
                        result_text = str(response).strip()
                    break  # 成功则跳出重试循环
                
                elif self.model_type == "local":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=300
                    )
                    result_text = response.choices[0].message.content.strip()
                    break  # 成功则跳出重试循环
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：1秒, 2秒, 4秒
                    logger.warning(f"预测失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"预测失败（已重试 {max_retries} 次）: {e}")
        
        # 如果所有重试都失败
        if last_error or result_text is None:
            error_msg = str(last_error) if last_error else "未收到响应"
            return {"sentiment": "unknown", "confidence": 0.0, "explanation": f"请求失败: {error_msg}"}
        
        # 解析JSON结果
        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                logger.warning(f"无法解析JSON结果: {result_text}")
                return {"sentiment": "unknown", "confidence": 0.5, "explanation": "解析失败"}
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}, 原始响应: {result_text[:200]}")
            return {"sentiment": "unknown", "confidence": 0.5, "explanation": f"JSON解析失败: {str(e)}"}
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        output_dir: str = "models/llm_fewshot"
    ) -> Dict[str, Any]:
        """
        训练Few-Shot模型（实际上是构建示例库）
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            output_dir: 输出目录
            
        Returns:
            训练结果
        """
        # 检查客户端是否初始化
        if not self.client:
            error_msg = f"❌ {self.model_type.upper()}客户端未初始化，无法进行训练！"
            logger.error("=" * 60)
            logger.error(error_msg)
            logger.error("请检查API Key是否正确设置")
            logger.error("=" * 60)
            raise RuntimeError(error_msg)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存Few-Shot示例库
        examples_path = output_path / "few_shot_examples.json"
        
        # 选择代表性的示例（每种情感选择1-2个）
        emotion_examples = {}
        for item in train_data:
            if 'emotion' in item and item['emotion']:
                emotion = str(item['emotion']).strip()
                # 提取主要情感（取第一个）
                main_emotion = emotion.split(',')[0].split('，')[0].strip()
                
                if main_emotion not in emotion_examples:
                    emotion_examples[main_emotion] = []
                
                if len(emotion_examples[main_emotion]) < 2:  # 每种情感最多2个示例
                    emotion_examples[main_emotion].append(item)
        
        # 展平示例列表
        examples = []
        for emotion, items in emotion_examples.items():
            examples.extend(items)
        
        # 保存示例
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Few-Shot示例库已保存: {examples_path}")
        logger.info(f"示例数量: {len(examples)}")
        
        # 在验证集上评估
        logger.info("在验证集上评估...")
        logger.info(f"将评估前10条数据（共{len(val_data)}条）以节省时间和API调用")
        correct = 0
        total = 0
        results = []
        
        for i, item in enumerate(val_data[:10], 1):  # 减少到10条以加快速度
            logger.info(f"评估进度: {i}/10")
            true_emotion = item.get('emotion', '')
            prediction = self.predict(item, examples)
            
            predicted_emotion = prediction.get('sentiment', '')

            # ---- 改进的模糊匹配逻辑（支持同义词和语义相似度）----
            # 1. 先用关键词提取，将长句情感描述拆成若干情感词
            true_keywords = set(self._extract_emotion_keywords(true_emotion))
            pred_keywords = set(self._extract_emotion_keywords(predicted_emotion))

            # 2. 使用同义词和语义相似度进行匹配
            matched_keywords, similarity_matches = self._match_keywords_with_similarity(
                true_keywords, pred_keywords, similarity_threshold=0.7
            )
            
            # 3. 判断是否匹配成功
            is_correct = False
            if matched_keywords:
                # 如果匹配到了至少一个关键词，认为预测正确
                is_correct = True
            else:
                # 退回到最原始的"包含"判断，作为补充
                if predicted_emotion and true_emotion and (
                    predicted_emotion in true_emotion or true_emotion in predicted_emotion
                ):
                    is_correct = True

            if is_correct:
                correct += 1
            
            # 格式化相似度匹配信息
            similarity_info = [
                {
                    'true_word': m[0],
                    'pred_word': m[1],
                    'similarity': round(m[2], 2)
                }
                for m in similarity_matches
            ]
            
            results.append({
                'poem': item.get('content', ''),
                'true_emotion': true_emotion,
                'predicted_emotion': predicted_emotion,
                'true_keywords': list(true_keywords),
                'predicted_keywords': list(pred_keywords),
                'matched_keywords': list(matched_keywords),
                'similarity_matches': similarity_info,
                'is_correct': is_correct,
                'confidence': prediction.get('confidence', 0.0),
                'explanation': prediction.get('explanation', '')
            })
            
            total += 1
            # 增加延迟以避免API限流（千问API建议在请求之间添加延迟）
            if i < 10:  # 不是最后一个请求时才延迟
                time.sleep(3)  # 增加到3秒延迟，减少限流风险
        
        accuracy = correct / total if total > 0 else 0.0
        
        # 保存评估结果
        eval_path = output_path / "evaluation_results.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': accuracy,
                'total': total,
                'correct': correct,
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"验证集准确率: {accuracy:.2%}")
        logger.info(f"评估结果已保存: {eval_path}")
        
        return {
            'accuracy': accuracy,
            'examples_count': len(examples),
            'output_dir': str(output_path)
        }


def load_data(data_dir: str = "data/train-data") -> List[Dict[str, Any]]:
    """加载数据"""
    data_dir = Path(data_dir)
    all_data = []
    
    json_files = list(data_dir.rglob("*.json"))
    logger.info(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                all_data.append(data)
        except Exception as e:
            logger.error(f"加载文件失败 {json_file}: {e}")
    
    logger.info(f"共加载 {len(all_data)} 条数据")
    return all_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于Few-Shot Learning的大模型训练')
    
    parser.add_argument('--model-type', type=str, default='qwen',
                       choices=['openai', 'qwen', 'local'],
                       help='模型类型（默认: qwen）')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API密钥（如果不提供，将从环境变量读取）')
    parser.add_argument('--train-data-dir', type=str, default='data/train-data',
                       help='训练数据目录')
    parser.add_argument('--output-dir', type=str, default='models/llm_fewshot',
                       help='输出目录')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='验证集比例')
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("基于Few-Shot Learning的大模型训练")
        logger.info("=" * 60)
        
        # 加载数据
        all_data = load_data(args.train_data_dir)
        
        if not all_data:
            logger.error("没有训练数据！")
            return 1
        
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(
            all_data, test_size=args.test_split, random_state=42
        )
        
        logger.info(f"训练集: {len(train_data)} 条")
        logger.info(f"验证集: {len(val_data)} 条")
        
        # 初始化训练器
        logger.info(f"正在初始化{args.model_type.upper()}客户端...")
        trainer = FewShotLLMTrainer(
            model_type=args.model_type,
            api_key=args.api_key
        )
        
        # 检查客户端是否初始化成功
        if not trainer.client:
            logger.error("=" * 60)
            logger.error(f"❌ {args.model_type.upper()}客户端初始化失败！")
            logger.error("请检查API Key是否正确设置")
            logger.error("=" * 60)
            return 1
        
        # 训练（构建Few-Shot示例库）
        result = trainer.train(
            train_data=train_data,
            val_data=val_data,
            output_dir=args.output_dir
        )
        
        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info(f"Few-Shot示例库: {result['examples_count']} 个示例")
        logger.info(f"验证集准确率: {result['accuracy']:.2%}")
        logger.info(f"输出目录: {result['output_dir']}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())


