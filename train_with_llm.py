"""
基于大语言模型的古诗词情感分析训练系统
支持多种大模型：OpenAI API、通义千问、本地LLM等
"""
import json
import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_llm.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


class LLMDataAugmenter:
    """使用大模型进行数据增强"""
    
    def __init__(self, model_type: str = "openai", api_key: Optional[str] = None):
        """
        初始化数据增强器
        
        Args:
            model_type: 模型类型 ("openai", "qwen", "local")
            api_key: API密钥（如果需要）
        """
        self.model_type = model_type
        self.api_key = api_key
        self.client = None
        
        if model_type == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            except ImportError:
                logger.warning("OpenAI库未安装，请运行: pip install openai")
        elif model_type == "qwen":
            try:
                import dashscope
                dashscope.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
                self.client = dashscope
            except ImportError:
                logger.warning("dashscope库未安装，请运行: pip install dashscope")
    
    def augment_data(self, poem_data: Dict[str, Any], num_augments: int = 2) -> List[Dict[str, Any]]:
        """
        使用大模型增强数据
        
        Args:
            poem_data: 原始诗词数据
            num_augments: 增强数量
            
        Returns:
            增强后的数据列表
        """
        if not self.client:
            logger.warning("大模型客户端未初始化，跳过数据增强")
            return []
        
        augmented = []
        content = poem_data.get('content', '')
        emotion = poem_data.get('emotion', '')
        
        prompt = f"""请基于以下古诗词，生成{num_augments}个相似情感但表达不同的古诗词句子。

原诗词：{content}
情感：{emotion}

要求：
1. 保持相同的情感倾向
2. 使用不同的意象和表达方式
3. 保持古诗词的韵律和风格
4. 每句不超过20字

请以JSON格式返回，格式：
[
    {{"content": "生成的诗词句子1", "emotion": "{emotion}"}},
    {{"content": "生成的诗词句子2", "emotion": "{emotion}"}}
]
"""
        
        try:
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
                result = response.choices[0].message.content
            elif self.model_type == "qwen":
                from dashscope import Generation
                response = Generation.call(
                    model="qwen-turbo",
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=500
                )
                result = response.output.text
            
            # 解析JSON结果
            import re
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                aug_data = json.loads(json_match.group())
                for item in aug_data:
                    item['title'] = poem_data.get('title', '') + '（增强）'
                    item['source'] = 'augmented'
                    augmented.append(item)
        except Exception as e:
            logger.error(f"数据增强失败: {e}")
        
        return augmented


class LLMTrainer:
    """基于大语言模型的训练器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        output_dir: str = "models/llm_sentiment_model",
        device: Optional[str] = None
    ):
        """
        初始化训练器
        
        Args:
            model_name: 预训练模型名称（支持HuggingFace模型）
            output_dir: 输出目录
            device: 设备 (cuda/cpu)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"模型名称: {model_name}")
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 根据模型类型选择架构
        if "instruct" in model_name.lower() or "chat" in model_name.lower():
            # 使用因果语言模型（用于生成式任务）
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model_type = "causal"
        else:
            # 使用序列分类模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=10,  # 临时值，会在训练时更新
                trust_remote_code=True
            )
            self.model_type = "classification"
        
        self.model.to(self.device)
        self.num_labels = None
    
    def prepare_prompt_data(
        self,
        texts: List[str],
        labels: List[int],
        id_to_emotion: Dict[int, str]
    ) -> List[str]:
        """
        准备提示词格式的数据（用于生成式模型）
        
        Args:
            texts: 文本列表
            labels: 标签列表
            id_to_emotion: ID到情感的映射
            
        Returns:
            格式化后的提示词列表
        """
        prompts = []
        for text, label_id in zip(texts, labels):
            emotion = id_to_emotion.get(label_id, "unknown")
            prompt = f"""请分析以下古诗词的情感。

古诗词：{text}

情感分析：{emotion}

请用一句话解释为什么这首诗表达的是{emotion}情感："""
            prompts.append(prompt)
        return prompts
    
    def train_with_prompts(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        id_to_emotion: Dict[int, str],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        max_length: int = 512
    ):
        """
        使用提示词方式训练生成式模型
        
        Args:
            train_texts: 训练文本
            train_labels: 训练标签
            val_texts: 验证文本
            val_labels: 验证标签
            id_to_emotion: ID到情感的映射
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            max_length: 最大长度
        """
        logger.info("准备提示词格式的训练数据...")
        
        # 准备提示词数据
        train_prompts = self.prepare_prompt_data(train_texts, train_labels, id_to_emotion)
        val_prompts = self.prepare_prompt_data(val_texts, val_labels, id_to_emotion)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        train_dataset = Dataset.from_dict({"text": train_prompts})
        val_dataset = Dataset.from_dict({"text": val_prompts})
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=42
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 因果语言建模，不是掩码语言建模
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        final_model_dir = self.output_dir / "final"
        trainer.save_model(str(final_model_dir))
        self.tokenizer.save_pretrained(str(final_model_dir))
        
        logger.info(f"模型已保存到: {final_model_dir}")
        return final_model_dir


class PoemLLMDataLoader:
    """古诗词数据加载器（支持大模型训练）"""
    
    @staticmethod
    def load_train_data(data_dir: str = "data/train-data") -> List[Dict[str, Any]]:
        """加载训练数据"""
        data_dir = Path(data_dir)
        all_data = []
        
        if not data_dir.exists():
            logger.error(f"训练数据目录不存在: {data_dir}")
            return all_data
        
        json_files = list(data_dir.rglob("*.json"))
        logger.info(f"找到 {len(json_files)} 个JSON文件")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'content' in item:
                            item['source_file'] = str(json_file.relative_to(data_dir))
                            all_data.append(item)
                elif isinstance(data, dict) and 'content' in data:
                    data['source_file'] = str(json_file.relative_to(data_dir))
                    all_data.append(data)
            except Exception as e:
                logger.error(f"加载文件失败 {json_file}: {e}")
        
        logger.info(f"共加载 {len(all_data)} 条训练数据")
        return all_data
    
    @staticmethod
    def extract_emotion_labels(data: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """提取情感标签（支持多标签）"""
        emotions = set()
        
        for item in data:
            if 'emotion' in item and item['emotion']:
                emotion_str = str(item['emotion']).strip()
                if emotion_str:
                    separators = [',', '，', ';', '；', ' ', '\t', '、']
                    for sep in separators:
                        if sep in emotion_str:
                            emotion_list = [e.strip() for e in emotion_str.split(sep) if e.strip()]
                            emotions.update(emotion_list)
                            break
                    else:
                        emotions.add(emotion_str)
        
        # 创建标签映射（只取前20个最常见的情感）
        from collections import Counter
        emotion_counter = Counter()
        for item in data:
            if 'emotion' in item and item['emotion']:
                emotion_str = str(item['emotion']).strip()
                if emotion_str:
                    separators = [',', '，', ';', '；', ' ', '\t', '、']
                    for sep in separators:
                        if sep in emotion_str:
                            emotion_list = [e.strip() for e in emotion_str.split(sep) if e.strip()]
                            emotion_counter.update(emotion_list)
                            break
                    else:
                        emotion_counter[emotion_str] += 1
        
        # 选择最常见的N个情感作为主要标签
        top_emotions = [emotion for emotion, _ in emotion_counter.most_common(15)]
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(top_emotions)}
        id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}
        
        logger.info(f"选择 {len(emotion_to_id)} 个主要情感类别")
        logger.info(f"情感类别: {list(emotion_to_id.keys())}")
        
        return emotion_to_id, id_to_emotion
    
    @staticmethod
    def prepare_sentiment_data(
        data: List[Dict[str, Any]],
        emotion_to_id: Dict[str, int]
    ) -> Tuple[List[str], List[int]]:
        """准备情感分析数据"""
        texts = []
        labels = []
        
        for item in data:
            if 'content' not in item:
                continue
            
            text = item['content']
            if not text or not isinstance(text, str):
                continue
            
            # 构建完整文本（包含标题）
            full_text = text
            if 'title' in item and item['title']:
                full_text = f"{item['title']}：{text}"
            
            texts.append(full_text)
            
            # 提取标签（取第一个匹配的情感）
            emotion_label = None
            if 'emotion' in item and item['emotion']:
                emotion_str = str(item['emotion']).strip()
                if emotion_str:
                    separators = [',', '，', ';', '；', ' ', '\t', '、']
                    for sep in separators:
                        if sep in emotion_str:
                            emotion_list = [e.strip() for e in emotion_str.split(sep) if e.strip()]
                            if emotion_list:
                                emotion_label = emotion_list[0]
                            break
                    else:
                        emotion_label = emotion_str
            
            # 映射到ID
            if emotion_label and emotion_label in emotion_to_id:
                labels.append(emotion_to_id[emotion_label])
            else:
                # 使用默认标签（第一个）
                labels.append(0)
        
        logger.info(f"准备了 {len(texts)} 条有效数据")
        return texts, labels


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于大语言模型的古诗词情感分析训练')
    
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2-0.5B-Instruct',
                       help='预训练模型名称')
    parser.add_argument('--train-data-dir', type=str, default='data/train-data',
                       help='训练数据目录')
    parser.add_argument('--output-dir', type=str, default='models/llm_sentiment_model',
                       help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--max-length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--test-split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--augment', action='store_true', help='是否使用数据增强')
    parser.add_argument('--augment-model', type=str, default='openai',
                       choices=['openai', 'qwen', 'local'],
                       help='数据增强使用的模型')
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("基于大语言模型的古诗词情感分析训练")
        logger.info("=" * 60)
        
        # 加载数据
        logger.info(f"加载训练数据: {args.train_data_dir}")
        train_data = PoemLLMDataLoader.load_train_data(args.train_data_dir)
        
        if not train_data:
            logger.error("没有训练数据！")
            return 1
        
        # 数据增强（可选）
        if args.augment:
            logger.info("开始数据增强...")
            augmenter = LLMDataAugmenter(model_type=args.augment_model)
            augmented_data = []
            for item in train_data[:10]:  # 只增强前10条作为示例
                aug = augmenter.augment_data(item, num_augments=2)
                augmented_data.extend(aug)
            train_data.extend(augmented_data)
            logger.info(f"数据增强后总数: {len(train_data)} 条")
        
        # 提取情感标签
        emotion_to_id, id_to_emotion = PoemLLMDataLoader.extract_emotion_labels(train_data)
        
        # 保存标签映射
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        label_map_path = output_dir / "label_mapping.json"
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump({
                'emotion_to_id': emotion_to_id,
                'id_to_emotion': id_to_emotion,
                'num_labels': len(emotion_to_id)
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"标签映射已保存到: {label_map_path}")
        
        # 准备数据
        logger.info("准备训练数据...")
        texts, labels = PoemLLMDataLoader.prepare_sentiment_data(train_data, emotion_to_id)
        
        if len(texts) == 0:
            logger.error("没有有效的训练数据！")
            return 1
        
        # 划分训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=args.test_split, random_state=42, stratify=labels
        )
        
        logger.info(f"训练集: {len(train_texts)} 条")
        logger.info(f"验证集: {len(val_texts)} 条")
        
        # 初始化训练器
        trainer = LLMTrainer(
            model_name=args.model_name,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # 训练模型
        logger.info("开始训练...")
        model_path = trainer.train_with_prompts(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            id_to_emotion=id_to_emotion,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length
        )
        
        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info(f"模型保存位置: {model_path}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())











