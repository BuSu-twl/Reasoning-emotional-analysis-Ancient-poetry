"""
古诗词情感分析训练脚本
基于baseline.py和poem_analysis.py的逻辑，使用Transformer模型进行训练
支持BERT、RoBERTa等预训练模型的情感分类任务
"""
import json
import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 配置日志
# 使用覆盖模式打开日志文件，确保每次训练只显示当前训练的日志
log_file_path = 'train_poem_sentiment.log'

# 清空或删除旧日志文件
def clear_log_file():
    """清空日志文件"""
    try:
        if os.path.exists(log_file_path):
            # 尝试删除
            try:
                os.remove(log_file_path)
            except (PermissionError, OSError):
                # 如果无法删除（可能被占用），尝试清空文件内容
                try:
                    with open(log_file_path, 'w', encoding='utf-8') as f:
                        f.write('')
                except Exception:
                    pass  # 如果还是失败，就继续
    except Exception:
        pass  # 忽略所有错误，继续执行

clear_log_file()

# 配置日志系统
# 先清除现有的handlers（如果存在）
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# 重新配置日志
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),  # 使用覆盖模式
            logging.StreamHandler()
        ],
        force=True  # Python 3.8+ 支持，强制重新配置日志系统
    )
except TypeError:
    # 如果 force 参数不支持（Python < 3.8），使用传统方式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

# 配置日志处理器，确保立即刷新
for handler in logger.handlers:
    handler.flush()
import sys
if hasattr(sys.stdout, 'flush'):
    sys.stdout.flush()


class PoemDataset(Dataset):
    """古诗词数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: tokenizer对象
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用tokenizer编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class PoemDataLoader:
    """古诗词数据加载器"""
    
    @staticmethod
    def load_train_data(data_dir: str = "data/train-data") -> List[Dict[str, Any]]:
        """
        加载训练数据
        
        Args:
            data_dir: 训练数据目录
            
        Returns:
            数据列表
        """
        data_dir = Path(data_dir)
        all_data = []
        
        if not data_dir.exists():
            logger.error(f"训练数据目录不存在: {data_dir}")
            return all_data
        
        # 递归查找所有JSON文件
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
        """
        从数据中提取情感标签（使用改进的标签映射）
        
        Args:
            data: 数据列表
            
        Returns:
            (emotion_to_id, id_to_emotion) 标签映射字典
        """
        # 使用改进的情感映射器
        try:
            from utils.emotion_mapper import EmotionMapper
            emotion_to_id = EmotionMapper.create_label_mapping(data)
        except ImportError:
            logger.warning("无法导入 EmotionMapper，使用原始方法")
            # 回退到原始方法
            emotions = set()
            for item in data:
                if 'emotion' in item and item['emotion']:
                    emotion_str = str(item['emotion']).strip()
                    if emotion_str:
                        separators = [',', '，', ';', '；', ' ', '\t']
                        for sep in separators:
                            if sep in emotion_str:
                                emotion_list = [e.strip() for e in emotion_str.split(sep) if e.strip()]
                                if emotion_list:
                                    emotions.add(emotion_list[0])
                                break
                        else:
                            emotions.add(emotion_str)
            emotion_to_id = {emotion: idx for idx, emotion in enumerate(sorted(emotions))}
        
        id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}
        
        logger.info(f"发现 {len(emotion_to_id)} 个情感类别（已标准化）")
        logger.info(f"情感类别: {list(emotion_to_id.keys())[:20]}...")  # 只显示前20个
        
        return emotion_to_id, id_to_emotion
    
    @staticmethod
    def prepare_sentiment_data(
        data: List[Dict[str, Any]], 
        emotion_to_id: Dict[str, int]
    ) -> Tuple[List[str], List[int]]:
        """
        准备情感分析数据（文本和标签）
        
        Args:
            data: 原始数据列表
            emotion_to_id: 情感标签到ID的映射
            
        Returns:
            (texts, labels) 文本列表和标签列表
        """
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
            
            # 提取标签
            emotion_label = None
            if 'emotion' in item and item['emotion']:
                emotion_str = str(item['emotion']).strip()
                if emotion_str:
                    # 使用改进的标签标准化
                    try:
                        from utils.emotion_mapper import EmotionMapper
                        emotion_str = EmotionMapper.normalize_emotion_label(emotion_str)
                    except ImportError:
                        pass  # 如果无法导入，使用原始字符串
                    separators = [',', '，', ';', '；', ' ', '\t']
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


class PoemTrainer:
    """古诗词情感分析训练器"""
    
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        output_dir: str = "models/sentiment_model",
        device: Optional[str] = None,
        local_model_dir: Optional[str] = None
    ):
        """
        初始化训练器
        
        Args:
            model_name: 预训练模型名称
            output_dir: 输出目录
            device: 设备 (cuda/cpu)，None则自动检测
            local_model_dir: 本地模型目录（如果提供，优先使用本地模型）
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 自动检测设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"模型名称: {model_name}")
        
        # 确定模型路径：优先使用本地模型
        model_path = None
        if local_model_dir:
            local_path = Path(local_model_dir)
            if local_path.exists() and (local_path / "config.json").exists():
                model_path = str(local_path)
                logger.info(f"使用本地模型: {model_path}")
            else:
                logger.warning(f"指定的本地模型目录不存在或不完整: {local_model_dir}，将尝试从HuggingFace下载")
        else:
            # 尝试在项目目录下查找本地模型
            possible_local_paths = [
                Path("models") / model_name,
                Path(model_name),
            ]
            for local_path in possible_local_paths:
                if local_path.exists() and (local_path / "config.json").exists():
                    model_path = str(local_path)
                    logger.info(f"发现本地模型，使用: {model_path}")
                    break
        
        # 如果找到了本地模型，使用本地路径；否则使用模型名称（会从HuggingFace下载）
        tokenizer_path = model_path if model_path else model_name
        model_load_path = model_path if model_path else model_name
        
        # 初始化tokenizer
        logger.info(f"正在加载tokenizer: {tokenizer_path}...")
        if model_path:
            logger.info("从本地加载tokenizer，无需下载")
        else:
            logger.info("如果首次使用，可能需要下载模型文件，请耐心等待...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=bool(model_path)  # 如果使用本地模型，只使用本地文件
            )
            logger.info("Tokenizer加载完成")
        except Exception as e:
            if model_path:
                logger.warning(f"从本地加载tokenizer失败: {e}，尝试从HuggingFace下载")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                raise
            logger.info("Tokenizer加载完成")
        
        self.model = None
        self.num_labels = None
        self.model_load_path = model_load_path
    
    def train_sentiment_model(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        num_labels: int,
        num_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        eval_strategy: str = "epoch",
        save_strategy: str = "epoch",
        warmup_steps: int = 10,  # 对于小数据集，使用较小的warmup steps
        weight_decay: float = 0.01
    ) -> Tuple[Any, Path]:
        """
        训练情感分类模型
        
        Args:
            train_texts: 训练文本列表
            train_labels: 训练标签列表
            val_texts: 验证文本列表
            val_labels: 验证标签列表
            num_labels: 标签数量
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            max_length: 最大序列长度
            eval_strategy: 评估策略
            save_strategy: 保存策略
            warmup_steps: 预热步数
            weight_decay: 权重衰减
            
        Returns:
            (model, model_path) 训练好的模型和保存路径
        """
        self.num_labels = num_labels
        
        # 初始化模型
        logger.info(f"初始化模型: {self.model_name}, 标签数: {num_labels}")
        logger.info("正在加载预训练模型...")
        # 检查是否应该使用本地模型
        use_local = False
        if hasattr(self, 'model_load_path') and self.model_load_path:
            model_path_obj = Path(self.model_load_path)
            if model_path_obj.exists() and (model_path_obj / "config.json").exists():
                use_local = True
        
        if use_local:
            logger.info(f"从本地加载模型: {self.model_load_path}，无需下载")
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_load_path,
                    num_labels=num_labels,
                    local_files_only=True
                )
            except Exception as e:
                logger.warning(f"从本地加载模型失败: {e}，尝试从HuggingFace下载")
                logger.info("如果首次使用，可能需要下载模型文件（约400MB），请耐心等待...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=num_labels
                )
        else:
            logger.info("如果首次使用，可能需要下载模型文件（约400MB），请耐心等待...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
        logger.info("模型加载完成，正在移动到设备...")
        self.model.to(self.device)
        logger.info(f"模型已移动到设备: {self.device}")
        
        # 创建数据集
        logger.info("正在创建训练数据集...")
        train_dataset = PoemDataset(train_texts, train_labels, self.tokenizer, max_length)
        logger.info(f"训练数据集创建完成，样本数: {len(train_dataset)}")
        
        logger.info("正在创建验证数据集...")
        val_dataset = PoemDataset(val_texts, val_labels, self.tokenizer, max_length)
        logger.info(f"验证数据集创建完成，样本数: {len(val_dataset)}")
        
        # 计算总步数
        total_steps = len(train_dataset) // batch_size * num_epochs
        logger.info(f"训练配置: {num_epochs}轮, 批次大小: {batch_size}, 总步数: {total_steps}")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=min(warmup_steps, len(train_dataset) // batch_size),  # 确保warmup不超过一个epoch
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,  # 减少到10步，让日志更频繁
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",  # 不使用wandb/tensorboard
            seed=42,
            fp16=torch.cuda.is_available(),  # 如果使用GPU，启用混合精度
            lr_scheduler_type="cosine",  # 使用cosine学习率衰减
            dataloader_num_workers=0,  # CPU训练时设为0
        )
        
        # 计算指标的函数
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            accuracy = accuracy_score(labels, predictions)
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # 创建Trainer
        logger.info("正在创建Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # 增加patience，给模型更多训练机会
        )
        logger.info("Trainer创建完成")
        
        # 开始训练
        logger.info("=" * 60)
        logger.info("开始训练...")
        logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        logger.info(f"训练轮数: {num_epochs}, 批次大小: {batch_size}")
        logger.info("=" * 60)
        trainer.train()
        logger.info("训练完成！")
        
        # 保存最终模型
        final_model_dir = self.output_dir / "final"
        trainer.save_model(str(final_model_dir))
        self.tokenizer.save_pretrained(str(final_model_dir))
        
        logger.info(f"模型已保存到: {final_model_dir}")
        
        return self.model, final_model_dir
    
    def evaluate_model(
        self,
        model_path: str,
        test_texts: List[str],
        test_labels: List[int],
        batch_size: int = 16,
        max_length: int = 512
    ) -> Tuple[float, Dict[str, Any]]:
        """
        评估模型
        
        Args:
            model_path: 模型路径
            test_texts: 测试文本列表
            test_labels: 测试标签列表
            batch_size: 批次大小
            max_length: 最大序列长度
            
        Returns:
            (accuracy, classification_report) 准确率和分类报告
        """
        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 创建数据集和数据加载器
        test_dataset = PoemDataset(test_texts, test_labels, tokenizer, max_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 预测
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(
            true_labels, 
            predictions, 
            output_dict=True,
            zero_division=0
        )
        
        logger.info(f"测试集准确率: {accuracy:.4f}")
        
        return accuracy, report


def train_test_split(texts, labels, test_size=0.2, random_state=42):
    """
    划分训练集和验证集
    如果某些类别样本数量过少（<2），则不使用分层抽样
    """
    if not labels or len(set(labels)) <= 1:
        logger.warning("标签数量不足或只有单一类别，无法进行分层抽样，将使用随机抽样。")
        return sklearn_train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )

    # 检查每个类别的样本数量
    from collections import Counter
    label_counts = Counter(labels)
    min_samples_per_class = min(label_counts.values())

    if min_samples_per_class < 2:
        logger.warning(
            f"数据集中存在少于2个样本的类别（最小样本数: {min_samples_per_class}），"
            "无法进行分层抽样，将使用随机抽样。请考虑增加数据量或合并稀疏类别。"
        )
        return sklearn_train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
    else:
        return sklearn_train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='古诗词情感分析模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python train_poem_sentiment.py --model-name bert-base-chinese
  python train_poem_sentiment.py --model-name bert-base-chinese --epochs 5 --batch-size 16
        """
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='bert-base-chinese',
        help='预训练模型名称 (默认: bert-base-chinese)'
    )
    parser.add_argument(
        '--train-data-dir',
        type=str,
        default='data/train-data',
        help='训练数据目录 (默认: data/train-data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/sentiment_model',
        help='模型输出目录 (默认: models/sentiment_model)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='训练轮数 (默认: 10，推荐10-20轮以获得更好效果)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='批次大小 (默认: 16)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='学习率 (默认: 2e-5，可尝试1e-5到3e-5)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='最大序列长度 (默认: 512)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='验证集比例 (默认: 0.2)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制数据量（用于测试）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备 (cuda/cpu)，None则自动检测'
    )
    parser.add_argument(
        '--local-model-dir',
        type=str,
        default=None,
        help='本地模型目录路径（如果提供，优先使用本地模型，例如: models/bert-base-chinese）'
    )

    args = parser.parse_args()
    
    try:
        logger.info("=" * 60)
        logger.info("古诗词情感分析模型训练")
        logger.info("=" * 60)
        
        # 加载数据
        logger.info(f"加载训练数据: {args.train_data_dir}")
        train_data = PoemDataLoader.load_train_data(args.train_data_dir)
        
        if not train_data:
            logger.error("没有训练数据！")
            return 1
        
        # 限制数据量（用于测试）
        if args.limit:
            train_data = train_data[:args.limit]
            logger.info(f"限制数据量为: {args.limit}")
        
        # 提取情感标签
        emotion_to_id, id_to_emotion = PoemDataLoader.extract_emotion_labels(train_data)
        
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
        texts, labels = PoemDataLoader.prepare_sentiment_data(train_data, emotion_to_id)
        
        if len(texts) == 0:
            logger.error("没有有效的训练数据！")
            return 1
        
        # 划分训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=args.test_split, random_state=42
        )
        
        logger.info(f"训练集: {len(train_texts)} 条")
        logger.info(f"验证集: {len(val_texts)} 条")
        
        # 初始化训练器
        logger.info("正在初始化训练器...")
        trainer = PoemTrainer(
            model_name=args.model_name,
            output_dir=args.output_dir,
            device=args.device,
            local_model_dir=args.local_model_dir
        )
        logger.info("训练器初始化完成")
        
        # 训练模型
        logger.info("准备开始训练模型...")
        model, model_path = trainer.train_sentiment_model(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            num_labels=len(emotion_to_id),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length
        )
        
        # 评估模型
        logger.info("评估模型...")
        accuracy, report = trainer.evaluate_model(
            model_path=str(model_path),
            test_texts=val_texts,
            test_labels=val_labels,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # 保存评估结果
        eval_result_path = model_path / "evaluation_results.json"
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': accuracy,
                'classification_report': report,
                'train_samples': len(train_texts),
                'val_samples': len(val_texts),
                'num_labels': len(emotion_to_id),
                'model_name': args.model_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info(f"模型保存位置: {model_path}")
        logger.info(f"验证集准确率: {accuracy:.4f}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

