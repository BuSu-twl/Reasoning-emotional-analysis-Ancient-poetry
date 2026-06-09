"""
古诗词处理工具模块
"""
import json
import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# 模拟阿里云NLP客户端
class AliyunNLPClient:
    """模拟阿里云NLP客户端（简化版）"""

    def __init__(self):
        logger.info("初始化模拟阿里云NLP客户端")

        # 情感关键词库
        self.sentiment_keywords = {
            'sad': ["愁", "泪", "悲", "伤", "思", "孤", "独", "寒", "冷", "暮", "夜"],
            'happy': ["喜", "乐", "欢", "笑", "春", "花", "明", "暖", "晴"],
            'peaceful': ["静", "闲", "宁", "淡", "平", "和"]
        }

        # 古诗常见实体
        self.poem_entities = {
            '自然景物': ['山', '水', '月', '光', '风', '雨', '花', '鸟', '云', '雪'],
            '器物': ['床', '船', '马', '剑', '酒', '琴', '书', '灯', '窗'],
            '地点': ['家', '乡', '城', '楼', '寺', '园', '庭'],
            '时间': ['春', '夏', '秋', '冬', '晨', '午', '暮', '夜']
        }

    @lru_cache(maxsize=1000)
    def segment_poem(self, poem_text):
        """模拟分词"""
        # 简单分词：按字符分割
        words = list(poem_text)

        result = {
            "Data": {
                "result": []
            }
        }

        for word in words:
            if word.strip():  # 跳过空格
                result["Data"]["result"].append({
                    "word": word,
                    "tags": "基本词-中文"
                })

        return result

    def extract_sentiment_features(self, poem_text):
        """提取情感特征"""
        # 简单特征提取
        features = {}

        # 基础统计
        features['char_count'] = len(poem_text)

        # 情感词统计
        words = list(poem_text)
        for sentiment, keyword_list in self.sentiment_keywords.items():
            count = sum(1 for word in words if word in keyword_list)
            features[f'{sentiment}_count'] = count

        # 实体词统计
        for category, entity_list in self.poem_entities.items():
            count = sum(1 for word in words if word in entity_list)
            features[f'{category}_count'] = count

        # 特殊标点
        features['comma_count'] = poem_text.count('，')
        features['period_count'] = poem_text.count('。')

        return features


class PoemDatasetProcessor:
    """古诗词数据集处理器"""

    def __init__(self, aliyun_client=None):
        self.aliyun_client = aliyun_client or AliyunNLPClient()
        self.feature_names = None

    def load_train_data(self, data_dir="data/train-data"):
        """加载训练数据"""
        data_dir = Path(data_dir)
        all_data = []

        if not data_dir.exists():
            logger.error(f"训练数据目录不存在: {data_dir}")
            return all_data

        # 查找JSON文件
        json_files = list(data_dir.rglob("*.json"))
        logger.info(f"找到 {len(json_files)} 个JSON文件")

        for json_file in json_files[:3]:  # 只处理前3个文件，加快速度
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    all_data.extend(data[:10])  # 每个文件只取前10条
                elif isinstance(data, dict):
                    all_data.append(data)

            except Exception as e:
                logger.error(f"加载文件失败 {json_file}: {e}")

        logger.info(f"共加载 {len(all_data)} 条训练数据")
        return all_data

    def extract_emotion_labels(self, data):
        """提取情感标签"""
        emotions = set()

        for item in data:
            if 'emotion' in item and item['emotion']:
                emotion_str = str(item['emotion']).strip()
                if emotion_str:
                    # 简单处理：取第一个情感
                    emotion = emotion_str.split(',')[0].strip()
                    emotions.add(emotion)

        # 如果没有找到情感标签，使用默认标签
        if not emotions:
            emotions = {"哀伤", "喜悦", "平静", "思念"}

        # 创建标签映射
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(sorted(emotions))}
        id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}

        logger.info(f"发现 {len(emotion_to_id)} 个情感类别: {list(emotion_to_id.keys())}")

        return emotion_to_id, id_to_emotion

    def prepare_training_data(self, train_data, emotion_to_id):
        """准备训练数据"""
        features_list = []
        labels = []

        logger.info("提取特征...")

        for i, item in enumerate(train_data):
            if i >= 50:  # 只处理50条数据，加快速度
                break

            if 'content' not in item:
                continue

            text = item['content']
            if not text or not isinstance(text, str):
                continue

            # 构建完整文本
            full_text = text
            if 'title' in item and item['title']:
                full_text = f"{item['title']}：{full_text}"

            # 提取特征
            features = self.aliyun_client.extract_sentiment_features(full_text)
            features_list.append(features)

            # 提取标签
            if 'emotion' in item and item['emotion']:
                emotion_str = str(item['emotion']).strip()
                if emotion_str:
                    emotion = emotion_str.split(',')[0].strip()
                    if emotion in emotion_to_id:
                        labels.append(emotion_to_id[emotion])
                    else:
                        labels.append(0)  # 默认标签
                else:
                    labels.append(0)
            else:
                labels.append(0)

        # 转换为DataFrame
        if features_list:
            df_features = pd.DataFrame(features_list)
            self.feature_names = df_features.columns.tolist()
            logger.info(f"特征提取完成，特征维度: {df_features.shape}")
            return df_features, np.array(labels), []
        else:
            return pd.DataFrame(), np.array([]), []


class PoemSentimentClassifier:
    """古诗词情感分类器（简化版）"""

    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        logger.info(f"初始化 {model_type} 分类器")

    def train(self, X, y, feature_names=None):
        """模拟训练"""
        logger.info(f"模拟训练，数据形状: {X.shape if hasattr(X, 'shape') else '未知'}")
        logger.info("训练完成（模拟）")
        return self

    def save(self, model_path):
        """保存模型"""
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        # 保存模拟模型信息
        info_file = model_path / "model_info.txt"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"模型类型: {self.model_type}\n")
            f.write(f"保存时间: 模拟保存\n")
            f.write("这是一个模拟模型，用于测试\n")

        logger.info(f"模拟模型已保存到: {model_path}")

    def load(self, model_path):
        """加载模型"""
        logger.info(f"模拟加载模型: {model_path}")
        return True