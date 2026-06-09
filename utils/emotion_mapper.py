"""
情感标签映射工具
将细粒度的情感标签映射到统一的粗粒度标签
"""
from typing import Dict, List, Optional
import re


class EmotionMapper:
    """情感标签映射器"""
    
    # 定义主要情感类别及其同义词
    EMOTION_CATEGORIES = {
        "思念": ["思念", "想念", "怀念", "思乡", "思恋", "眷恋", "思乡之情", "思乡与", "思念与", "思念、"],
        "孤独": ["孤独", "孤寂", "寂寞", "孤单", "孤苦", "寂寥", "孤独与", "孤独、"],
        "悲伤": ["悲伤", "悲哀", "悲痛", "悲凉", "悲苦", "悲愤", "悲愁", "哀伤", "哀怨", "哀痛", "哀伤与"],
        "忧愁": ["忧愁", "忧虑", "担忧", "愁苦", "惆怅", "愁闷", "愁思", "苦闷", "愁苦", "愁怅"],
        "无奈": ["无奈", "无助", "无力", "无望", "无可奈何", "失望", "无奈与"],
        "期待": ["期待", "期望", "盼望", "渴望", "向往", "希冀", "渴望"],
        "伤感": ["伤感", "感伤", "伤怀", "伤离", "伤别", "伤情", "伤感与", "伤感、"],
        "豪情": ["豪情", "豪迈", "豪放", "豪壮", "豪气", "壮志", "豪情壮志"],
        "闲适": ["闲适", "悠闲", "安逸", "舒适", "惬意", "自在", "悠然自得", "恬静"],
        "宁静": ["宁静", "平静", "安静", "宁和", "平和", "淡泊", "宁静、"],
        "喜悦": ["喜悦", "高兴", "快乐", "愉快", "欢乐", "欣喜", "愉悦"],
        "遗憾": ["遗憾", "惋惜", "可惜", "懊悔", "悔恨"],
        "离别": ["离别", "惜别", "分别", "告别", "离愁", "离愁别绪", "依依惜别"],
        "怀旧": ["怀旧", "怀念", "追忆", "回忆", "怀古"],
        "愤怒": ["愤怒", "愤慨", "愤懑", "愤恨", "气愤", "恼怒", "怨恨"],
        "热爱": ["热爱", "喜爱", "喜欢", "钟爱", "珍爱"],
        "感慨": ["感慨", "感叹", "感怀", "感念"],
        "其他": []  # 其他未分类的情感
    }
    
    @classmethod
    def map_to_category(cls, emotion_str: str) -> str:
        """
        将细粒度情感标签映射到粗粒度类别
        
        Args:
            emotion_str: 原始情感标签字符串
            
        Returns:
            映射后的粗粒度类别
        """
        if not emotion_str or not emotion_str.strip():
            return "其他"
        
        emotion_str = emotion_str.strip()
        
        # 提取所有情感词
        separators = [',', '，', ';', '；', ' ', '\t', '、', '与', '和', '及']
        emotion_words = []
        
        # 先按分隔符分割
        parts = re.split(r'[,，;；\s\t、与和及]+', emotion_str)
        for part in parts:
            part = part.strip()
            if part:
                emotion_words.append(part)
        
        # 如果没有分割出多个词，尝试直接匹配
        if not emotion_words:
            emotion_words = [emotion_str]
        
        # 统计每个类别匹配的分数
        category_scores = {}
        for category, keywords in cls.EMOTION_CATEGORIES.items():
            score = 0
            for word in emotion_words:
                for keyword in keywords:
                    if keyword in word or word in keyword:
                        score += 1
                        break
            if score > 0:
                category_scores[category] = score
        
        # 如果有匹配的类别，返回得分最高的
        if category_scores:
            # 按分数排序，如果分数相同，优先选择更常见的类别
            sorted_categories = sorted(
                category_scores.items(), 
                key=lambda x: (x[1], x[0] in ["思念", "孤独", "悲伤", "忧愁"]), 
                reverse=True
            )
            return sorted_categories[0][0]
        
        # 如果没有匹配，尝试模糊匹配
        for category, keywords in cls.EMOTION_CATEGORIES.items():
            for keyword in keywords:
                if keyword in emotion_str or emotion_str in keyword:
                    return category
        
        return "其他"
    
    @classmethod
    def normalize_emotion_label(cls, emotion_str: str) -> str:
        """
        标准化情感标签（提取主要情感）
        
        Args:
            emotion_str: 原始情感标签字符串
            
        Returns:
            标准化后的情感标签
        """
        if not emotion_str or not emotion_str.strip():
            return "其他"
        
        emotion_str = emotion_str.strip()
        
        # 先映射到类别
        category = cls.map_to_category(emotion_str)
        
        # 如果映射到"其他"，尝试提取第一个有意义的情感词
        if category == "其他":
            separators = [',', '，', ';', '；', ' ', '\t', '、', '与', '和', '及']
            parts = re.split(r'[,，;；\s\t、与和及]+', emotion_str)
            for part in parts:
                part = part.strip()
                if part and len(part) >= 2:  # 至少2个字符
                    # 再次尝试映射
                    mapped = cls.map_to_category(part)
                    if mapped != "其他":
                        return mapped
            # 如果还是找不到，返回原始字符串的前10个字符
            return emotion_str[:10] if len(emotion_str) > 10 else emotion_str
        
        return category
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """获取所有情感类别"""
        return [cat for cat in cls.EMOTION_CATEGORIES.keys() if cat != "其他"]
    
    @classmethod
    def create_label_mapping(cls, data: List[Dict]) -> Dict[str, int]:
        """
        从数据创建标签映射
        
        Args:
            data: 数据列表
            
        Returns:
            标签到ID的映射字典
        """
        # 收集所有标准化后的标签
        normalized_labels = set()
        
        for item in data:
            if 'emotion' in item and item['emotion']:
                emotion_str = str(item['emotion']).strip()
                if emotion_str:
                    normalized = cls.normalize_emotion_label(emotion_str)
                    normalized_labels.add(normalized)
        
        # 如果没有标签，使用默认标签
        if not normalized_labels:
            normalized_labels = {"思念", "孤独", "悲伤", "忧愁", "其他"}
        
        # 创建映射（按字母顺序排序，但"其他"放在最后）
        sorted_labels = sorted([l for l in normalized_labels if l != "其他"])
        if "其他" in normalized_labels:
            sorted_labels.append("其他")
        
        emotion_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        
        return emotion_to_id









