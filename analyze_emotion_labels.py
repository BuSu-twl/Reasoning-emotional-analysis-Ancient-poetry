"""
分析情感标签分布，为重新划分标签做准备
"""
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

def load_all_data(data_dir: str = "data/train-data") -> List[Dict[str, Any]]:
    """加载所有训练数据"""
    data_dir = Path(data_dir)
    all_data = []
    json_files = list(data_dir.rglob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                all_data.append(data)
        except Exception as e:
            print(f"加载失败 {json_file}: {e}")
    
    return all_data

def analyze_emotion_distribution(data: List[Dict[str, Any]]):
    """分析情感标签分布"""
    emotion_counter = Counter()
    emotion_full_counter = Counter()
    
    for item in data:
        if 'emotion' in item and item['emotion']:
            emotion_str = str(item['emotion']).strip()
            if emotion_str:
                # 统计完整标签
                emotion_full_counter[emotion_str] += 1
                
                # 提取主要情感词（去除标点和连接词）
                import re
                # 分割多种分隔符
                parts = re.split(r'[,，;；、\s]+', emotion_str)
                for part in parts:
                    part = part.strip()
                    if part:
                        # 提取核心情感词（去除"对"、"的"等修饰词）
                        core_words = []
                        for word in part.split():
                            # 去除常见的修饰词
                            if word not in ['对', '的', '与', '和', '及', '以及', '等']:
                                core_words.append(word)
                        if core_words:
                            emotion_counter[' '.join(core_words)] += 1
    
    print("=" * 60)
    print("情感标签分析报告")
    print("=" * 60)
    print(f"\n总数据量: {len(data)} 条")
    print(f"有情感标签的数据: {sum(1 for item in data if item.get('emotion'))} 条")
    print(f"唯一完整标签数: {len(emotion_full_counter)}")
    print(f"提取的核心情感词数: {len(emotion_counter)}")
    
    print("\n" + "=" * 60)
    print("最常见的完整标签（前20个）:")
    print("=" * 60)
    for emotion, count in emotion_full_counter.most_common(20):
        print(f"{count:3d}次: {emotion}")
    
    print("\n" + "=" * 60)
    print("最常见的核心情感词（前30个）:")
    print("=" * 60)
    for emotion, count in emotion_counter.most_common(30):
        print(f"{count:3d}次: {emotion}")
    
    return emotion_full_counter, emotion_counter

def suggest_label_mapping(emotion_counter: Counter) -> Dict[str, str]:
    """建议标签映射（将细粒度标签映射到粗粒度标签）"""
    
    # 定义主要情感类别
    main_categories = {
        "思念": ["思念", "想念", "怀念", "思乡", "思恋", "眷恋"],
        "孤独": ["孤独", "孤寂", "寂寞", "孤单", "孤苦", "寂寥"],
        "悲伤": ["悲伤", "悲哀", "悲痛", "悲凉", "悲苦", "悲愤", "悲愁", "哀伤", "哀怨", "哀痛"],
        "忧愁": ["忧愁", "忧虑", "担忧", "愁苦", "惆怅", "愁闷", "愁思", "苦闷"],
        "愤怒": ["愤怒", "愤慨", "愤懑", "愤恨", "气愤", "恼怒"],
        "无奈": ["无奈", "无助", "无力", "无望", "无可奈何", "失望"],
        "期待": ["期待", "期望", "盼望", "渴望", "向往", "希冀"],
        "伤感": ["伤感", "感伤", "伤怀", "伤离", "伤别", "伤情"],
        "豪情": ["豪情", "豪迈", "豪放", "豪壮", "豪气", "壮志"],
        "陶醉": ["陶醉", "沉醉", "迷恋", "痴迷", "着迷"],
        "闲适": ["闲适", "悠闲", "安逸", "舒适", "惬意", "自在"],
        "赞美": ["赞美", "称赞", "赞赏", "赞叹", "颂扬", "歌颂"],
        "遗憾": ["遗憾", "惋惜", "可惜", "懊悔", "悔恨"],
        "喜悦": ["喜悦", "高兴", "快乐", "愉快", "欢乐", "欣喜"],
        "宁静": ["宁静", "平静", "安静", "宁和", "平和", "淡泊"],
        "热爱": ["热爱", "喜爱", "喜欢", "钟爱", "珍爱"],
        "感慨": ["感慨", "感叹", "感怀", "感念"],
        "离别": ["离别", "惜别", "分别", "告别"],
        "怀旧": ["怀旧", "怀念", "追忆", "回忆"],
        "其他": []  # 其他未分类的情感
    }
    
    # 创建反向映射（从核心词到主类别）
    word_to_category = {}
    for category, words in main_categories.items():
        for word in words:
            word_to_category[word] = category
    
    # 统计每个类别的数量
    category_counter = Counter()
    for word, count in emotion_counter.items():
        # 查找匹配的类别
        matched = False
        for key_word, category in word_to_category.items():
            if key_word in word:
                category_counter[category] += count
                matched = True
                break
        if not matched:
            category_counter["其他"] += count
    
    print("\n" + "=" * 60)
    print("建议的标签映射（粗粒度分类）:")
    print("=" * 60)
    for category, count in category_counter.most_common():
        print(f"{count:3d}次: {category}")
    
    return main_categories

if __name__ == "__main__":
    # 加载数据
    print("正在加载数据...")
    data = load_all_data("data/train-data")
    
    # 分析标签分布
    emotion_full_counter, emotion_counter = analyze_emotion_distribution(data)
    
    # 建议标签映射
    label_mapping = suggest_label_mapping(emotion_counter)
    
    # 保存分析结果
    output = {
        "total_samples": len(data),
        "unique_full_labels": len(emotion_full_counter),
        "unique_core_words": len(emotion_counter),
        "top_full_labels": dict(emotion_full_counter.most_common(50)),
        "top_core_words": dict(emotion_counter.most_common(50)),
        "suggested_categories": label_mapping
    }
    
    with open("emotion_analysis_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n分析结果已保存到: emotion_analysis_result.json")









