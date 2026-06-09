"""
将TXT目录下的txt文件转换成训练集格式
根据train_data目录下的训练集格式进行转换
"""
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_poem_type(content: str) -> str:
    """
    根据内容判断诗词类型
    
    Args:
        content: 诗词内容
        
    Returns:
        类型字符串：唐诗/宋词/其他
    """
    # 移除标点和空格
    clean_content = re.sub(r'[，。、；：！？\s\n]', '', content)
    
    # 计算每行的字符数
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return "其他"
    
    # 统计每行字数
    line_lengths = [len(re.sub(r'[，。、；：！？\s]', '', line)) for line in lines]
    
    # 判断规则
    avg_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0
    
    # 唐诗：通常是4句或8句，每句5字或7字
    # 宋词：行数不固定，每行字数变化较大
    if len(lines) == 4 or len(lines) == 8:
        if all(5 <= length <= 7 for length in line_lengths):
            return "唐诗"
    
    # 如果行数较多且字数变化大，可能是宋词
    if len(lines) > 4 and max(line_lengths) - min(line_lengths) > 3:
        return "宋词"
    
    # 默认判断为唐诗
    if all(4 <= length <= 7 for length in line_lengths):
        return "唐诗"
    
    return "其他"


def classify_poem_by_length(content: str) -> str:
    """
    根据诗词长度和格式分类到具体子类别
    
    Args:
        content: 诗词内容
        
    Returns:
        子类别：五言绝句/七言绝句/五言律诗/七言律诗/北宋词/南宋词/其他
    """
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return "其他"
    
    # 计算每行字数（去除标点）
    line_lengths = [len(re.sub(r'[，。、；：！？\s]', '', line)) for line in lines]
    avg_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0
    
    # 唐诗分类
    if len(lines) == 4:  # 绝句
        if 4 <= avg_length <= 5:
            return "五言绝句"
        elif 6 <= avg_length <= 7:
            return "七言绝句"
    elif len(lines) == 8:  # 律诗
        if 4 <= avg_length <= 5:
            return "五言律诗"
        elif 6 <= avg_length <= 7:
            return "七言律诗"
    
    # 宋词分类（简单判断）
    poem_type = detect_poem_type(content)
    if poem_type == "宋词":
        # 根据内容长度简单判断（可以后续优化）
        if len(content) < 100:
            return "北宋词"
        else:
            return "南宋词"
    
    return "其他"


def extract_title_from_content(content: str) -> str:
    """
    从内容中提取可能的标题
    
    Args:
        content: 诗词内容
        
    Returns:
        标题字符串
    """
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return "无题"
    
    # 取第一行的前几个字作为标题（如果第一行较短）
    first_line = lines[0]
    # 移除标点
    first_line_clean = re.sub(r'[，。、；：！？\s]', '', first_line)
    
    # 如果第一行很短（2-4字），可能是标题
    if 2 <= len(first_line_clean) <= 4:
        return first_line_clean
    
    # 否则取前2-3个字，但避免取到不完整的词
    if len(first_line_clean) >= 2:
        # 优先取2个字（更常见）
        if len(first_line_clean) >= 2:
            return first_line_clean[:2]
    
    return "无题"


def parse_txt_file(txt_path: Path) -> List[Dict[str, Any]]:
    """
    解析TXT文件，提取诗词内容
    
    Args:
        txt_path: TXT文件路径
        
    Returns:
        诗词数据列表
    """
    logger.info(f"正在解析文件: {txt_path}")
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"读取文件失败 {txt_path}: {e}")
        return []
    
    # 按空行分割诗词
    # 先按双换行分割
    poems_raw = re.split(r'\n\s*\n+', content)
    
    # 如果分割后只有一个，尝试按单换行分割（可能是连续的诗）
    if len(poems_raw) == 1:
        # 尝试识别：如果连续多行且每行字数相似，可能是一首诗
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        poems_raw = []
        current_poem = []
        
        for i, line in enumerate(lines):
            line_clean = re.sub(r'[，。、；：！？\s]', '', line)
            # 如果当前行是诗词内容（有标点或字数在合理范围）
            if '，' in line or '。' in line or (4 <= len(line_clean) <= 7):
                current_poem.append(line)
            else:
                # 如果当前行不是诗词内容，且之前有内容，保存之前的诗
                if current_poem:
                    poems_raw.append('\n'.join(current_poem))
                    current_poem = []
                # 如果当前行可能是标题（短且无标点）
                if len(line_clean) <= 6 and '，' not in line and '。' not in line:
                    # 可能是标题，暂时忽略
                    pass
        
        # 保存最后一首诗
        if current_poem:
            poems_raw.append('\n'.join(current_poem))
    
    poems = []
    for poem_raw in poems_raw:
        poem_raw = poem_raw.strip()
        if not poem_raw:
            continue
        
        # 清理内容：移除多余空格，保留换行
        lines = [line.strip() for line in poem_raw.split('\n') if line.strip()]
        if not lines:
            continue
        
        # 检查第一行是否是标题（短且无标点）
        title = None
        content_lines = lines
        
        first_line = lines[0]
        first_line_clean = re.sub(r'[，。、；：！？\s]', '', first_line)
        
        # 如果第一行可能是标题（2-6字，无标点，且不是诗词内容）
        if (2 <= len(first_line_clean) <= 6 and 
            '，' not in first_line and '。' not in first_line and
            len(lines) > 1 and
            # 确保不是诗词的第一句（通常诗词第一句会有标点或字数较多）
            not any(c in first_line for c in ['，', '。', '？', '！', '：'])):
            title = first_line_clean
            content_lines = lines[1:]
        
        # 如果没有识别到标题，从内容中提取（取第一句的前2-3个字）
        if not title:
            title = extract_title_from_content('\n'.join(content_lines))
        
        # 构建内容（保留换行）
        content = '\n'.join(content_lines)
        
        # 移除注释（如：一作：xxx）
        content = re.sub(r'\([^)]*一作[^)]*\)', '', content)
        content = re.sub(r'（[^）]*一作[^）]*）', '', content)
        content = re.sub(r'\([^)]*通[^)]*\)', '', content)
        content = re.sub(r'（[^）]*通[^）]*）', '', content)
        
        # 清理多余空格
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r' \n', '\n', content)
        content = content.strip()
        
        if not content or len(content) < 4:
            continue
        
        poems.append({
            'title': title,
            'content': content,
            'keywords': {},  # 可选，后续可以手动添加
            'trans': '',  # 可选，后续可以手动添加
            'emotion': ''  # 可选，后续可以手动添加或使用分析工具
        })
    
    logger.info(f"从 {txt_path.name} 中提取了 {len(poems)} 首诗")
    return poems


def save_to_train_data(poems: List[Dict[str, Any]], output_dir: Path, category: str = None):
    """
    保存诗词数据到指定目录
    
    Args:
        poems: 诗词数据列表
        output_dir: 输出目录
        category: 分类类别（如：五言绝句、七言绝句等）
    """
    if not poems:
        logger.warning("没有诗词数据需要保存")
        return
    
    # 如果没有指定类别，根据内容自动分类
    if not category:
        # 统计各类别数量
        category_counts = defaultdict(int)
        for poem in poems:
            cat = classify_poem_by_length(poem['content'])
            category_counts[cat] += 1
        
        # 使用最多的类别
        if category_counts:
            category = max(category_counts.items(), key=lambda x: x[1])[0]
        else:
            category = "其他"
    
    # 确定保存路径（保存到 add_data 目录下，按类别分类）
    if category in ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]:
        save_path = output_dir / "唐诗" / category / "train.json"
    elif category in ["北宋词", "南宋词"]:
        save_path = output_dir / "宋词" / category / "train.json"
    else:
        # 其他类别统一保存到 add_data 根目录
        save_path = output_dir / "train.json"
    
    # 创建目录
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果文件已存在，读取现有数据
    existing_data = []
    if save_path.exists():
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        except Exception as e:
            logger.warning(f"读取现有文件失败 {save_path}: {e}")
            existing_data = []
    
    # 合并数据（避免重复：根据标题和内容判断）
    existing_content_set = set()
    for item in existing_data:
        # 使用标题+内容的前20个字符作为唯一标识
        content_preview = item.get('content', '')[:20]
        title = item.get('title', '')
        existing_content_set.add((title, content_preview))
    
    new_poems = []
    for poem in poems:
        content_preview = poem.get('content', '')[:20]
        title = poem.get('title', '')
        if (title, content_preview) not in existing_content_set:
            new_poems.append(poem)
            existing_content_set.add((title, content_preview))
    
    if new_poems:
        existing_data.extend(new_poems)
        
        # 保存
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"已保存 {len(new_poems)} 首新诗到 {save_path}")
        logger.info(f"文件中共有 {len(existing_data)} 首诗")
    else:
        logger.info(f"没有新诗需要添加（可能已存在）")


def convert_txt_to_train_data(txt_dir: str = "data/TXT", output_dir: str = "data/add_data"):
    """
    将TXT目录下的所有txt文件转换成训练集格式
    
    Args:
        txt_dir: TXT文件目录
        output_dir: 输出目录（训练集目录）
    """
    txt_dir_path = Path(txt_dir)
    output_dir_path = Path(output_dir)
    
    if not txt_dir_path.exists():
        logger.error(f"TXT目录不存在: {txt_dir_path}")
        return
    
    # 查找所有txt文件
    txt_files = list(txt_dir_path.glob("*.txt"))
    if not txt_files:
        logger.warning(f"在 {txt_dir_path} 中未找到txt文件")
        return
    
    logger.info(f"找到 {len(txt_files)} 个txt文件")
    
    all_poems = []
    
    # 处理每个txt文件
    for txt_file in txt_files:
        poems = parse_txt_file(txt_file)
        if poems:
            all_poems.extend(poems)
    
    logger.info(f"共提取了 {len(all_poems)} 首诗")
    
    # 按类别分组保存
    poems_by_category = defaultdict(list)
    for poem in all_poems:
        category = classify_poem_by_length(poem['content'])
        poems_by_category[category].append(poem)
    
    # 保存到对应目录
    for category, poems in poems_by_category.items():
        save_to_train_data(poems, output_dir_path, category)
    
    logger.info("=" * 60)
    logger.info("转换完成！")
    logger.info(f"共处理 {len(all_poems)} 首诗")
    logger.info("分类统计：")
    for category, poems in poems_by_category.items():
        logger.info(f"  {category}: {len(poems)} 首")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将TXT文件转换成训练集格式")
    parser.add_argument(
        "--txt-dir",
        type=str,
        default="data/TXT",
        help="TXT文件目录（默认: data/TXT）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/add_data",
        help="输出目录（默认: data/add_data）"
    )
    
    args = parser.parse_args()
    
    convert_txt_to_train_data(args.txt_dir, args.output_dir)

