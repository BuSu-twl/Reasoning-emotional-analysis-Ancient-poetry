import json
import os
import time
import logging
from functools import lru_cache
from aliyunsdkcore.client import AcsClient
from aliyunsdkalinlp.request.v20200629.GetWsCustomizedChGeneralRequest import GetWsCustomizedChGeneralRequest

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置阿里云访问凭证（建议通过环境变量设置，避免密钥泄露）
access_key_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
access_key_secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')

# 初始化AcsClient（仅在凭证存在时创建）
client = None
if access_key_id and access_key_secret:
    try:
        client = AcsClient(
            access_key_id,
            access_key_secret,
            "cn-hangzhou"  # 阿里云NLP服务所在区域
        )
    except Exception as e:
        logger.warning(f"初始化AcsClient失败: {e}")
        client = None
else:
    logger.warning("未找到阿里云访问凭证，AcsClient未初始化")

# 定义古诗常见实体类别
POEM_ENTITIES = {
    '自然景物': ['山', '水', '月', '光', '风', '雨', '花', '鸟', '云', '雪', '星', '日', '天', '地'],
    '器物': ['床', '船', '马', '剑', '酒', '琴', '书', '灯', '窗', '门', '桌', '椅'],
    '地点': ['家', '乡', '城', '楼', '寺', '园', '庭', '宫', '殿', '台', '阁'],
    '时间': ['春', '夏', '秋', '冬', '晨', '午', '暮', '夜', '晓', '夕']
}


def get_aliyun_client(access_key_id=None, access_key_secret=None):
    """获取或创建阿里云客户端"""
    # 如果提供了参数，使用参数创建新客户端
    if access_key_id and access_key_secret:
        try:
            return AcsClient(
                access_key_id,
                access_key_secret,
                "cn-hangzhou"
            )
        except Exception as e:
            logger.error(f"创建AcsClient失败: {e}")
            return None
    
    # 否则使用全局客户端
    global client
    if client is None:
        # 尝试从环境变量创建
        env_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
        env_secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
        if env_id and env_secret:
            try:
                client = AcsClient(env_id, env_secret, "cn-hangzhou")
            except Exception as e:
                logger.error(f"从环境变量创建AcsClient失败: {e}")
                return None
    return client


def _segment_poem_impl(poem_text, acs_client):
    """实际执行分词操作的内部函数"""
    if acs_client is None:
        logger.error("AcsClient未初始化，无法进行分词")
        return None
    
    request = GetWsCustomizedChGeneralRequest()
    request.set_Text(poem_text)
    request.set_ServiceCode("alinlp")  # 固定值
    request.set_TokenizerId("GENERAL_CHN")  # 中文分词器
    request.set_OutType("1")  # 0:大粒度, 1:中粒度(默认), 2:小粒度

    try:
        response = acs_client.do_action_with_exception(request)
        result = json.loads(response)

        # 验证返回结果结构
        if not isinstance(result, dict):
            logger.error(f"API返回结果不是字典格式 - {result}")
            return None

        # 检查是否有错误信息
        if "Code" in result or "Message" in result:
            logger.error(f"阿里云NLP返回错误: {result}")
            return None

        # 确保Data字段存在且为字典
        if "Data" in result:
            if isinstance(result["Data"], str):
                try:
                    result["Data"] = json.loads(result["Data"])
                except json.JSONDecodeError:
                    logger.error(f"Data字段无法解析为字典 - {result['Data']}")
                    return None
            elif not isinstance(result["Data"], dict):
                logger.error(f"Data字段类型错误 - {type(result['Data'])}")
                return None
        else:
            logger.error("返回结果中缺少Data字段")
            return None

        return result
    except Exception as e:
        logger.error(f"调用阿里云NLP分词API时出错: {e}")
        return None


@lru_cache(maxsize=100)
def _segment_poem_cached(poem_text):
    """缓存版本的分词函数（仅缓存poem_text）"""
    acs_client = get_aliyun_client()
    return _segment_poem_impl(poem_text, acs_client)


def segment_poem(poem_text, acs_client=None):
    """调用阿里云NLP基础服务进行中文分词（带缓存）"""
    # 如果提供了客户端，直接使用（不缓存）
    if acs_client is not None:
        return _segment_poem_impl(poem_text, acs_client)
    
    # 否则使用缓存版本
    return _segment_poem_cached(poem_text)


def analyze_poem_sentiment(poem_text, segmentation_result=None, acs_client=None):
    """基于分词结果进行情感分析（返回中文标签，与训练模型兼容）"""
    # 如果未提供分词结果，先进行分词
    if segmentation_result is None:
        segmentation_result = segment_poem(poem_text, acs_client)

    # 检查分词结果是否有效
    if not segmentation_result or "Data" not in segmentation_result:
        return {"sentiment": "其他", "confidence": 0}

    # 确保Data是字典类型
    if not isinstance(segmentation_result["Data"], dict):
        logger.error(f"Data字段类型错误 - {type(segmentation_result['Data'])}")
        return {"sentiment": "其他", "confidence": 0}

    # 检查result字段是否存在且为列表
    if "result" not in segmentation_result["Data"] or not isinstance(segmentation_result["Data"]["result"], list):
        logger.error(f"result字段缺失或类型错误 - {segmentation_result['Data'].get('result')}")
        return {"sentiment": "其他", "confidence": 0}

    # 提取分词结果
    words = []
    for item in segmentation_result["Data"]["result"]:
        if isinstance(item, dict) and "word" in item:
            words.append(item["word"])
        else:
            logger.warning(f"分词结果项格式错误 - {item}")

    # 使用改进的情感映射器进行更准确的分析
    try:
        from utils.emotion_mapper import EmotionMapper
        
        # 定义情感关键词库（与EmotionMapper的类别对应）
        # 包含单字词和复合词
        emotion_keywords = {
            "思念": ["思", "念", "想", "怀", "乡", "恋", "眷", "思乡", "思念", "怀念", "怀乡"],
            "孤独": ["孤", "独", "寂", "寞", "单", "寥", "孤独", "寂寞", "孤单", "孤寂", "寂寥"],
            "悲伤": ["悲", "哀", "伤", "痛", "苦", "凄", "凉", "断肠", "断肠人", "悲凉", "凄凉", "悲伤", "哀伤", "痛苦"],
            "忧愁": ["愁", "忧", "虑", "惆", "怅", "闷", "忧愁", "惆怅", "愁苦", "忧闷"],
            "无奈": ["无", "奈", "失", "望", "无望", "无奈", "失望", "无助"],
            "期待": ["期", "待", "盼", "望", "渴", "向", "往", "期待", "盼望", "渴望", "向往"],
            "伤感": ["伤", "感", "怀", "离", "别", "伤感", "离别", "感伤", "离愁"],
            "豪情": ["豪", "迈", "放", "壮", "气", "志", "豪迈", "豪情", "壮志", "豪放"],
            "闲适": ["闲", "适", "悠", "逸", "惬", "意", "闲适", "悠闲", "安逸", "惬意"],
            "宁静": ["静", "宁", "淡", "和", "安", "泊", "宁静", "安静", "淡泊", "平和"],
            "喜悦": ["喜", "乐", "欢", "笑", "悦", "欣", "喜悦", "欢乐", "快乐", "欢欣"],
            "遗憾": ["遗", "憾", "惋", "惜", "悔", "遗憾", "惋惜", "后悔"],
            "离别": ["离", "别", "分", "告", "惜别", "离别", "分别", "告别"],
            "怀旧": ["怀", "旧", "忆", "追", "怀旧", "回忆", "追忆"],
            "愤怒": ["愤", "怒", "恨", "怨", "恼", "愤怒", "怨恨", "恼怒"],
            "热爱": ["热", "爱", "喜", "钟", "珍", "热爱", "喜爱", "钟爱", "珍惜"],
            "感慨": ["感", "慨", "叹", "怀", "感慨", "感叹", "慨叹"]
        }
        
        # 定义意象词库（古诗词中常见的意象词，往往带有特定情感色彩）
        # 扩展版：包含更多古诗词常见意象
        image_words = {
            "悲伤": ["枯藤", "老树", "昏鸦", "瘦马", "夕阳", "残阳", "暮色", "西风", "秋风", "寒鸦", 
                    "断肠", "断肠人", "天涯", "天涯路", "落花", "流水", "孤雁", "孤舟", "残月", "冷月",
                    "寒山", "荒原", "衰草", "败叶", "凋零", "零落", "萧瑟", "凄凉", "荒凉", "寂寥",
                    "暮霭", "寒烟", "孤鸿", "孤影", "残灯", "寒灯", "夜雨", "秋雨", "霜雪", "寒霜"],
            "忧愁": ["愁云", "愁绪", "暮云", "阴云", "寒江", "冷月", "孤灯", "空山", "愁眉", "愁容",
                    "愁思", "愁肠", "愁苦", "愁闷", "忧心", "忧思", "忧虑", "忧忡", "惆怅", "怅惘",
                    "苦闷", "郁闷", "烦忧", "烦闷", "焦虑", "不安", "忐忑", "彷徨"],
            "离别": ["长亭", "短亭", "杨柳", "折柳", "南浦", "灞桥", "阳关", "关山", "送别", "惜别",
                    "离别", "分别", "告别", "分手", "远行", "远游", "离愁", "别绪", "离情", "别情",
                    "离歌", "别曲", "离亭", "别馆", "离筵", "别宴"],
            "思念": ["明月", "归雁", "鸿雁", "书信", "家书", "故园", "故里", "故人", "故乡", "家乡",
                    "思乡", "怀乡", "乡愁", "乡思", "乡情", "归心", "归思", "归梦", "思君", "念君",
                    "忆君", "怀君", "思亲", "念亲", "忆亲", "怀亲", "思友", "念友", "忆友", "怀友"],
            "孤独": ["孤雁", "孤舟", "孤灯", "独坐", "独行", "空山", "空庭", "孤影", "独影", "孤身",
                    "独身", "孤寂", "独寂", "孤苦", "独苦", "孤零", "独零", "孤寡", "独寡", "孤单",
                    "独单", "孤凄", "独凄", "孤冷", "独冷", "孤寒", "独寒", "孤清", "独清"],
            "豪情": ["大江", "长河", "高山", "险峰", "战马", "宝剑", "金戈", "铁马", "豪迈", "豪放",
                    "豪壮", "豪气", "壮志", "雄心", "壮心", "雄心", "豪情", "豪志", "豪言", "豪语",
                    "豪举", "豪行", "豪饮", "豪歌", "豪笑", "豪情", "豪兴", "豪意", "豪心", "豪胆"],
            "宁静": ["青山", "绿水", "白云", "清风", "明月", "松竹", "梅兰", "幽谷", "幽静", "宁静",
                    "安静", "寂静", "清静", "恬静", "静谧", "宁谧", "安谧", "静穆", "肃静", "沉静",
                    "淡泊", "淡雅", "清淡", "清雅", "清幽", "幽雅", "雅静", "闲静", "静好", "静美"],
            "喜悦": ["春风", "春雨", "春花", "新绿", "莺啼", "燕舞", "鸟语", "花香", "欢快", "快乐",
                    "愉快", "欢乐", "欣喜", "喜悦", "高兴", "开心", "欢欣", "欢畅", "欢愉", "欢娱",
                    "欢歌", "欢笑", "欢庆", "欢聚", "欢宴", "欢会", "欢心", "欢意", "欢情", "欢兴"]
        }
        
        # 统计各类情感关键词出现次数（改进版：考虑上下文和位置）
        emotion_scores = {}
        poem_length = len(poem_text)
        
        for category, keywords in emotion_keywords.items():
            score = 0
            # 先检查复合词（更准确，权重更高）
            for keyword in keywords:
                if len(keyword) > 1:
                    # 计算关键词在文本中的位置（开头和结尾的情感词权重更高）
                    positions = []
                    start = 0
                    while True:
                        pos = poem_text.find(keyword, start)
                        if pos == -1:
                            break
                        positions.append(pos)
                        start = pos + 1
                    
                    for pos in positions:
                        # 开头或结尾的情感词权重更高（古诗词的情感往往在开头或结尾表达）
                        if pos < poem_length * 0.2 or pos > poem_length * 0.8:
                            score += 3  # 开头或结尾的复合词权重最高
                        else:
                            score += 2  # 中间的复合词权重较高
                        
                        # 检查是否在句首或句尾（标点符号后或前）
                        if pos > 0 and poem_text[pos-1] in ['，', '。', '、', '；', '：', '！', '？', '\n']:
                            score += 1  # 句首额外加分
                        if pos + len(keyword) < poem_length and poem_text[pos + len(keyword)] in ['，', '。', '、', '；', '：', '！', '？', '\n']:
                            score += 1  # 句尾额外加分
            
            # 再检查单字词（权重较低，但也要考虑）
            for word in words:
                for keyword in keywords:
                    if len(keyword) == 1 and (keyword in word or word in keyword):
                        score += 1
                        break
            
            if score > 0:
                emotion_scores[category] = score
        
        # 统计意象词（权重较高，考虑意象词的组合效果）
        for category, images in image_words.items():
            image_count = 0
            found_images = []
            for image in images:
                if image in poem_text:
                    image_count += 1
                    found_images.append(image)
            
            if image_count > 0:
                # 基础权重
                base_score = image_count * 2
                # 如果多个意象词同时出现，说明情感更强烈（组合效应）
                if image_count >= 3:
                    base_score += 3  # 3个以上意象词额外加分
                elif image_count >= 2:
                    base_score += 1  # 2个意象词额外加分
                
                emotion_scores[category] = emotion_scores.get(category, 0) + base_score
        
        # 特殊模式识别（在原文中查找）
        special_patterns = {
            "思念": ["思", "念", "想", "怀", "乡", "思乡", "怀乡"],
            "孤独": ["孤", "独", "寂", "孤独", "寂寞", "孤单"],
            "悲伤": ["悲", "哀", "伤", "断肠", "断肠人", "悲凉", "凄凉"],
            "忧愁": ["愁", "忧", "惆", "忧愁", "惆怅"],
            "豪情": ["豪", "壮", "志", "豪迈", "豪情", "壮志"],
            "宁静": ["静", "淡", "泊", "宁静", "淡泊"],
            "期待": ["期", "盼", "望", "期待", "盼望"]
        }
        
        for category, patterns in special_patterns.items():
            # 优先匹配复合词
            for pattern in patterns:
                if len(pattern) > 1 and pattern in poem_text:
                    emotion_scores[category] = emotion_scores.get(category, 0) + 2
                    break
        
        # 确定主要情感（改进版：考虑情感强度和区分度）
        if emotion_scores:
            # 按分数排序
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            main_emotion, max_score = sorted_emotions[0]
            total_score = sum(emotion_scores.values())
            
            # 计算第二高分（用于判断区分度）
            second_score = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0
            score_gap = max_score - second_score  # 分数差距
            
            # 改进的置信度计算
            base_confidence = max_score / total_score if total_score > 0 else 0.5
            
            # 根据最高分调整置信度
            if max_score >= 10:
                confidence = min(0.95, base_confidence + 0.25)
            elif max_score >= 6:
                confidence = min(0.9, base_confidence + 0.2)
            elif max_score >= 3:
                confidence = min(0.85, base_confidence + 0.15)
            elif max_score >= 2:
                confidence = min(0.8, base_confidence + 0.1)
            else:
                confidence = max(0.5, min(0.75, base_confidence + 0.05))
            
            # 如果分数差距大，说明情感更明确，提高置信度
            if score_gap >= 5:
                confidence = min(0.95, confidence + 0.1)
            elif score_gap >= 3:
                confidence = min(0.9, confidence + 0.05)
            
            # 如果只有一个情感类别匹配，适当提高置信度
            if len(emotion_scores) == 1:
                confidence = min(0.9, confidence + 0.1)
            
            # 如果最高分明显高于其他所有分数之和，进一步提高置信度
            other_scores = total_score - max_score
            if max_score > other_scores * 2:
                confidence = min(0.95, confidence + 0.05)
            
            return {"sentiment": main_emotion, "confidence": confidence}
        else:
            # 如果没有匹配到任何情感关键词，尝试基于意象词进行基础判断
            # 扩展的悲伤意象检查
            sad_images = ["枯", "老", "昏", "瘦", "夕", "断肠", "天涯", "西风", "秋", "寒", "冷", 
                         "残", "衰", "败", "凋", "零", "落", "孤", "独", "寂", "寥", "凄", "凉"]
            sad_count = sum(1 for img in sad_images if img in poem_text)
            if sad_count >= 3:
                return {"sentiment": "悲伤", "confidence": 0.65}
            elif sad_count >= 2:
                return {"sentiment": "悲伤", "confidence": 0.55}
            elif sad_count >= 1:
                return {"sentiment": "悲伤", "confidence": 0.45}
            
            return {"sentiment": "其他", "confidence": 0.3}
    
    except ImportError:
        # 回退到原始方法（但返回中文标签）
        # 定义情感关键词库（扩展版）
        sad_words = ["愁", "泪", "悲", "伤", "思", "孤", "独", "寒", "冷", "暮", "夜", "苦", "痛", "哀", "怨", "恨", "凄", "凉", "寂"]
        happy_words = ["喜", "乐", "欢", "笑", "春", "花", "明", "暖", "晴", "美", "好", "兴", "悦", "欣", 
                       "豪", "达", "旷", "洒", "脱", "放", "畅", "快", "胜", "强", "勇", "敢", "无", "畏", "任"]
        peaceful_words = ["静", "闲", "宁", "淡", "和", "安", "悠", "清", "远", "空", "禅", "悟", "淡泊"]
        
        # 扩展：包含常见的情感表达词
        def match_sentiment_word(word, word_list):
            """检查词是否在情感词列表中，或包含情感词"""
            if not word:
                return False
            if word in word_list:
                return True
            if len(word) > 1:
                for emotion_word in word_list:
                    if len(emotion_word) > 1 and emotion_word in word:
                        return True
            return False

        # 统计情感关键词出现次数
        sad_count = sum(1 for word in words if match_sentiment_word(word, sad_words))
        happy_count = sum(1 for word in words if match_sentiment_word(word, happy_words))
        peaceful_count = sum(1 for word in words if match_sentiment_word(word, peaceful_words))

        # 特殊模式识别
        special_patterns = {
            "豪情": ["谁怕", "胜", "轻胜", "任", "豪", "达", "旷"],
            "宁静": ["任平生", "淡泊", "宁静"]
        }
        
        has_haoyi_pattern = any(pattern in poem_text for pattern in special_patterns["豪情"])
        has_peaceful_pattern = any(pattern in poem_text for pattern in special_patterns["宁静"])
        
        # 判断情感倾向（映射到中文标签）
        total = sad_count + happy_count + peaceful_count
        if total == 0:
            if has_haoyi_pattern:
                return {"sentiment": "豪情", "confidence": 0.65}
            elif has_peaceful_pattern:
                return {"sentiment": "宁静", "confidence": 0.55}
            return {"sentiment": "其他", "confidence": 0.3}
        
        # 调整计数
        if has_haoyi_pattern:
            happy_count += 1
            total += 1
        if has_peaceful_pattern and "任平生" in poem_text:
            peaceful_count += 1
            total += 1

        # 计算置信度
        max_count = max(sad_count, happy_count, peaceful_count)
        base_confidence = max_count / total if total > 0 else 0.5
        
        if max_count >= 3:
            confidence = min(0.85, base_confidence + 0.15)
        elif max_count >= 2:
            confidence = min(0.8, base_confidence + 0.1)
        else:
            confidence = max(0.4, min(0.7, base_confidence))
        
        if total <= 2:
            confidence = max(0.3, confidence - 0.2)

        # 确定情感标签（返回中文）
        if sad_count > happy_count and sad_count > peaceful_count:
            return {"sentiment": "悲伤", "confidence": confidence}
        elif happy_count > sad_count and happy_count > peaceful_count:
            return {"sentiment": "喜悦", "confidence": confidence}
        elif peaceful_count > sad_count and peaceful_count > happy_count:
            return {"sentiment": "宁静", "confidence": confidence}
        else:
            # 多种情感并存的情况
            if sad_count > 0 and happy_count > 0:
                return {"sentiment": "其他", "confidence": confidence}
            elif sad_count > 0 and peaceful_count > 0:
                return {"sentiment": "忧愁", "confidence": confidence}
            elif happy_count > 0 and peaceful_count > 0:
                return {"sentiment": "闲适", "confidence": confidence}
            else:
                return {"sentiment": "其他", "confidence": 0.3}


def extract_poem_keywords(poem_text, segmentation_result=None, acs_client=None):
    """提取古诗词关键词"""
    # 如果未提供分词结果，先进行分词
    if segmentation_result is None:
        segmentation_result = segment_poem(poem_text, acs_client)

    # 检查分词结果是否有效
    if not segmentation_result or "Data" not in segmentation_result:
        return {"keywords": []}

    # 确保Data是字典类型
    if not isinstance(segmentation_result["Data"], dict):
        return {"keywords": []}

    # 检查result字段是否存在且为列表
    if "result" not in segmentation_result["Data"] or not isinstance(segmentation_result["Data"]["result"], list):
        return {"keywords": []}

    # 提取名词和形容词作为关键词
    keywords = []
    for item in segmentation_result["Data"]["result"]:
        if isinstance(item, dict):
            word = item.get("word", "")
            tags = item.get("tags", "")
            
            # 跳过单字符（除非是重要的单字词）
            if len(word) == 1:
                continue
            
            # 处理tags可能是字符串或列表的情况
            tags_str = ""
            if isinstance(tags, str):
                tags_str = tags
            elif isinstance(tags, list):
                tags_str = " ".join(str(t) for t in tags)
            
            # 检查是否是名词、形容词或其他有意义的词
            if ("基本词-中文" in tags_str or 
                "形容词" in tags_str or 
                "名词" in tags_str or
                "动词" in tags_str):
                if word and word not in keywords:  # 避免重复
                    keywords.append(word)

    # 如果关键词太少，返回所有长度>=2的词
    if len(keywords) < 3:
        keywords = []
        for item in segmentation_result["Data"]["result"]:
            if isinstance(item, dict):
                word = item.get("word", "")
                if len(word) >= 2 and word not in keywords:
                    keywords.append(word)

    # 返回前5个关键词
    return {"keywords": keywords[:5]}


def recognize_poem_entities(poem_text, segmentation_result=None, acs_client=None):
    """识别古诗词中的实体"""
    # 如果未提供分词结果，先进行分词
    if segmentation_result is None:
        segmentation_result = segment_poem(poem_text, acs_client)

    # 检查分词结果是否有效
    if not segmentation_result or "Data" not in segmentation_result:
        return {"entities": []}

    # 确保Data是字典类型
    if not isinstance(segmentation_result["Data"], dict):
        return {"entities": []}

    # 检查result字段是否存在且为列表
    if "result" not in segmentation_result["Data"] or not isinstance(segmentation_result["Data"]["result"], list):
        return {"entities": []}

    # 识别实体
    entities = []
    for item in segmentation_result["Data"]["result"]:
        if isinstance(item, dict):
            word = item["word"]
            # 检查是否是人名
            if "人名" in item.get("tags", ""):
                entities.append({"text": word, "type": "人名"})
            # 检查是否是地名
            elif "地名" in item.get("tags", ""):
                entities.append({"text": word, "type": "地名"})
            # 检查是否是时间词
            elif "时间词" in item.get("tags", ""):
                entities.append({"text": word, "type": "时间"})
            # 检查是否是古诗常见实体
            else:
                for category, entity_list in POEM_ENTITIES.items():
                    if word in entity_list:
                        entities.append({"text": word, "type": category})
                        break

    return {"entities": entities}


def infer_poem_sentiment_reasoning(poem_text, access_key_id=None, access_key_secret=None):
    """调用阿里云NLP进行古诗词情感推理分析"""
    # 获取客户端
    acs_client = None
    if access_key_id and access_key_secret:
        acs_client = get_aliyun_client(access_key_id, access_key_secret)
    else:
        acs_client = get_aliyun_client()
    
    if acs_client is None:
        logger.error("无法获取AcsClient，分析失败")
        return {
            "sentiment": {"sentiment": "unknown", "confidence": 0},
            "keywords": {"keywords": []},
            "entities": {"entities": []}
        }
    
    # 先进行分词
    segmentation_result = segment_poem(poem_text, acs_client)

    # 检查分词结果是否有效
    if not segmentation_result:
        return {
            "sentiment": {"sentiment": "unknown", "confidence": 0},
            "keywords": {"keywords": []},
            "entities": {"entities": []}
        }

    # 进行情感分析
    sentiment_result = analyze_poem_sentiment(poem_text, segmentation_result, acs_client)

    # 提取关键词
    keyword_result = extract_poem_keywords(poem_text, segmentation_result, acs_client)

    # 识别实体
    entity_result = recognize_poem_entities(poem_text, segmentation_result, acs_client)

    # 构建综合分析结果
    analysis_result = {
        "sentiment": sentiment_result,
        "keywords": keyword_result,
        "entities": entity_result
    }

    return analysis_result


def generate_analysis_report(analysis_result):
    """生成古诗词情感分析报告"""
    report = "## 古诗词情感分析报告\n\n"

    # 情感分析结果
    sentiment = analysis_result["sentiment"]
    if sentiment:
        sentiment_label = sentiment.get("sentiment", "unknown")
        confidence = sentiment.get("confidence", 0)

        report += f"### 情感分析\n"
        report += f"**情感倾向**: {sentiment_label}\n"
        report += f"**置信度**: {confidence:.2f}\n\n"

        # 根据情感标签提供解释
        if sentiment_label == "sad":
            report += "这首诗表达了深沉的忧伤和思念之情，可能与离别、思乡或人生感慨有关。\n"
        elif sentiment_label == "happy":
            report += "这首诗洋溢着喜悦和轻松的情绪，可能描绘了美好的景物或愉快的经历。\n"
        elif sentiment_label == "peaceful":
            report += "这首诗营造出宁静平和的氛围，可能描绘了自然景色或表达了淡泊的心境。\n"
        elif sentiment_label == "angry":
            report += "这首诗流露出愤怒或不满的情绪，可能与社会现实或个人遭遇有关。\n"
        elif sentiment_label == "mixed":
            report += "这首诗融合了多种情感，既有忧伤也有喜悦，体现了复杂的心境。\n"
        elif sentiment_label == "melancholic":
            report += "这首诗带有淡淡的忧伤，但整体氛围仍保持平和，可能表达了对往事的追忆。\n"
        elif sentiment_label == "serene":
            report += "这首诗既平静又愉悦，展现了诗人对自然美景的欣赏和内心的满足。\n"
        else:
            report += "这首诗的情感表达较为复杂，需要结合具体意象进行深入分析。\n"
    else:
        report += "### 情感分析\n情感分析未能获取有效结果。\n\n"

    # 关键词提取结果
    keywords = analysis_result["keywords"]
    if keywords and "keywords" in keywords:
        keyword_list = keywords["keywords"]

        report += f"\n### 关键词提取\n"
        report += "提取到的关键意象: " + ", ".join(keyword_list) + "\n"
    else:
        report += "\n### 关键词提取\n关键词提取未能获取有效结果。\n"

    # 实体识别结果
    entities = analysis_result["entities"]
    if entities and "entities" in entities:
        entity_list = entities["entities"]

        report += f"\n### 实体识别\n"
        if entity_list:
            report += "识别到的实体:\n"
            for entity in entity_list:
                report += f"- **{entity['text']}**: {entity['type']}\n"
        else:
            report += "未识别到明显实体。\n"
    else:
        report += "\n### 实体识别\n实体识别未能获取有效结果。\n"

    return report


# 示例使用
if __name__ == "__main__":
    # 示例古诗词
    poem1 = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    poem2 = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"

    # 分析并打印第一首诗
    print(f"正在分析古诗词: '{poem1}'\n")
    analysis_result1 = infer_poem_sentiment_reasoning(poem1)
    report1 = generate_analysis_report(analysis_result1)
    print(report1)

    # 添加分隔线
    print("\n" + "=" * 50 + "\n")

    # 分析并打印第二首诗
    print(f"正在分析古诗词: '{poem2}'\n")
    analysis_result2 = infer_poem_sentiment_reasoning(poem2)
    report2 = generate_analysis_report(analysis_result2)
    print(report2)