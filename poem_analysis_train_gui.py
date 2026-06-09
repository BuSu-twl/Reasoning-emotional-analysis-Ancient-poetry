"""
古诗词情感分析综合可视化界面
整合训练、分析、测试、预测等所有功能
使用 Streamlit 创建统一的 Web 界面
"""
import streamlit as st
import os
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import threading
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="古诗词情感分析系统",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .poem-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .sentiment-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    .keyword-tag {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    .entity-tag {
        display: inline-block;
        background-color: #2196F3;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== 工具函数 ====================

def check_dependencies() -> List[str]:
    """检查必要的依赖"""
    missing = []
    dependencies = {
        "torch": "torch",
        "transformers": "transformers",
        "sklearn": "scikit-learn",
        "plotly": "plotly",
        "pandas": "pandas"
    }
    
    for module_name, package_name in dependencies.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    return missing


def get_model_list() -> List[Dict]:
    """获取已训练的模型列表"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    models: List[Dict] = []
    
    def is_valid_model_dir(model_path: Path) -> bool:
        """检查目录是否是有效的模型目录（包含config.json）"""
        if not model_path.exists() or not model_path.is_dir():
            return False
        # 检查是否有config.json（模型配置文件）
        config_file = model_path / "config.json"
        return config_file.exists()

    # ========== 新结构：支持多个版本的 sentiment_model ==========
    sentiment_model_root = models_dir / "sentiment_model"
    if sentiment_model_root.exists():
        # 1) 兼容旧结构：models/sentiment_model/final
        legacy_final = sentiment_model_root / "final"
        legacy_label = sentiment_model_root / "label_mapping.json"
        if is_valid_model_dir(legacy_final):
            models.append({
                "name": "sentiment_model",
                "path": str(legacy_final),
                "label_map": str(legacy_label) if legacy_label.exists() else None,
                "type": "transformer"
            })

        # 2) 新结构：models/sentiment_model/<run_name>/final
        for run_dir in sentiment_model_root.iterdir():
            if not run_dir.is_dir() or run_dir.name == "final":
                continue
            
            # 跳过 checkpoint 目录（后面单独处理）
            if run_dir.name.startswith("checkpoint-"):
                continue
                
            final_dir = run_dir / "final"
            label_map = run_dir / "label_mapping.json"
            if is_valid_model_dir(final_dir):
                models.append({
                    "name": f"sentiment_model_{run_dir.name}",
                    "path": str(final_dir),
                    "label_map": str(label_map) if label_map.exists() else None,
                    "type": "transformer"
                })
        
        # 3) checkpoint 目录也可以作为模型使用（训练过程中的检查点）
        for checkpoint_dir in sentiment_model_root.iterdir():
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("checkpoint-"):
                if is_valid_model_dir(checkpoint_dir):
                    label_map = sentiment_model_root / "label_mapping.json"
                    models.append({
                        "name": f"sentiment_model_{checkpoint_dir.name}",
                        "path": str(checkpoint_dir),
                        "label_map": str(label_map) if label_map.exists() else None,
                        "type": "transformer"
                    })

    # ========== llm_sentiment_model（大模型微调）==========
    llm_sentiment_root = models_dir / "llm_sentiment_model"
    if llm_sentiment_root.exists():
        # 检查是否有 final 目录
        llm_final = llm_sentiment_root / "final"
        llm_label = llm_sentiment_root / "label_mapping.json"
        if is_valid_model_dir(llm_final):
            models.append({
                "name": "llm_sentiment_model",
                "path": str(llm_final),
                "label_map": str(llm_label) if llm_label.exists() else None,
                "type": "transformer"
            })
        
        # 检查是否有 run_* 子目录
        for run_dir in llm_sentiment_root.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                run_final = run_dir / "final"
                run_label = run_dir / "label_mapping.json"
                if is_valid_model_dir(run_final):
                    models.append({
                        "name": f"llm_sentiment_model_{run_dir.name}",
                        "path": str(run_final),
                        "label_map": str(run_label) if run_label.exists() else None,
                        "type": "transformer"
                    })
        
        # 检查是否直接包含模型文件（即使没有final目录）
        if is_valid_model_dir(llm_sentiment_root):
            # 避免重复添加（如果已经有 final 目录）
            if not llm_final.exists():
                models.append({
                    "name": "llm_sentiment_model (direct)",
                    "path": str(llm_sentiment_root),
                    "label_map": str(llm_label) if llm_label.exists() else None,
                    "type": "transformer"
                })

    # ========== llm_fewshot（Few-Shot Learning模型）==========
    llm_fewshot_root = models_dir / "llm_fewshot"
    if llm_fewshot_root.exists():
        # Few-Shot模型包含few_shot_examples.json和evaluation_results.json
        examples_file = llm_fewshot_root / "few_shot_examples.json"
        eval_file = llm_fewshot_root / "evaluation_results.json"
        
        # 检查是否有主模型文件
        if examples_file.exists() or eval_file.exists():
            models.append({
                "name": "llm_fewshot",
                "path": str(llm_fewshot_root),
                "label_map": None,  # Few-Shot模型不使用标签映射文件
                "type": "llm_fewshot",
                "examples_file": str(examples_file) if examples_file.exists() else None,
                "eval_file": str(eval_file) if eval_file.exists() else None
            })
        
        # 检查是否有 run_* 子目录（不同训练运行的记录）
        for run_dir in llm_fewshot_root.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                run_examples = run_dir / "few_shot_examples.json"
                run_eval = run_dir / "evaluation_results.json"
                if run_examples.exists() or run_eval.exists():
                    models.append({
                        "name": f"llm_fewshot_{run_dir.name}",
                        "path": str(run_dir),
                        "label_map": None,
                        "type": "llm_fewshot",
                        "examples_file": str(run_examples) if run_examples.exists() else None,
                        "eval_file": str(run_eval) if run_eval.exists() else None
                    })

    # ========== 其他模型目录（通用扫描）==========
    excluded_dirs = {"sentiment_model", "bert-base-chinese", "llm_fewshot", "aliyun_models", "llm_sentiment_model"}
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in excluded_dirs:
            continue
        
        # 检查是否有 final 目录
        final_model = model_dir / "final"
        label_map = model_dir / "label_mapping.json"
        if is_valid_model_dir(final_model):
            models.append({
                "name": model_dir.name,
                "path": str(final_model),
                "label_map": str(label_map) if label_map.exists() else None,
                "type": "transformer"
            })
        
        # 也检查是否有 run_* 子目录
        for sub_dir in model_dir.iterdir():
            if sub_dir.is_dir() and sub_dir.name.startswith("run_"):
                run_final = sub_dir / "final"
                run_label = sub_dir / "label_mapping.json"
                if is_valid_model_dir(run_final):
                    models.append({
                        "name": f"{model_dir.name}_{sub_dir.name}",
                        "path": str(run_final),
                        "label_map": str(run_label) if run_label.exists() else None,
                        "type": "transformer"
                    })
        
        # 检查是否直接包含模型文件（旧版本可能直接保存在目录下）
        if is_valid_model_dir(model_dir):
            # 避免重复添加（如果已经有 final 目录）
            if not (model_dir / "final").exists():
                label_map_direct = model_dir / "label_mapping.json"
                models.append({
                    "name": f"{model_dir.name} (direct)",
                    "path": str(model_dir),
                    "label_map": str(label_map_direct) if label_map_direct.exists() else None,
                    "type": "transformer"
                })

    return models


def load_label_mapping(label_map_path: Optional[str]) -> Optional[Dict]:
    """加载标签映射"""
    if not label_map_path or not Path(label_map_path).exists():
        return None
    try:
        with open(label_map_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def get_sentiment_color(sentiment: str) -> str:
    """根据情感类型返回对应的颜色"""
    color_map = {
        "happy": "#4CAF50",
        "sad": "#F44336",
        "peaceful": "#2196F3",
        "mixed": "#FF9800",
        "melancholic": "#9C27B0",
        "serene": "#00BCD4",
        "unknown": "#9E9E9E"
    }
    return color_map.get(sentiment.lower(), "#9E9E9E")


def get_sentiment_emoji(sentiment: str) -> str:
    """根据情感类型返回对应的表情"""
    emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "peaceful": "😌",
        "mixed": "😐",
        "melancholic": "😔",
        "serene": "😇",
        "unknown": "🤔"
    }
    return emoji_map.get(sentiment.lower(), "🤔")


# ==================== 数据加载功能 ====================

@st.cache_data
def load_training_data(data_dir: str = "data/train-data") -> Tuple[List[Dict], int]:
    """加载训练数据（带缓存，避免重复加载）"""
    try:
        from train_poem_sentiment import PoemDataLoader
        data = PoemDataLoader.load_train_data(data_dir)
        return data, len(data)
    except Exception as e:
        st.warning(f"无法加载数据: {e}")
        return [], 0


def get_data_statistics(data: List[Dict]) -> Dict:
    """获取数据统计信息"""
    stats = {
        "total": len(data),
        "with_emotion": sum(1 for item in data if item.get('emotion')),
        "with_title": sum(1 for item in data if item.get('title')),
        "emotion_types": {}
    }
    
    for item in data:
        if item.get('emotion'):
            emotion = str(item['emotion']).strip()
            # 提取第一个情感标签
            separators = [',', '，', ';', '；', ' ', '\t']
            for sep in separators:
                if sep in emotion:
                    emotion = emotion.split(sep)[0].strip()
                    break
            stats["emotion_types"][emotion] = stats["emotion_types"].get(emotion, 0) + 1
    
    return stats


# ==================== 训练功能 ====================

def run_training_process(config: Dict) -> subprocess.Popen:
    """运行训练进程"""
    training_mode = config.get("training_mode", "🤖 传统Transformer训练")
    
    # 根据训练模式选择日志文件和基础输出目录
    if training_mode == "🌟 大模型训练（Few-Shot）":
        log_file = Path("train_llm_fewshot.log")
        script_name = "train_llm_fewshot.py"
    elif training_mode == "🚀 大模型微调训练":
        log_file = Path("train_llm.log")
        script_name = "train_with_llm.py"
    else:
        log_file = Path("train_poem_sentiment.log")
        script_name = "train_poem_sentiment.py"
    
    # 尝试清空日志文件
    if log_file.exists():
        try:
            log_file.unlink()
        except (PermissionError, OSError):
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write('')
            except Exception:
                pass
    
    # 为当前训练运行生成唯一的输出目录，避免覆盖历史模型
    base_output_dir = None
    if training_mode == "🌟 大模型训练（Few-Shot）":
        base_output_dir = Path(config.get("output_dir", "models/llm_fewshot"))
    elif training_mode == "🚀 大模型微调训练":
        base_output_dir = Path(config.get("output_dir", "models/llm_sentiment_model"))
    else:
        base_output_dir = Path(config.get("output_dir", "models/sentiment_model"))

    from datetime import datetime
    # 如果基础目录已存在，则在其下创建带时间戳的子目录
    if base_output_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_output_dir = base_output_dir / f"run_{timestamp}"
    else:
        effective_output_dir = base_output_dir

    effective_output_dir_str = str(effective_output_dir)

    # 构建命令
    if training_mode == "🌟 大模型训练（Few-Shot）":
        cmd = [
            sys.executable,
            script_name,
            "--model-type", config.get("model_type", "openai"),
            "--train-data-dir", config.get("train_data_dir", "data/train-data"),
            "--output-dir", effective_output_dir_str,
            "--test-split", str(config.get("test_split", 0.2))
        ]
        if config.get("api_key"):
            cmd.extend(["--api-key", config["api_key"]])
    elif training_mode == "🚀 大模型微调训练":
        cmd = [
            sys.executable,
            script_name,
            "--model-name", config.get("model_name", "Qwen/Qwen2-0.5B-Instruct"),
            "--train-data-dir", config.get("train_data_dir", "data/train-data"),
            "--output-dir", effective_output_dir_str,
            "--epochs", str(config.get("epochs", 3)),
            "--batch-size", str(config.get("batch_size", 4)),
            "--learning-rate", str(config.get("learning_rate", 5e-5)),
            "--max-length", str(config.get("max_length", 512)),
            "--test-split", str(config.get("test_split", 0.2))
        ]
        if config.get("device"):
            cmd.extend(["--device", config["device"]])
        if config.get("augment"):
            cmd.append("--augment")
    else:
        # 传统训练
        cmd = [
            sys.executable,
            script_name,
            "--model-name", config.get("model_name", "bert-base-chinese"),
            "--train-data-dir", config.get("train_data_dir", "data/train-data"),
            "--output-dir", effective_output_dir_str,
            "--epochs", str(config.get("epochs", 3)),
            "--batch-size", str(config.get("batch_size", 16)),
            "--learning-rate", str(config.get("learning_rate", 2e-5)),
            "--max-length", str(config.get("max_length", 512)),
            "--test-split", str(config.get("test_split", 0.2))
        ]
        if config.get("limit"):
            cmd.extend(["--limit", str(config["limit"])])
        if config.get("device"):
            cmd.extend(["--device", config["device"]])
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )
    
    return process


def show_training_page():
    """显示训练页面"""
    st.markdown('<h1 class="main-header">🏋️ 模型训练</h1>', unsafe_allow_html=True)
    
    # 训练模式选择
    training_mode = st.radio(
        "训练模式",
        ["🤖 传统Transformer训练", "🌟 大模型训练（Few-Shot）", "🚀 大模型微调训练"],
        horizontal=True,
        help="选择训练方式：传统训练适合大数据集，大模型训练适合小数据集"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 训练配置")
        
        if training_mode == "🤖 传统Transformer训练":
            # 传统训练配置
            model_name = st.selectbox(
                "预训练模型",
                [
                    "bert-base-chinese",
                    "hfl/chinese-bert-wwm-ext",
                    "hfl/chinese-roberta-wwm-ext",
                    "hfl/chinese-macbert-base"
                ],
                help="选择用于微调的预训练模型"
            )
        elif training_mode == "🌟 大模型训练（Few-Shot）":
            # Few-Shot训练配置
            st.info("💡 Few-Shot Learning适合小数据集（164条），无需GPU，使用大模型API")
            
            model_type = st.selectbox(
                "大模型类型",
                ["openai", "qwen", "local"],
                help="选择使用的大模型API"
            )
            
            if model_type == "openai":
                api_key = st.text_input("OpenAI API Key", type="password", 
                                       help="留空则使用环境变量 OPENAI_API_KEY")
            elif model_type == "qwen":
                api_key = st.text_input("通义千问 API Key", type="password",
                                       help="留空则使用环境变量 DASHSCOPE_API_KEY")
            else:
                api_key = None
                st.text_input("本地LLM服务地址", value=os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1"),
                             help="设置环境变量 LOCAL_LLM_URL")
            
            model_name = model_type  # 用于标识
            
        else:  # 大模型微调训练
            st.info("💡 大模型微调需要GPU资源，适合有充足计算资源的情况")
            
            model_name = st.selectbox(
                "预训练大模型",
                [
                    "Qwen/Qwen2-0.5B-Instruct",
                    "Qwen/Qwen2-1.5B-Instruct",
                    "Qwen/Qwen2-7B-Instruct",
                    "THUDM/chatglm3-6b"
                ],
                help="选择用于微调的大语言模型（需要GPU）"
            )
        
        # 训练参数（根据模式显示不同参数）
        if training_mode == "🌟 大模型训练（Few-Shot）":
            # Few-Shot训练不需要这些参数
            st.info("Few-Shot训练会自动构建示例库，无需设置训练参数")
            epochs = 0
            batch_size = 0
            learning_rate = 0
            max_length = 512
            output_dir = st.text_input("输出目录", value="models/llm_fewshot")
            test_split = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        else:
            # 传统训练或大模型微调的参数
            col_a, col_b = st.columns(2)
            with col_a:
                epochs = st.number_input("训练轮数", min_value=1, max_value=20, value=3)
                batch_size = st.number_input("批次大小", min_value=1, max_value=64, 
                                            value=4 if training_mode == "🚀 大模型微调训练" else 16)
            with col_b:
                learning_rate = st.number_input("学习率", min_value=1e-6, max_value=1e-2, 
                                                value=5e-5 if training_mode == "🚀 大模型微调训练" else 2e-5, 
                                                format="%e")
                max_length = st.number_input("最大序列长度", min_value=128, max_value=1024, value=512)
            
            if training_mode == "🚀 大模型微调训练":
                output_dir = st.text_input("输出目录", value="models/llm_sentiment_model")
            else:
                output_dir = st.text_input("输出目录", value="models/sentiment_model")
            test_split = st.slider("验证集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        
        # 高级选项（仅传统训练和大模型微调显示）
        if training_mode != "🌟 大模型训练（Few-Shot）":
            with st.expander("🔧 高级选项"):
                # 从环境变量获取设备选择（如果通过启动脚本设置了）
                env_device = os.environ.get('TRAINING_DEVICE', None)
                device_options = ["auto", "cpu", "cuda"]
                default_index = 0
                
                # 如果环境变量设置了设备，使用对应的索引
                # 注意：启动脚本会设置实际检测到的设备（cpu/cuda），而不是"auto"
                if env_device and env_device in device_options:
                    default_index = device_options.index(env_device)
                
                device_help = "训练设备选择"
                if env_device:
                    device_help += f" (启动脚本已设置: {env_device})"
                
                device = st.selectbox(
                    "设备", 
                    device_options, 
                    index=default_index,
                    help=device_help
                )
                limit_data = st.number_input("限制数据量（用于测试）", min_value=0, value=0, 
                                            help="设为0表示使用全部数据")
        else:
            # Few-Shot训练不需要这些选项
            device = "auto"
            limit_data = 0
        
        # 数据信息
        st.markdown("### 📊 数据信息")
        train_data_dir = Path("data/train-data")
        if train_data_dir.exists():
            json_files = list(train_data_dir.rglob("*.json"))
            st.info(f"📁 找到 {len(json_files)} 个训练数据文件")
            
            train_data, data_count = load_training_data(str(train_data_dir))
            if data_count > 0:
                st.success(f"✅ 共加载 {data_count} 条训练数据")
                
                # 显示数据统计
                stats = get_data_statistics(train_data)
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("总数据量", stats["total"])
                with col_stat2:
                    st.metric("有情感标签", stats["with_emotion"])
                with col_stat3:
                    st.metric("情感类型数", len(stats["emotion_types"]))
                
                # 显示情感分布
                if stats["emotion_types"]:
                    st.markdown("#### 情感分布")
                    emotion_df = pd.DataFrame(
                        list(stats["emotion_types"].items()),
                        columns=["情感", "数量"]
                    )
                    fig = px.bar(emotion_df, x="情感", y="数量", title="情感标签分布")
                    st.plotly_chart(fig, width='stretch')
        else:
            st.error("❌ 训练数据目录不存在: data/train-data")
    
    with col2:
        st.markdown("### ⚡ 快速操作")
        
        if training_mode == "🌟 大模型训练（Few-Shot）":
            st.info("""
            **Few-Shot训练特点：**
            - ✅ 适合小数据集（164条）
            - ✅ 无需GPU
            - ✅ 训练速度快
            - ✅ 可处理分散标签（227个）
            """)
        elif training_mode == "🚀 大模型微调训练":
            st.warning("""
            **大模型微调要求：**
            - ⚠️ 需要GPU资源
            - ⚠️ 训练时间较长
            - ✅ 可本地部署
            """)
        else:
            if st.button("🚀 快速训练（1轮）", width='stretch'):
                st.session_state['train_config'] = {
                    "training_mode": training_mode,
                    "model_name": model_name,
                    "epochs": 1,
                    "batch_size": 8,
                    "learning_rate": 2e-5,
                    "max_length": 512,
                    "output_dir": output_dir,
                    "test_split": 0.2,
                    "device": None if device == "auto" else device,
                    "limit": 100
                }
                st.session_state['start_training'] = True
                st.rerun()
            
            if st.button("⚡ 标准训练（3轮）", width='stretch'):
                st.session_state['train_config'] = {
                    "training_mode": training_mode,
                    "model_name": model_name,
                    "epochs": 3,
                    "batch_size": 16,
                    "learning_rate": 2e-5,
                    "max_length": 512,
                    "output_dir": output_dir,
                    "test_split": 0.2,
                    "device": None if device == "auto" else device,
                    "limit": 0
                }
                st.session_state['start_training'] = True
                st.rerun()
    
    # 训练按钮
    st.markdown("---")
    col_start, col_stop = st.columns(2)
    
    with col_start:
        if st.button("▶️ 开始训练", type="primary", width='stretch'):
            train_config = {
                "training_mode": training_mode,
                "model_name": model_name,
                "output_dir": output_dir,
                "test_split": test_split,
                "train_data_dir": "data/train-data"
            }
            
            if training_mode == "🌟 大模型训练（Few-Shot）":
                train_config.update({
                    "model_type": model_type,
                    "api_key": api_key if api_key else None
                })
            else:
                train_config.update({
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_length": max_length,
                    "device": None if device == "auto" else device,
                    "limit": limit_data if limit_data > 0 else None
                })
            
            st.session_state['train_config'] = train_config
            st.session_state['start_training'] = True
            st.rerun()
    
    with col_stop:
        if st.button("⏹️ 停止训练", width='stretch'):
            if 'training_process' in st.session_state:
                st.session_state['training_process'].terminate()
                st.session_state['training'] = False
            st.info("训练已停止")
    
    # 处理训练
    if st.session_state.get('start_training', False) and 'train_config' in st.session_state:
        config = st.session_state['train_config']
        st.session_state['start_training'] = False
        
        try:
            # 运行训练进程
            process = run_training_process(config)
            st.session_state['training_process'] = process
            st.session_state['training'] = True
            st.session_state['training_complete'] = False
            
            # 使用后台线程等待进程完成（不阻塞UI）
            def wait_for_training():
                return_code = process.wait()
                st.session_state['training_return_code'] = return_code
                st.session_state['training'] = False
                st.session_state['training_complete'] = True
                # 不在后台线程中调用 st.rerun()，而是设置标志让主线程检查
                # st.rerun() 会在主线程的下一次循环中自动触发
            
            # 检查是否已有运行中的线程
            if 'training_thread' not in st.session_state:
                thread = threading.Thread(target=wait_for_training, daemon=True)
                thread.start()
                st.session_state['training_thread'] = thread
            else:
                # 如果线程已存在，检查是否还活着
                existing_thread = st.session_state.get('training_thread')
                if existing_thread is None or not existing_thread.is_alive():
                    thread = threading.Thread(target=wait_for_training, daemon=True)
                    thread.start()
                    st.session_state['training_thread'] = thread
            
        except Exception as e:
            st.error(f"训练过程中出错: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # 显示训练状态和日志
    st.markdown("### 📝 训练日志")
    # 根据训练模式选择日志文件
    training_mode = st.session_state.get('train_config', {}).get('training_mode', '🤖 传统Transformer训练')
    if training_mode == "🌟 大模型训练（Few-Shot）":
        log_file = Path("train_llm_fewshot.log")
    elif training_mode == "🚀 大模型微调训练":
        log_file = Path("train_llm.log")
    else:
        log_file = Path("train_poem_sentiment.log")
    
    # 训练状态提示
    if st.session_state.get('training', False):
        st.info("⏳ 训练正在进行中，请稍候...\n\n日志文件会实时更新，您可以点击下方「刷新日志」按钮查看最新进度。")
    
    # 读取并显示日志文件（训练中或训练完成都可以显示）
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                final_logs = f.read()
            
            if final_logs.strip():
                # 只显示最后500行，避免界面过长
                log_lines = final_logs.split('\n')
                if len(log_lines) > 500:
                    display_logs = '\n'.join(log_lines[-500:])
                    st.info(f"显示最后 500 行日志（共 {len(log_lines)} 行）")
                else:
                    display_logs = final_logs
                st.code(display_logs, language="text")
        except Exception as e:
            st.warning(f"读取日志文件失败: {e}")
    elif st.session_state.get('training', False):
        st.info("日志文件尚未生成，训练可能刚刚开始...")
    
    # 训练完成后的处理（检查训练状态）
    if st.session_state.get('training_complete', False):
        st.session_state['training_complete'] = False
        return_code = st.session_state.get('training_return_code', -1)
        
        if return_code == 0:
            st.success("✅ 训练完成！")
            st.balloons()
            st.rerun()  # 在主线程中刷新界面
        else:
            st.error(f"❌ 训练失败，返回码: {return_code}")
            st.rerun()  # 在主线程中刷新界面
    
    # 定期检查训练进程状态（如果训练正在进行）
    if st.session_state.get('training', False):
        training_process = st.session_state.get('training_process')
        if training_process and training_process.poll() is not None:
            # 进程已完成，但线程可能还没更新状态
            return_code = training_process.returncode
            st.session_state['training_return_code'] = return_code
            st.session_state['training'] = False
            st.session_state['training_complete'] = True
            st.rerun()
    
    # 刷新按钮（始终显示，方便查看日志）
    col_refresh1, col_refresh2 = st.columns([1, 4])
    with col_refresh1:
        if st.button("🔄 刷新日志", help="刷新并查看最新的训练日志"):
            st.rerun()


# ==================== 分析功能 ====================

def analyze_with_poem_analysis(text: str, access_key_id: str = None, access_key_secret: str = None):
    """使用poem_analysis.py进行分析"""
    try:
        from poem_analysis import infer_poem_sentiment_reasoning, generate_analysis_report
        
        # 如果没有提供凭证，尝试从环境变量获取
        if not access_key_id or not access_key_secret:
            access_key_id = access_key_id or os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID')
            access_key_secret = access_key_secret or os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
            if not access_key_id or not access_key_secret:
                raise ValueError("未找到阿里云凭证，请通过界面配置或设置环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        
        # 直接传递凭证给分析函数，避免模块重载
        result = infer_poem_sentiment_reasoning(text, access_key_id, access_key_secret)
        report = generate_analysis_report(result)
        
        return result, report
    except ValueError as e:
        st.error(f"配置错误: {e}")
        return None, None
    except Exception as e:
        st.error(f"分析失败: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def analyze_with_baseline(text: str, api_key: str = None):
    """使用通义千问API进行分析"""
    try:
        import dashscope
        
        # 获取API Key（优先使用传入的，否则从环境变量获取）
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            st.error("❌ 通义千问 API Key 未设置！\n\n请使用以下方式之一设置：\n1. 在下方输入框中输入 API Key\n2. 设置环境变量: DASHSCOPE_API_KEY")
            return None, None
        
        # 设置API Key
        dashscope.api_key = api_key
        
        prompt = f"""
你是一个古诗词专家，请分析以下古诗词的情感：

{text}

请以JSON格式返回分析结果，包含：
- sentiment: 情感类型（如：happy, sad, peaceful等）
- confidence: 置信度（0-1之间的浮点数）
- keywords: 关键词列表
- explanation: 简要解释
"""
        
        # 调用通义千问API
        from dashscope import Generation
        response = Generation.call(
            model="qwen-turbo",
            prompt=prompt,
            temperature=0.3,
            max_tokens=800,
            timeout=60
        )
        
        # 检查响应状态
        if response.status_code != 200:
            error_msg = f"通义千问API调用失败: {response.message}"
            st.error(error_msg)
            return None, None
        
        # 获取响应文本
        if hasattr(response, 'output') and hasattr(response.output, 'text'):
            content = response.output.text.strip()
        elif hasattr(response, 'output') and hasattr(response.output, 'choices'):
            if response.output.choices and len(response.output.choices) > 0:
                content = response.output.choices[0].message.content.strip()
            else:
                st.error("响应中没有有效的文本内容")
                return None, None
        else:
            content = str(response).strip()
        
        # 返回结果
        return {"content": content}, None
        
    except ImportError:
        st.error("❌ dashscope库未安装！\n\n请运行: pip install dashscope")
        return None, None
    except Exception as e:
        st.warning(f"LLM分析失败: {e}")
        return None, None


def show_analysis_page():
    """显示分析页面"""
    st.markdown('<h1 class="main-header">📜 古诗词情感分析</h1>', unsafe_allow_html=True)
    
    # 分析模式选择
    analysis_mode = st.radio(
        "分析模式",
        ["阿里云NLP分析", "大语言模型分析", "已训练模型预测"],
        horizontal=True
    )
    
    
    # 输入模式选择
    input_mode = st.radio(
        "输入模式",
        ["📝 简单输入", "📋 完整格式（测试集格式）"],
        horizontal=True,
        help="简单输入：只输入诗词内容；完整格式：包含标题、作者、选择题等"
    )
    
    # 输入区域
    if input_mode == "📝 简单输入":
        col_input, col_example = st.columns([2, 1])
        
        # 初始化示例选择
        if 'selected_example_text' not in st.session_state:
            st.session_state['selected_example_text'] = ""
        
        with col_input:
            st.markdown("### 📝 输入古诗词")
            poem_text = st.text_area(
                "请输入古诗词内容",
                value=st.session_state.get('selected_example_text', ''),
                height=150,
                placeholder="例如：床前明月光，疑是地上霜。举头望明月，低头思故乡。",
                key="poem_input_area"
            )
        
        with col_example:
            st.markdown("### 📚 示例")
            examples = [
                ("静夜思", "床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
                ("春晓", "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"),
                ("登鹳雀楼", "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。")
            ]
            
            for title, content in examples:
                if st.button(f"📖 {title}", key=f"example_{title}"):
                    st.session_state['selected_example_text'] = content
                    st.rerun()
        
        # 用于分析的文本
        analysis_text = poem_text
        poem_data = {"content": poem_text}
        
    else:  # 完整格式输入
        st.markdown("### 📋 完整格式输入（测试集格式）")
        
        # 从测试集加载示例（放在最前面，以便填充数据）
        test_data_path = Path("data/eval-data/eval_data.json")
        loaded_example = None
        
        if test_data_path.exists():
            try:
                with open(test_data_path, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                
                if isinstance(test_data, list) and len(test_data) > 0:
                    test_examples = test_data[:20]  # 显示前20个
                    
                    # 选择框
                    example_options = [f"{ex.get('title', '未知')} - {ex.get('author', '')}" for ex in test_examples]
                    selected_example_idx = st.selectbox(
                        "选择测试集示例",
                        options=range(len(test_examples)),
                        format_func=lambda x: example_options[x],
                        help="从测试集中选择示例自动填充",
                        key="test_example_select"
                    )
                    
                    if st.button("📥 加载示例", key="load_test_example"):
                        loaded_example = test_examples[selected_example_idx]
                        st.session_state['loaded_test_example'] = loaded_example
                        st.rerun()
                    
                    # 如果session_state中有加载的示例，使用它
                    if 'loaded_test_example' in st.session_state:
                        loaded_example = st.session_state['loaded_test_example']
            except Exception as e:
                st.warning(f"加载测试集失败: {e}")
        
        # 使用加载的示例或空值作为默认值
        default_title = loaded_example.get('title', '') if loaded_example else ''
        default_author = loaded_example.get('author', '') if loaded_example else ''
        default_content = loaded_example.get('content', '') if loaded_example else ''
        default_qa_words = '\n'.join(loaded_example.get('qa_words', [])) if loaded_example else ''
        default_qa_sents = '\n'.join(loaded_example.get('qa_sents', [])) if loaded_example else ''
        default_choose = loaded_example.get('choose', {}) if loaded_example else {}
        
        col_title, col_author = st.columns(2)
        with col_title:
            title = st.text_input(
                "标题", 
                value=default_title,
                key="full_title_input",
                placeholder="例如：李端公"
            )
        with col_author:
            author = st.text_input(
                "作者", 
                value=default_author,
                key="full_author_input",
                placeholder="例如：卢纶"
            )
        
        content = st.text_area(
            "诗词内容 *",
            value=default_content,
            key="full_content_input",
            height=150,
            placeholder="例如：故关衰草遍，离别自堪悲。路出寒云外，人归暮雪时。少孤为客早，多难识君迟。掩泪空相向，风尘何处期。",
            help="必填项"
        )
        
        # 扩展信息（可选）
        with st.expander("📚 扩展信息（可选）", expanded=False):
            qa_words_input = st.text_area(
                "需要解释的词语（每行一个）",
                value=default_qa_words,
                key="full_qa_words_input",
                height=100,
                placeholder="例如：\n衰草\n故关\n风尘",
                help="每行输入一个词语"
            )
            
            qa_sents_input = st.text_area(
                "需要翻译的句子（每行一个）",
                value=default_qa_sents,
                key="full_qa_sents_input",
                height=100,
                placeholder="例如：\n故关衰草遍，离别自堪悲\n掩泪空相向，风尘何处期。",
                help="每行输入一个句子"
            )
            
            st.markdown("**选择题选项**")
            col_a, col_b = st.columns(2)
            col_c, col_d = st.columns(2)
            
            with col_a:
                option_a = st.text_input(
                    "选项 A", 
                    value=default_choose.get('A', ''),
                    key="full_option_a",
                    placeholder="例如：欢快的重逢"
                )
            with col_b:
                option_b = st.text_input(
                    "选项 B", 
                    value=default_choose.get('B', ''),
                    key="full_option_b",
                    placeholder="例如：仕途的无奈"
                )
            with col_c:
                option_c = st.text_input(
                    "选项 C", 
                    value=default_choose.get('C', ''),
                    key="full_option_c",
                    placeholder="例如：对未来的期待"
                )
            with col_d:
                option_d = st.text_input(
                    "选项 D", 
                    value=default_choose.get('D', ''),
                    key="full_option_d",
                    placeholder="例如：惜别的感伤"
                )
        
        # 清除已加载的示例（避免下次自动填充）
        if 'loaded_test_example' in st.session_state:
            del st.session_state['loaded_test_example']
        
        # 构建完整数据
        poem_data = {
            "title": title,
            "author": author,
            "content": content
        }
        
        if qa_words_input.strip():
            qa_words = [w.strip() for w in qa_words_input.split('\n') if w.strip()]
            poem_data["qa_words"] = qa_words
        
        if qa_sents_input.strip():
            qa_sents = [s.strip() for s in qa_sents_input.split('\n') if s.strip()]
            poem_data["qa_sents"] = qa_sents
        
        if option_a or option_b or option_c or option_d:
            choose = {}
            if option_a:
                choose["A"] = option_a
            if option_b:
                choose["B"] = option_b
            if option_c:
                choose["C"] = option_c
            if option_d:
                choose["D"] = option_d
            if choose:
                poem_data["choose"] = choose
        
        # 用于分析的文本（包含标题和内容）
        if title:
            analysis_text = f"{title}：{content}" if content else content
        else:
            analysis_text = content
        
        # 重新读取输入值（因为Streamlit的widget值在rerun后会保持）
        # 需要确保在expander中也能访问到正确的值
        if st.session_state.get('test_example_loaded'):
            # 如果已加载，值已经在上面设置了
            pass
        else:
            # 如果没有加载，使用空值（已在上面定义）
            pass
    
    # 分析按钮
    if st.button("🔍 开始分析", type="primary", width='stretch'):
        # 检查输入是否有效
        if input_mode == "📝 简单输入":
            if not poem_text.strip():
                st.warning("请输入古诗词内容")
                return
        else:
            if not content.strip():
                st.warning("请输入诗词内容（必填）")
                return
        
        with st.spinner("正在分析..."):
            if analysis_mode == "阿里云NLP分析":
                # 获取阿里云配置（使用全局配置）
                use_env_global = st.session_state.get('use_env_global', True)
                aliyun_config = st.session_state.get('global_aliyun_config')
                
                access_key_id = None
                access_key_secret = None
                
                if not use_env_global and aliyun_config:
                    # 使用界面中输入的配置
                    access_key_id = aliyun_config.get('access_key_id')
                    access_key_secret = aliyun_config.get('access_key_secret')
                # 如果 use_env_global 为 True，则 access_key_id 和 access_key_secret 保持为 None
                # analyze_with_poem_analysis 会使用环境变量
                
                # 检查配置是否有效
                if not use_env_global:
                    if not access_key_id or not access_key_secret:
                        st.error("请先在侧边栏输入阿里云 Access Key ID 和 Secret，或勾选'使用环境变量'")
                        return
                else:
                    # 检查环境变量是否存在
                    env_access_key_id = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID')
                    env_access_key_secret = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
                    if not env_access_key_id or not env_access_key_secret:
                        st.error("环境变量中未找到阿里云凭证。请设置 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET，或在侧边栏取消勾选'使用环境变量'并手动输入凭证。")
                        return
                
                result, report = analyze_with_poem_analysis(
                    analysis_text, 
                    access_key_id=access_key_id, 
                    access_key_secret=access_key_secret
                )
                if result:
                    display_analysis_result_aliyun(result, report, poem_data if input_mode == "📋 完整格式（测试集格式）" else None)
                else:
                    st.warning("分析失败，请检查侧边栏的阿里云配置")
            
            elif analysis_mode == "大语言模型分析":
                    st.info("💡 使用通义千问API进行大语言模型分析")
                    with st.expander("🔑 API配置", expanded=True):
                        use_env_key = st.checkbox("使用环境变量", value=True, 
                                                 key="use_env_qwen_key",
                                                 help="如果勾选，将使用环境变量 DASHSCOPE_API_KEY 中的API Key")
                        api_key = None
                        if not use_env_key:
                            api_key = st.text_input(
                                "通义千问 API Key", 
                                type="password",
                                key="qwen_api_key_input",
                                help="留空则使用环境变量 DASHSCOPE_API_KEY，可以从 https://dashscope.console.aliyun.com/ 获取"
                            )
                            # 如果手动输入为空，尝试从环境变量获取（作为备用）
                            if not api_key:
                                api_key = os.getenv("DASHSCOPE_API_KEY")
                        else:
                            api_key = os.getenv("DASHSCOPE_API_KEY")
                            if api_key:
                                st.success("✅ 已从环境变量读取 API Key")
                            else:
                                st.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置，请在下方取消勾选并手动输入")
                    
                    result, _ = analyze_with_baseline(analysis_text, api_key)
                    if result:
                        st.markdown("### 📊 分析结果")
                        st.json(result)
                        # 如果使用完整格式，显示选择题匹配
                        if input_mode == "📋 完整格式（测试集格式）" and poem_data.get('choose'):
                            # 尝试从result中提取情感标签
                            sentiment_label = "unknown"
                            if isinstance(result, dict):
                                if 'sentiment' in result:
                                    sentiment_label = str(result.get('sentiment', 'unknown'))
                                elif 'content' in result:
                                    # 尝试从content中解析
                                    import re
                                    sentiment_match = re.search(r'"sentiment":\s*"([^"]+)"', str(result.get('content', '')))
                                    if sentiment_match:
                                        sentiment_label = sentiment_match.group(1)
                            display_choice_matching(sentiment_label, poem_data.get('choose'))
            
            elif analysis_mode == "已训练模型预测":
                models = get_model_list()
                if not models:
                    st.warning("没有可用的模型，请先训练一个模型")
                else:
                    model_name = st.selectbox("选择模型", [m["name"] for m in models], 
                                             key="model_select_analysis")
                    selected_model = next(m for m in models if m["name"] == model_name)
                    predict_with_model(analysis_text, selected_model)
                    # 如果使用完整格式，显示选择题匹配
                    if input_mode == "📋 完整格式（测试集格式）" and poem_data.get('choose'):
                        # 需要从预测结果中获取情感标签
                        pass


def display_analysis_result_aliyun(result: Dict, report: str, poem_data: Optional[Dict] = None):
    """显示阿里云NLP分析结果"""
    st.markdown("### 📊 分析结果")
    
    sentiment = result.get("sentiment", {})
    sentiment_label = sentiment.get("sentiment", "unknown")
    confidence = sentiment.get("confidence", 0)
    
    # 如果提供了完整数据，显示标题和作者
    if poem_data:
        col_title_author = st.columns(2)
        with col_title_author[0]:
            if poem_data.get('title'):
                st.markdown(f"**标题**: {poem_data['title']}")
        with col_title_author[1]:
            if poem_data.get('author'):
                st.markdown(f"**作者**: {poem_data['author']}")
        st.markdown("---")
    
    # 情感显示
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**情感类型**: {get_sentiment_emoji(sentiment_label)} {sentiment_label}")
        st.markdown(f"**置信度**: {confidence:.2%}")
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "置信度 (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': get_sentiment_color(sentiment_label)},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}]}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, width='stretch')
    
    # 如果提供了选择题，显示匹配结果
    if poem_data and poem_data.get('choose'):
        display_choice_matching(sentiment_label, poem_data.get('choose'))
    
    # 关键词和实体
    keywords = result.get("keywords", {}).get("keywords", [])
    entities = result.get("entities", {}).get("entities", [])
    
    if keywords:
        st.markdown("### 🔑 关键词")
        keyword_html = " ".join([f'<span class="keyword-tag">{k}</span>' for k in keywords])
        st.markdown(keyword_html, unsafe_allow_html=True)
    
    # 如果提供了qa_words，显示并对比
    if poem_data and poem_data.get('qa_words'):
        st.markdown("### 📖 需要解释的词语")
        qa_words = poem_data.get('qa_words', [])
        for word in qa_words:
            if word in keywords:
                st.success(f"✅ {word} - 已识别为关键词")
            else:
                st.info(f"ℹ️ {word} - 未在关键词中")
    
    if entities:
        st.markdown("### 🏷️ 实体识别")
        entity_html = " ".join([f'<span class="entity-tag">{e["text"]} ({e["type"]})</span>' 
                               for e in entities])
        st.markdown(entity_html, unsafe_allow_html=True)
    
    # 如果提供了qa_sents，显示
    if poem_data and poem_data.get('qa_sents'):
        st.markdown("### 📝 需要翻译的句子")
        for i, sent in enumerate(poem_data.get('qa_sents', []), 1):
            st.text(f"{i}. {sent}")
    
    # 详细报告
    if report:
        st.markdown("### 📄 详细报告")
        st.markdown(report)


def display_choice_matching(sentiment_label: str, choices: Dict[str, str]):
    """显示情感标签与选择题选项的匹配"""
    st.markdown("### 🎯 选择题匹配")
    
    # 简单的关键词匹配
    sentiment_lower = sentiment_label.lower()
    
    matches = []
    for option, text in choices.items():
        # 检查情感标签是否与选项文本相关
        text_lower = text.lower()
        score = 0
        
        # 情感关键词映射
        emotion_keywords = {
            "sad": ["悲伤", "伤感", "哀伤", "愁", "悲", "伤", "惜别", "离别", "离愁"],
            "happy": ["欢快", "快乐", "喜悦", "愉快", "高兴"],
            "peaceful": ["宁静", "平静", "淡泊", "闲适"],
            "melancholic": ["无奈", "惆怅", "感慨"],
            "mixed": ["期待", "希望", "向往"]
        }
        
        # 检查选项文本中的关键词
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                if emotion in sentiment_lower or any(kw in sentiment_lower for kw in keywords):
                    score += 1
        
        # 直接文本匹配
        if sentiment_lower in text_lower or any(word in text_lower for word in sentiment_lower.split()):
            score += 2
        
        matches.append({
            "option": option,
            "text": text,
            "score": score
        })
    
    # 按分数排序
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # 显示匹配结果
    for match in matches:
        if match['score'] > 0:
            st.success(f"**{match['option']}**: {match['text']} (匹配度: {match['score']})")
        else:
            st.info(f"**{match['option']}**: {match['text']}")
    
    # 推荐答案
    if matches and matches[0]['score'] > 0:
        st.markdown(f"**💡 推荐答案**: {matches[0]['option']} - {matches[0]['text']}")


def predict_with_model(text: str, model_info: Dict):
    """使用已训练模型进行预测"""
    model_type = model_info.get('type', 'transformer')
    
    # Few-Shot模型需要特殊处理（需要API调用）
    if model_type == "llm_fewshot":
        st.warning("⚠️ Few-Shot Learning模型需要使用大模型API进行预测")
        st.info("""
        **Few-Shot模型说明**：
        - 该模型基于Few-Shot Learning方法，需要通过大模型API进行推理
        - 请使用"🎯 多任务理解"页面中的功能，或使用 `train_llm_fewshot.py` 脚本进行预测
        - 预测时需要API密钥（OpenAI或通义千问）
        """)
        
        # 显示模型信息
        examples_file = model_info.get('examples_file')
        eval_file = model_info.get('eval_file')
        
        if eval_file and Path(eval_file).exists():
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                    accuracy = eval_data.get('accuracy', 0)
                    st.info(f"模型验证准确率: {accuracy:.2%}")
            except Exception:
                pass
        
        if examples_file and Path(examples_file).exists():
            try:
                with open(examples_file, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                    st.info(f"Few-Shot示例数量: {len(examples)} 个")
            except Exception:
                pass
        
        return
    
    # Transformer模型的标准预测流程
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_info['path'])
        model = AutoModelForSequenceClassification.from_pretrained(model_info['path'])
        model.eval()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_id = torch.argmax(logits, dim=1).item()
        
        label_map = load_label_mapping(model_info.get('label_map'))
        if label_map:
            id_to_emotion = label_map.get('id_to_emotion', {})
            predicted_emotion = id_to_emotion.get(str(predicted_id), "未知")
            confidence = probabilities[0][predicted_id].item()
            
            st.markdown("### 📊 预测结果")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**预测情感**: {predicted_emotion}")
                st.success(f"**置信度**: {confidence:.2%}")
            
            with col2:
                # 显示所有类别的概率
                st.markdown("**所有类别概率**")
                prob_dict = {id_to_emotion.get(str(i), f"类别{i}"): probabilities[0][i].item() 
                            for i in range(len(probabilities[0]))}
                prob_df = pd.DataFrame(list(prob_dict.items()), columns=["情感", "概率"])
                fig = px.bar(prob_df, x="情感", y="概率", title="情感概率分布")
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("无法加载标签映射")
    except Exception as e:
        st.error(f"预测失败: {e}")


# ==================== 测试功能 ====================

def show_testing_page():
    """显示测试页面"""
    st.markdown('<h1 class="main-header">🧪 模型测试</h1>', unsafe_allow_html=True)
    
    models = get_model_list()
    if not models:
        st.warning("⚠️ 没有找到已训练的模型，请先训练一个模型")
        return
    
    # 模型选择
    model_names = [m["name"] for m in models]
    selected_model_name = st.selectbox("选择模型", model_names)
    selected_model = next(m for m in models if m["name"] == selected_model_name)
    
    # 显示模型信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**模型路径**: {selected_model['path']}")
    with col2:
        label_map = load_label_mapping(selected_model.get('label_map'))
        if label_map:
            num_labels = len(label_map.get('id_to_emotion', {}))
            st.info(f"**标签数量**: {num_labels}")
    with col3:
        model_path = Path(selected_model['path'])
        if model_path.exists():
            size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024*1024)
            st.info(f"**模型大小**: {size_mb:.1f} MB")
    
    # 测试模式
    test_mode = st.radio("测试模式", ["单文本测试", "批量测试"], horizontal=True)
    
    if test_mode == "单文本测试":
        test_text = st.text_area("输入古诗词", height=100,
                                placeholder="例如：床前明月光，疑是地上霜。举头望明月，低头思故乡。")
        
        if st.button("🔍 预测", type="primary"):
            if test_text.strip():
                predict_with_model(test_text, selected_model)
            else:
                st.warning("请输入测试文本")
    
    else:
        st.markdown("### 📁 批量测试")
        uploaded_file = st.file_uploader("上传JSON文件", type=['json'])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    st.info(f"已加载 {len(data)} 条数据")
                    if st.button("开始批量预测"):
                        batch_predict(data, selected_model)
                else:
                    st.error("JSON文件格式错误，应为数组格式")
            except Exception as e:
                st.error(f"文件解析失败: {e}")


def batch_predict(data: List[Dict], model_info: Dict):
    """批量预测"""
    model_type = model_info.get('type', 'transformer')
    
    # Few-Shot模型不支持批量预测（需要API调用）
    if model_type == "llm_fewshot":
        st.warning("⚠️ Few-Shot Learning模型暂不支持批量预测功能")
        st.info("Few-Shot模型需要通过大模型API进行预测，请使用 `train_llm_fewshot.py` 脚本进行批量预测")
        return
    
    # Transformer模型的标准批量预测流程
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_info['path'])
        model = AutoModelForSequenceClassification.from_pretrained(model_info['path'])
        model.eval()
        
        label_map = load_label_mapping(model_info.get('label_map'))
        id_to_emotion = label_map.get('id_to_emotion', {}) if label_map else {}
        
        results = []
        progress_bar = st.progress(0)
        
        for i, item in enumerate(data):
            text = item.get('content', '')
            if not text:
                continue
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_id = torch.argmax(logits, dim=1).item()
                confidence = torch.softmax(logits, dim=1)[0][predicted_id].item()
            
            predicted_emotion = id_to_emotion.get(str(predicted_id), "未知")
            results.append({
                **item,
                "predicted_emotion": predicted_emotion,
                "confidence": confidence
            })
            
            progress_bar.progress((i + 1) / len(data))
        
        # 显示结果
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        # 下载结果
        csv = results_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载结果 (CSV)", csv, "predictions.csv", "text/csv")
        
    except Exception as e:
        st.error(f"批量预测失败: {e}")


# ==================== 多任务理解功能 ====================

def show_multitask_page():
    """显示多任务理解页面"""
    st.markdown('<h1 class="main-header">🎯 多任务理解</h1>', unsafe_allow_html=True)
    
    st.info("""
    **多任务理解功能**支持以下任务：
    1. **词语释义** - 对古诗词中的关键词进行注释
    2. **句子翻译** - 将古诗词翻译成白话文
    3. **情感推理** - 从多个选项中选择最符合诗歌情感的一项
    
    该功能使用 Few-Shot Learning 方法，基于大语言模型的推理能力。
    """)
    
    # 任务类型选择
    task_type = st.radio(
        "选择任务类型",
        ["词语释义", "句子翻译", "情感推理"],
        horizontal=True
    )
    
    # 模型配置
    st.markdown("### 🔧 模型配置")
    col_model, col_key = st.columns(2)
    
    with col_model:
        model_type = st.selectbox(
            "大模型类型",
            ["qwen", "openai", "local"],
            help="选择使用的大模型API"
        )
    
    with col_key:
        use_env_key = st.checkbox("使用环境变量", value=True, help="从环境变量读取API Key")
        api_key = None
        if not use_env_key:
            if model_type == "openai":
                api_key = st.text_input("OpenAI API Key", type="password")
            elif model_type == "qwen":
                api_key = st.text_input("通义千问 API Key", type="password")
    
    # 输入区域
    st.markdown("### 📝 输入古诗词")
    col_title, col_author = st.columns(2)
    with col_title:
        title = st.text_input("标题", placeholder="例如：静夜思")
    with col_author:
        author = st.text_input("作者", placeholder="例如：李白")
    
    content = st.text_area(
        "诗词内容 *",
        height=150,
        placeholder="例如：床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        help="必填项"
    )
    
    # 根据任务类型显示不同的输入选项
    if task_type == "词语释义":
        st.markdown("### 📖 需要解释的词语（可选）")
        keywords_input = st.text_area(
            "输入需要解释的词语，每行一个",
            height=100,
            placeholder="例如：\n明月\n故乡",
            help="如果不输入，系统会自动识别关键词"
        )
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()] if keywords_input else []
    
    elif task_type == "句子翻译":
        st.info("💡 系统会将整首诗词翻译成白话文")
    
    elif task_type == "情感推理":
        st.markdown("### 🎯 选择题选项")
        col_a, col_b = st.columns(2)
        col_c, col_d = st.columns(2)
        
        with col_a:
            option_a = st.text_input("选项 A", placeholder="例如：欢快的重逢")
        with col_b:
            option_b = st.text_input("选项 B", placeholder="例如：仕途的无奈")
        with col_c:
            option_c = st.text_input("选项 C", placeholder="例如：对未来的期待")
        with col_d:
            option_d = st.text_input("选项 D", placeholder="例如：惜别的感伤")
        
        options = {}
        if option_a:
            options["A"] = option_a
        if option_b:
            options["B"] = option_b
        if option_c:
            options["C"] = option_c
        if option_d:
            options["D"] = option_d
    
    # 执行按钮
    if st.button("🚀 开始分析", type="primary", width='stretch'):
        if not content.strip():
            st.warning("请输入诗词内容（必填）")
            return
        
        if task_type == "情感推理" and not options:
            st.warning("请至少输入一个选项")
            return
        
        try:
            # 导入多任务训练器
            from train_poem_multitask import PoemMultiTaskTrainer, load_train_data, load_eval_choice_data
            
            # 初始化训练器
            with st.spinner("正在初始化模型..."):
                trainer = PoemMultiTaskTrainer(model_type=model_type, api_key=api_key)
                if not trainer.client:
                    st.error(f"❌ {model_type.upper()}客户端初始化失败，请检查API Key配置")
                    return
            
            # 准备数据
            poem_data = {
                "title": title,
                "author": author,
                "content": content
            }
            
            # 加载示例数据
            with st.spinner("正在加载示例数据..."):
                train_data = load_train_data("data/train-data")
                eval_data = load_eval_choice_data("data/eval-data/eval_data.json")
            
            # 根据任务类型执行
            if task_type == "词语释义":
                # 选择示例（包含keywords字段的数据）
                examples = [item for item in train_data if item.get("keywords")][:3]
                
                with st.spinner("正在分析词语释义..."):
                    result = trainer.predict_phrase(poem_data, examples)
                
                st.markdown("### 📊 分析结果")
                if result:
                    st.json(result)
                    # 美化显示
                    st.markdown("#### 词语释义")
                    for word, meaning in result.items():
                        st.markdown(f"**{word}**: {meaning}")
                else:
                    st.warning("未能提取到词语释义，请检查输入")
            
            elif task_type == "句子翻译":
                # 选择示例（包含trans字段的数据）
                examples = [item for item in train_data if item.get("trans")][:3]
                
                with st.spinner("正在翻译..."):
                    result = trainer.predict_sentence(poem_data, examples)
                
                st.markdown("### 📊 翻译结果")
                st.markdown(f"**原文**: {content}")
                st.markdown("---")
                st.markdown(f"**译文**: {result}")
            
            elif task_type == "情感推理":
                # 选择示例（优先选择有答案的，如果没有则使用有选择题选项的）
                examples_with_answer = [item for item in eval_data if item.get("answer") and item.get("choose")]
                if examples_with_answer:
                    examples = examples_with_answer[:3]
                else:
                    # 如果没有带答案的示例，使用有选择题选项的作为示例
                    examples = [item for item in eval_data if item.get("choose")][:3]
                
                if not examples:
                    st.warning("未找到情感推理示例数据，请确保 data/eval-data/eval_data.json 存在且包含选择题选项（choose字段）")
                    return
                
                # 如果没有带答案的示例，给出提示但继续执行
                if not examples_with_answer:
                    st.info("💡 提示：示例数据中没有答案字段，系统将使用Few-Shot Learning进行推理，准确度可能略有影响")
                
                with st.spinner("正在推理情感..."):
                    result = trainer.predict_emotion(poem_data, examples, options)
                
                st.markdown("### 📊 推理结果")
                st.success(f"**预测选项**: {result}")
                
                # 显示所有选项
                st.markdown("#### 所有选项")
                for opt_key, opt_text in options.items():
                    if opt_key == result:
                        st.success(f"**{opt_key}**: {opt_text} ✅")
                    else:
                        st.text(f"{opt_key}: {opt_text}")
        
        except ImportError as e:
            st.error(f"导入失败: {e}")
            st.info("请确保 train_poem_multitask.py 文件存在")
        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc())


# ==================== 模型管理功能 ====================

def show_model_management_page():
    """显示模型管理页面"""
    st.markdown('<h1 class="main-header">📊 模型管理</h1>', unsafe_allow_html=True)
    
    models = get_model_list()
    
    if not models:
        st.warning("⚠️ 没有找到已训练的模型")
        return
    
    st.markdown("### 📦 已训练模型列表")
    
    for model in models:
        model_type = model.get('type', 'unknown')
        with st.expander(f"📦 {model['name']} ({model_type})", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text(f"路径: {model['path']}")
                st.text(f"类型: {model_type}")
            
            with col2:
                # 对于llm_fewshot类型，显示评估结果
                if model_type == "llm_fewshot":
                    eval_file = model.get('eval_file')
                    if eval_file and Path(eval_file).exists():
                        try:
                            with open(eval_file, 'r', encoding='utf-8') as f:
                                eval_data = json.load(f)
                                accuracy = eval_data.get('accuracy', 0)
                                total = eval_data.get('total', 0)
                                correct = eval_data.get('correct', 0)
                                st.metric("准确率", f"{accuracy:.2%}")
                                st.text(f"测试样本: {total} 条")
                                st.text(f"正确预测: {correct} 条")
                        except Exception as e:
                            st.warning(f"无法加载评估结果: {e}")
                    else:
                        st.info("Few-Shot Learning模型")
                        st.text("（评估结果文件不存在）")
                    
                    # 显示Few-Shot示例数量
                    examples_file = model.get('examples_file')
                    if examples_file and Path(examples_file).exists():
                        try:
                            with open(examples_file, 'r', encoding='utf-8') as f:
                                examples = json.load(f)
                                st.text(f"Few-Shot示例: {len(examples)} 个")
                        except Exception:
                            pass
                else:
                    # 对于transformer类型，显示标签映射
                    label_map = load_label_mapping(model.get('label_map'))
                    if label_map:
                        num_labels = len(label_map.get('id_to_emotion', {}))
                        st.metric("标签数", num_labels)
                        
                        # 显示标签列表
                        with st.expander("查看标签"):
                            for id_val, emotion in label_map.get('id_to_emotion', {}).items():
                                st.text(f"{id_val}: {emotion}")
                    else:
                        st.info("标签映射文件不存在")
            
            with col3:
                model_path = Path(model['path'])
                if model_path.exists():
                    size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024*1024)
                    st.metric("模型大小", f"{size_mb:.1f} MB")
                
                if st.button("🗑️ 删除", key=f"delete_{model['name']}"):
                    import shutil
                    model_dir = Path(model['path'])
                    if model_dir.exists():
                        try:
                            shutil.rmtree(model_dir)
                            st.success(f"已删除模型: {model['name']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {e}")
                    else:
                        st.warning(f"模型目录不存在: {model_dir}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ 系统配置")
        
        # 依赖检查
        st.markdown("### 📦 依赖检查")
        missing = check_dependencies()
        if missing:
            st.error(f"缺少依赖: {', '.join(missing)}")
            st.info("请运行: pip install -r requirements_train.txt")
        else:
            st.success("✅ 所有依赖已安装")
        
        st.markdown("---")
        
        # 阿里云配置（全局配置，可在各个页面使用）
        st.markdown("### 🔑 阿里云凭证")
        use_env_global = st.checkbox("使用环境变量", value=True, key="use_env_global", 
                                     help="如果勾选，将使用环境变量中的凭证")
        
        access_key_id_global = None
        access_key_secret_global = None
        
        if not use_env_global:
            access_key_id_global = st.text_input("Access Key ID", type="default", 
                                                 key="global_aliyun_access_key_id")
            access_key_secret_global = st.text_input("Access Key Secret", type="password", 
                                                     key="global_aliyun_access_key_secret")
        
        # 保存到session_state
        if not use_env_global and access_key_id_global and access_key_secret_global:
            st.session_state['global_aliyun_config'] = {
                'access_key_id': access_key_id_global,
                'access_key_secret': access_key_secret_global
            }
        else:
            st.session_state['global_aliyun_config'] = None
        
        st.markdown("---")
        
        # 页面选择
        st.markdown("### 📑 功能导航")
        page = st.radio(
            "选择功能",
            ["📜 情感分析", "🏋️ 模型训练", "🧪 模型测试", "📊 模型管理", "🎯 多任务理解"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ 关于")
        st.info("""
        古诗词情感分析系统
        
        整合功能：
        - 情感分析（阿里云NLP/通义千问/已训练模型）
        - 模型训练（Transformer/Few-Shot/微调）
        - 模型测试与预测
        - 模型管理
        - 多任务理解（词语释义/句子翻译/情感推理）
        """)
    
    # 主内容区
    if page == "📜 情感分析":
        show_analysis_page()
    elif page == "🏋️ 模型训练":
        show_training_page()
    elif page == "🧪 模型测试":
        show_testing_page()
    elif page == "📊 模型管理":
        show_model_management_page()
    elif page == "🎯 多任务理解":
        show_multitask_page()


if __name__ == "__main__":
    # 初始化session state
    if 'training' not in st.session_state:
        st.session_state['training'] = False
    if 'start_training' not in st.session_state:
        st.session_state['start_training'] = False
    if 'poem_title' not in st.session_state:
        st.session_state['poem_title'] = ''
    if 'poem_content' not in st.session_state:
        st.session_state['poem_content'] = ''
    if 'poem_options' not in st.session_state:
        st.session_state['poem_options'] = {}
    
    main()

