# 📜 古诗词情感分析系统

一个基于深度学习和 NLP 技术的古诗词情感分析系统，支持可视化界面、传统 Transformer 训练和大模型 Few-Shot / 微调训练。

## ✨ 功能特点

- 🎨 **综合可视化界面** - 基于 Streamlit 的 Web 界面，整合分析、训练、测试、模型管理
- 🤖 **多种训练方案**  
  - 传统 Transformer 训练（本地 BERT/RoBERTa 等）  
  - Few-Shot 大模型训练（OpenAI / 通义千问 / 本地 LLM）  
  - 大模型微调训练（如 Qwen2、ChatGLM 等）
- 📊 **丰富展示** - 情感分析、关键词提取、实体识别、训练日志和可视化图表
- 🔧 **一键启动** - 脚本和批处理封装，配置简单

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置阿里云 NLP 凭证（如使用阿里云分析方案）

**Windows (PowerShell):**
```powershell
$env:ALIBABA_CLOUD_ACCESS_KEY_ID="your_key_id"
$env:ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_key_secret"
```

**Linux/Mac:**
```bash
export ALIBABA_CLOUD_ACCESS_KEY_ID="your_key_id"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your_key_secret"
```

### 3. 启动综合可视化界面

**Windows:**
- 双击 `启动综合界面.bat`

**命令行:**
```bash
python run_integrated_gui.py
# 或直接运行 Streamlit
streamlit run poem_analysis_train_gui.py
```

浏览器会自动打开（或手动访问）：
```text
http://localhost:8501
```

## 📖 详细文档

- `启动指南.md`：整体启动与环境配置说明
- `综合界面使用说明.md`：综合可视化界面详细说明
- `训练脚本说明.md`、`训练脚本选择指南.md`：各训练脚本的使用与选择建议
- `README_大模型训练.md`、`大模型训练系统说明.md`：大模型 Few-Shot 与微调训练说明
- `完整文档.md`：项目完整文档索引

## 🗂️ 项目结构（核心）

```text
├── run_integrated_gui.py          # 综合界面启动脚本（推荐入口）
├── poem_analysis_train_gui.py     # 综合可视化界面（分析 + 训练 + 测试 + 模型管理）
├── poem_analysis.py               # 阿里云 NLP 分析逻辑
├── train_poem_sentiment.py        # 传统 Transformer 情感分类训练脚本
├── train_llm_fewshot.py           # 大模型 Few-Shot 训练脚本
├── train_with_llm.py              # 大模型微调训练脚本
├── train_poem_multitask.py        # 多任务 Few-Shot 训练脚本
├── baseline_improved.py           # 基于 LLM 的分析脚本（需要本地/远程 LLM 服务）
├── utils/
│   ├── emotion_mapper.py          # 情感标签归一化与映射
│   └── poem_processor.py          # 传统特征工程相关工具
├── data/
│   ├── train-data/                # 训练数据
│   └── eval-data/                 # 评估/测试数据
├── models/                        # 训练输出及模型目录
│   ├── sentiment_model/           # 传统 Transformer 模型及多次训练版本
│   ├── llm_fewshot/               # Few-Shot 示例库与评估结果
│   └── llm_sentiment_model/       # 大模型微调结果
└── requirements.txt               # 统一依赖列表
```

（旧的 `run_gui.py` / `poem_analysis_gui.py` / `requirements_gui.txt` 已废弃，具体见 `已删除文件说明.md`。）

## 💡 使用场景

- **学习研究** - 古诗词情感分析、Few-Shot Learning 与大模型训练
- **教学演示** - 深度学习 / NLP / 大模型课程的综合案例
- **课程设计** - 深度学习课程设计或毕业设计项目

## 📝 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

- 阿里云 NLP 服务
- 通义千问 / Qwen 系列大模型
- OpenAI API 及兼容生态
- Streamlit、PyTorch、Transformers 等开源项目

---

**开始使用：** `python run_integrated_gui.py` 🚀

## 📄 许可证

[MIT](LICENSE)

