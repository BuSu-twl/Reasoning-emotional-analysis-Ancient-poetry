# API Key 设置指南

## 🔑 问题说明

运行 `train_llm_fewshot.py` 时出现错误：
```
ERROR - 初始化OpenAI客户端失败: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable
ERROR - 客户端未初始化
```

这是因为缺少 API Key 导致的。

---

## ✅ 解决方案

### 方案一：使用通义千问（推荐，国内用户）

通义千问是阿里云的大模型服务，对国内用户更友好。

#### 1. 获取 API Key

访问：https://dashscope.console.aliyun.com/
- 注册/登录阿里云账号
- 开通 DashScope 服务
- 创建 API Key

#### 2. 设置 API Key

**Windows PowerShell:**
```powershell
$env:DASHSCOPE_API_KEY="sk-你的API密钥"
```

**Windows CMD:**
```cmd
set DASHSCOPE_API_KEY=sk-你的API密钥
```

**Linux/Mac:**
```bash
export DASHSCOPE_API_KEY="sk-你的API密钥"
```

#### 3. 运行训练

```bash
python train_llm_fewshot.py --model-type qwen
```

---

### 方案二：使用 OpenAI

#### 1. 获取 API Key

访问：https://platform.openai.com/api-keys
- 注册/登录 OpenAI 账号
- 创建 API Key

#### 2. 设置 API Key

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-你的API密钥"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=sk-你的API密钥
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-你的API密钥"
```

#### 3. 运行训练

```bash
python train_llm_fewshot.py --model-type openai
```

---

### 方案三：命令行参数（临时使用）

如果不想设置环境变量，可以直接在命令行中传递 API Key：

**通义千问:**
```bash
python train_llm_fewshot.py --model-type qwen --api-key sk-你的API密钥
```

**OpenAI:**
```bash
python train_llm_fewshot.py --model-type openai --api-key sk-你的API密钥
```

⚠️ **注意**: 这种方式会在命令行历史中暴露 API Key，不建议在生产环境使用。

---

## 🎯 推荐配置（Windows PowerShell）

### 一次性设置（当前会话）

```powershell
# 设置通义千问 API Key
$env:DASHSCOPE_API_KEY="sk-你的API密钥"

# 运行训练
python train_llm_fewshot.py --model-type qwen
```

### 永久设置（推荐）

1. **创建环境变量配置文件**

在用户目录下创建或编辑 `.env` 文件（PowerShell）：
```powershell
# 打开 PowerShell 配置文件
notepad $PROFILE
```

2. **添加以下内容**

```powershell
# 设置通义千问 API Key
$env:DASHSCOPE_API_KEY="sk-你的API密钥"

# 如果需要使用 OpenAI
# $env:OPENAI_API_KEY="sk-你的API密钥"
```

3. **重新加载配置**

```powershell
. $PROFILE
```

---

## 🔍 验证 API Key 是否设置成功

### PowerShell
```powershell
# 检查通义千问
echo $env:DASHSCOPE_API_KEY

# 检查 OpenAI
echo $env:OPENAI_API_KEY
```

### Python 验证
```python
import os
print("DASHSCOPE_API_KEY:", os.getenv("DASHSCOPE_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
```

---

## 📝 使用 GUI 界面设置

如果你使用 `run_integrated_gui.py` 启动的 GUI 界面：

1. 启动 GUI：
   ```bash
   python run_integrated_gui.py
   ```

2. 在 GUI 界面中：
   - 选择"训练"页面
   - 选择"Few-Shot Learning"
   - 选择模型类型（通义千问/OpenAI）
   - **在 API Key 输入框中输入你的 API Key**
   - 点击"开始训练"

GUI 界面会自动将 API Key 传递给训练脚本。

---

## ⚠️ 常见问题

### 1. API Key 设置后仍然报错

**可能原因：**
- 环境变量设置后没有重新打开终端
- 使用了错误的变量名（注意大小写）
- API Key 格式错误

**解决方法：**
- 在当前终端中重新设置环境变量
- 检查变量名是否正确
- 确认 API Key 以 `sk-` 开头

### 2. 使用命令行参数仍然报错

**可能原因：**
- API Key 格式错误
- 模型类型选择错误

**解决方法：**
- 确认 API Key 完整且正确
- 确认 `--model-type` 参数与 API Key 类型匹配（qwen 对应 DASHSCOPE_API_KEY，openai 对应 OPENAI_API_KEY）

### 3. 训练时出现超时错误

**可能原因：**
- 网络连接不稳定
- API 服务限流

**解决方法：**
- 检查网络连接
- 等待一段时间后重试
- 代码已包含重试机制，会自动重试3次

---

## 📚 相关文档

- [训练脚本选择指南.md](./训练脚本选择指南.md)
- [README_大模型训练.md](./README_大模型训练.md)

---

## 🎉 快速开始示例

### 使用通义千问（推荐）

```powershell
# 1. 设置 API Key
$env:DASHSCOPE_API_KEY="sk-你的API密钥"

# 2. 运行训练
python train_llm_fewshot.py --model-type qwen

# 3. 查看结果
# 训练完成后，结果保存在 models/llm_fewshot/ 目录
```

### 使用 OpenAI

```powershell
# 1. 设置 API Key
$env:OPENAI_API_KEY="sk-你的API密钥"

# 2. 运行训练
python train_llm_fewshot.py --model-type openai

# 3. 查看结果
# 训练完成后，结果保存在 models/llm_fewshot/ 目录
```

---

**修复完成！** 现在代码会在 API Key 未设置时给出清晰的错误提示，并提前退出，避免浪费时间和 API 调用。










