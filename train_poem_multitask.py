"""
多任务大模型 Few-Shot 训练与评估

任务：
1) 古诗词短语释义（词语注释）
2) 古诗词句子翻译（白话文译文）
3) 古诗词情感多选推理

依赖：
- 与 train_llm_fewshot.py 相同的模型初始化方式（openai / qwen / local）
- 数据来源：data/train-data 下的训练集；情感多选使用 data/eval-data/eval_data.json
"""
import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_poem_multitask.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


# ------------------ 工具函数 ------------------ #
def load_train_data(data_dir: str = "data/train-data") -> List[Dict[str, Any]]:
    """加载训练数据"""
    data_dir = Path(data_dir)
    all_data: List[Dict[str, Any]] = []
    json_files = list(data_dir.rglob("*.json"))
    logger.info(f"找到 {len(json_files)} 个训练 JSON 文件")
    for js in json_files:
        try:
            with open(js, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                all_data.append(data)
        except Exception as e:
            logger.error(f"加载失败 {js}: {e}")
    logger.info(f"共加载训练样本 {len(all_data)} 条")
    return all_data


def load_eval_choice_data(path: str = "data/eval-data/eval_data.json") -> List[Dict[str, Any]]:
    """加载情感多选评测数据"""
    p = Path(path)
    if not p.exists():
        logger.warning(f"未找到情感多选数据: {path}")
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("eval_data.json 格式非列表，已忽略")
        return []
    logger.info(f"情感多选评测数据: {len(data)} 条")
    return data


def extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON 块"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


# ------------------ 训练器 ------------------ #
class PoemMultiTaskTrainer:
    def __init__(self, model_type: str = "openai", api_key: Optional[str] = None):
        self.model_type = model_type
        self.api_key = api_key
        self.client = None
        self.model_name = None
        self._init_client()

    # 模型初始化（与 train_llm_fewshot.py 保持一致）
    def _init_client(self):
        if self.model_type == "openai":
            try:
                from openai import OpenAI

                key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    logger.error("❌ OpenAI API Key 未设置")
                    return
                self.client = OpenAI(api_key=key)
                self.model_name = "gpt-3.5-turbo"
                logger.info(f"✅ OpenAI 初始化成功，模型: {self.model_name}")
            except Exception as e:
                logger.error(f"初始化 OpenAI 失败: {e}")
                self.client = None

        elif self.model_type == "qwen":
            try:
                import dashscope

                key = self.api_key or os.getenv("DASHSCOPE_API_KEY")
                if not key:
                    logger.error("❌ 通义千问 API Key 未设置")
                    return
                dashscope.api_key = key
                self.client = dashscope
                self.model_name = "qwen-turbo"
                logger.info(f"✅ 通义千问初始化成功，模型: {self.model_name}")
            except Exception as e:
                logger.error(f"初始化通义千问失败: {e}")
                self.client = None

        elif self.model_type == "local":
            try:
                from openai import OpenAI

                url = os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")
                self.client = OpenAI(base_url=url, api_key="EMPTY")
                self.model_name = "local-llm"
                logger.info(f"✅ 本地 LLM 初始化成功，模型: {self.model_name}")
            except Exception as e:
                logger.error(f"初始化本地 LLM 失败: {e}")
                self.client = None

    def _chat(self, prompt: str) -> str:
        """统一的聊天接口"""
        if not self.client:
            raise RuntimeError("客户端未初始化")
        if self.model_type in ("openai", "local"):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        elif self.model_type == "qwen":
            from dashscope import Generation

            resp = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=0.3,
                max_tokens=800,
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Qwen 调用失败: {resp.message}")
            return resp.output.text.strip()
        else:
            raise RuntimeError("未知的 model_type")

    # ------------------ Prompt 构建 ------------------ #
    def build_phrase_prompt(self, poem: Dict[str, Any], examples: List[Dict[str, Any]], max_examples: int = 3) -> str:
        prompt = "你是古诗词词语注释专家。根据示例，对给定诗词中出现的关键词做简明释义，返回 JSON。\n\n示例：\n"
        for i, ex in enumerate(examples[:max_examples], 1):
            kw = ex.get("keywords", {})
            if not kw:
                continue
            prompt += f"示例{i} 标题：{ex.get('title','')}\n"
            prompt += f"内容：{ex.get('content','')}\n"
            prompt += "释义：{\n"
            for k, v in kw.items():
                prompt += f'  "{k}": "{v}",\n'
            prompt += "}\n\n"

        prompt += "请为下述诗词提供每个关键词的释义，返回 JSON（键为词语，值为释义）：\n"
        prompt += f"标题：{poem.get('title','')}\n"
        prompt += f"内容：{poem.get('content','')}\n"
        prompt += "只返回 JSON，不要其他文字。"
        return prompt

    def build_sentence_prompt(self, poem: Dict[str, Any], examples: List[Dict[str, Any]], max_examples: int = 3) -> str:
        prompt = "你是古诗词翻译专家。根据示例，把诗句翻译成简洁流畅的白话文。\n\n示例：\n"
        for i, ex in enumerate(examples[:max_examples], 1):
            trans = ex.get("trans", "")
            if not trans:
                continue
            prompt += f"示例{i} 标题：{ex.get('title','')}\n"
            prompt += f"内容：{ex.get('content','')}\n"
            prompt += f"译文：{trans}\n\n"
        prompt += "请翻译下述诗词，直接给出白话文译文：\n"
        prompt += f"标题：{poem.get('title','')}\n"
        prompt += f"内容：{poem.get('content','')}\n"
        prompt += "只返回译文，不要其他文字。"
        return prompt

    def build_emotion_prompt(
        self,
        poem: Dict[str, Any],
        examples: List[Dict[str, Any]],
        options: Dict[str, str],
        max_examples: int = 3,
    ) -> str:
        prompt = "你是古诗词情感分析专家。根据示例，在给定选项中选择最符合诗意的情感，返回选项字母。\n\n示例：\n"
        for i, ex in enumerate(examples[:max_examples], 1):
            choose = ex.get("choose", {})
            if not choose:
                continue
            prompt += f"示例{i} 标题：{ex.get('title','')}\n"
            prompt += f"内容：{ex.get('content','')}\n"
            prompt += "选项：\n"
            for k, v in choose.items():
                prompt += f"{k}: {v}\n"
            # 如果有答案，显示答案；如果没有，跳过答案行
            answer = ex.get('answer', '')
            if answer:
                prompt += f"正确答案：{answer}\n\n"
            else:
                prompt += "\n"
        prompt += "请为下述诗词选择最符合的情感选项，只返回选项字母（如 A/B/C/D）：\n"
        prompt += f"标题：{poem.get('title','')}\n"
        prompt += f"内容：{poem.get('content','')}\n"
        prompt += "选项：\n"
        for k, v in options.items():
            prompt += f"{k}: {v}\n"
        prompt += "只返回字母，不要其他内容。"
        return prompt

    # ------------------ 预测接口 ------------------ #
    def predict_phrase(self, poem: Dict[str, Any], examples: List[Dict[str, Any]]) -> Dict[str, str]:
        prompt = self.build_phrase_prompt(poem, examples)
        out = self._chat(prompt)
        parsed = extract_json(out)
        return parsed or {}

    def predict_sentence(self, poem: Dict[str, Any], examples: List[Dict[str, Any]]) -> str:
        prompt = self.build_sentence_prompt(poem, examples)
        return self._chat(prompt)

    def predict_emotion(self, poem: Dict[str, Any], examples: List[Dict[str, Any]], options: Dict[str, str]) -> str:
        prompt = self.build_emotion_prompt(poem, examples, options)
        out = self._chat(prompt)
        m = re.search(r"[A-D]", out)
        return m.group() if m else out.strip()

    # ------------------ 评估 ------------------ #
    @staticmethod
    def _phrase_score(pred: Dict[str, str], truth: Dict[str, str]) -> float:
        if not truth:
            return 0.0
        overlap = set(pred.keys()) & set(truth.keys())
        return len(overlap) / len(truth)

    @staticmethod
    def _sentence_overlap(pred: str, truth: str) -> float:
        if not pred or not truth:
            return 0.0
        # 以字符集合简单衡量重叠度
        s1, s2 = set(pred), set(truth)
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0.0

    def evaluate(
        self,
        train_data: List[Dict[str, Any]],
        eval_choice: List[Dict[str, Any]],
        sample_size: int = 5,
    ) -> Dict[str, Any]:
        """分别对三个任务做小样本评估"""
        if not self.client:
            raise RuntimeError("客户端未初始化")

        output = {"phrase": [], "sentence": [], "emotion": []}

        # 划分训练/验证
        train, val = train_test_split(train_data, test_size=0.2, random_state=42)

        # 1) 词语释义
        val_phrase = [x for x in val if x.get("keywords")] [:sample_size]
        examples_phrase = [x for x in train if x.get("keywords")] [:3]
        for item in val_phrase:
            pred = self.predict_phrase(item, examples_phrase)
            score = self._phrase_score(pred, item.get("keywords", {}))
            output["phrase"].append(
                {
                    "title": item.get("title", ""),
                    "pred": pred,
                    "truth": item.get("keywords", {}),
                    "score": round(score, 3),
                }
            )
            time.sleep(1)

        # 2) 句子翻译
        val_sent = [x for x in val if x.get("trans")] [:sample_size]
        examples_sent = [x for x in train if x.get("trans")] [:3]
        for item in val_sent:
            pred = self.predict_sentence(item, examples_sent)
            score = self._sentence_overlap(pred, item.get("trans", ""))
            output["sentence"].append(
                {
                    "title": item.get("title", ""),
                    "pred": pred,
                    "truth": item.get("trans", ""),
                    "score": round(score, 3),
                }
            )
            time.sleep(1)

        # 3) 情感多选
        val_emotion = eval_choice[:sample_size] if eval_choice else []
        examples_emotion = eval_choice[-3:] if len(eval_choice) >= 3 else val_emotion
        for item in val_emotion:
            pred = self.predict_emotion(item, examples_emotion, item.get("choose", {}))
            correct = pred == item.get("answer", "")
            output["emotion"].append(
                {
                    "idx": item.get("idx"),
                    "title": item.get("title", ""),
                    "pred": pred,
                    "truth": item.get("answer", ""),
                    "correct": correct,
                    "options": item.get("choose", {}),
                }
            )
            time.sleep(1)

        return output


# ------------------ 主入口 ------------------ #
def main():
    parser = argparse.ArgumentParser(description="古诗词多任务 Few-Shot 训练与评估")
    parser.add_argument("--model-type", type=str, default="qwen", choices=["openai", "qwen", "local"])
    parser.add_argument("--api-key", type=str, default=None, help="API Key，未提供则读取环境变量")
    parser.add_argument("--train-data-dir", type=str, default="data/train-data", help="训练数据目录")
    parser.add_argument("--eval-choice-path", type=str, default="data/eval-data/eval_data.json", help="情感多选评测集")
    parser.add_argument("--output-dir", type=str, default="models/poem_multitask", help="输出目录")
    parser.add_argument("--sample-size", type=int, default=5, help="每个任务评估样本数（减少费用）")

    args = parser.parse_args()

    trainer = PoemMultiTaskTrainer(model_type=args.model_type, api_key=args.api_key)
    if not trainer.client:
        logger.error("客户端未初始化，退出")
        return 1

    train_data = load_train_data(args.train_data_dir)
    eval_choice = load_eval_choice_data(args.eval_choice_path)

    results = trainer.evaluate(train_data, eval_choice, sample_size=args.sample_size)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multitask_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总展示
    phrase_avg = sum(r["score"] for r in results["phrase"]) / len(results["phrase"]) if results["phrase"] else 0
    sent_avg = sum(r["score"] for r in results["sentence"]) / len(results["sentence"]) if results["sentence"] else 0
    emo_acc = sum(1 for r in results["emotion"] if r["correct"]) / len(results["emotion"]) if results["emotion"] else 0

    logger.info("=" * 60)
    logger.info(f"词语释义 平均覆盖率: {phrase_avg:.2f}")
    logger.info(f"句子翻译 字符重叠度: {sent_avg:.2f}")
    logger.info(f"情感多选 准确率: {emo_acc:.2%}")
    logger.info(f"结果文件: {out_path}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



