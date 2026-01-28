#!/usr/bin/env python3
"""
PPTEval 批量评估脚本

使用 PPTEval 框架对生成的 PPT 进行批量评估。
评估维度：Content（内容）、Design（设计）、Coherence（连贯性）

使用方法:
    python batch_eval.py --config batch_eval_config.yaml
    python batch_eval.py --config batch_eval_config.yaml --dry-run
    python batch_eval.py --config batch_eval_config.yaml --resume
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import yaml
from pdf2image import convert_from_path
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# 直接导入本仓库的 paths 模块
from paths import get_data_source_dirs, get_valid_item_dirs


def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_eval_{timestamp}.log"

    logger = logging.getLogger("batch_eval")
    logger.setLevel(log_level)

    # 文件处理器
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(log_level)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_script_dir() -> Path:
    """获取脚本所在目录"""
    return Path(__file__).resolve().parent


def resolve_data_root(config: dict) -> Path:
    """解析数据根目录"""
    data_root = config.get("data_root", ".")
    script_dir = get_script_dir()
    
    # 如果是相对路径，相对于脚本目录解析
    data_root_path = Path(data_root)
    if not data_root_path.is_absolute():
        data_root_path = (script_dir / data_root_path).resolve()
    
    return data_root_path



def find_ppt_file(data_dir: Path, config: dict) -> Path | None:
    """在数据目录下查找生成的 PPT 文件"""
    result_dir = config.get("generation_result_dir", "generation_task/results")
    ppt_filenames = config.get("ppt_filenames", ["slides.pptx", "slides.pdf"])
    
    result_path = data_dir / result_dir
    
    for filename in ppt_filenames:
        ppt_file = result_path / filename
        if ppt_file.exists():
            return ppt_file
    
    return None


def collect_eval_cases(
    data_root: Path,
    config: dict,
    logger: logging.Logger
) -> list[dict]:
    """收集所有待评估的用例"""
    cases = []
    
    # 获取数据源目录（使用本仓库的 paths 模块）
    source_dirs = get_data_source_dirs(base_dir=data_root)
    
    exclude_dirs = set(config.get("exclude_dirs", []))
    filter_keywords = config.get("filter_keywords", [])
    max_cases = config.get("max_cases", 0)
    
    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.debug(f"Source dir not found: {source_dir}")
            continue
        
        # 获取有效的子目录
        item_dirs = get_valid_item_dirs(source_dir, skip_names=exclude_dirs)
        
        for item_dir in item_dirs:
            # 应用关键词过滤
            if filter_keywords:
                item_path_str = str(item_dir)
                if not any(kw in item_path_str for kw in filter_keywords):
                    continue
            
            # 查找 PPT 文件
            ppt_file = find_ppt_file(item_dir, config)
            if ppt_file is None:
                logger.debug(f"No PPT file found in {item_dir}")
                continue
            
            cases.append({
                "data_dir": item_dir,
                "ppt_file": ppt_file,
                "relative_path": item_dir.relative_to(data_root),
            })
            
            if max_cases > 0 and len(cases) >= max_cases:
                break
        
        if max_cases > 0 and len(cases) >= max_cases:
            break
    
    logger.info(f"Collected {len(cases)} evaluation cases")
    return cases


def load_state(state_file: Path) -> dict:
    """加载状态文件"""
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_state(state_file: Path, state: dict):
    """保存状态文件"""
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def get_score_filename(model_name: str) -> str:
    """生成分数文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 简化模型名称
    model_abbr = model_name.split("/")[-1].replace(".", "_")
    return f"{model_abbr}_{timestamp}_score"


def pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """将 PDF 转换为图片"""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=72)
    image_paths = []
    for i, img in enumerate(images):
        img_path = output_dir / f"slide_{i + 1:04d}.jpg"
        img.save(str(img_path))
        image_paths.append(img_path)
    return image_paths


async def ppt_to_images_async(ppt_path: Path, output_dir: Path) -> list[Path]:
    """将 PPTX 转换为图片（异步）"""
    import tempfile
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果已存在图片，直接返回
    existing_images = sorted(glob(str(output_dir / "slide_*.jpg")))
    if existing_images:
        return [Path(p) for p in existing_images]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 使用 LibreOffice 转换为 PDF
        command = [
            "soffice",
            "--headless",
            "--convert-to", "pdf",
            str(ppt_path),
            "--outdir", temp_dir,
        ]
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"soffice failed: {stderr.decode()}")
        
        # 找到生成的 PDF
        pdf_files = list(Path(temp_dir).glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(f"No PDF generated from {ppt_path}")
        
        # PDF 转图片
        return pdf_to_images(pdf_files[0], output_dir)


async def convert_to_images(ppt_path: Path, output_dir: Path) -> list[Path]:
    """将 PPT/PDF 转换为图片"""
    suffix = ppt_path.suffix.lower()
    
    # 如果已存在图片，直接返回
    existing_images = sorted(glob(str(output_dir / "slide_*.jpg")))
    if existing_images:
        return [Path(p) for p in existing_images]
    
    if suffix == ".pdf":
        return pdf_to_images(ppt_path, output_dir)
    elif suffix in [".pptx", ".ppt"]:
        return await ppt_to_images_async(ppt_path, output_dir)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


class PPTEvaluator:
    """PPT 评估器"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._init_models()
        self._load_prompts()
    
    def _init_models(self):
        """初始化模型"""
        from pptagent.llms import AsyncLLM
        
        model_config = self.config.get("model", {})
        
        api_base = model_config.get("api_base") or os.environ.get("API_BASE")
        language_model = model_config.get("language_model") or os.environ.get("LANGUAGE_MODEL", "gpt-4.1")
        vision_model = model_config.get("vision_model") or os.environ.get("VISION_MODEL", "gpt-4.1")
        
        self.language_model = AsyncLLM(language_model, api_base)
        self.vision_model = AsyncLLM(vision_model, api_base)
        self.model_name = language_model
        
        self.logger.info(f"Language model: {language_model}")
        self.logger.info(f"Vision model: {vision_model}")
        if api_base:
            self.logger.info(f"API base: {api_base}")
        
        # 检查 API Key 状态
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            self.logger.info(f"API key: {api_key[:8]}...{api_key[-4:]} (length: {len(api_key)})")
        else:
            self.logger.warning("OPENAI_API_KEY environment variable is not set!")
    
    def _load_prompts(self):
        """加载评估提示词"""
        from jinja2 import Template
        from pptagent.utils import package_join
        
        self.text_scorer = Template(
            open(package_join("prompts", "ppteval", "ppteval_content.txt"), encoding="utf-8").read()
        )
        self.vision_scorer = Template(
            open(package_join("prompts", "ppteval", "ppteval_style.txt"), encoding="utf-8").read()
        )
        self.style_descriptor = open(
            package_join("prompts", "ppteval", "ppteval_describe_style.txt"), encoding="utf-8"
        ).read()
        self.content_descriptor = open(
            package_join("prompts", "ppteval", "ppteval_describe_content.txt"), encoding="utf-8"
        ).read()
        self.ppt_extractor = Template(
            open(package_join("prompts", "ppteval", "ppteval_extract.txt"), encoding="utf-8").read()
        )
        self.logic_scorer = Template(
            open(package_join("prompts", "ppteval", "ppteval_coherence.txt"), encoding="utf-8").read()
        )
        # 文本提取提示词（用于 PDF 图片提取文字，模仿原流程 to_text() 输出格式）
        self.text_extractor_prompt = """Please extract all text content from this slide image. Output the text in the following format:

Title: [slide title if present]
[All text content from the slide, preserving bullet points and structure]

Be accurate and extract the actual text, do not describe or summarize."""
    
    async def _call_vision_model(self, prompt: str, image_path: str, task_name: str = "vision") -> str:
        """调用视觉模型并记录输入输出"""
        self.logger.debug(f"[{task_name}] Vision model input:")
        self.logger.debug(f"  Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"  Prompt: {prompt}")
        self.logger.debug(f"  Image: {image_path}")
        
        try:
            result = await self.vision_model(prompt, image_path)
            self.logger.debug(f"[{task_name}] Vision model output:")
            self.logger.debug(f"  Result: {result[:300]}..." if len(str(result)) > 300 else f"  Result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"[{task_name}] Vision model error: {e}")
            raise
    
    async def _call_language_model(self, prompt: str, task_name: str = "language", return_json: bool = False):
        """调用语言模型并记录输入输出"""
        self.logger.debug(f"[{task_name}] Language model input:")
        self.logger.debug(f"  Prompt: {prompt[:500]}..." if len(prompt) > 500 else f"  Prompt: {prompt}")
        
        try:
            result = await self.language_model(prompt, return_json=return_json)
            self.logger.debug(f"[{task_name}] Language model output:")
            self.logger.debug(f"  Result: {str(result)[:300]}..." if len(str(result)) > 300 else f"  Result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"[{task_name}] Language model error: {e}")
            raise

    async def describe_slide(self, image_path: Path) -> dict:
        """描述幻灯片的内容和风格"""
        descr_file = image_path.with_suffix(".json")
        
        if descr_file.exists():
            with open(descr_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        style_descr = await self._call_vision_model(self.style_descriptor, str(image_path), "describe_style")
        content_descr = await self._call_vision_model(self.content_descriptor, str(image_path), "describe_content")
        
        descr = {"content": content_descr, "style": style_descr}
        with open(descr_file, "w", encoding="utf-8") as f:
            json.dump(descr, f, indent=2, ensure_ascii=False)
        
        return descr
    
    async def eval_slide_design(self, descr: dict) -> dict:
        """评估幻灯片设计"""
        prompt = self.vision_scorer.render(descr=descr["style"])
        return await self._call_language_model(prompt, "eval_design", return_json=True)
    
    async def eval_slide_content(self, descr: dict) -> dict:
        """评估幻灯片内容"""
        prompt = self.text_scorer.render(descr=descr["content"])
        return await self._call_language_model(prompt, "eval_content", return_json=True)
    
    async def eval_coherence(self, ppt_path: Path, slide_images: list[Path]) -> dict:
        """评估演示文稿的连贯性
        
        对于 PPTX 文件：使用原流程 Presentation.from_file().to_text() 提取文本
        对于 PDF 文件：使用图片描述汇总作为替代
        """
        extracted_file = ppt_path.parent / (ppt_path.stem + "_extracted.json")
        
        if extracted_file.exists():
            with open(extracted_file, "r", encoding="utf-8") as f:
                extracted = json.load(f)
        else:
            # 根据文件类型选择不同的提取策略
            if ppt_path.suffix.lower() in [".pptx", ".ppt"]:
                # PPTX: 使用原流程 Presentation.from_file().to_text()
                presentation_text = self._extract_text_from_pptx(ppt_path)
                # 如果 PPTX 解析失败，回退到 Vision Model 提取
                if not presentation_text:
                    presentation_text = await self._extract_text_from_images(slide_images)
            else:
                # PDF: 使用 Vision Model 提取实际文字
                presentation_text = await self._extract_text_from_images(slide_images)
            
            prompt = self.ppt_extractor.render(presentation=presentation_text)
            extracted = await self._call_language_model(prompt, "extract_structure", return_json=True)
            
            with open(extracted_file, "w", encoding="utf-8") as f:
                json.dump(extracted, f, indent=2, ensure_ascii=False)
        
        # 评估连贯性
        prompt = self.logic_scorer.render(presentation=extracted)
        return await self._call_language_model(prompt, "eval_coherence", return_json=True)
    
    def _extract_text_from_pptx(self, ppt_path: Path) -> str:
        """从 PPTX 文件提取文本（原流程）"""
        import tempfile
        from pptagent.presentation import Presentation
        from pptagent.utils import Config
        
        tmp_config = Config(tempfile.mkdtemp())
        try:
            presentation = Presentation.from_file(str(ppt_path), tmp_config)
            return presentation.to_text()
        except Exception as e:
            self.logger.warning(f"Failed to parse PPTX structure: {e}, falling back to image extraction")
            return ""
    
    async def _extract_text_from_images(self, slide_images: list[Path]) -> str:
        """使用 Vision Model 从图片提取实际文字（模仿原流程 to_text() 逻辑）"""
        slide_contents = []
        for i, img_path in enumerate(slide_images):
            # 检查是否已有提取结果
            text_file = img_path.with_suffix(".txt")
            if text_file.exists():
                with open(text_file, "r", encoding="utf-8") as f:
                    slide_text = f.read()
            else:
                # 使用 Vision Model 提取文字
                slide_text = await self._call_vision_model(
                    self.text_extractor_prompt, str(img_path), f"extract_text_slide_{i+1}"
                )
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(slide_text)
            
            slide_contents.append(f"Slide {i+1} of {len(slide_images)}\n{slide_text}")
        
        return "\n----\n".join(slide_contents)
    
    async def evaluate(self, case: dict) -> dict:
        """评估单个用例"""
        ppt_path = case["ppt_file"]
        data_dir = case["data_dir"]
        case_name = case.get("relative_path", ppt_path.name)
        
        self.logger.debug(f"[{case_name}] Starting evaluation...")
        
        # 图片输出目录
        images_dir = ppt_path.parent / (ppt_path.stem + "_images")
        
        # 转换为图片
        self.logger.debug(f"[{case_name}] Converting PPT to images...")
        slide_images = await convert_to_images(ppt_path, images_dir)
        
        if not slide_images:
            raise ValueError(f"No slide images generated from {ppt_path}")
        
        self.logger.debug(f"[{case_name}] Generated {len(slide_images)} slide images")
        
        # 评估每张幻灯片
        evals = {"vision": {}, "content": {}}
        
        for i, img_path in enumerate(slide_images):
            self.logger.debug(f"[{case_name}] Evaluating slide {i+1}/{len(slide_images)}...")
            descr = await self.describe_slide(img_path)
            
            img_key = str(img_path.name)
            evals["vision"][img_key] = await self.eval_slide_design(descr)
            evals["content"][img_key] = await self.eval_slide_content(descr)
        
        # 评估连贯性
        self.logger.debug(f"[{case_name}] Evaluating coherence...")
        evals["logic"] = await self.eval_coherence(ppt_path, slide_images)
        
        # 计算平均分
        scores = self._calculate_scores(evals)
        
        # 清理中间文件
        self._cleanup_intermediate_files(ppt_path, images_dir, slide_images)
        
        return {
            "evals": evals,
            "scores": scores,
            "ppt_file": str(ppt_path),
            "num_slides": len(slide_images),
        }
    
    def _cleanup_intermediate_files(self, ppt_path: Path, images_dir: Path, slide_images: list[Path]):
        """清理中间文件，只保留评分结果"""
        import shutil
        
        try:
            # 删除幻灯片图片的描述 JSON 文件
            for img_path in slide_images:
                json_file = img_path.with_suffix(".json")
                if json_file.exists():
                    json_file.unlink()
                    self.logger.debug(f"Deleted: {json_file}")
                
                # 删除文字提取 txt 文件
                txt_file = img_path.with_suffix(".txt")
                if txt_file.exists():
                    txt_file.unlink()
                    self.logger.debug(f"Deleted: {txt_file}")
            
            # 删除提取的结构文件
            extracted_file = ppt_path.parent / (ppt_path.stem + "_extracted.json")
            if extracted_file.exists():
                extracted_file.unlink()
                self.logger.debug(f"Deleted: {extracted_file}")
            
            # 删除图片目录
            if images_dir.exists():
                shutil.rmtree(images_dir)
                self.logger.debug(f"Deleted directory: {images_dir}")
            
            self.logger.info(f"Cleaned up intermediate files for {ppt_path.name}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup intermediate files: {e}")
    
    def _calculate_scores(self, evals: dict) -> dict:
        """计算各维度平均分"""
        scores = {}
        
        # Vision/Design 分数
        vision_scores = [
            v.get("score", 0) for v in evals["vision"].values()
            if isinstance(v, dict)
        ]
        scores["design"] = sum(vision_scores) / len(vision_scores) if vision_scores else 0
        
        # Content 分数
        content_scores = [
            v.get("score", 0) for v in evals["content"].values()
            if isinstance(v, dict)
        ]
        scores["content"] = sum(content_scores) / len(content_scores) if content_scores else 0
        
        # Coherence/Logic 分数
        if isinstance(evals.get("logic"), dict):
            scores["coherence"] = evals["logic"].get("score", 0)
        else:
            scores["coherence"] = 0
        
        # 计算三个维度的平均分
        dim_scores = [scores["design"], scores["content"], scores["coherence"]]
        
        # 算术平均
        scores["arithmetic_mean"] = sum(dim_scores) / len(dim_scores)
        
        # 几何平均（避免零值问题）
        if all(s > 0 for s in dim_scores):
            scores["geometric_mean"] = math.pow(
                dim_scores[0] * dim_scores[1] * dim_scores[2], 
                1/3
            )
        else:
            scores["geometric_mean"] = 0.0
        
        # 保留 overall 作为算术平均的别名（向后兼容）
        scores["overall"] = scores["arithmetic_mean"]
        
        return scores


async def run_evaluation(
    cases: list[dict],
    evaluator: PPTEvaluator,
    config: dict,
    state: dict,
    state_file: Path,
    logger: logging.Logger,
    resume: bool = False,
) -> dict:
    """运行批量评估"""
    max_concurrent = config.get("max_concurrent", 4)
    timeout = config.get("timeout", 600)
    overwrite = config.get("overwrite", False)
    
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {"success": [], "failed": []}
    
    async def eval_case(case: dict) -> tuple[dict, dict | None, str | None]:
        case_key = str(case["relative_path"])
        
        # 检查是否已完成
        if not overwrite and case_key in state["completed"]:
            if resume:
                logger.debug(f"Skipping completed: {case_key}")
                return case, state["completed"][case_key], None
        
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    evaluator.evaluate(case),
                    timeout=timeout
                )
                
                # 保存分数
                score_dir = case["data_dir"] / config.get("output", {}).get("score_dir", "generation_task/results")
                score_dir.mkdir(parents=True, exist_ok=True)
                score_filename = get_score_filename(evaluator.model_name)
                score_file = score_dir / score_filename
                
                with open(score_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                result["score_file"] = str(score_file)
                
                return case, result, None
                
            except Exception as e:
                import traceback
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                logger.error(f"Failed to evaluate {case_key}:")
                logger.error(f"  Error: {error_msg}")
                logger.error(f"  Traceback:\n{error_traceback}")
                return case, None, f"{error_msg}\n{error_traceback}"
    
    # 并发执行评估
    tasks = [eval_case(case) for case in cases]
    
    for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
        case, result, error = await coro
        case_key = str(case["relative_path"])
        
        if result is not None:
            state["completed"][case_key] = result
            results["success"].append(case_key)
            logger.info(f"✓ {case_key}: {result['scores']}")
        else:
            state["failed"][case_key] = error
            results["failed"].append(case_key)
        
        # 定期保存状态
        save_state(state_file, state)
    
    return results


def print_summary(results: dict, state: dict, logger: logging.Logger):
    """打印评估摘要"""
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    
    total = len(results["success"]) + len(results["failed"])
    logger.info(f"Total cases: {total}")
    logger.info(f"Success: {len(results['success'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    
    if results["success"]:
        # 计算总体平均分
        all_scores = {"design": [], "content": [], "coherence": [], "overall": []}
        for case_key in results["success"]:
            if case_key in state["completed"]:
                scores = state["completed"][case_key].get("scores", {})
                for dim in all_scores:
                    if dim in scores:
                        all_scores[dim].append(scores[dim])
        
        logger.info("\nAverage Scores:")
        for dim, scores in all_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                logger.info(f"  {dim.capitalize()}: {avg:.2f}")


def main():
    parser = argparse.ArgumentParser(description="PPTEval 批量评估脚本")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="batch_eval_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行，只显示将评估的用例"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑，跳过已完成的用例"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有待评估用例"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试日志"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = get_script_dir() / config_path
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = get_script_dir() / config.get("log_dir", "batch_eval_logs")
    logger = setup_logging(log_dir, log_level)
    
    logger.info(f"Loaded config from: {config_path}")
    
    # 解析数据根目录
    data_root = resolve_data_root(config)
    logger.info(f"Data root: {data_root}")
    
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        sys.exit(1)
    
    # 收集评估用例
    cases = collect_eval_cases(data_root, config, logger)
    
    if args.list or args.dry_run:
        logger.info("\nEvaluation cases:")
        for i, case in enumerate(cases, 1):
            logger.info(f"  {i}. {case['relative_path']} -> {case['ppt_file'].name}")
        
        if args.dry_run:
            logger.info(f"\nDry run complete. Would evaluate {len(cases)} cases.")
        return
    
    if not cases:
        logger.warning("No cases to evaluate")
        return
    
    # 加载状态
    state_file = get_script_dir() / config.get("state_file", "batch_eval_state.json")
    state = load_state(state_file) if args.resume else {"completed": {}, "failed": {}}
    
    # 初始化评估器
    evaluator = PPTEvaluator(config, logger)
    
    # 运行评估
    logger.info(f"Starting evaluation of {len(cases)} cases...")
    results = asyncio.run(
        run_evaluation(cases, evaluator, config, state, state_file, logger, args.resume)
    )
    
    # 保存最终状态
    save_state(state_file, state)
    
    # 打印摘要
    print_summary(results, state, logger)
    
    logger.info(f"\nState saved to: {state_file}")


if __name__ == "__main__":
    main()
