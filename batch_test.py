#!/usr/bin/env python3
"""
Freeform 模式批量测试脚本

用法：
    python batch_test.py [--config config.yaml] [--resume]

功能：
    - 从配置文件加载参数
    - 支持并发控制
    - 支持断点续跑
    - 自动扫描数据目录，排除 unused 目录
"""

import argparse
import asyncio
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml

from deeppresenter.main import AgentLoop
from deeppresenter.utils.typings import ConvertType, InputRequest


@dataclass
class TestCase:
    """单个测试用例"""

    case_id: str
    data_dir: Path
    instruction: str
    attachments: list[str]
    output_path: Path
    status: Literal["pending", "running", "success", "failed", "skipped"] = "pending"
    error_message: str = ""
    start_time: str = ""
    end_time: str = ""


@dataclass
class BatchConfig:
    """批量测试配置"""

    # 数据目录
    data_root: str = "/home/cxs/PPT_Agent/Benchmark/data_results/PPTAgent_v2.0.0"
    # 输出模式
    convert_type: str = "deeppresenter"  # deeppresenter 或 pptagent
    # 并发数
    max_concurrent: int = 1
    # 每个任务的超时时间（秒）
    timeout: int = 1800
    # 是否覆盖已有结果
    overwrite: bool = False
    # 排除的目录名
    exclude_dirs: list[str] = field(default_factory=lambda: ["unused"])
    # 输出文件名
    output_filename: str = "slides.pdf"
    # 状态文件路径（用于断点续跑）
    state_file: str = "batch_test_state.json"
    # 日志目录
    log_dir: str = "batch_test_logs"
    # 筛选条件：只运行包含这些关键词的用例（为空则运行所有）
    filter_keywords: list[str] = field(default_factory=list)
    # 最大运行数量（0 表示不限制）
    max_cases: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "BatchConfig":
        """从 YAML 文件加载配置"""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str):
        """保存配置到 YAML 文件"""
        data = {
            "data_root": self.data_root,
            "convert_type": self.convert_type,
            "max_concurrent": self.max_concurrent,
            "timeout": self.timeout,
            "overwrite": self.overwrite,
            "exclude_dirs": self.exclude_dirs,
            "output_filename": self.output_filename,
            "state_file": self.state_file,
            "log_dir": self.log_dir,
            "filter_keywords": self.filter_keywords,
            "max_cases": self.max_cases,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


class BatchTestRunner:
    """批量测试运行器"""

    def __init__(self, config: BatchConfig, resume: bool = False):
        self.config = config
        self.resume = resume
        self.test_cases: list[TestCase] = []
        self.state_file = Path(config.state_file)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def scan_test_cases(self) -> list[TestCase]:
        """扫描数据目录，发现所有测试用例"""
        data_root = Path(self.config.data_root)
        test_cases = []

        # 获取数据源目录列表
        source_dirs = self._get_data_source_dirs(data_root)

        # 扫描每个数据源目录下的一级文件夹
        for source_dir in source_dirs:
            if not source_dir.exists():
                print(f"警告: 数据源目录不存在 {source_dir}")
                continue

            for item in source_dir.iterdir():
                if not item.is_dir():
                    continue
                # 跳过隐藏目录和排除目录
                if item.name.startswith(".") or item.name in self.config.exclude_dirs:
                    continue

                # 检查是否是数据目录
                if self._is_data_dir(item):
                    case = self._create_test_case(item)
                    if case:
                        test_cases.append(case)

        # 应用筛选条件
        if self.config.filter_keywords:
            test_cases = [
                case
                for case in test_cases
                if any(kw in case.case_id for kw in self.config.filter_keywords)
            ]

        # 限制最大数量
        if self.config.max_cases > 0:
            test_cases = test_cases[: self.config.max_cases]

        return test_cases

    def _get_data_source_dirs(self, data_root: Path) -> list[Path]:
        """从 paths.py 获取数据源目录列表"""
        paths_file = data_root / "utils" / "paths.py"

        if not paths_file.exists():
            print(f"警告: paths.py 不存在于 {paths_file}，将回退到扫描 data_root")
            # 回退：返回 data_root 下所有一级目录
            return [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")]

        try:
            # 动态加载 paths.py 模块
            spec = importlib.util.spec_from_file_location("paths", paths_file)
            if spec is None:
                raise ValueError(f"spec is None for {paths_file}")
            paths_module = importlib.util.module_from_spec(spec)
            if paths_module is None:
                raise ValueError(f"paths_module is None for {paths_file}")
            if spec.loader is None:
                raise ValueError(f"spec.loader is None for {paths_file}")
            spec.loader.exec_module(paths_module)

            # 调用 get_data_source_dirs 函数
            return paths_module.get_data_source_dirs(base_dir=data_root)
        except Exception as e:
            print(f"警告: 加载 paths.py 失败: {e}，将回退到扫描 data_root")
            return [d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    def _is_data_dir(self, path: Path) -> bool:
        """检查是否是有效的数据目录"""
        generation_task = path / "generation_task"
        prompt_file = generation_task / "generation_prompt.md"
        return generation_task.exists() and prompt_file.exists()

    def _create_test_case(self, data_dir: Path) -> TestCase | None:
        """创建测试用例"""
        prompt_file = data_dir / "generation_task" / "generation_prompt.md"

        # 读取指令
        try:
            instruction = prompt_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"警告: 无法读取 {prompt_file}: {e}")
            return None

        # 查找附件
        attachments = self._find_attachments(data_dir)
        if not attachments:
            print(f"警告: {data_dir} 没有找到附件文件")
            # 有些用例可能不需要附件，继续处理
            pass

        # 生成用例 ID
        # 替换 / 为 _ 并移除冒号（冒号会导致 Docker -v 参数解析失败）
        relative_path = data_dir.relative_to(self.config.data_root)
        case_id = str(relative_path).replace("/", "_").replace(":", "")

        # 输出路径
        output_path = data_dir / "generation_task" / "results" / self.config.output_filename

        return TestCase(
            case_id=case_id,
            data_dir=data_dir,
            instruction=instruction,
            attachments=attachments,
            output_path=output_path,
        )

    def _find_attachments(self, data_dir: Path) -> list[str]:
        """查找附件文件"""
        attachments = []

        # 1. 先检查 material.pdf 或 material.md
        for ext in ["pdf", "md"]:
            single_material = data_dir / f"material.{ext}"
            if single_material.exists():
                attachments.append(str(single_material))
                return attachments

        # 2. 检查 material_1.pdf, material_2.pdf, ... 格式
        pattern = re.compile(r"material_(\d+)\.(pdf|md)$")
        material_files = []
        for f in data_dir.iterdir():
            if f.is_file():
                match = pattern.match(f.name)
                if match:
                    idx = int(match.group(1))
                    material_files.append((idx, str(f)))

        # 按序号排序
        material_files.sort(key=lambda x: x[0])
        attachments = [f[1] for f in material_files]

        if attachments:
            return attachments

        # 3. 回退：查找目录下所有 PDF/MD 文件（排除 generation_task 子目录）
        for f in data_dir.iterdir():
            if f.is_file() and f.suffix.lower() in [".pdf", ".md"]:
                attachments.append(str(f))

        # 按文件名排序
        attachments.sort()
        return attachments

    def load_state(self) -> dict:
        """加载状态文件"""
        if self.state_file.exists():
            with open(self.state_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_state(self):
        """保存状态文件"""
        state = {
            "last_update": datetime.now().isoformat(),
            "total_cases": len(self.test_cases),
            "cases": {},
        }
        for case in self.test_cases:
            state["cases"][case.case_id] = {
                "status": case.status,
                "error_message": case.error_message,
                "start_time": case.start_time,
                "end_time": case.end_time,
                "output_path": str(case.output_path),
            }

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def apply_saved_state(self, state: dict):
        """应用保存的状态（用于断点续跑）"""
        saved_cases = state.get("cases", {})
        for case in self.test_cases:
            if case.case_id in saved_cases:
                saved = saved_cases[case.case_id]
                if saved["status"] == "success":
                    case.status = "skipped"
                    print(f"跳过已完成: {case.case_id}")

    async def run_single_case(
        self,
        case: TestCase,
        semaphore: asyncio.Semaphore,
    ) -> TestCase:
        """运行单个测试用例"""
        async with semaphore:
            # 检查是否需要跳过
            if case.status == "skipped":
                return case

            # 检查是否已有输出且不覆盖
            if case.output_path.exists() and not self.config.overwrite:
                case.status = "skipped"
                print(f"跳过已存在: {case.case_id}")
                return case

            case.status = "running"
            case.start_time = datetime.now().isoformat()
            
            # 更新进度
            self.completed_count = getattr(self, 'completed_count', 0)
            total = getattr(self, 'total_pending', 1)
            progress = f"[{self.completed_count + 1}/{total}]"
            
            print(f"\n{'='*60}")
            print(f"{progress} 开始运行: {case.case_id}")
            print(f"时间: {case.start_time}")
            print(f"附件数: {len(case.attachments)}")
            if case.attachments:
                for att in case.attachments[:3]:  # 只显示前3个附件
                    print(f"  - {Path(att).name}")
                if len(case.attachments) > 3:
                    print(f"  ... 还有 {len(case.attachments) - 3} 个附件")
            print(f"{'='*60}")

            # 创建 AgentLoop
            session_id = f"batch/{datetime.now().strftime('%Y%m%d')}/{case.case_id}"

            try:
                loop = AgentLoop(session_id=session_id)

                # 确定转换类型
                convert_type = (
                    ConvertType.DEEPPRESENTER
                    if self.config.convert_type == "deeppresenter"
                    else ConvertType.PPTAGENT
                )

                request = InputRequest(
                    instruction=case.instruction,
                    attachments=case.attachments,
                    convert_type=convert_type,
                )

                # 运行（带超时）
                output_file = None
                async with asyncio.timeout(self.config.timeout):
                    async for msg in loop.run(request):
                        if isinstance(msg, (str, Path)):
                            output_file = Path(msg)
                            print(f"生成输出: {output_file}")

                # 复制输出文件到目标位置
                if output_file and output_file.exists():
                    case.output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(output_file, case.output_path)
                    case.status = "success"
                    print(f"✓ 成功: {case.case_id} -> {case.output_path}")
                else:
                    case.status = "failed"
                    case.error_message = "未生成输出文件"
                    print(f"✗ 失败: {case.case_id} - 未生成输出文件")

            except asyncio.TimeoutError:
                case.status = "failed"
                case.error_message = f"超时 ({self.config.timeout}s)"
                print(f"✗ 超时: {case.case_id}")

            except Exception as e:
                case.status = "failed"
                case.error_message = str(e)
                print(f"✗ 错误: {case.case_id} - {e}")

            finally:
                case.end_time = datetime.now().isoformat()
                # 更新完成计数
                if hasattr(self, 'completed_count'):
                    self.completed_count += 1
                # 每完成一个用例就保存状态
                self.save_state()

            return case

    async def run_all(self):
        """运行所有测试用例"""
        # 扫描测试用例
        print("正在扫描测试用例...")
        self.test_cases = self.scan_test_cases()
        print(f"发现 {len(self.test_cases)} 个测试用例")

        if not self.test_cases:
            print("没有找到测试用例")
            return

        # 断点续跑
        if self.resume:
            state = self.load_state()
            if state:
                print(f"加载状态文件: {self.state_file}")
                self.apply_saved_state(state)

        # 检查已存在的输出
        if not self.config.overwrite:
            for case in self.test_cases:
                if case.status == "pending" and case.output_path.exists():
                    case.status = "skipped"

        # 统计需要运行的用例
        pending_cases = [c for c in self.test_cases if c.status == "pending"]
        skipped_cases = [c for c in self.test_cases if c.status == "skipped"]
        print(f"需要运行: {len(pending_cases)} 个用例")
        print(f"已跳过: {len(skipped_cases)} 个用例 (已完成或已存在)")

        if not pending_cases:
            print("所有用例已完成或跳过")
            self.print_summary()
            return

        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # 保存初始状态
        self.save_state()

        # 并发运行所有用例
        print(f"\n开始批量测试 (并发数: {self.config.max_concurrent})...")
        start_time = datetime.now()

        # 使用计数器跟踪进度
        self.completed_count = 0
        self.total_pending = len(pending_cases)

        tasks = [self.run_single_case(case, semaphore) for case in self.test_cases]
        await asyncio.gather(*tasks)

        end_time = datetime.now()
        duration = end_time - start_time

        # 保存最终状态
        self.save_state()

        # 打印总结
        print(f"\n{'='*60}")
        print(f"批量测试完成")
        print(f"总耗时: {duration}")
        self.print_summary()

    def print_summary(self):
        """打印测试总结"""
        success = sum(1 for c in self.test_cases if c.status == "success")
        failed = sum(1 for c in self.test_cases if c.status == "failed")
        skipped = sum(1 for c in self.test_cases if c.status == "skipped")
        pending = sum(1 for c in self.test_cases if c.status == "pending")

        print(f"\n测试总结:")
        print(f"  总数: {len(self.test_cases)}")
        print(f"  成功: {success}")
        print(f"  失败: {failed}")
        print(f"  跳过: {skipped}")
        print(f"  待处理: {pending}")

        if failed > 0:
            print(f"\n失败用例:")
            for case in self.test_cases:
                if case.status == "failed":
                    print(f"  - {case.case_id}: {case.error_message}")


def create_default_config(path: str):
    """创建默认配置文件"""
    config = BatchConfig()
    config.to_yaml(path)
    print(f"已创建默认配置文件: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Freeform 模式批量测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有测试用例
  python batch_test.py --list

  # 试运行，查看将运行的用例
  python batch_test.py --dry-run

  # 运行所有测试用例
  python batch_test.py

  # 断点续跑
  python batch_test.py --resume

  # 运行单个用例（用于调试）
  python batch_test.py --run-one education_CSAPP-Lectures_2015Fall_Lecture01

  # 使用自定义配置文件
  python batch_test.py --config my_config.yaml
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="batch_test_config.yaml",
        help="配置文件路径 (默认: batch_test_config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑模式，跳过已成功的用例",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="创建默认配置文件",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出测试用例，不运行",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式，显示将要运行的用例",
    )
    parser.add_argument(
        "--run-one",
        type=str,
        metavar="CASE_ID",
        help="运行单个指定的测试用例（用于调试）",
    )

    args = parser.parse_args()

    # 创建默认配置
    if args.create_config:
        create_default_config(args.config)
        return

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        print(f"使用 --create-config 创建默认配置文件")
        # 使用默认配置
        config = BatchConfig()
    else:
        print(f"加载配置: {config_path}")
        config = BatchConfig.from_yaml(str(config_path))

    # 创建运行器
    runner = BatchTestRunner(config, resume=args.resume)

    # 仅列出用例
    if args.list:
        cases = runner.scan_test_cases()
        print(f"\n发现 {len(cases)} 个测试用例:")
        for i, case in enumerate(cases, 1):
            print(f"  {i:3d}. {case.case_id}")
            print(f"       附件: {len(case.attachments)} 个")
            print(f"       输出: {case.output_path}")
        return

    # 试运行
    if args.dry_run:
        cases = runner.scan_test_cases()
        print(f"\n试运行模式 - 将运行 {len(cases)} 个测试用例:")
        for case in cases:
            exists = "✓ 已存在" if case.output_path.exists() else "○ 待生成"
            print(f"  {exists} {case.case_id}")
        return

    # 运行单个用例
    if args.run_one:
        cases = runner.scan_test_cases()
        target_case = None
        for case in cases:
            if case.case_id == args.run_one:
                target_case = case
                break
        
        if target_case is None:
            print(f"错误: 找不到用例 '{args.run_one}'")
            print(f"提示: 使用 --list 查看所有可用的用例 ID")
            sys.exit(1)
        
        print(f"\n运行单个用例: {target_case.case_id}")
        print(f"数据目录: {target_case.data_dir}")
        print(f"附件: {target_case.attachments}")
        print(f"输出: {target_case.output_path}")
        
        runner.test_cases = [target_case]
        runner.total_pending = 1
        runner.completed_count = 0
        asyncio.run(runner.run_all())
        return

    # 运行批量测试
    asyncio.run(runner.run_all())


if __name__ == "__main__":
    main()
