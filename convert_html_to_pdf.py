#!/usr/bin/env python3
"""
直接将已有的 HTML slides 转换为 PDF，无需重跑整个流程
"""

import asyncio
import sys
from glob import glob
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "deeppresenter"))

from deeppresenter.utils.webview import PlaywrightConverter


async def convert_slides_to_pdf(
    slides_dir: Path,
    output_pdf: Path,
    aspect_ratio: str = "widescreen",
):
    """将 slides 目录中的 HTML 文件转换为 PDF"""
    html_files = list(slides_dir.glob("*.html"))
    if not html_files:
        print(f"错误: {slides_dir} 中没有找到 HTML 文件")
        return False

    print(f"找到 {len(html_files)} 个 HTML 文件")
    print(f"输出: {output_pdf}")

    async with PlaywrightConverter() as converter:
        await converter.convert_to_pdf(html_files, output_pdf, aspect_ratio)

    print(f"✓ 转换完成: {output_pdf}")
    return True


async def main():
    # 4 个需要重新转换的用例（路径来自 batch_test_state.json）
    cases = [
        {
            "name": "ICLR-25_Accelerated training",
            "slides_dir": "/home/cxs/.cache/deeppresenter/batch/20260127/academic_artificial_intellgence_ICLR-25_Accelerated training through iterative gradient propagation along the residual path/slides",
            "output_pdf": "/home/cxs/PPT_Agent/Benchmark/data_results/PPTAgent_v2.0.0/academic/artificial_intellgence/ICLR-25/Accelerated training through iterative gradient propagation along the residual path/generation_task/results/slides.pdf",
        },
        {
            "name": "THU_DSA_Lecture6",
            "slides_dir": "/home/cxs/.cache/deeppresenter/batch/20260127/education_THU_DSA_Lecture6/slides",
            "output_pdf": "/home/cxs/PPT_Agent/Benchmark/data_results/PPTAgent_v2.0.0/education/THU_DSA/Lecture6/generation_task/results/slides.pdf",
        },
        {
            "name": "THU_DSA_Lecture9",
            "slides_dir": "/home/cxs/.cache/deeppresenter/batch/20260127/education_THU_DSA_Lecture9/slides",
            "output_pdf": "/home/cxs/PPT_Agent/Benchmark/data_results/PPTAgent_v2.0.0/education/THU_DSA/Lecture9/generation_task/results/slides.pdf",
        },
        {
            "name": "World_Bank_GPE",
            "slides_dir": "/home/cxs/.cache/deeppresenter/batch/20260127/finance_and_economics_World_Bank_GPE_GPE_Jan_2025/slide_deck",
            "output_pdf": "/home/cxs/PPT_Agent/Benchmark/data_results/PPTAgent_v2.0.0/finance_and_economics/World_Bank_GPE/GPE_Jan_2025/generation_task/results/slides.pdf",
        },
    ]

    success_count = 0
    for case in cases:
        print(f"\n{'='*60}")
        print(f"处理: {case['name']}")
        print(f"{'='*60}")

        slides_dir = Path(case["slides_dir"])
        output_pdf = Path(case["output_pdf"])

        if not slides_dir.exists():
            print(f"错误: slides 目录不存在: {slides_dir}")
            continue

        # 确保输出目录存在
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        try:
            await convert_slides_to_pdf(slides_dir, output_pdf)
            success_count += 1
        except Exception as e:
            print(f"✗ 转换失败: {e}")

    print(f"\n{'='*60}")
    print(f"完成: {success_count}/{len(cases)} 个用例转换成功")


if __name__ == "__main__":
    asyncio.run(main())
