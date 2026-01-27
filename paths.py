from os import PathLike
from pathlib import Path


"""
路径说明
- 本文件里的字符串路径是“相对 data 根目录”的相对路径，而不是相对当前工作目录(cwd)。
- 这样无论你从哪里运行代码，都能稳定解析到正确位置。

如需切换到另一份同构目录结构的数据集：
- 方式1：调用 get_data_source_dirs(base_dir=Path("/your/other/data"))
"""


DATA_SOURCE_DIRS_REL = [
    # academic
    'academic/academic_talks/F1000-talks',
    'academic/artificial_intellgence/CVPR_2023_Highlight',
    'academic/artificial_intellgence/CVPR_2024_Oral',
    'academic/artificial_intellgence/CVPR_2025_Oral',
    'academic/artificial_intellgence/ICLR-24',
    'academic/artificial_intellgence/ICLR-25',
    'academic/artificial_intellgence/ICML-24',
    'academic/artificial_intellgence/ICML-25',
    'academic/artificial_intellgence/NeurIPS_2024_Oral',
    'academic/computer_system/FAST_2025',
    'academic/computer_system/NSDI_2025',
    'academic/computer_system/OSDI_2025',
    'academic/computer_system/USENIX',
    'academic/NBER_conferences',
    
    # advertisement
    'advertisement/Apple_iPad',
    'advertisement/Apple_iPhone',
    'advertisement/Apple_Mac',
    'advertisement/BMW',
    'advertisement/lenovo',
    
    # education
    'education/Computer_science_lectures',
    'education/CSAPP-Lectures_2015Fall',
    'education/MIT-Financing_Economic_Development',
    'education/MIT-the_human_brain',
    'education/THU_DSA',
    
    # finance_and_economics
    'finance_and_economics/Alphabet_Investor_Relations',
    'finance_and_economics/JPMorgan_Chase',
    'finance_and_economics/Microsoft_press_release',
    'finance_and_economics/OECD_Economic_Outlook',
    'finance_and_economics/TESLA_update_letter',
    'finance_and_economics/World_Bank_GPE',
    
    # talk
    'talk/middle_school_presentation',
    'talk/ted_chinese',
    'talk/US_presidental_speech',
]

DATA_SOURCE_DIRS_REL = [Path(rel) for rel in DATA_SOURCE_DIRS_REL]


def get_data_source_dirs(base_dir: PathLike | None = None) -> list[Path]:
    """
    返回数据源目录（绝对路径）。
    - base_dir=None 时，默认使用 DATA_ROOT（本仓库的 data 目录）
    - 传入 base_dir 时，可切换到另一份同构数据目录
    """
    # data 根目录默认值：.../data/utils/paths.py -> parents[1] == .../data
    default_root = Path(__file__).resolve().parents[1]
    root = Path(base_dir or default_root).expanduser().resolve()
    return [root / rel for rel in DATA_SOURCE_DIRS_REL]


# 默认跳过的目录名称
SKIP_NAMES = {"__pycache__", "meta_prompts", "unused"}


def get_valid_item_dirs(src_dir: Path, skip_names: set[str] | None = None) -> list[Path]:
    """
    获取目录下有效的子目录列表。
    排除：
    - 非目录项
    - skip_names 中的名称（默认为 SKIP_NAMES）
    - 以 . 开头的隐藏目录
    
    Args:
        src_dir: 源目录路径
        skip_names: 要跳过的目录名称集合，默认为 SKIP_NAMES
        
    Returns:
        有效的子目录路径列表
        
    Raises:
        OSError: 如果无法读取目录
    """
    if skip_names is None:
        skip_names = SKIP_NAMES
    
    return [
        p
        for p in src_dir.iterdir()
        if p.is_dir() and p.name not in skip_names and not p.name.startswith(".")
    ]

