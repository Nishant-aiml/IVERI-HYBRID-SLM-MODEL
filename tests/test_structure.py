# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for verifying the exact project structure of IVERI CORE."""

from __future__ import annotations

from pathlib import Path


def test_required_directories_exist(project_root: Path) -> None:
    """Test that all required directories exist in the project."""
    required_dirs = [
        "configs",
        "core",
        "model",
        "model/blt",
        "model/mamba2",
        "model/mor",
        "model/titans",
        "model/moe",
        "data",
        "training",
        "evaluation",
        "baselines",
        "utils",
        "utils/validation",
        "scripts",
        "tests",
        "quality",
        "docs",
        "docs/architecture",
        "docs/phases",
        "docs/benchmarks",
        "docs/research",
        "docs/decisions",
        "docs/reports",
        "experiments",
        "experiments/phase_0",
        "research_log",
        "reports",
        "reports/phase_0/tests",
        "reports/phase_0/quality",
        "reports/phase_0/validation",
    ]

    for rel_path in required_dirs:
        dir_path = project_root / rel_path
        assert dir_path.exists(), f"Missing directory: {rel_path}"
        assert dir_path.is_dir(), f"Path is not a directory: {rel_path}"


def test_init_files_exist(project_root: Path) -> None:
    """Test that __init__.py files are present in all package directories."""
    required_inits = [
        "configs",
        "core",
        "model",
        "model/blt",
        "model/mamba2",
        "model/mor",
        "model/titans",
        "model/moe",
        "data",
        "training",
        "evaluation",
        "baselines",
        "utils",
        "utils/validation",
        "scripts",
        "tests",
    ]

    for rel_path in required_inits:
        init_file = project_root / rel_path / "__init__.py"
        assert init_file.exists(), f"Missing __init__.py in: {rel_path}"
        assert init_file.is_file(), f"Path is not a file: {rel_path}/__init__.py"


def test_infrastructure_files_exist(project_root: Path) -> None:
    """Test that root infrastructure and package files exist."""
    required_files = [
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        ".gitignore",
        "README.md",
        "train.py",
    ]

    for rel_path in required_files:
        file_path = project_root / rel_path
        assert file_path.exists(), f"Missing core infrastructure file: {rel_path}"
        assert file_path.is_file(), f"Path is not a file: {rel_path}"


def test_core_files_exist(project_root: Path) -> None:
    """Test that core package infrastructure modules exist."""
    core_files = [
        "core/constants.py",
        "core/exceptions.py",
        "core/interfaces.py",
        "core/registry.py",
        "core/factory.py",
    ]

    for rel_path in core_files:
        file_path = project_root / rel_path
        assert file_path.exists(), f"Missing core package file: {rel_path}"
        assert file_path.is_file(), f"Path is not a file: {rel_path}"


def test_config_files_exist(project_root: Path) -> None:
    """Test that configuration system files exist."""
    assert (project_root / "configs/base_config.py").is_file()


def test_utils_files_exist(project_root: Path) -> None:
    """Test that utility files exist."""
    utils_files = [
        "utils/logging.py",
        "utils/validation/__init__.py",
        "utils/validation/tensors.py",
        "utils/validation/gradients.py",
        "utils/validation/configs.py",
        "utils/validation/memory.py",
    ]

    for rel_path in utils_files:
        file_path = project_root / rel_path
        assert file_path.exists(), f"Missing utility file: {rel_path}"
        assert file_path.is_file(), f"Path is not a file: {rel_path}"


def test_no_model_implementation_files(project_root: Path) -> None:
    """Verify that no actual model implementations exist in Phase 0."""
    # Verify model/titans structure (allowing lr_gen.py, updater.py, and memory.py in Phase 1.7)
    titans_dir = project_root / "model/titans"
    titans_files = sorted([f.name for f in titans_dir.iterdir() if f.is_file()])
    allowed_titans_files = {
        tuple(sorted(["__init__.py"])),
        tuple(sorted(["__init__.py", "lr_gen.py"])),
        tuple(sorted(["__init__.py", "lr_gen.py", "updater.py"])),
        tuple(sorted(["__init__.py", "lr_gen.py", "updater.py", "memory.py"])),
    }
    assert (
        tuple(titans_files) in allowed_titans_files
    ), f"Unexpected files in model/titans: {titans_files}"

    # Verify model/blt structure (allowing entropy_model.py, patcher.py, encoder.py, and decoder.py in Phase 1.6)
    blt_dir = project_root / "model/blt"
    blt_files = sorted([f.name for f in blt_dir.iterdir() if f.is_file()])
    allowed_blt_files = {
        tuple(sorted(["__init__.py"])),
        tuple(sorted(["__init__.py", "entropy_model.py"])),
        tuple(sorted(["__init__.py", "entropy_model.py", "patcher.py"])),
        tuple(sorted(["__init__.py", "entropy_model.py", "patcher.py", "encoder.py"])),
        tuple(
            sorted(["__init__.py", "entropy_model.py", "patcher.py", "encoder.py", "decoder.py"])
        ),
    }
    assert tuple(blt_files) in allowed_blt_files, f"Unexpected files in model/blt: {blt_files}"

    # Verify model/mor structure (allowing router.py, recursion.py, and kv_cache.py in Phase 1.5)
    mor_dir = project_root / "model/mor"
    mor_files = sorted([f.name for f in mor_dir.iterdir() if f.is_file()])
    allowed_mor_files = {
        tuple(sorted(["__init__.py"])),
        tuple(sorted(["__init__.py", "router.py"])),
        tuple(sorted(["__init__.py", "router.py", "recursion.py"])),
        tuple(sorted(["__init__.py", "router.py", "recursion.py", "kv_cache.py"])),
    }
    assert tuple(mor_files) in allowed_mor_files, f"Unexpected files in model/mor: {mor_files}"

    # Verify model/mamba2 structure (allowing math.py, scan.py, and block.py in Phase 1.3)
    mamba2_dir = project_root / "model/mamba2"
    mamba2_files = sorted([f.name for f in mamba2_dir.iterdir() if f.is_file()])
    allowed_mamba2_combinations = {
        tuple(sorted(["__init__.py"])),
        tuple(sorted(["__init__.py", "math.py"])),
        tuple(sorted(["__init__.py", "math.py", "scan.py"])),
        tuple(sorted(["__init__.py", "math.py", "scan.py", "block.py"])),
    }
    assert (
        tuple(mamba2_files) in allowed_mamba2_combinations
    ), f"Unexpected files in model/mamba2: {mamba2_files}"

    # Verify model/moe structure (allowing router.py and experts.py in Phase 1.2)
    moe_dir = project_root / "model/moe"
    moe_files = sorted([f.name for f in moe_dir.iterdir() if f.is_file()])
    # In Wave 1, router.py exists; in Wave 2, experts.py is added.
    allowed_moe_files = {
        tuple(sorted(["__init__.py"])),
        tuple(sorted(["__init__.py", "router.py"])),
        tuple(sorted(["__init__.py", "router.py", "experts.py"])),
    }
    assert tuple(moe_files) in allowed_moe_files, f"Unexpected files in model/moe: {moe_files}"

    # Check files in model root (allowing norms.py, rope.py, swiglu.py, and attention.py)
    model_root = project_root / "model"
    files = sorted([f.name for f in model_root.iterdir() if f.is_file()])
    expected = sorted(
        [
            "__init__.py",
            "norms.py",
            "rope.py",
            "swiglu.py",
            "attention.py",
            "backbone.py",
            "iveri_core.py",
        ]
    )
    assert files == expected, f"Unexpected files in model/: {files}"


def test_docs_structure(project_root: Path) -> None:
    """Verify documentation file locations."""
    assert (project_root / "docs/dependency_graph.md").is_file()
    assert (project_root / "docs/phases/phase_0_plan.md").is_file()


def test_experiments_structure(project_root: Path) -> None:
    """Verify experiments folder structure."""
    assert (project_root / "experiments/README.md").is_file()
    assert (project_root / "experiments/phase_0").is_dir()


def test_reports_structure(project_root: Path) -> None:
    """Verify reports output directories exist."""
    assert (project_root / "reports/phase_0/tests").is_dir()
    assert (project_root / "reports/phase_0/quality").is_dir()
    assert (project_root / "reports/phase_0/validation").is_dir()
