from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()


def collect_files(relative_root: str):
    """Collect regular files under a directory, preserving structure in share/."""
    base = ROOT / relative_root
    if not base.exists():
        return []

    collected = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        install_dir = Path("share") / "RT-COSMIK" / path.parent.relative_to(ROOT)
        collected.append((str(install_dir), [str(path.relative_to(ROOT))]))
    return collected


script_files = [
    str(path.relative_to(ROOT))
    for path in (ROOT / "scripts").rglob("*.py")
    if path.is_file() and path.name != "__init__.py"
]

readme_path = ROOT / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="RT-COSMIK",
    version="0.1.0",
    description="Real-Time Constrained and Open-Source Multibody Inverse Kinematics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    scripts=script_files,
    include_package_data=True,
    data_files=[
        ("share/RT-COSMIK", ["settings.py"]),
        *collect_files("config"),
        *collect_files("weights"),
    ],
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        'dataclasses; python_version<"3.7"',
    ],
    python_requires=">=3.8",
)
