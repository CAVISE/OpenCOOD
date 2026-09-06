from pathlib import Path
from runpy import run_path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
__version__ = run_path(ROOT / "opencood" / "version.py")["__version__"]


def _read_requirements_file():
    """Return package requirements, excluding pip-only options."""
    requirements = []
    for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith(("#", "-")):
            requirements.append(line)
    return requirements


setup(
    name="opencood",
    version=__version__,
    packages=find_packages(),
    url="https://github.com/CAVISE/opencood.git",
    license="MIT",
    author="CAVISE",
    author_email="CAVISE",
    description="An opensource pytorch framework for autonomous driving cooperative detection. Forked from original OpenCOOD",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "opencood": ["hypes_yaml/*.yaml"],
        "logreplay": ["hypes_yaml/*.yaml"],
    },
    python_requires=">=3.12",
    install_requires=_read_requirements_file(),
)
