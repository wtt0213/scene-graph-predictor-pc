from pathlib import Path
from setuptools import find_packages, setup

PROJECT_NAME = "scene-graph-predictor-pc"
PACKAGE_NAME = PROJECT_NAME.replace("-", "_")
DESCRIPTION = "3D Scene Graph Models"


if __name__ == "__main__":
    version = "0.1.0"

    print(f"Building {PROJECT_NAME}-{version}")

    setup(
        name=PROJECT_NAME,
        version=version,
        author="Ziqin Wang",
        author_email="wzqbuaa@qq.com",
        url=f"https://github.com/wtt0213/{PROJECT_NAME}",
        download_url=f"https://github.com/wtt0213/{PROJECT_NAME}/tags",
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=("tests",)),
        package_data={PACKAGE_NAME: ["data/*.txt", "config/*.json"]},
        zip_safe=False,
        python_requires=">=3.8",
        install_requires=[
            "trimesh",
            "open3d",
            'aiofiles',
            'fastapi',
            "uvicorn[standard]",
            "python-multipart",
        ],
    )