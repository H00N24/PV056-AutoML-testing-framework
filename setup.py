from setuptools import find_packages, setup

setup(
    name="pv056_2019",
    version="1.0",
    description="Testing",
    url="https://github.com/H00N24/PV056-AutoML-testing-framework",
    author="Robert Kolcun, Ondrej Kurak",
    author_email="487564@mail.muni.cz, okurak@mail.muni.cz",
    license="MIT",
    packages=find_packages(include=["pv056_2019", "pv056_2019.*"]),
    include_package_data=True,
    install_requires=["pandas", "numpy", "liac-arff", "sklearn", "pydantic"],
    entry_points={
        "console_scripts": [
            "pv056-enrich-data=pv056_2019.enrich_data:main",
            "pv056-run-clf=pv056_2019.main_clf:main",
            "pv056-statistics=pv056_2019.statistics:main",
        ]
    },
    zip_safe=False,
)
