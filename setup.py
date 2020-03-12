from setuptools import setup,find_packages

setup(
        name = "rank_fit",
        version="0.0.1",
        author="Devin Greene",
        author_email="devin@greene.cz",
        description="Algorithms for ranking contestants based on "\
        "win-lose data",
        packages = find_packages(exclude = 
    ( 'rank_fit.tools', 'rank_fit.compare_methods' )),
        python_requires=">=3.6")
