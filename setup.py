from setuptools import setup, find_packages

setup(
    name='telco_churn_package',           # you can pick your package name 
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)

