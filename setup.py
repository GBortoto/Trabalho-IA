import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IA_Project",
    version="0.0.1",
    author="Vinícius Teixeira , Alex Fogaça , Guilherme Bortoto , Antonio Batista , Kenny Takahashi",
    description="Pacote contendo o projeto de Inteligência Artificial sobre data mining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GBortoto/Trabalho-IA",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3.5",
        "Operating System :: OS Independent",
    ),
    package_data={
        '' : ['*.txt'],
    },
    scripts=['KMeans.py','Matrix.py', 'ProcessTexts.py',
    'Silhouette.py','SOM.py','XMeans.py','KMeansPlotter.py',
    'Main.py','SOMBackup.py'],
    install_requires=['numpy','os','string','tensorflow',
                      'matplotlib','math','scipy',
                      'sklearn','nltk','pylab'],
)
