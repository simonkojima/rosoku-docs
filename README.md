# rosoku.github.io

## docs generation

### Requirements for docs generation

```
pip install git+https://github.com/NeuroTechX/moabb.git#egg=moabb
pip install sphinx pydata-sphinx-theme sphinx-multiversion sphinx-gallery numpydoc
```

1. Run following in docs

```

sphinx-apidoc -f -o ../docs/source ../nearby && make html

```

2. Run following in rosoku root

```

sphinx-build -b html docs/source ~/git/rosoku-docs/latest

```