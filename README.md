# rosoku.github.io

## docs generation

### Requirements for docs generation

Install rosoku first.

```
#pip install git+https://github.com/NeuroTechX/moabb.git#egg=moabb
pip install sphinx pydata-sphinx-theme sphinx-multiversion sphinx-gallery numpydoc moabb braindecode
```

1. Run following in docs

```

sphinx-apidoc -f -o ../docs/source ../nearby && make html

```

2. Run following in rosoku root

```

sphinx-build -b html docs/source ~/git/rosoku-docs/latest

```