# Contributing

Contributions are welcome. Please open an issue or pull request on
[GitHub](https://github.com/agc888/goatpy).

## Development setup

```bash
git clone https://github.com/agc888/goatpy.git
cd goatpy
pip install -e ".[dev,docs,napari]"
```

## Running tests

```bash
pytest tests/
```

## Building the docs locally

```bash
cd docs
pip install -r requirements.txt
make html
open build/html/index.html
```
