## Schism

[![CI-Core](https://github.com/EdCaunt/schism/actions/workflows/pytest_core.yml/badge.svg)](https://github.com/EdCaunt/schism/actions/workflows/pytest_core.yml)

Schism is a set of utilities used for the implementation of
immersed boundaries in Devito. The intention is to build a high-level
abstractions to simplify the process of imposing boundary conditions
on non-grid-conforming topographies. By making a suitably versatile,
generic tool for constructing immersed boundaries, the integration of
immersed boundary methods into higher level applications with minimal
additional complexity is made possible.

This repository is currently a work in process building on my previous effort
[DevitoBoundary](https://github.com/devitocodes/devitoboundary).

In order to download, install and use Devito follow the instructions
listed [here](https://github.com/devitocodes/devito).


## Quickstart
In order to install Schism:
*Requirements:* A working Devito installation.

```
conda activate devito
git clone https://github.com/EdCaunt/schism
cd schism
pip install -e .
```

## Get in touch

If you're using Schism or Devito, we would like to hear from
you. Whether you are facing issues or just trying it out, join the
[conversation](devitocodes.slack.com/).
