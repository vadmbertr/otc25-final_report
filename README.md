# OTC 2025 Final Report

Repository hosting the final report of the OTC 2025.

The report is written using [Quarto](https://quarto.org/), allowing to combine both report text and code, and produce either a HTML or PDF manuscript.

- [index](index.qmd) contains the report structure,
- the different sections are in the `sections/` folder. A section can be a `.qmd` (Quarto Markdown) or a `.ipynb` file,
- static images are in the `images/` folder,
- the data (or part of it) is in the `data/` folder,
- the Python modules are in the `src/` folder,
- output files (e.g. index.html or index.pdf) are in the `_manuscript/` folder,
- the file `_quarto.yml` contains the Quarto configuration.

Instructions for installing Quarto can be found [in Quarto Get Started tutorial](https://quarto.org/docs/get-started/).

Compiling the report can then be done using the command line:

```bash
quarto render
```

And HTML preview is achieved using:

```bash
quarto preview
```
