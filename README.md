# Bill2Watt

*Bill2Watt* repository contains a Python package called *bill2watt*; whcih focuses on data-driven prediction of hourly load profiles from time-of-use energy bills. Additionally, the repository contains some show-case examples of how to use the package and data that can be used for testing various predictors (see licensing details [here](#license)).

## Description

The *bill2watt* package provides functionality for training data-driven models to predict hourly load profiles of heterogeneous and numerous electricity end users. 
It leverages the time-of-use energy bill, which offers valuable insights into the consumption patterns of the end users. 
The package implements multiple models to effectively map energy bills to their corresponding load profiles, including standard load profiles, classification, and regression.

In addition to the package, the repository includes an exploratory data analysis (EDA) notebook ([eda.ipynb](eda.ipynb)) that demonstrates data analysis techniques and provides insights into the dataset's characteristics and electricity consumption patterns. 
The EDA notebook serves as an example of how to explore the data related to load profiles.
Moreover a notebook to test the various predictors in *bill2watt* is provided ([tests.ipynb](tests.ipynb)), that demonstrates how the predictors can be used on a training/testing dataset and compares the performance of the various predictors according to multiple metrics. 
The testing notebook can be used to have some insight on how the package 'bill2watt' works.

## Table of Contents

- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Usage

To use the *bill2watt* package, follow these steps:

1. Download or clone the repository to your local machine.
2. Refer to the provided documentation (available [here](https://cadema-polito.github.io/Bill2Watt/index.html)) to learn how to utilize the *bill2watt* package for predicting load profiles from time-of-use energy bills.
3. Run the Jupyter Notebook file [eda.ipynb](eda.ipynb) to access the exploratory data analysis (EDA) notebook and explore the dataset used for load profile analysis.
4. Run the Jupyter Notebook file [tests.ipynb](tests.ipynb) to evaluate the various predictors in the *bill2watt* package and compare their performances based on various metrics.
5. Additional examples and guides on using the *bill2watt* package will be provided soon.

If you have any questions or need further assistance, please contact the project author.

Thank you!

## License

- Code: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Data: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

### Code
The code in the *bill2watt* package and in the rest of the repository is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. This means that you are free to use, modify, and distribute the code for any purpose, including commercial use, as long as you provide attribution to the original author(s), by citing the following reference:

> P. Lazzeroni, G. Lorenti and M. Repetto, "A data-driven approach to predict hourly load profiles from time-of-use electricity bills," in IEEE Access, 2023, doi: 10.1109/ACCESS.2023.3286020. [[Crossref]](https://doi.org/10.1109/ACCESS.2023.3286020).

### Data
The data contained in the CSV files throughout the repository are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license. 
This means that you are free to use the data for academic purposes, such as research and educational activities, but you are not allowed to use it for commercial purposes without obtaining explicit permission.
Please note that the data contained in the CSV files should be referenced as follows:

> P. Lazzeroni, G. Lorenti and M. Repetto, "A data-driven approach to predict hourly load profiles from time-of-use electricity bills", in IEEE Access, 2023, doi: 10.1109/ACCESS.2023.3286020. [[Crossref]](https://doi.org/10.1109/ACCESS.2023.3286020)

For more details, please refer to the [LICENSE.md](LICENSE.md) file for the code license and the [LICENSE_DATA.md](LICENSE_DATA.md) file for the data license.

## Contact

Corresponding Author: Gianmarco Lorenti [gianmarco.lorenti@polito.it](mailto:gianmarco.lorenti@polito.it).
