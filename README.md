# BOOM : Tephrochronology dataset of the Southern and Austral Volcanic Zones of the Andes

This repository contains jupyter notebooks aimed to process the information in the BOOM dataset (link here when is back up), maintained by the [ESPRI](https://mesocentre.ipsl.fr/ "https://mesocentre.ipsl.fr/#") server from the Institute Pierre-Simon Laplace, France. The notebooks process the geochemical composition information contained in the dataset, with three aims:

- **UncertaintyAndStandards**.ipynb: Estimate the precision and accuracy of the geochemical composition analyses contained in the BOOM dataset.
- **CheckNormalizations**.ipynb: Establish if the major element composition of samples in which different major elements have been analyzed (e.g. samples where MnO, Cl and/or P2O5 have not been analyzed, versus samples that have) are comparable or not.
- **MachineLearning**.ipynb and **BOOMDataset_Preprocessing**.ipynb: Evaluating the use of machine learning classification algortihms as a way of identifying the volcanic source of "unclassified" tephra deposits.

The notebooks and the dataset are fully described in Martínez Fontaine et al. *in prep* (linked when is published).

Additional exploration of the BOOM dataset, as well as downloading subsets of it, can be done with the BOOM explorer (link here when is up).
