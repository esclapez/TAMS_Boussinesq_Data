# TAMS - Boussinesq: data and processing scripts

This repository contains data and processing scripts for the
paper entitle:
`Constructing efficient score functions for rare event simulation in high-dimensional ocean-climate models`
by L. Esclapez, V. Jacques-Dumas, R. B\"orner, L. Soucasse and H. A. Dijkstra

submitted for publication to Chaos.

Note that the full TAMS databases from the numerous runs listed in the
manuscript are not provided here due to disk space limitations. 
Only the metadata from those runs, used in producing the paper's figures
are included, as well as the results of the POD decomposition upon which
the data-driven score function is constructed.

Some of the scripts require Python classes defined in the Boussinesq example
of pyTAMS:
https://github.com/nlesc-eTAOC/pyTAMS/tree/main/examples/Boussinesq

Specifically, you need to copy/symlink the podscore.py file into the current data_autonomous
subdirectory.

