# Test systems

# `predictions`

Reference and predicted pose structures for PROTAC test systems, i.e. every system
contains both, PPI and PLI.

# `ref_metrics.csv`

This file contains different metrics for the test systems in `predictions` obtained
via the PINDER/PLINDER evaluation pipeline.

# `pdb`

These are multimeric systems taken from the PDB.
In contrast to the `predictions` they contain multiple instances of the same entity
to better test atom matching.

# `dockq`

This directory contains test structures for tests of the `DockQScore` metric.
Corresponding structures are in the same subdirectory containing the respective pose and reference complex.

- **1A2K**:
  Reference structure originates from PDB.
  Originates from PINDER.
- **6S0A**:
  Reference structure originates from PDB.
  Pose was docked with *DockGPT*.
  Originates from PINDER.
- **5O2Z**:
  Reference structure originates from PDB.
  Pose was docked with *DiffDock*.
  Originates from PINDER.
- **2E31**:
  Reference structure originates from PDB.
  Pose was docked with *GeoDock*.
  Originates from PINDER.
- **6X1G**:
  Reference structure originates from PDB.
  Pose was docked with *HDock*.
  Originates from PINDER.
- **6J6J**:
  Streptavidin bound to biotin.
  Reference (`6J6J`) and pose (`1SWK`) structure originate from PDB.
  The moposedel is actually just a mutant structure