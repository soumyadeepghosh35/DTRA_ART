#!/bin/bash
# A support script for automated testing of ART's Jupyter notebooks.
#
# THIS IS EXPENSIVE. Anticipated runtime for all of ART's notebooks is ~11 hours at the time of
# writing.
#
# This script only verifies that the notebooks run to completion, and to save on compute
# resources, stops after the first failure.  It doesn't verify notebooks' results,
# some of which are independently verified by ART's integration tests, at a cost of ~1.5 hour
# runtime.  See https://github.com/JBEI/AutomatedRecommendationTool/blob/master/docs/Developing.md

set -euo pipefail

function run_notebook() {
  echo "************************************************************************"
  echo "* Testing $1 ..."
  echo "************************************************************************"
  time jupyter nbconvert --to notebook --execute "$1"
}

echo "Testing tutorial notebooks..."
for f in *.ipynb
do
  run_notebook "$f"
done

echo "Testing notebooks published with the ART paper..."
cd paper/
for f in *.ipynb
do
  run_notebook "$f"
done

echo "Done testing notebooks!  All notebooks ran to completion."
