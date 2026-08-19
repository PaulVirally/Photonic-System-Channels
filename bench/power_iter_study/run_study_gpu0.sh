#!/bin/bash

# GPU 0 half of the power-iteration study. Greens must already exist:
#
#   STAGES=greens bash bench/power_iter_study/run_study.sh 0
#
# Then run this and run_study_gpu1.sh together.

STAGES=${STAGES:-study} exec bash "$(dirname "$0")/run_study.sh" 0
