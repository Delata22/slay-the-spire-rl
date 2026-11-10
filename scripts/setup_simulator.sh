#!/usr/bin/env bash
# Builds the headless C++ simulator (sts_lightspeed) with the Python bindings used here.
#   bash scripts/setup_simulator.sh            # -> external/sts_lightspeed/build/slaythespire*.so
# Requirements: git, cmake >= 3.16, a C++17 compiler, Python 3.10+ headers.
# Pinned versions: sts_lightspeed @7476a81 + the RL hooks patch from sts-rl-agent @54d7402
# (MIT, adds clone / init_encounter / status bindings), then patches/scumsearch_fixed_uct.patch
# (our fix of the reference MCTS, opt-in with fixed=True). pybind11 is bumped to v2.13.6 because
# the pinned v2.7.1 does not compile against Python >= 3.11.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/external"; mkdir -p "$EXT"; cd "$EXT"
PY="${PYTHON:-$(command -v python3)}"

[ -d sts_lightspeed ] || git clone --recursive https://github.com/gamerpuppy/sts_lightspeed
[ -d sts-rl-agent ] || git clone https://github.com/destire-mio/sts-rl-agent
(cd sts-rl-agent && git checkout -q 54d7402)

cd sts_lightspeed
git checkout -q 7476a81
git submodule update --init --recursive
git apply --check ../sts-rl-agent/sim_patch/sim_rl_hooks.patch 2>/dev/null \
  && git apply ../sts-rl-agent/sim_patch/sim_rl_hooks.patch || echo "patch already applied"
# our fix of the MCTS baseline: mcts_recommend(bc, sims, fixed=True); fixed=False is the original search
git apply --check "$ROOT/patches/scumsearch_fixed_uct.patch" 2>/dev/null \
  && git apply "$ROOT/patches/scumsearch_fixed_uct.patch" || echo "MCTS fix already applied"
(cd pybind11 && git fetch -q --tags && git checkout -q v2.13.6)

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$PY"
make -j"$(nproc)" slaythespire
echo
echo "Built: $(ls "$PWD"/slaythespire*.so)"
echo "Either keep this default location or export STS_LIGHTSPEED_BUILD=$PWD"
