#!/usr/bin/env bash
set -euo pipefail
export ARM=global
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_p0001_lookback_probe.sh"
