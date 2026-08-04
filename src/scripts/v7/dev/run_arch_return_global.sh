#!/usr/bin/env bash
set -euo pipefail
export ARM=global
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_arch_return_40step.sh"

