#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run step01 + step02 on a single RAW in a sandbox, with optional no-refine compare.

Usage:
  scripts/debug_rotation_case.sh /path/to/file.ARW [CASE_ID]
  RAW_SRC=/path/to/file.ARW CASE_ID=optional_name scripts/debug_rotation_case.sh

Optional env vars:
  CASE_ROOT=tmp/cases
  MAX_REFINE_ABS_DEG=3.0
  RUN_NO_REFINE_COMPARE=false

Examples:
  scripts/debug_rotation_case.sh /path/to/DSC08438.ARW
  scripts/debug_rotation_case.sh /path/to/DSC08438.ARW DSC08438
  RAW_SRC="/mnt/c/.../DSC01234.ARW" CASE_ID="DSC01234" scripts/debug_rotation_case.sh
  RAW_SRC="/mnt/c/.../DSC01234.ARW" CASE_ID="DSC01234" RUN_NO_REFINE_COMPARE=true scripts/debug_rotation_case.sh
EOF
  exit 0
fi

RAW_SRC="${RAW_SRC:-${1:-}}"
CASE_ID="${CASE_ID:-${2:-}}"

if [[ -z "${RAW_SRC}" ]]; then
  echo "set RAW_SRC or pass a RAW path as first arg" >&2
  exit 1
fi

if [[ -z "${CASE_ID}" ]]; then
  raw_base="$(basename "${RAW_SRC}")"
  CASE_ID="${raw_base%.*}"
fi

CASE_ROOT="${CASE_ROOT:-tmp/cases}"
MAX_REFINE_ABS_DEG="${MAX_REFINE_ABS_DEG:-3.0}"
RUN_NO_REFINE_COMPARE="${RUN_NO_REFINE_COMPARE:-false}"
CASE_DIR="${CASE_ROOT}/${CASE_ID}"
RAW_DIR="${CASE_DIR}/raw"
STEP01_DIR="${CASE_DIR}/step01"
STEP02_DIR="${CASE_DIR}/step02"
STEP02_NO_REFINE_DIR="${CASE_DIR}/step02_no_refine"

is_true() {
  case "${1,,}" in
    1 | true | yes | y | on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -f "${RAW_SRC}" ]]; then
  echo "RAW not found: ${RAW_SRC}" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}" "${STEP01_DIR}" "${STEP02_DIR}"

RAW_COPY="${RAW_DIR}/${CASE_ID}.ARW"
if [[ -e "${RAW_COPY}" && "${RAW_SRC}" -ef "${RAW_COPY}" ]]; then
  echo "RAW already in case raw dir at ${RAW_COPY}; skipping copy."
else
  cp -f "${RAW_SRC}" "${RAW_COPY}"
fi

echo "== Step 01 =="
cargo run --release --example step01_prepare -- \
  "${RAW_DIR}" \
  --raw "${RAW_COPY}" \
  --out-dir "${STEP01_DIR}"

echo "== Step 02 (configured) =="
step02_args=(
  --input "${STEP01_DIR}/next.json"
  --out-dir "${STEP02_DIR}"
  --max-refine-abs-deg "${MAX_REFINE_ABS_DEG}"
)
cargo run --release --example step02_detect_stub -- "${step02_args[@]}"

if is_true "${RUN_NO_REFINE_COMPARE}"; then
  echo "== Step 02 (no refine compare) =="
  mkdir -p "${STEP02_NO_REFINE_DIR}"
  step02_no_refine_args=(
    --input "${STEP01_DIR}/next.json"
    --out-dir "${STEP02_NO_REFINE_DIR}"
    # Force a no-op refine baseline by clamping applied correction to 0.
    --max-refine-abs-deg "0.0"
  )
  cargo run --release --example step02_detect_stub -- "${step02_no_refine_args[@]}"
fi

echo
echo "Done."
echo "Inspect:"
echo "  ${STEP02_DIR}/overlay_initial.jpg"
echo "  ${STEP02_DIR}/overlay.jpg"
echo "  ${STEP02_DIR}/result.json"
echo "  ${STEP02_DIR}/rotation_decision.jpg"
if is_true "${RUN_NO_REFINE_COMPARE}"; then
  echo "  ${STEP02_NO_REFINE_DIR}/overlay.jpg"
  echo "  ${STEP02_NO_REFINE_DIR}/result.json"
fi
