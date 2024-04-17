#!/bin/bash
set -eu # Increase bash strictness
set -o pipefail

export REVIEWDOG_VERSION=v0.15.0

echo "Installing reviewdog..."
wget -O - -q https://raw.githubusercontent.com/reviewdog/reviewdog/master/install.sh | sh -s -- -b /tmp "${REVIEWDOG_VERSION}"

flake8_exe="$(which flake8)"

echo "Flake8 version: $(${flake8_exe} --version)"

echo "Checking python code with the flake8 linter and reviewdog..."
exit_val="0"
cd ${WORKDIR}
${flake8_exe} . 2>&1 | /tmp/reviewdog -f="flake8" \
    -name="flake8" \
    -reporter="github-pr-review" \
    -fail-on-error="false" \
    -level="error"
    -tee || exit_val="$?"

echo "Clean up reviewdog..."
rm /tmp/reviewdog

if [[ "${exit_val}" -ne '0' ]]; then
  exit 1
fi