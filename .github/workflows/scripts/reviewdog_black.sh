#!/bin/bash
set -eu # Increase bash strictness
set -o pipefail

export REVIEWDOG_VERSION=v0.15.0

echo "Installing reviewdog..."
wget -O - -q https://raw.githubusercontent.com/reviewdog/reviewdog/master/install.sh | sh -s -- -b /tmp "${REVIEWDOG_VERSION}"

black_exe="$(which black)"

echo "Black version: $(${black_exe} --version)"

echo "Checking python code with the black and reviewdog..."
exit_val="0"
cd ${WORKDIR}

${black_exe} --diff --quiet --check . | /tmp/reviewdog -f="diff" \
    -f.diff.strip=0 \
    -name="black" \
    -reporter="github-pr-review" \
    -fail-on-error="false" \
    -level="error" \
    -tee || exit_val="$?"

echo "Clean up reviewdog..."
rm /tmp/reviewdog

if [[ "${exit_val}" -ne '0' ]]; then
  exit 1
fi