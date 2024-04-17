#!/bin/bash
set -eu # Increase bash strictness
set -o pipefail

export REVIEWDOG_VERSION=v0.15.0

echo ${CI_PULL_REQUEST}
echo ${CI_COMMIT}
echo ${CI_REPO_OWNER}
echo ${CI_REPO_NAME}
echo ${CI_REPOSITORY}
echo ${CI_BRANCH}

echo "Installing reviewdog..."
wget -O - -q https://raw.githubusercontent.com/reviewdog/reviewdog/master/install.sh | sh -s -- -b /tmp "${REVIEWDOG_VERSION}"

echo "Flake8 version: $(flake8 --version)"

echo "Checking python code with the flake8 linter and reviewdog..."

flake_exit_val="0"
flake8_output="$(flake8 ${WORKDIR} --config=.flake8 2>&1)" || flake_exit_val="$?"

echo "flake8 output:"
echo "${flake8_output}"

exit_val="0"
echo "${flake8_output}" | /tmp/reviewdog -f="flake8" \
    -name="flake8" \
    -reporter="github-pr-review" \
    -filter-mode="added" \
    -fail_on-error="false" \
    -level="error" || exit_val="$?"

echo "Clean up reviewdog..."
rm /tmp/reviewdog

if [[ "${exit_val}" -ne '0' ]]; then
  exit 1
fi