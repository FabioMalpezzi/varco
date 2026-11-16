#!/bin/sh
# What can be checked without an Apple GPU, and what the CI workflow runs:
#   1. every shell script parses (sh -n);
#   2. every Python file compiles;
#   3. the comparators of the tests fail on results that are wrong, missing
#      or not in their inventory (scripts/check-comparators.py);
#   4. the links between the Markdown files hold (scripts/check-links.py);
#   5. scripts/autostart.sh starts each server where the containers of the
#      engine must find it, and with ENGINE=kiri on Kiri's socket and on no
#      TCP port (scripts/check-autostart.sh, with stand-ins for launchd and
#      for the servers; on a Mac only, skipped elsewhere);
#   6. the patches apply to their upstream at the pinned commit (needs the
#      network: skipped with --offline).
# It does not run the bridges: those need a Mac with an Apple GPU, and are
# tested with scripts/test-all.sh.
#   scripts/check.sh [--offline]
cd "$(dirname "$0")/.." || exit 1
status=0
fail() { echo "FAILED: $*"; status=1; }

echo "== shell scripts"
for f in $(git ls-files -co --exclude-standard '*.sh'); do
  sh -n "$f" || fail "$f"
done

echo "== Python files"
for f in $(git ls-files -co --exclude-standard '*.py'); do
  python3 -c "import sys; compile(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1], 'exec')" "$f" || fail "$f"
done

echo "== comparators"
python3 scripts/check-comparators.py || status=1

echo "== links"
python3 scripts/check-links.py || status=1

echo "== autostart"
sh scripts/check-autostart.sh || status=1

if [ "$1" = --offline ]; then
  echo "== patches: skipped (--offline)"
  exit $status
fi
echo "== patches"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
. mlx/versions.sh
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$MLX_TAG" "$MLX_REPO" "$work/mlx" || fail "cloning MLX"
if [ "$(git -C "$work/mlx" rev-parse HEAD)" != "$MLX_COMMIT" ]; then
  fail "MLX $MLX_TAG is not at the pinned commit $MLX_COMMIT"
fi
for patch in mlx/patches/*.patch; do
  git -C "$work/mlx" apply --index "$PWD/$patch" && echo "applies to MLX $MLX_TAG: $patch" || fail "$patch"
done
. kiri/versions.sh
git clone --quiet "$KIRI_REPO" "$work/kiri" && git -C "$work/kiri" checkout --quiet "$KIRI_COMMIT" || fail "cloning Kiri at $KIRI_COMMIT"
for patch in kiri/patches/*.patch; do
  git -C "$work/kiri" apply --check "$PWD/$patch" && echo "applies alone to Kiri $KIRI_COMMIT: $patch" || fail "$patch alone"
done
for patch in kiri/patches/*.patch; do
  git -C "$work/kiri" apply --index "$PWD/$patch" || fail "$patch after the ones before it"
done
[ $status = 0 ] && echo "apply together to Kiri $KIRI_COMMIT: kiri/patches/*.patch"
exit $status
