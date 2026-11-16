#!/bin/sh
# Clone Kiri at the commit the patches are against and apply them.
#   kiri/apply.sh [DIR]        (default ./kiri-src, next to this script)
# What you get is Kiri's source with the GPU channel and the memory-headroom
# change, not a build: see README.md for building and running it.
set -e
cd "$(dirname "$0")"
. ./versions.sh
dir=${1:-kiri-src}
if [ -e "$dir" ]; then
  echo "$dir already exists: remove it, or give another directory" >&2
  exit 1
fi
git clone --quiet "$KIRI_REPO" "$dir"
git -C "$dir" checkout --quiet "$KIRI_COMMIT"
for patch in patches/*.patch; do
  git -C "$dir" apply --index "$PWD/$patch"
  echo "applied $patch"
done
echo "$dir is Kiri at $KIRI_COMMIT with the patches applied (staged, not committed)"
