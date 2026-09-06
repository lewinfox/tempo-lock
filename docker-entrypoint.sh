#!/bin/sh
# Fly (and plain `docker run -v`) attach volumes owned by root, so the data dir has to be
# chowned at boot rather than at build time. Start as root, fix the mount, then drop to
# the unprivileged app user for the actual server.
set -e

DATA="${TEMPOLOCK_DATA:-/data}"

if [ "$(id -u)" = "0" ]; then
  mkdir -p "$DATA"
  chown app:app "$DATA"
  exec setpriv --reuid=app --regid=app --init-groups "$@"
fi

exec "$@"
