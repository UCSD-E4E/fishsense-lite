#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -e

#
# Always install local overrides first
#
/app/docker/docker-bootstrap.sh

if [ "$SUPERSET_LOAD_EXAMPLES" = "yes" ]; then
    STEP_CNT=4
else
    STEP_CNT=3
fi

echo_step() {
cat <<EOF
######################################################################
Init Step ${1}/${STEP_CNT} [${2}] -- ${3}
######################################################################
EOF
}
ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin}"
# If Cypress run – overwrite the password for admin and export env variables
if [ "$CYPRESS_CONFIG" == "true" ]; then
    ADMIN_PASSWORD="general"
    export SUPERSET_TESTENV=true
    export POSTGRES_DB=superset_cypress
    export SUPERSET__SQLALCHEMY_DATABASE_URI=postgresql+psycopg2://superset:superset@db:5432/superset_cypress
fi
# Initialize the database
echo_step "1" "Starting" "Applying DB migrations"
superset db upgrade
echo_step "1" "Complete" "Applying DB migrations"

# Create an admin user
echo_step "2" "Starting" "Setting up admin user ( admin / $ADMIN_PASSWORD )"
if [ "$CYPRESS_CONFIG" == "true" ]; then
    superset load_test_users
else
    superset fab create-admin \
        --username admin \
        --email admin@superset.com \
        --password "$ADMIN_PASSWORD" \
        --firstname Superset \
        --lastname Admin
fi
echo_step "2" "Complete" "Setting up admin user"
# Create default roles and permissions
echo_step "3" "Starting" "Setting up roles and perms"
superset init
echo_step "3" "Complete" "Setting up roles and perms"

if [ "$SUPERSET_LOAD_EXAMPLES" = "yes" ]; then
    # Load some data to play with
    echo_step "4" "Starting" "Loading examples"


    # If Cypress run which consumes superset_test_config – load required data for tests
    if [ "$CYPRESS_CONFIG" == "true" ]; then
        superset load_examples --load-test-data
    else
        superset load_examples
    fi
    echo_step "4" "Complete" "Loading examples"
fi

# ── Step 5: import committed dashboard assets (IaC) ──────────────────────────
# Superset dashboards-as-code: databases/datasets/charts/dashboards live as YAML
# under docker/assets/ and are re-imported on every converge (idempotent —
# import overwrites by UUID). The FishSense DB connection's password is injected
# from $DATABASE_PASSWORD at import time via a placeholder, so the secret never
# lives in git. A failed import must NOT fail init (Superset still comes up), so
# this is best-effort.
ASSETS_SRC=/app/docker/assets
if [ -d "$ASSETS_SRC" ]; then
    echo_step "5" "Starting" "Importing committed dashboard assets"
    # Whole block runs in a subshell with `set +e` and a trailing `|| echo` so a
    # bad bundle can NEVER fail init — Superset (which depends on init completing
    # successfully) must still come up. The script runs under `set -e`, so
    # without this containment a single failed command here takes Superset down.
    (
        set +e
        BUNDLE="$(mktemp -d)"
        cp -r "$ASSETS_SRC"/. "$BUNDLE"/
        # nix-store bind mounts carry 1970 mtimes and `zip` rejects timestamps
        # before 1980 ("ZIP does not support timestamps before 1980"), so bump
        # every copied file to now before archiving.
        find "$BUNDLE" -exec touch {} +
        # Inject the superset DB-role password into the FishSense connection URI.
        sed -i "s|__DB_PASSWORD__|${DATABASE_PASSWORD}|g" "$BUNDLE"/databases/*.yaml
        ZIP=/tmp/fishsense-assets.zip
        python -c "import shutil,sys; shutil.make_archive('/tmp/fishsense-assets','zip',sys.argv[1])" "$BUNDLE"
        superset import-assets -p "$ZIP"
        rm -rf "$BUNDLE" "$ZIP"
    ) || echo "WARN: dashboard asset import failed — Superset still starts (fix the bundle + re-converge)"
    echo_step "5" "Complete" "Importing committed dashboard assets"
fi
