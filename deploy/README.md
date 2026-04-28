# fishsense-web-services
FishSense Web Services - Deploy

# Local development (devcontainer)

The repo ships a devcontainer that brings up a self-contained local stack
(postgres + temporal + fishsense-api + nginx) so you can iterate on
workflows + activities without touching prod.

1. `cp deploy/.env.local.example deploy/.env`
2. Edit `FISHSENSE_DUMP_PATH` to point at your prod backup (`pg_dump -Fc`).
3. Open the repo in VSCode and "Reopen in Container".

The first boot of the postgres service runs
`deploy/pg_volumes/scripts.local/00_restore.sh`, which `pg_restore`s the dump
into a named docker volume (NOT under the repo — the data is large). To start
clean later: `docker volume rm fishsense-local_pg_data`.

The local stack lives in `deploy/compose.local.yml` and is intentionally
**not** layered on the prod `compose.yml` — prod's Authentik/mTLS/letsencrypt
coupling makes that messier than a separate file.

# Deploy Procedure
1. Ensure `//e4e-nas.ucsd.edu/fishsense_data/REEF/data` is mounted as a docker volume named `fishsense_data_reef`.
2. Ensure `//e4e-nas.ucsd.edu/fishsense/Fishsense Lite Calibration Parameters` is mounted as a docker volume named `fishsense_lens_cal`.
3. Ensure `//e4e-nas.ucsd.edu/fishsense_process_work` is mounted as a docker volume named `fishsense_process_work`.
4. Ensure `.secrets/postgres_admin_password.txt` is populated
4. Ensure `.secrets/temporal_database_password.env` is populated with `POSTGRES_PWD`.
4. Ensure `.secrets/temporal_ui.env` is populated with `TEMPORAL_AUTH_CLIENT_ID` and `TEMPORAL_AUTH_CLIENT_SECRET`.
5. Ensure `spider_volumes/config/.secrets.toml` is populated with the following:
- `label_studio.api_key`
6. Ensure `.env` is populated.  Example:
```
USER_ID=1001
GROUP_ID=1001
```
7. Ensure `spider_volumes/data`, `spider_volumes/logs`, `spider_volumes/cache` exist
8. Ensure `label_studio_reporter/config/config.toml` and `label_studio_reporter/config/gcloud_credentials.json` exist
9. Ensure `label_studio_reporter/cache` and `label_studio_reporter/logs` exist