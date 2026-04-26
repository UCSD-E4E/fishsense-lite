# fishsense-web-services
FishSense Web Services - Deploy

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