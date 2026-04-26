# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# This file is included in the final Docker image and SHOULD be overridden when
# deploying the image to prod. Settings configured here are intended for use in local
# development environments. Also note that superset_config_docker.py is imported
# as a final step as a means to override "defaults" configured here
#
import logging
import os
import sys

from celery.schedules import crontab
from flask_caching.backends.filesystemcache import FileSystemCache

from flask_appbuilder.security.manager import AUTH_OAUTH

logger = logging.getLogger()

DATABASE_DIALECT = os.getenv("DATABASE_DIALECT")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_DB = os.getenv("DATABASE_DB")

# EXAMPLES_USER = os.getenv("EXAMPLES_USER")
# EXAMPLES_PASSWORD = os.getenv("EXAMPLES_PASSWORD")
# EXAMPLES_HOST = os.getenv("EXAMPLES_HOST")
# EXAMPLES_PORT = os.getenv("EXAMPLES_PORT")
# EXAMPLES_DB = os.getenv("EXAMPLES_DB")

# The SQLAlchemy connection string.
SQLALCHEMY_DATABASE_URI = (
    f"{DATABASE_DIALECT}://"
    f"{DATABASE_USER}:{DATABASE_PASSWORD}@"
    f"{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_DB}"
)

# # Use environment variable if set, otherwise construct from components
# # This MUST take precedence over any other configuration
# SQLALCHEMY_EXAMPLES_URI = os.getenv(
#     "SUPERSET__SQLALCHEMY_EXAMPLES_URI",
#     (
#         f"{DATABASE_DIALECT}://"
#         f"{EXAMPLES_USER}:{EXAMPLES_PASSWORD}@"
#         f"{EXAMPLES_HOST}:{EXAMPLES_PORT}/{EXAMPLES_DB}"
#     ),
# )


REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_CELERY_DB = os.getenv("REDIS_CELERY_DB", "0")
REDIS_RESULTS_DB = os.getenv("REDIS_RESULTS_DB", "1")

RESULTS_BACKEND = FileSystemCache("/app/superset_home/sqllab")

CACHE_CONFIG = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
    "CACHE_KEY_PREFIX": "superset_",
    "CACHE_REDIS_HOST": REDIS_HOST,
    "CACHE_REDIS_PORT": REDIS_PORT,
    "CACHE_REDIS_DB": REDIS_RESULTS_DB,
}
DATA_CACHE_CONFIG = CACHE_CONFIG
THUMBNAIL_CACHE_CONFIG = CACHE_CONFIG


class CeleryConfig:
    broker_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_CELERY_DB}"
    imports = (
        "superset.sql_lab",
        "superset.tasks.scheduler",
        "superset.tasks.thumbnails",
        "superset.tasks.cache",
    )
    result_backend = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_RESULTS_DB}"
    worker_prefetch_multiplier = 1
    task_acks_late = False
    beat_schedule = {
        "reports.scheduler": {
            "task": "reports.scheduler",
            "schedule": crontab(minute="*", hour="*"),
        },
        "reports.prune_log": {
            "task": "reports.prune_log",
            "schedule": crontab(minute=10, hour=0),
        },
    }


CELERY_CONFIG = CeleryConfig

SLACK_API_TOKEN = os.getenv("SLACK_API_TOKEN", "")

SMTP_HOST = "smtp.ucsd.edu" # change to your host
SMTP_PORT = 25 # your port, e.g. 587
SMTP_STARTTLS = False
SMTP_SSL_SERVER_AUTH = False # If you're using an SMTP server with a valid certificate
SMTP_SSL = False
SMTP_MAIL_FROM = os.getenv("SMTP_MAIL_FROM", "")
SMTP_USER = None
SMTP_PASSWORD = None
EMAIL_NOTIFICATIONS = True
EMAIL_REPORTS_SUBJECT_PREFIX = "[Superset] "

FEATURE_FLAGS = {
  "ALERT_REPORTS": True,
  "ALERT_REPORT_SLACK_V2": True,  # use the v2 upload path
}
ALERT_REPORTS_NOTIFICATION_DRY_RUN = False

WEBDRIVER_BASEURL = f"http://fishsense_superset:8088{os.environ.get('SUPERSET_APP_ROOT', '/')}/"  # When using docker compose baseurl should be http://superset_nginx{ENV{BASEPATH}}/  # noqa: E501
# The base URL for the email report hyperlinks.
WEBDRIVER_BASEURL_USER_FRIENDLY = (
    f"http://fishsense_superset:8088{os.environ.get('SUPERSET_APP_ROOT', '/')}/"
)
WEBDRIVER_TYPE = "chrome"
WEBDRIVER_OPTION_ARGS = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--window-size=1920,1080",
]
SQLLAB_CTAS_NO_LIMIT = True
ENABLE_PROXY_FIX = True

log_level_text = os.getenv("SUPERSET_LOG_LEVEL", "INFO")
LOG_LEVEL = getattr(logging, log_level_text.upper(), logging.INFO)

# Set the authentication type to OAuth
AUTH_TYPE = AUTH_OAUTH

OAUTH_PROVIDERS = [
    {
        'name': 'authentik',
        'icon': 'fa-address-card',
        'token_key': 'access_token',
        'remote_app': {
            'client_id': os.environ.get('AUTHENTIK_KEY'),
            'client_secret': os.environ.get('AUTHENTIK_SECRET'),
            'api_base_url': 'https://auth.fabricant.ucsd.edu/application/o/fishsense-analytics/',
            'client_kwargs':{
              'scope': 'email profile'
            },
            'jwks_uri': 'https://auth.fabricant.ucsd.edu/application/o/fishsense-analytics/jwks/',
            'request_token_url': None,
            'access_token_url': 'https://auth.fabricant.ucsd.edu/application/o/token/',
            'authorize_url': 'https://auth.fabricant.ucsd.edu/application/o/authorize/'
        }
    }
]

# Will allow user self registration, allowing to create Flask users from Authorized User
AUTH_USER_REGISTRATION = True
ALLOW_LOCAL_USER_LOGIN = True 

# The default user self registration role
AUTH_USER_REGISTRATION_ROLE = "Public"

WEBDRIVER_AUTH = {"username": os.getenv("WEBDRIVER_AUTH_USERNAME", ""), "password":  os.getenv("WEBDRIVER_AUTH_PASSWORD", "")}

if os.getenv("CYPRESS_CONFIG") == "true":
    # When running the service as a cypress backend, we need to import the config
    # located @ tests/integration_tests/superset_test_config.py
    base_dir = os.path.dirname(__file__)
    module_folder = os.path.abspath(
        os.path.join(base_dir, "../../tests/integration_tests/")
    )
    sys.path.insert(0, module_folder)
    from superset_test_config import *  # noqa

    sys.path.pop(0)

#
# Optionally import superset_config_docker.py (which will have been included on
# the PYTHONPATH) in order to allow for local settings to be overridden
#
try:
    import superset_config_docker
    from superset_config_docker import *  # noqa: F403

    logger.info(
        f"Loaded your Docker configuration at [{superset_config_docker.__file__}]"
    )
except ImportError:
    logger.info("Using default Docker config...")