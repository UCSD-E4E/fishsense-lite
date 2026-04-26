CREATE TABLE canonical_dives(
    path TEXT PRIMARY KEY,
    date DATE,
    invalid_image BOOLEAN,
    multiple_date BOOLEAN,
    checksum TEXT
);

CREATE TABLE dives (
    path TEXT PRIMARY KEY,
    date DATE,
    invalid_image BOOLEAN,
    multiple_date BOOLEAN,
    checksum TEXT
);

CREATE TABLE images (
    path TEXT PRIMARY KEY,
    dive TEXT REFERENCES dives (path),
    camera_sn TEXT,
    image_md5 TEXT,
    laser_task_id BIGINT
);
