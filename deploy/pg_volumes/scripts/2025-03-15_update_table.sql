CREATE TABLE data_paths (
    "idx" INT PRIMARY KEY,
    "path" TEXT NOT NULL
);

INSERT INTO data_paths (idx, path)
VALUES (0, '/mnt/fishsense_data_reef/REEF/data/');

ALTER TABLE images
ADD data_path INT NULL REFERENCES data_paths;

UPDATE images
SET data_path = 0;

CREATE INDEX "images_image_md5" ON "images" ("image_md5");

ALTER TABLE dives
ADD data_path INT NULL REFERENCES data_paths;

ALTER TABLE canonical_dives
ADD data_path INT NULL REFERENCES data_paths;

UPDATE dives
SET data_path = 0;

UPDATE canonical_dives
SET data_path = 0;

ALTER TABLE images
ADD ignore BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE images
ADD date TIMESTAMP NULL;