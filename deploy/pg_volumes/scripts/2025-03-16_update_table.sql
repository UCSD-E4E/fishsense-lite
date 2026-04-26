CREATE TABLE cameras (
    "idx" INT PRIMARY KEY,
    "serial_number" TEXT,
    "name" TEXT
);

ALTER TABLE dives
ADD camera INT NULL REFERENCES cameras;
ALTER TABLE canonical_dives
ADD camera INT NULL REFERENCES cameras;


ALTER TABLE "canonical_dives"
ADD "site" text NULL;
COMMENT ON TABLE "canonical_dives" IS '';

CREATE TABLE "dive_slates" (
  "idx" serial NOT NULL,
  PRIMARY KEY ("idx"),
  "name" text NOT NULL,
  "scan_path" text NOT NULL
);

ALTER TABLE "canonical_dives"
ADD "slate_id" INTEGER NULL;

ALTER TABLE "canonical_dives"
ADD "priority" TEXT NOT NULL DEFAULT 'LOW';

CREATE TABLE "priorities" (
  "idx" integer NOT NULL,
  "name" TEXT NOT NULL
);

INSERT INTO "priorities" ("idx", "name")
VALUES
  (0, 'HIGH'),
  (1, 'MEDIUM'),
  (2, 'LOW')
;

ALTER TABLE "cameras"
ADD "lens_cal_path" TEXT NULL;

ALTER TABLE "images"
ADD "headtail_task_id" BIGINT NULL;