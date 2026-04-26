-- Creates job management tables
CREATE TABLE jobs (
    job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    worker TEXT NOT NULL,
    created_on TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
    job_type TEXT,
    job_status INTEGER NOT NULL DEFAULT 0,
    job_progress INTEGER NOT NULL DEFAULT 0
)
;
COMMENT ON COLUMN jobs.job_status IS '0=pending, 1=in_progress, 2=cancelled, 3=failed';

ALTER TABLE images
ADD COLUMN preprocess_job_id uuid NULL REFERENCES jobs(job_id) ON DELETE SET NULL ON UPDATE CASCADE;
ALTER TABLE images
ADD COLUMN preprocess_laser_job_id uuid NULL REFERENCES jobs(job_id) ON DELETE SET NULL ON UPDATE CASCADE;
