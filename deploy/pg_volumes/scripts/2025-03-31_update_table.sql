ALTER TABLE jobs
ADD COLUMN expiration
    TIMESTAMP
    NOT NULL
;
ALTER TABLE jobs
ADD COLUMN origin
    TEXT NOT NULL
;
COMMENT ON COLUMN jobs.job_status IS '0=pending, 1=in_progress, 2=cancelled, 3=failed, 4=expired';