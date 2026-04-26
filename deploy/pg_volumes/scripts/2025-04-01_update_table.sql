ALTER TABLE images
ADD COLUMN field_cal_ignore BOOL NOT NULL DEFAULT false
;
COMMENT ON COLUMN images.field_cal_ignore IS 'Suppresses field calibration processing'
;
GRANT UPDATE(field_cal_ignore) ON images TO ccrutchf
;