CREATE SEQUENCE lens_cal_idx_seq
    INCREMENT 1
    MINVALUE 1
    MAXVALUE 0x7FFFFFFFFFFFFFFF
;

CREATE TABLE lens_cal (
    idx BIGINT PRIMARY KEY DEFAULT NEXTVAL('lens_cal_idx_seq'),
    camera_idx INTEGER NOT NULL REFERENCES cameras(idx),
    parameters JSONB NOT NULL,
    calibration_data TEXT REFERENCES canonical_dives(checksum)
)
;
CREATE SEQUENCE depth_cal_idx_seq
    INCREMENT 1
    MINVALUE 1
    MAXVALUE 0x7FFFFFFFFFFFFFFF
;
CREATE TABLE depth_cal (
    idx BIGINT PRIMARY KEY DEFAULT NEXTVAL('lens_cal_idx_seq'),
    lens_cal BIGINT REFERENCES lens_cal(idx),
    parameters JSONB NOT NULL,
    calibration_data TEXT NOT NULL REFERENCES canonical_dives(checksum)
)
;

ALTER TABLE lens_cal
ADD COLUMN date DATE NOT NULL;
ALTER TABLE depth_cal
ADD COLUMN date DATE NOT NULL;

GRANT SELECT, UPDATE, INSERT ON TABLE lens_cal TO ccrutchf;
GRANT SELECT, UPDATE, INSERT ON TABLE depth_cal TO ccrutchf;