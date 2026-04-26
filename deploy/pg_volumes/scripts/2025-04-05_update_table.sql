ALTER TABLE data_paths
    ADD COLUMN unc_path TEXT
;

UPDATE data_paths
SET unc_path = (
    SELECT replace(path, '/mnt/fishsense_data_reef/', '//e4e-nas.ucsd.edu/fishsense_data/')
    FROM data_paths
)
;
ALTER TABLE data_paths
    ALTER COLUMN unc_path SET NOT NULL
;