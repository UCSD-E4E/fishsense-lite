CREATE TABLE laser_labels (
    cksum TEXT PRIMARY KEY,
    task_id BIGINT,
    x INT,
    y INT
);

CREATE TABLE headtail_labels (
    cksum TEXT PRIMARY KEY,
    task_id BIGINT,
    head_x INT,
    head_y INT,
    tail_x INT,
    tail_y INT
);