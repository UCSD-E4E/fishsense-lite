CREATE DATABASE temporal_db;
CREATE DATABASE temporal_visibility;

CREATE ROLE temporal WITH
    LOGIN
    ENCRYPTED PASSWORD 'SCRAM-SHA-256$4096:RmtF7A7akHbDRSIUOevvkg==$/rcSoWSMPGyaf8bYN9GPB3gZMLuCrJol+ouaYrLoStg=:k4a8HG+9goUifvjuQYgU0P6EizMWp6ezGioYp7rMTuQ=';
GRANT ALL PRIVILEGES ON DATABASE temporal_db TO temporal;
GRANT ALL PRIVILEGES ON DATABASE temporal_visibility TO temporal;
\c temporal_db postgres
\c temporal_visibility postgres
GRANT ALL ON SCHEMA public TO temporal;