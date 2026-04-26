CREATE DATABASE superset;

CREATE ROLE superset WITH
    LOGIN
    ENCRYPTED PASSWORD 'SCRAM-SHA-256$4096:zkRRL43OLyaN90M2isKnuw==$pS6POyjzuDvV1vfu3Vju+SpITzzCRK5utSZDXwArEOo=:g21y+miAakYZI9CsgPxS4TR/iIR/wmSL8vvYP+cwgWM=';
GRANT ALL PRIVILEGES ON DATABASE superset TO superset;
\c superset postgres
GRANT ALL ON SCHEMA public TO superset;

-- Provide read only access to the fishsense database so that we can report on it
\c fishsense postgres

GRANT CONNECT ON DATABASE fishsense TO superset;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO superset;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO superset;

ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO superset;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON SEQUENCES TO superset;