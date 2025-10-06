import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="airdb",
    user="airuser",
    password="airpass"
)
cur = conn.cursor()

print("Connected to DB, creating tables...")

cur.execute("""
CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT PRIMARY KEY,
    name TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    metadata JSONB
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS pollutant_readings (
    id SERIAL PRIMARY KEY,
    station_id TEXT REFERENCES stations(station_id),
    ts TIMESTAMPTZ,
    pm25 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    no2 DOUBLE PRECISION,
    so2 DOUBLE PRECISION,
    o3 DOUBLE PRECISION,
    aqi INTEGER,
    raw_metadata JSONB
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS fire_hotspots (
    id TEXT PRIMARY KEY,
    ts TIMESTAMPTZ,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    source TEXT
);
""")

conn.commit()
cur.close()
conn.close()

print("✅ Tables created successfully!")
