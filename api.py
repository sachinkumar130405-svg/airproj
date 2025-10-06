import psycopg2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS so your HTML map can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Database connection helper
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="airq",
        user="airuser",
        password="airpass"
    )
    return conn


@app.get("/")
def root():
    return {"status": "ok", "message": "Air Quality API is running"}

@app.get("/stations")
@app.get("/stations")
def get_stations():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, lat, lon, pm25, pm10 FROM stations LIMIT 100;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    def calc_aqi(pm25):
        if pm25 <= 50:
            return 50
        elif pm25 <= 100:
            return 100
        elif pm25 <= 200:
            return 200
        elif pm25 <= 300:
            return 300
        else:
            return 400

    stations = [
        {
            "name": r[0],
            "lat": r[1],
            "lon": r[2],
            "aqi": calc_aqi(r[3])
        }
        for r in rows
    ]
    return stations

