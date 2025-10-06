import psycopg2

try:
    conn = psycopg2.connect(
        dbname="airquality",
        user="airuser",
        password="airpass",
        host="localhost",
        port=5432
    )
    print("✅ Database connection successful!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)
