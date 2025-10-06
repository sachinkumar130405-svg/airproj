import requests

url = "https://api.openaq.org/v3.1/measurements"
params = {
    "coordinates": "28.6139,77.2090",
    "radius": 50000,
    "parameter": "pm25",
    "limit": 5
}

r = requests.get(url, params=params)
print("Status:", r.status_code)
print(r.text[:500])
