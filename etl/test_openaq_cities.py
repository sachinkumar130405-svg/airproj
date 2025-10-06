import requests

API_KEY = "64dfd2fb0a935eeef8f9432d3639a172faf290240ef4598c4549941beafa411b"
headers = {"X-API-Key": API_KEY}

url = "https://api.openaq.org/v3/cities"
params = {"country": "IN", "limit": 50}

r = requests.get(url, headers=headers, params=params)
r.raise_for_status()
for c in r.json().get("results", []):
    print(c["city"])
