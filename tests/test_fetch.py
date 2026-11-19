from pipeline.fetch import Fetcher

f = Fetcher()
r = f.get("https://boards-api.greenhouse.io/v1/boards/stripe/jobs?content=true")
print(r.status, r.is_json, len(r.json()["jobs"]))