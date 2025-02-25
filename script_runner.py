import redis

try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("✅ Connected to Redis!")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
