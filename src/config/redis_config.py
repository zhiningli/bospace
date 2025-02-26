import redis

def get_redis_client():
    """Returns a Redis client instance"""
    return redis.Redis(host="localhost", port=6379, decode_responses=True)

# Global Redis client instance
redis_client = get_redis_client()
