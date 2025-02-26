from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.config import redis_client
import asyncio

router = APIRouter()

@router.websocket("/task-updates/{task_id}")
async def websocket_task_updates(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time task updates from Redis Streams.
    Supports both `bo_task_updates` and `search_space_updates`.
    """

    await websocket.accept()
    streams = ["bo_task_updates", "search_space_updates"]
    last_id = "$" 

    print(f"✅ WebSocket connected for task_id: {task_id}")

    try:
        while True:
            messages = redis_client.xread({stream: last_id for stream in streams}, count=1, block=5000)

            if messages:
                for stream, entries in messages:
                    last_id = entries[-1][0]  

                    for entry in entries:
                        entry_id, data = entry
                        message_dict = {key: value for key, value in data.items()}

                        if message_dict.get("task_id") == task_id:
                            print(f"📡 Sending update to WebSocket: {message_dict}") 
                            await websocket.send_json(message_dict)

                            if message_dict.get("status") == "COMPLETED":
                                print(f"🎉 Task {task_id} is completed. Closing WebSocket...")
                                await websocket.close()
                                return

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print(f"❌ Client disconnected from task {task_id}")

