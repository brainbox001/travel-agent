# server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json
from main import graph
from langchain.schema import BaseMessage
from memory import memory, USER_COUNT


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def serialize_event(event):
    """Convert LangChain messages and other objects to JSON-serializable format."""
    if isinstance(event, BaseMessage):
        return {
            "type": event.type,
            "content": event.content,
            "additional_kwargs": event.additional_kwargs,
        }
    if isinstance(event, dict):
        if "brain" in event:
            return {"brain" : True}
        elif "tools" in event:
            return { "tools" : True}
        return {k: serialize_event(v) for k, v in event.items()}
    if isinstance(event, list):
        return [serialize_event(v) for v in event]
    return event


@app.get("/chat")
async def chat_home(thread_id: str = None):
    global USER_COUNT
    if thread_id is None:
        thread_id = str(USER_COUNT)
        USER_COUNT += 1
    # print(f"user count now {USER_COUNT}")

    config = {"configurable": {"thread_id": thread_id}}

    # print(f"the thread id - {thread_id}")

    try:
        past_convo = await memory.aget(config=config)
        # print(f"past convo before serializing : {past_convo}")
        serialized = []

        if past_convo is not None:
            event = past_convo['channel_values']
            messages = []
            if event is not None:
                messages = event['messages']
                serialized = serialize_event(messages)
        response = {
            "thread_id" : thread_id,
            "data" : serialized
        }
        # SSE format
        # print(f"past convo after serializing : {serialized}")
        return response
    except Exception as e:
        print(f"error {e}")
        return {'error': 'An error occured, try again'}


@app.post("/chat")
async def chat_endpoint(request: Request, thread_id: str):
    body = await request.json()
    user_input = body.get("input", {})
    config = {"configurable": {"thread_id": thread_id}}

    async def event_stream():
        try:
            async for event in graph.astream(user_input, config=config):
                # Serialize event to JSON-serializable format

                # print(f"################################\nEvent before serializing : {event}\n###############################")
                serialized = serialize_event(event)
                # SSE format
                # print(f"serialized event : {serialized}")
                yield f"data: {json.dumps(serialized)}\n\n"
        except Exception as e:
            print(f"error {e}")
            yield f"data: {json.dumps({'error': 'An error occured, please try again'})}\n\n"
        finally:
            # Signal completion
            yield "event: end\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=7860)
