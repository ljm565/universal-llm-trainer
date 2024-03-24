# import time
# from threading import Thread
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# import uvicorn


# app = FastAPI()

# class Chatter:
#     def __init__(self):
#         self.examples = list(range(10))


#     def generate(self):
#         for i in self.examples:
#             print(i)
#             time.sleep(1)


#     async def inference(self):
#         thread = Thread(target=self.generate)
#         thread.start()



# chatter = Chatter()

# @app.get("/")
# async def main():
#     return StreamingResponse(chatter.generate())


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



import time
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from threading import Thread
import uvicorn

app = FastAPI()

class Chatter:
    def __init__(self):
        self.examples = list(range(10))

    async def generate(self, websocket):
        # for i in self.examples:
        #     await websocket.send_text(str(i))
        #     time.sleep(1)

        async for message in websocket.iter_text():
            # 여기서 서버에서 수행할 작업 수행
            # 이 예시에서는 받은 메시지를 그대로 다시 클라이언트로 전송
            for m in message:
                await websocket.send_text(m)
                time.sleep(0.01)

chatter = Chatter()


# async def generate(ws):
#     await ws.send_text('hi')
    


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # data = await websocket.receive_text()
    # print(data)
    await chatter.generate(websocket)
    # chatter_thread = Thread(target=chatter.generate, args=(websocket,))
    # chatter_thread.start()
   



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)