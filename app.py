from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import httpx
import uuid
import json
import time
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NovelAI Cloud Queue Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_QUEUE_SIZE = 100
MAX_CONCURRENT_TASKS = 5

task_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
tasks = OrderedDict()
active_tasks = 0

class GenerationRequest(BaseModel):
    input: str
    model: str = "nai-diffusion-3"
    action: str = "generate"
    parameters: dict = {}
    api_key: str = ""
    greeting: str = "正在生成中~"
    negative_prompt: str = ""
    use_new_shared_trial: bool = True

class TaskStatus(BaseModel):
    taskId: str
    status: str
    position: int = 0
    imageData: str = ""
    error: str = ""

async def process_tasks():
    global active_tasks
    while True:
        if active_tasks >= MAX_CONCURRENT_TASKS:
            await asyncio.sleep(0.5)
            continue
        
        try:
            task_id = await task_queue.get()
            active_tasks += 1
            await process_single_task(task_id)
            active_tasks -= 1
            task_queue.task_done()
        except Exception as e:
            logger.error(f"任务处理循环错误: {e}")
            await asyncio.sleep(1)

async def process_single_task(task_id):
    global tasks
    
    if task_id not in tasks:
        return
    
    task = tasks[task_id]
    request = task["request"]
    
    try:
        tasks[task_id]["status"] = "running"
        logger.info(f"开始处理任务: {task_id}")
        
        image_data = await call_novelai_api(request)
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["imageData"] = image_data
        logger.info(f"任务完成: {task_id}")
        
        for ws in task.get("websockets", []):
            try:
                await ws.send_json({
                    "success": True,
                    "id": task_id,
                    "imageData": image_data,
                    "prompt": request.input,
                    "change": request.input
                })
            except:
                pass
                
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        logger.error(f"任务失败: {task_id}, 错误: {e}")
        
        for ws in task.get("websockets", []):
            try:
                await ws.send_json({
                    "success": False,
                    "id": task_id,
                    "error": str(e)
                })
            except:
                pass

async def call_novelai_api(request: GenerationRequest):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {request.api_key}"
    }
    
    payload = {
        "input": request.input,
        "model": request.model,
        "action": request.action,
        "parameters": request.parameters,
        "negative_prompt": request.negative_prompt
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.novelai.net/ai/generate-image",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        data = response.json()
        
        if "imageBase64" in data:
            return f"data:image/png;base64,{data['imageBase64']}"
        elif "images" in data and data["images"]:
            return f"data:image/png;base64,{data['images'][0]}"
        else:
            raise ValueError("API 返回不包含图片数据")

@app.get("/ping")
async def ping():
    return {"status": "alive"}

@app.get("/stats")
async def stats():
    return {
        "queue_size": task_queue.qsize(),
        "active_tasks": active_tasks,
        "total_tasks": len(tasks),
        "max_queue_size": MAX_QUEUE_SIZE,
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS
    }

@app.post("/queue")
async def submit_task(request: GenerationRequest):
    if task_queue.full():
        raise HTTPException(status_code=429, detail="队列已满，请稍后重试")
    
    task_id = str(uuid.uuid4())[:12]
    
    tasks[task_id] = {
        "status": "queued",
        "request": request,
        "websockets": [],
        "imageData": "",
        "error": ""
    }
    
    await task_queue.put(task_id)
    
    return {
        "taskId": task_id,
        "status": "queued",
        "position": task_queue.qsize(),
        "greeting": request.greeting
    }

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    
    if task["status"] == "completed":
        return {
            "taskId": task_id,
            "status": "completed",
            "imageData": task["imageData"]
        }
    elif task["status"] == "failed":
        return {
            "taskId": task_id,
            "status": "failed",
            "error": task["error"]
        }
    else:
        return {
            "taskId": task_id,
            "status": task["status"],
            "position": task_queue.qsize()
        }

@app.post("/api/predict")
async def api_predict(request: GenerationRequest):
    if task_queue.full():
        raise HTTPException(status_code=429, detail="队列已满")
    
    task_id = str(uuid.uuid4())[:12]
    
    tasks[task_id] = {
        "status": "queued",
        "request": request,
        "websockets": [],
        "imageData": "",
        "error": ""
    }
    
    await task_queue.put(task_id)
    
    while task_id in tasks:
        task = tasks[task_id]
        if task["status"] == "completed":
            return {
                "success": True,
                "id": task_id,
                "imageData": task["imageData"],
                "prompt": request.input,
                "change": request.input
            }
        elif task["status"] == "failed":
            raise HTTPException(status_code=500, detail=task["error"])
        
        await asyncio.sleep(0.5)
    
    raise HTTPException(status_code=404, detail="任务未找到")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket 连接已建立")
    
    task_id = None
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                request_data = json.loads(data)
                
                if task_queue.full():
                    await websocket.send_json({"success": False, "error": "队列已满"})
                    continue
                
                task_id = str(uuid.uuid4())[:12]
                
                request = GenerationRequest(
                    input=request_data.get("input", ""),
                    model=request_data.get("model", "nai-diffusion-3"),
                    action=request_data.get("action", "generate"),
                    parameters=request_data.get("parameters", {}),
                    api_key=request_data.get("api_key", ""),
                    greeting=request_data.get("greeting", "正在生成中~"),
                    negative_prompt=request_data.get("negative_prompt", ""),
                    use_new_shared_trial=request_data.get("use_new_shared_trial", True)
                )
                
                tasks[task_id] = {
                    "status": "queued",
                    "request": request,
                    "websockets": [websocket],
                    "imageData": "",
                    "error": ""
                }
                
                await task_queue.put(task_id)
                
                await websocket.send_json({
                    "type": "submitted",
                    "taskId": task_id,
                    "position": task_queue.qsize(),
                    "greeting": request.greeting
                })
                
            except json.JSONDecodeError:
                await websocket.send_json({"success": False, "error": "无效的 JSON 格式"})
            except Exception as e:
                await websocket.send_json({"success": False, "error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
        if task_id and task_id in tasks:
            tasks[task_id]["websockets"] = [
                ws for ws in tasks[task_id]["websockets"] if ws != websocket
            ]
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")

@app.get("/")
async def root():
    return {
        "message": "NovelAI Cloud Queue Service",
        "endpoints": {
            "POST /queue": "提交生成任务",
            "GET /status/{task_id}": "查询任务状态",
            "POST /api/predict": "提交任务并等待结果",
            "GET /ping": "健康检查",
            "GET /stats": "队列统计",
            "WebSocket /ws": "实时任务推送"
        }
    }

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_tasks())
    logger.info("队列服务已启动")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("队列服务已关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)