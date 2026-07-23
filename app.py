from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uuid
import logging
import os

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

TURSO_DB_URL = os.environ.get("TURSO_DB_URL")

def get_db_connection():
    import sqlite3
    conn = sqlite3.connect("queue.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    
    try:
        if hasattr(conn, 'execute'):
            cursor = conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    input TEXT NOT NULL,
                    model TEXT NOT NULL,
                    action TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    greeting TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    use_new_shared_trial INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    position INTEGER NOT NULL DEFAULT 0,
                    result TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            conn.commit()
        else:
            cursor = conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    input TEXT NOT NULL,
                    model TEXT NOT NULL,
                    action TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    greeting TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    use_new_shared_trial INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    position INTEGER NOT NULL DEFAULT 0,
                    result TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
    finally:
        if hasattr(conn, 'close'):
            conn.close()

init_db()

class GenerationRequest(BaseModel):
    input: str
    model: str = "nai-diffusion-3"
    action: str = "generate"
    parameters: dict = {}
    api_key: str = ""
    greeting: str = "正在生成中~"
    negative_prompt: str = ""
    use_new_shared_trial: bool = True

async def call_novelai_api(request: GenerationRequest):
    if not request.api_key or request.api_key.strip() == "":
        raise ValueError("API key 为空，请在客户端设置有效的 NovelAI API key")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {request.api_key.strip()}"
    }
    
    payload = {
        "input": request.input,
        "model": request.model,
        "action": request.action,
        "parameters": request.parameters
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                "https://image.novelai.net/ai/generate-image",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 401:
                raise ValueError(f"API key 无效或已过期 (HTTP {response.status_code})")
            elif response.status_code == 403:
                raise ValueError(f"访问被拒绝，请检查 API key 和账户权限 (HTTP {response.status_code})")
            elif response.status_code == 429:
                raise ValueError(f"请求过于频繁，请稍后重试 (HTTP {response.status_code})")
            elif response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('detail', error_data.get('error', str(error_data)))
                except:
                    try:
                        error_msg = response.text
                    except:
                        error_msg = str(response.content[:500])
                raise ValueError(f"NovelAI API 错误 (HTTP {response.status_code}): {error_msg}")
            
            content_type = response.headers.get('content-type', '')
            
            if 'json' in content_type.lower():
                try:
                    data = response.json()
                    if "imageBase64" in data:
                        return f"data:image/png;base64,{data['imageBase64']}"
                    elif "images" in data and data["images"]:
                        return f"data:image/png;base64,{data['images'][0]}"
                    else:
                        raise ValueError(f"API 返回不包含图片数据: {str(data)[:200]}")
                except Exception as e:
                    raise ValueError(f"解析 JSON 响应失败: {str(e)}")
            
            elif 'zip' in content_type.lower() or response.content[:4] == b'PK\x03\x04':
                import base64
                zip_base64 = base64.b64encode(response.content).decode('ascii')
                return f"data:application/zip;base64,{zip_base64}"
            
            elif 'image' in content_type.lower():
                import base64
                img_base64 = base64.b64encode(response.content).decode('ascii')
                return f"data:{content_type};base64,{img_base64}"
            
            else:
                try:
                    text = response.text
                    if text.strip():
                        raise ValueError(f"API 返回未知格式: {text[:200]}")
                except:
                    pass
                raise ValueError(f"API 返回未知内容类型: {content_type}")
                
        except httpx.ConnectError:
            raise ValueError("无法连接到 NovelAI API，请检查网络连接")
        except httpx.TimeoutException:
            raise ValueError("请求超时，请重试")
        except Exception as e:
            raise ValueError(f"请求失败: {str(e)}")

def row_to_dict(row):
    if hasattr(row, 'keys'):
        return dict(row)
    elif hasattr(row, '__dict__'):
        return row.__dict__
    else:
        return dict(zip(['id', 'input', 'model', 'action', 'parameters', 'api_key', 'greeting', 'negative_prompt', 'use_new_shared_trial', 'status', 'position', 'result', 'error', 'created_at', 'started_at', 'completed_at'], row))

async def process_next_task():
    conn = get_db_connection()
    
    try:
        if hasattr(conn, 'execute'):
            conn.execute('''
                UPDATE tasks 
                SET status = 'processing', started_at = CURRENT_TIMESTAMP 
                WHERE id = (
                    SELECT id FROM tasks 
                    WHERE status = 'pending' 
                    ORDER BY created_at ASC 
                    LIMIT 1
                )
                AND NOT EXISTS (
                    SELECT 1 FROM tasks WHERE status = 'processing'
                )
            ''')
            conn.commit()
            
            cursor = conn.execute('''
                SELECT * FROM tasks 
                WHERE status = 'processing'
                ORDER BY started_at DESC 
                LIMIT 1
            ''')
            task = cursor.fetchone()
        else:
            await conn.execute('''
                UPDATE tasks 
                SET status = 'processing', started_at = CURRENT_TIMESTAMP 
                WHERE id = (
                    SELECT id FROM tasks 
                    WHERE status = 'pending' 
                    ORDER BY created_at ASC 
                    LIMIT 1
                )
                AND NOT EXISTS (
                    SELECT 1 FROM tasks WHERE status = 'processing'
                )
            ''')
            
            result = await conn.execute('''
                SELECT * FROM tasks 
                WHERE status = 'processing'
                ORDER BY started_at DESC 
                LIMIT 1
            ''')
            task = result.rows[0] if result.rows else None
        
        if not task:
            logger.info("队列为空或已有任务在处理中")
            return
        
        task_dict = row_to_dict(task)
        task_id = task_dict['id']
        logger.info(f"开始处理任务: {task_id}")
        
        req = GenerationRequest(
            input=task_dict['input'],
            model=task_dict['model'],
            action=task_dict['action'],
            parameters=eval(task_dict['parameters']),
            api_key=task_dict['api_key'],
            greeting=task_dict['greeting'],
            negative_prompt=task_dict['negative_prompt'],
            use_new_shared_trial=bool(task_dict['use_new_shared_trial'])
        )
        
        try:
            image_data = await call_novelai_api(req)
            
            if hasattr(conn, 'execute'):
                conn.execute('''
                    UPDATE tasks 
                    SET status = 'completed', result = ?, completed_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (image_data, task_id))
                conn.commit()
            else:
                await conn.execute('''
                    UPDATE tasks 
                    SET status = 'completed', result = ?, completed_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (image_data, task_id))
            
            logger.info(f"任务完成: {task_id}")
        except Exception as e:
            if hasattr(conn, 'execute'):
                conn.execute('''
                    UPDATE tasks 
                    SET status = 'failed', error = ?, completed_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (str(e), task_id))
                conn.commit()
            else:
                await conn.execute('''
                    UPDATE tasks 
                    SET status = 'failed', error = ?, completed_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (str(e), task_id))
            
            logger.error(f"任务失败: {task_id}, 错误: {e}")
        
        if hasattr(conn, 'execute'):
            conn.execute('''
                UPDATE tasks 
                SET position = (SELECT COUNT(*) FROM tasks t2 WHERE t2.created_at < tasks.created_at AND t2.status = 'pending') + 1
                WHERE status = 'pending'
            ''')
            conn.commit()
        else:
            await conn.execute('''
                UPDATE tasks 
                SET position = (SELECT COUNT(*) FROM tasks t2 WHERE t2.created_at < tasks.created_at AND t2.status = 'pending') + 1
                WHERE status = 'pending'
            ''')
        
    finally:
        if hasattr(conn, 'close'):
            conn.close()

@app.get("/ping")
async def ping():
    return {"status": "alive", "db_type": "turso" if TURSO_DB_URL else "sqlite"}

@app.get("/stats")
async def stats():
    conn = get_db_connection()
    
    try:
        if hasattr(conn, 'execute'):
            cursor = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            pending_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'processing'")
            processing_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
            completed_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'failed'")
            failed_count = cursor.fetchone()[0]
        else:
            result = await conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            pending_count = result.rows[0][0]
            
            result = await conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'processing'")
            processing_count = result.rows[0][0]
            
            result = await conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
            completed_count = result.rows[0][0]
            
            result = await conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'failed'")
            failed_count = result.rows[0][0]
        
        return {
            "service": "NovelAI Cloud Queue Service",
            "status": "running",
            "db_type": "turso" if TURSO_DB_URL else "sqlite",
            "pending_tasks": pending_count,
            "processing_tasks": processing_count,
            "completed_tasks": completed_count,
            "failed_tasks": failed_count
        }
    finally:
        if hasattr(conn, 'close'):
            conn.close()

@app.post("/queue")
async def submit_task(request: GenerationRequest):
    task_id = str(uuid.uuid4())[:12]
    
    conn = get_db_connection()
    
    try:
        if hasattr(conn, 'execute'):
            cursor = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            position = cursor.fetchone()[0] + 1
            
            conn.execute('''
                INSERT INTO tasks 
                (id, input, model, action, parameters, api_key, greeting, negative_prompt, use_new_shared_trial, status, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            ''', (
                task_id,
                request.input,
                request.model,
                request.action,
                str(request.parameters),
                request.api_key,
                request.greeting,
                request.negative_prompt,
                int(request.use_new_shared_trial),
                position
            ))
            conn.commit()
        else:
            result = await conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            position = result.rows[0][0] + 1
            
            await conn.execute('''
                INSERT INTO tasks 
                (id, input, model, action, parameters, api_key, greeting, negative_prompt, use_new_shared_trial, status, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            ''', (
                task_id,
                request.input,
                request.model,
                request.action,
                str(request.parameters),
                request.api_key,
                request.greeting,
                request.negative_prompt,
                int(request.use_new_shared_trial),
                position
            ))
        
        logger.info(f"任务入队: {task_id}, 位置: {position}")
        
        await process_next_task()
        
        if hasattr(conn, 'execute'):
            cursor = conn.execute('SELECT status, result FROM tasks WHERE id = ?', (task_id,))
            task = cursor.fetchone()
        else:
            result = await conn.execute('SELECT status, result FROM tasks WHERE id = ?', (task_id,))
            task = result.rows[0] if result.rows else None
        
        if task and task[0] == 'completed':
            return {
                "taskId": task_id,
                "position": position,
                "status": "completed",
                "completed": True,
                "result": {"imageBase64": task[1]},
                "imageData": task[1],
                "greeting": request.greeting
            }
        
        return {
            "taskId": task_id,
            "position": position,
            "status": "submitted",
            "greeting": request.greeting
        }
    finally:
        if hasattr(conn, 'close'):
            conn.close()

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    conn = get_db_connection()
    
    try:
        if hasattr(conn, 'execute'):
            cursor = conn.execute('''
                SELECT id, status, position, result, error, created_at, started_at, completed_at 
                FROM tasks 
                WHERE id = ?
            ''', (task_id,))
            
            task = cursor.fetchone()
        else:
            result = await conn.execute('''
                SELECT id, status, position, result, error, created_at, started_at, completed_at 
                FROM tasks 
                WHERE id = ?
            ''', (task_id,))
            
            task = result.rows[0] if result.rows else None
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        await process_next_task()
        
        task_dict = row_to_dict(task)
        
        response = {
            "taskId": task_dict['id'],
            "status": task_dict['status'],
            "position": task_dict['position']
        }
        
        if task_dict['status'] == 'completed':
            response["completed"] = True
            response["result"] = {"imageBase64": task_dict['result']}
            response["imageData"] = task_dict['result']
        elif task_dict['status'] == 'failed':
            response["failed"] = True
            response["error"] = task_dict['error']
        elif task_dict['status'] == 'processing':
            response["completed"] = False
        
        return response
    finally:
        if hasattr(conn, 'close'):
            conn.close()

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    conn = get_db_connection()
    
    try:
        if hasattr(conn, 'execute'):
            cursor = conn.execute('''
                UPDATE tasks 
                SET status = 'cancelled' 
                WHERE id = ? AND status = 'pending'
            ''', (task_id,))
            
            affected = cursor.rowcount
            conn.commit()
            
            conn.execute('''
                UPDATE tasks 
                SET position = (SELECT COUNT(*) FROM tasks t2 WHERE t2.created_at < tasks.created_at AND t2.status = 'pending') + 1
                WHERE status = 'pending'
            ''')
            conn.commit()
        else:
            result = await conn.execute('''
                UPDATE tasks 
                SET status = 'cancelled' 
                WHERE id = ? AND status = 'pending'
            ''', (task_id,))
            
            affected = result.rows_affected
            
            await conn.execute('''
                UPDATE tasks 
                SET position = (SELECT COUNT(*) FROM tasks t2 WHERE t2.created_at < tasks.created_at AND t2.status = 'pending') + 1
                WHERE status = 'pending'
            ''')
        
        if affected > 0:
            return {"success": True, "message": "任务已取消"}
        else:
            return {"success": False, "message": "任务不存在或已在处理中"}
    finally:
        if hasattr(conn, 'close'):
            conn.close()

@app.post("/api/predict")
async def api_predict(request: GenerationRequest):
    task_id = str(uuid.uuid4())[:12]
    
    conn = get_db_connection()
    
    try:
        if hasattr(conn, 'execute'):
            cursor = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            position = cursor.fetchone()[0] + 1
            
            conn.execute('''
                INSERT INTO tasks 
                (id, input, model, action, parameters, api_key, greeting, negative_prompt, use_new_shared_trial, status, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            ''', (
                task_id,
                request.input,
                request.model,
                request.action,
                str(request.parameters),
                request.api_key,
                request.greeting,
                request.negative_prompt,
                int(request.use_new_shared_trial),
                position
            ))
            conn.commit()
        else:
            result = await conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            position = result.rows[0][0] + 1
            
            await conn.execute('''
                INSERT INTO tasks 
                (id, input, model, action, parameters, api_key, greeting, negative_prompt, use_new_shared_trial, status, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            ''', (
                task_id,
                request.input,
                request.model,
                request.action,
                str(request.parameters),
                request.api_key,
                request.greeting,
                request.negative_prompt,
                int(request.use_new_shared_trial),
                position
            ))
        
        logger.info(f"任务入队 (/api/predict): {task_id}, 位置: {position}")
        
        await process_next_task()
        
        return {
            "success": True,
            "id": task_id,
            "position": position,
            "status": "submitted",
            "prompt": request.input
        }
    finally:
        if hasattr(conn, 'close'):
            conn.close()

async def handle_generic_request(body: dict):
    task_id = str(uuid.uuid4())[:12]
    
    input_text = body.get('input', body.get('prompt', body.get('text', '')))
    model = body.get('model', body.get('model_name', 'nai-diffusion-3'))
    action = body.get('action', 'generate')
    parameters = body.get('parameters', body.get('params', {}))
    api_key = body.get('api_key', body.get('apiKey', body.get('token', body.get('key', ''))))
    greeting = body.get('greeting', body.get('message', '正在生成中~'))
    negative_prompt = body.get('negative_prompt', body.get('negativePrompt', body.get('negative', '')))
    use_new_shared_trial = body.get('use_new_shared_trial', True)
    
    conn = get_db_connection()
    
    try:
        if hasattr(conn, 'execute'):
            cursor = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            position = cursor.fetchone()[0] + 1
            
            conn.execute('''
                INSERT INTO tasks 
                (id, input, model, action, parameters, api_key, greeting, negative_prompt, use_new_shared_trial, status, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            ''', (
                task_id, input_text, model, action, str(parameters), api_key, 
                greeting, negative_prompt, int(use_new_shared_trial), position
            ))
            conn.commit()
        else:
            result = await conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
            position = result.rows[0][0] + 1
            
            await conn.execute('''
                INSERT INTO tasks 
                (id, input, model, action, parameters, api_key, greeting, negative_prompt, use_new_shared_trial, status, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            ''', (
                task_id, input_text, model, action, str(parameters), api_key, 
                greeting, negative_prompt, int(use_new_shared_trial), position
            ))
        
        logger.info(f"任务入队 (通用端点): {task_id}, 位置: {position}")
        
        await process_next_task()
        
        if hasattr(conn, 'execute'):
            cursor = conn.execute('SELECT status, result FROM tasks WHERE id = ?', (task_id,))
            task = cursor.fetchone()
        else:
            result = await conn.execute('SELECT status, result FROM tasks WHERE id = ?', (task_id,))
            task = result.rows[0] if result.rows else None
        
        if task and task[0] == 'completed':
            return {
                "success": True,
                "taskId": task_id,
                "position": position,
                "status": "completed",
                "completed": True,
                "result": {"imageBase64": task[1]},
                "imageData": task[1]
            }
        
        return {
            "success": True,
            "taskId": task_id,
            "position": position,
            "status": "submitted"
        }
    finally:
        if hasattr(conn, 'close'):
            conn.close()

@app.post("/api/generate")
async def api_generate(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.post("/v1/generate")
async def v1_generate(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.post("/submit")
async def submit(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.post("/task")
async def task(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.post("/api/queue")
async def api_queue(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.post("/v1/queue")
async def v1_queue(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.post("/ai/generate-image")
async def ai_generate_image(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.post("/join-queue")
async def join_queue(request: Request):
    body = await request.json()
    return await handle_generic_request(body)

@app.get("/")
async def root():
    return {
        "message": "NovelAI Cloud Queue Service",
        "db_type": "turso" if TURSO_DB_URL else "sqlite",
        "endpoints": {
            "POST /queue": "提交生成任务（入队）",
            "POST /api/queue": "提交任务（兼容通用API）",
            "POST /v1/queue": "提交任务（兼容通用API）",
            "POST /api/generate": "提交任务（兼容通用API）",
            "POST /generate": "提交任务（兼容通用API）",
            "POST /v1/generate": "提交任务（兼容通用API）",
            "POST /submit": "提交任务（兼容通用API）",
            "POST /task": "提交任务（兼容通用API）",
            "POST /ai/generate-image": "提交任务（兼容 NovelAI 原生API格式）",
            "POST /join-queue": "提交任务（兼容酒馆智绘姬插件）",
            "POST /api/predict": "提交任务（兼容 Gradio API）",
            "GET /status/{task_id}": "查询任务状态（轮询）",
            "POST /cancel/{task_id}": "取消任务",
            "GET /ping": "健康检查",
            "GET /stats": "队列统计"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)