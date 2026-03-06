import os
import sys
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# 将项目根目录添加到 sys.path 以便导入 document_store
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_store import DocumentStore, KnowledgeDocument
from agent_types.common import LLMConfig

app = FastAPI()
# 使用绝对路径定位模板目录，确保从不同工作目录启动服务都能找到模板
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

# 初始化 DocumentStore
def get_store():
    # 路径基于项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")
    db_path = os.path.join(base_dir, "data", "document_db")
    
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
        cfg = full_cfg.get("embedding_url")
    
    if not cfg:
        raise RuntimeError("Config file does not contain 'embedding_url'")
    
    config = LLMConfig.model_validate(cfg)
    return DocumentStore(
        db_path=db_path,
        embedding_service_url=config,
        table_name="test_knowledge"  # 默认使用测试表名
    )

store = get_store()

@app.get("/", response_class=HTMLResponse)
async def list_documents(
    request: Request, 
    skip: int = 0, 
    limit: int = 100, 
    query: str = None,
    top_k: int = 10,
    vector_weight: float = 0.7,
    doc_type: str = None
):
    if query:
        # 语义检索模式：使用 top_k、vector_weight 和 doc_type 过滤
        results = store.search(
            query, 
            top_k=top_k, 
            vector_weight=vector_weight, 
            doc_type=doc_type if doc_type else None
        )
    else:
        # 列表浏览模式：使用 skip, limit 和 doc_type 过滤
        results = store.list_documents(
            type=doc_type if doc_type else None,
            skip=skip, 
            limit=limit
        )
        
    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": results,
        "skip": skip,
        "limit": limit,
        "query": query or "",
        "top_k": top_k,
        "vector_weight": vector_weight,
        "doc_type": doc_type or ""
    })

@app.post("/delete/{doc_id}")
async def delete_document(doc_id: str):
    store.delete_document(doc_id)
    return {"status": "success"}

@app.post("/update/{doc_id}")
async def update_document(doc_id: str, request: Request):
    data = await request.json()
    # 先获取原文档以保留元数据
    old_doc = store.get_document(doc_id)
    
    new_doc = KnowledgeDocument(
        id=doc_id,
        text=data.get("text", old_doc.text),
        type=data.get("type", old_doc.type),
        metadata=old_doc.metadata,
        created_at=old_doc.created_at
    )
    store.upsert_document(new_doc)
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
