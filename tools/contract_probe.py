from __future__ import annotations
import json, sys, types, inspect

def install_stubs():
    psycopg2=types.ModuleType("psycopg2")
    psycopg2.connect=lambda **kwargs: None
    sys.modules["psycopg2"]=psycopg2
    google=sys.modules.get("google") or types.ModuleType("google")
    if not hasattr(google,"__path__"): google.__path__=[]
    cloud=types.ModuleType("google.cloud"); cloud.__path__=[]
    tasks=types.ModuleType("google.cloud.tasks_v2")
    class CloudTasksClient: pass
    tasks.CloudTasksClient=CloudTasksClient
    cloud.tasks_v2=tasks; google.cloud=cloud
    sys.modules["google"]=google
    sys.modules["google.cloud"]=cloud
    sys.modules["google.cloud.tasks_v2"]=tasks

install_stubs()
import main

model_names=[
 "IngestRequest","IndexDocumentRequest","IngestUsageMonthRequest","SearchRequest",
 "AskRequest","RootCauseRequest","DraftPSOptions","DraftPSRequest",
 "DeleteDocumentRequest","DeleteCompanyIndexRequest","StructuredSourceIngestRequest",
]

def model_contract(cls):
    fields=getattr(cls,"model_fields",None)
    if fields is not None:
        rows={}
        for name,f in fields.items():
            rows[name]={"required":bool(f.is_required()),"default":repr(f.default),"annotation":str(f.annotation)}
        schema=cls.model_json_schema()
    else:
        rows={}
        for name,f in cls.__fields__.items():
            rows[name]={"required":bool(f.required),"default":repr(f.default),"annotation":str(f.outer_type_)}
        schema=cls.schema()
    return {"fields":rows,"schema":schema}

routes=[]
for r in main.app.routes:
    path=getattr(r,"path","")
    if path.startswith("/v1/") or path in {"/ping","/version"}:
        routes.append({"path":path,"methods":sorted(getattr(r,"methods",[]) or []),"name":getattr(r,"name","")})
routes.sort(key=lambda x:(x["path"],x["methods"]))

scope_cases=[]
cases=[
 ("c","m",None,None,None),
 (" c "," m "," d ",None,None),
 ("c","",None,["a"," b "],None),
 ("c","m","d",["x"],"machine_all"),
 ("c","m","d",["x"],"company_general"),
 ("c","",None,"a,b", "document_ids"),
]
for args in cases:
    try: value=main._resolve_query_scope(*args); error=None
    except Exception as exc:
        value=None; error={"type":type(exc).__name__,"status_code":getattr(exc,"status_code",None),"detail":getattr(exc,"detail",str(exc))}
    scope_cases.append({"args":args,"value":value,"error":error})
for args in [("","m",None,None,None),("c","",None,None,"machine_all"),("c","m",None,None,"bad")]:
    try: value=main._resolve_query_scope(*args); error=None
    except Exception as exc:
        value=None; error={"type":type(exc).__name__,"status_code":getattr(exc,"status_code",None),"detail":getattr(exc,"detail",str(exc))}
    scope_cases.append({"args":args,"value":value,"error":error})

out={
 "routes":routes,
 "models":{n:model_contract(getattr(main,n)) for n in model_names},
 "scope_cases":scope_cases,
 "sentinel":main.COMPANY_GENERAL_MACHINE_SENTINEL,
 "scope_signatures":{
   n:str(inspect.signature(getattr(main,n))) for n in ["_normalize_document_ids","_normalize_ai_scope","_resolve_query_scope"]
 },
 "markers":{
   "core":main.ASSISTANT_CORE_V2_CODE_MARKER,
   "release":main.ASSISTANT_CORE_V2_RELEASE_ID,
   "ask_render":main.ASSISTANT_ASK_UI_RENDER_VERSION,
 },
}
print(json.dumps(out,sort_keys=True,ensure_ascii=False,default=str))
