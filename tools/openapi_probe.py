from __future__ import annotations
import json, sys, types

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

import main
print(json.dumps(main.app.openapi(), sort_keys=True, ensure_ascii=False, separators=(",",":")))
