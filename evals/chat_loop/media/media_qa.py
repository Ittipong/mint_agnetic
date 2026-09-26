"""Send slip images (/chat/stream image_b64s) or voice clips (/chat/voice)
through the tunnel:  python3 media_qa.py slip_transfer.jpg receipt_711.jpg v1.m4a
"""
import base64, json, os, sys, time, uuid, subprocess
BASE="https://chat.minttechdev.uk"; U="ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
def parse(raw):
    ev=None; out=[]; data=[]
    for line in raw.splitlines():
        if line.startswith("event:"): ev=line[6:].strip()
        elif line.startswith("data:"): data.append(line[5:].lstrip(" ") if line[5:].startswith(" ") else line[5:])
        elif line=="":
            if ev: out.append((ev,"\n".join(data)))
            ev=None; data=[]
    return out
def show(name, evs, secs):
    ans="".join(d for e,d in evs if e=="answer_token")
    print(f"\n#### {name} ({secs}s)")
    for e,d in evs:
        if e=="block":
            b=json.loads(d)
            if b.get("type") in ("answer","suggestions"): continue
            print("  [block]", json.dumps(b,ensure_ascii=False)[:900]); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_block.json"),"w").write(json.dumps(b,ensure_ascii=False))
        elif e=="error": print("  !!ERROR", d[:300])
    print("  <<<", ans[:600])
def slip(path):
    body={"user_id":U,"thread_id":f"qa-slip-{uuid.uuid4().hex[:6]}","message":"","image_b64s":[base64.b64encode(open(path,'rb').read()).decode()]}
    open('/tmp/_slip.json','w').write(json.dumps(body))
    t=time.time(); r=subprocess.run(["curl","-s","-N","-m","120","-X","POST",BASE+"/chat/stream","-H","Content-Type: application/json","--data-binary","@/tmp/_slip.json"],capture_output=True,text=True)
    show(path, parse(r.stdout.replace("\r","")), round(time.time()-t,1))
def voice(path):
    # VOICE_WALLET_ID = the wallet picked in the chat input (optional form field)
    wal = ["-F", f"wallet_id={os.environ['VOICE_WALLET_ID']}"] if os.environ.get("VOICE_WALLET_ID") else []
    t=time.time(); r=subprocess.run(["curl","-s","-N","-m","120","-X","POST",BASE+"/chat/voice","-F",f"user_id={U}","-F",f"thread_id=qa-voice-{uuid.uuid4().hex[:6]}",*wal,"-F",f"audio=@{path};type=audio/mp4"],capture_output=True,text=True)
    show(path, parse(r.stdout.replace("\r","")), round(time.time()-t,1))
for p in sys.argv[1:]:
    (slip if p.endswith(".jpg") else voice)(p)
