import json,glob,re,sys
pat=sys.argv[1] if len(sys.argv)>1 else '*'
for f in sorted(glob.glob(f'results/{pat}.json')):
    r=json.load(open(f)); print(f"\n######## {r['id']}")
    for t in r['turns']:
        print(f">>> {t['msg']} ({t['secs']}s)")
        if t['errors']: print("  !!ERR", t['errors'])
        for b in t['blocks']:
            bt=b.get('type')
            if bt in ('answer','suggestions','status'): continue
            s=json.dumps(b,ensure_ascii=False)
            s=re.sub(r'"(icon|sync_id|transaction_sync_id|proposal_id|category_sync_id|wallet_sync_id|id)": "[^"]*",?','',s)
            print("  [",bt,"]",s[:500])
        a=re.sub(r'\{\{(wallet|category|goal|budget):[0-9a-f-]+\}\}\s*','',t['answer'])
        print("<<<",a[:1400].replace('\n\n','\n'))
