import sys, json
for l in sys.stdin:
    if not l.startswith('{'): print(l.strip()[:300]); continue
    r = json.loads(l)
    print('M=%-4d ort %.2f(%s) | ' % (r['M'], r['ort_ms'], r['ort_err_vs_ref']) + ' | '.join('%s %.2f(%s)' % (k[:-3].replace('v2_', ''), v, r[k[:-3] + '_err']) for k, v in r.items() if k.endswith('_ms') and k != 'ort_ms'))
