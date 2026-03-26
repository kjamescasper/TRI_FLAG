f = r'C:\Users\judge\OneDrive\Desktop\Lab Code\acegen-open\scripts\reinvent\reinvent.py'
content = open(f).read()
patched = content.replace('    PrioritizedSampler,\n', '    RandomSampler,\n')
if 'RandomSampler' in patched and 'PrioritizedSampler' not in patched:
    open(f, 'w').write(patched)
    print('patched ok')
else:
    print('ERROR: check manually')
    print('PrioritizedSampler still present:', 'PrioritizedSampler' in patched)
    print('RandomSampler present:', 'RandomSampler' in patched)