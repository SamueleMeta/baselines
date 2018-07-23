from screenutils import list_screens

for s in list_screens():
    if s.name.startswith('s_'):
        s.kill()
