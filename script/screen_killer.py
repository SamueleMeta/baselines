from screenutils import list_screens

for s in list_screens():
    if s.name != 'tensorboard':
        s.kill()
