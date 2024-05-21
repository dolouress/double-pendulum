from acrobot_used import AcrobotEnv

acrobot = AcrobotEnv(render_mode='human')
acrobot.reset()
while(True):
    acrobot.step(-1)

