



# ========================== 可视化相关的代码 ==========================

DRAW_SCALE = 40
DRAW_POINT = 2


window = pyglet.window.Window()
batch = pyglet.graphics.Batch()

def entity_draw_comment(entity,base_y):
    rA_label = pyglet.text.Label(f'{entity.name}',
                            font_name='Times New Roman',
                            font_size=10,
                            color = (0,0,0,255),
                            x=0, y=window.height-4-base_y,
                            anchor_x='left', anchor_y='top',
                            batch=batch                          
                            )
    rA_Circle = pyglet.shapes.Circle(250,window.height-12-base_y,7,batch=batch)
    rA_Circle.color = entity.color
    rA_Circle.opacity = 128
    batch.draw()

def entity_draw_body():
    draw_group = []    
    draw_group.append(pyglet.shapes.Circle(env.redA.get_simulate_position()[0],env.redA.get_simulate_position()[1],env.redA.explore_size*DRAW_SCALE,batch=batch))
    draw_group[0].opacity = 10
    draw_group[0].color = env.redA.color

    draw_group.append(pyglet.shapes.Circle(env.redB1.get_simulate_position()[0],env.redB1.get_simulate_position()[1],env.redB1.explore_size*DRAW_SCALE,batch=batch))
    draw_group[1].opacity = 128
    draw_group[1].color = env.redB1.color
    
    draw_group.append(pyglet.shapes.Circle(env.redB2.get_simulate_position()[0],env.redB2.get_simulate_position()[1],env.redB2.explore_size*DRAW_SCALE,batch=batch))
    draw_group[2].opacity = 128
    draw_group[2].color = env.redB2.color

    draw_group.append(pyglet.shapes.Circle(env.blueA.get_simulate_position()[0],env.blueA.get_simulate_position()[1],env.blueA.explore_size*DRAW_SCALE,batch=batch))
    draw_group[3].opacity = 128
    draw_group[3].color = env.blueA.color
    batch.draw()

    draw_point = []
    draw_point.append(pyglet.shapes.Circle(env.redA.get_simulate_position()[0],env.redA.get_simulate_position()[1],DRAW_POINT,batch=batch))
    draw_point[0].opacity = 255
    draw_point[0].color = (255,255,255)

    draw_point.append(pyglet.shapes.Circle(env.redB1.get_simulate_position()[0],env.redB1.get_simulate_position()[1],DRAW_POINT,batch=batch))
    draw_point[1].opacity = 255
    draw_point[1].color = (255,255,255)
    
    draw_point.append(pyglet.shapes.Circle(env.redB2.get_simulate_position()[0],env.redB2.get_simulate_position()[1],DRAW_POINT,batch=batch))
    draw_point[2].opacity = 255
    draw_point[2].color = (255,255,255)

    draw_point.append(pyglet.shapes.Circle(env.blueA.get_simulate_position()[0],env.blueA.get_simulate_position()[1],DRAW_POINT,batch=batch))
    draw_point[3].opacity = 255
    draw_point[3].color = (255,255,255)
    batch.draw()


@window.event
def on_draw():
    pyglet.gl.glClearColor(1, 1, 1, 1)
    window.clear()
    entity_list = [env.redA,env.redB1,env.redB2,env.blueA]
    entity_draw_comment(env.redA,0)
    entity_draw_comment(env.redB1,16)
    entity_draw_comment(env.redB2,16*2)
    entity_draw_comment(env.blueA,16*3)

    entity_draw_body()

i = 0
init_state = env.reset()
print(init_state)
print(env.action_space)
print(env.observation_space)
def update(dt):
    global i
    if i < 488:
        env.step(env.action_space)
        # print(env.get_overall_observation())        
        if env.done:
            init_state = env.reset()
            # print(init_state)
        i += 1
    else:
        i=0
        env.reset()

pyglet.clock.schedule_interval(update, 0.1)
pyglet.app.run()