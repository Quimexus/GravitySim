import taichi as ti
import pygame
import sys


ti.init(arch=ti.gpu)


NUM_PARTICLES = 200000
INITIAL_WIDTH = 1080
INITIAL_HEIGHT = 720

# Base Constants
GRAVITY_STRENGTH = 1000.0
FRICTION = 0.99

# Attraction Multiplier
ATTRACTION_MULTIPLIER_NORMAL = 1.0
ATTRACTION_MULTIPLIER_CLICK = 10.0  # <--- New: 10x multiplier on click

# Speed Control
TARGET_FPS = 165
TIME_SCALE = 0.1

# Pygame
pygame.init()
clock = pygame.time.Clock()

# Taichi
pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)

# Initial screen setup
surface = pygame.display.set_mode((INITIAL_WIDTH, INITIAL_HEIGHT), pygame.RESIZABLE)
canvas_ti = ti.Vector.field(3, dtype=ti.f32, shape=surface.get_size())

# Global display state
global_zoom_level = 1.0
global_aspect_ratio = INITIAL_WIDTH / INITIAL_HEIGHT

@ti.kernel
def initialize_particles():
    for i in pos:
        pos[i] = ti.Vector([0.5, 0.5])
        # Smaller initial velocity for a slower look
        vel[i] = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.005


@ti.kernel
def update_physics(mouse_world_x: ti.f32, mouse_world_y: ti.f32, dt: ti.f32,
                   is_attracting_10x: ti.i32):
    mouse_loc = ti.Vector([mouse_world_x, mouse_world_y])

    # Determine the current attraction multiplier
    multiplier = ATTRACTION_MULTIPLIER_NORMAL
    if is_attracting_10x == 1:
        multiplier = ATTRACTION_MULTIPLIER_CLICK

    effective_gravity_const = GRAVITY_STRENGTH * multiplier * 0.00005  # Combined constant

    for i in pos:
        diff = mouse_loc - pos[i]
        distance_sq = diff.norm_sqr()

        force = ti.Vector([0.0, 0.0])
        if distance_sq > 0.00001:
            force = diff.normalized() * effective_gravity_const / distance_sq

        vel[i] += force * dt
        vel[i] *= (1.0 - (1.0 - FRICTION) * dt * 10.0)
        pos[i] += vel[i] * dt


@ti.kernel
def render_to_canvas(zoom: ti.f32, aspect_ratio: ti.f32, current_width: ti.i32, current_height: ti.i32,
                     canvas: ti.template()):
    for i, j in canvas:
        canvas[i, j] = ti.Vector([0.0, 0.0, 1.0])

    particle_color = ti.Vector([1.0, 1.0, 1.0])
    for i in range(NUM_PARTICLES):
        centered_x = pos[i].x - 0.5
        centered_y = pos[i].y - 0.5

        zoomed_x = centered_x * zoom
        zoomed_y = centered_y * zoom

        screen_x_norm = (zoomed_x / aspect_ratio) + 0.5
        screen_y_norm = zoomed_y + 0.5

        x = ti.cast(screen_x_norm * current_width, ti.i32)
        y = ti.cast(screen_y_norm * current_height, ti.i32)

        if 0 <= x < current_width and 0 <= y < current_height:
            canvas[x, y] = particle_color


# MAIN LOOP
def main():
    global global_zoom_level, global_aspect_ratio, surface, canvas_ti

    initialize_particles()

    last_time = pygame.time.get_ticks()
    is_left_mouse_down = False

    while True:
        current_time = pygame.time.get_ticks()
        frame_dt = (current_time - last_time) / 1000.0
        last_time = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.VIDEORESIZE:
                new_width, new_height = event.size
                surface = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                canvas_ti = ti.Vector.field(3, dtype=ti.f32, shape=(new_width, new_height))
                global_aspect_ratio = new_width / new_height

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    initialize_particles()
                elif event.key == pygame.K_f:
                    # Toggle fullscreen
                    flags = surface.get_flags()
                    if flags & pygame.FULLSCREEN:
                        surface = pygame.display.set_mode((INITIAL_WIDTH, INITIAL_HEIGHT), pygame.RESIZABLE)
                    else:
                        surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.RESIZABLE)

                    new_width, new_height = surface.get_size()
                    canvas_ti = ti.Vector.field(3, dtype=ti.f32, shape=(new_width, new_height))
                    global_aspect_ratio = new_width / new_height

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    global_zoom_level *= 1.1
                elif event.button == 5:
                    global_zoom_level /= 1.1
                elif event.button == 1:
                    is_left_mouse_down = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_left_mouse_down = False

        # PHYSICS

        physics_dt = frame_dt * TIME_SCALE

        current_width, current_height = surface.get_size()

        # Mouse World Coordinate Conversion
        mouse_x_screen, mouse_y_screen = pygame.mouse.get_pos()
        mouse_screen_norm_x = mouse_x_screen / current_width
        mouse_screen_norm_y = mouse_y_screen / current_height

        centered_x = mouse_screen_norm_x - 0.5
        centered_y = mouse_screen_norm_y - 0.5
        world_x_pre_zoom = centered_x * global_aspect_ratio
        mouse_world_x = (world_x_pre_zoom / global_zoom_level) + 0.5
        mouse_world_y = (centered_y / global_zoom_level) + 0.5

        attract_10x_flag = 1 if is_left_mouse_down else 0

        update_physics(mouse_world_x, mouse_world_y, physics_dt, attract_10x_flag)

        render_to_canvas(global_zoom_level, global_aspect_ratio, current_width, current_height, canvas_ti)

        canvas_np = canvas_ti.to_numpy()
        pygame.surfarray.blit_array(surface, (canvas_np * 255).astype('uint8'))

        # UI
        pygame.display.set_caption(
            f"Particles (Pygame+Taichi) | FPS: {clock.get_fps():.1f} / {TARGET_FPS} | Time Scale: {TIME_SCALE:.2f}x | Click Multiplier: {'10x' if is_left_mouse_down else '1x'}")

        pygame.display.flip()

        # FPS
        clock.tick(TARGET_FPS)


if __name__ == "__main__":
    main()
