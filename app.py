import streamlit as st
import math
import sys

import numpy as np
import pygame

W, H = 720, 780         
FPS = 60
TEKS = "I love you!"        
FOKUS, JARAK = 700.0, 5
CY = H // 2 + 50           
STEP_BUNGA = 4  
STEP_TEKS = 3              

rng = np.random.default_rng()

PALET = np.array([
    (255, 90, 120), (255, 200, 60), (90, 200, 255),
    (150, 255, 150), (200, 120, 255), (255, 255, 255),
], dtype=float)


def ease_out(x):
    x = float(np.clip(x, 0.0, 1.0))
    return 1 - (1 - x) ** 3


def plot(buf, xs, ys, cols, w=3, h=3):
    """Gambar titik-titik berukuran w x h ke frame buffer."""
    for dx in range(w):
        for dy in range(h):
            x = xs + dx
            y = ys + dy
            m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            buf[x[m], y[m]] = cols[m]


def grid(origin, u, v, nu, nv):
    a = np.linspace(0, 1, nu)
    b = np.linspace(0, 1, nv)
    A, B = np.meshgrid(a, b, indexing="ij")
    return (np.array(origin, float)[None, :]
            + A.reshape(-1, 1) * np.array(u, float)[None, :]
            + B.reshape(-1, 1) * np.array(v, float)[None, :])


def build_box():
    """Titik-titik permukaan kado. part: 0=badan, 1=dinding dalam, 2=tutup, 3=pita."""
    P, N, C, PART = [], [], [], []
    dens = 22

    def add(pts, normal, base, part, ribbon=True):
        base = np.array(base, float)
        if ribbon:
            rb = (np.abs(pts[:, 0]) < 0.13) | (np.abs(pts[:, 2]) < 0.13)
            col = np.where(rb[:, None], np.array((255, 235, 170), float), base[None, :])
        else:
            col = np.tile(base, (len(pts), 1))
        P.append(pts)
        N.append(np.tile(np.array(normal, float), (len(pts), 1)))
        C.append(col)
        PART.append(np.full(len(pts), part))

    hw, y0, y1 = 1.0, -1.0, 0.25
    nu, nv = int(2 * hw * dens), int((y1 - y0) * dens)
    body = (236, 84, 130)
    add(grid((-hw, y0, hw), (2 * hw, 0, 0), (0, y1 - y0, 0), nu, nv), (0, 0, 1), body, 0)
    add(grid((-hw, y0, -hw), (2 * hw, 0, 0), (0, y1 - y0, 0), nu, nv), (0, 0, -1), body, 0)
    add(grid((hw, y0, -hw), (0, 0, 2 * hw), (0, y1 - y0, 0), nu, nv), (1, 0, 0), body, 0)
    add(grid((-hw, y0, -hw), (0, 0, 2 * hw), (0, y1 - y0, 0), nu, nv), (-1, 0, 0), body, 0)

    hi = hw - 0.06
    nui = int(2 * hi * dens)
    dalam = (250, 180, 80)
    add(grid((-hi, y0, hi), (2 * hi, 0, 0), (0, y1 - y0, 0), nui, nv), (0, 0, -1), dalam, 1, False)
    add(grid((-hi, y0, -hi), (2 * hi, 0, 0), (0, y1 - y0, 0), nui, nv), (0, 0, 1), dalam, 1, False)
    add(grid((hi, y0, -hi), (0, 0, 2 * hi), (0, y1 - y0, 0), nui, nv), (-1, 0, 0), dalam, 1, False)
    add(grid((-hi, y0, -hi), (0, 0, 2 * hi), (0, y1 - y0, 0), nui, nv), (1, 0, 0), dalam, 1, False)
    add(grid((-hi, y0 + 0.06, -hi), (2 * hi, 0, 0), (0, 0, 2 * hi), nui, nui),
        (0, 1, 0), (255, 215, 120), 1, False)

    lw, ly0, ly1 = 1.1, 0.25, 0.65
    nul, nvl = int(2 * lw * dens), int((ly1 - ly0) * dens)
    lid = (255, 120, 160)
    add(grid((-lw, ly0, lw), (2 * lw, 0, 0), (0, ly1 - ly0, 0), nul, nvl), (0, 0, 1), lid, 2)
    add(grid((-lw, ly0, -lw), (2 * lw, 0, 0), (0, ly1 - ly0, 0), nul, nvl), (0, 0, -1), lid, 2)
    add(grid((lw, ly0, -lw), (0, 0, 2 * lw), (0, ly1 - ly0, 0), nul, nvl), (1, 0, 0), lid, 2)
    add(grid((-lw, ly0, -lw), (0, 0, 2 * lw), (0, ly1 - ly0, 0), nul, nvl), (-1, 0, 0), lid, 2)
    add(grid((-lw, ly1, -lw), (2 * lw, 0, 0), (0, 0, 2 * lw), nul, nul), (0, 1, 0), lid, 2)

    # pita di atas tutup
    t = np.linspace(0, 2 * math.pi, 60)
    for s in (-1, 1):
        for k in range(3):
            r = 1.0 - k * 0.18
            x = s * 0.38 + 0.32 * r * np.cos(t)
            y = ly1 + 0.22 + 0.2 * r * np.sin(t)
            z = 0.05 * np.sin(2 * t)
            pts = np.column_stack([x, y, z])
            P.append(pts)
            N.append(np.zeros_like(pts))
            C.append(np.tile(np.array((255, 225, 140), float), (len(pts), 1)))
            PART.append(np.full(len(pts), 3))

    return (np.concatenate(P), np.concatenate(N), np.concatenate(C), np.concatenate(PART))


def draw_box(buf, box, t, ang, tilt, open_t):
    P, Nrm, C, PART = box
    opened = open_t is not None
    ot = open_t if opened else 0.0

    keep = np.ones(len(P), bool)
    if not opened:
        keep &= PART != 1

    lid_t = max(0.0, ot - 0.5)
    lid_y = 1.5 * lid_t + 9.0 * lid_t ** 2
    lid_x = 1.2 * lid_t
    if lid_y > 9:
        keep &= PART < 2

    body_oy = -5.0 * ease_out((ot - 1.4) / 1.4) if opened else 0.0
    if body_oy < -4.8:
        keep &= PART >= 2
    if not keep.any():
        return

    pk = PART[keep]
    p = P[keep].copy()
    lid = pk >= 2
    p[lid, 0] += lid_x
    p[lid, 1] += lid_y
    p[~lid, 1] += body_oy
    p[:, 1] += 0.06 * math.sin(t * 2) if not opened else 0.0
    if opened and ot < 0.5:
        p[:, 0] += 0.035 * math.sin(ot * 80)

    ca, sa = math.cos(ang), math.sin(ang)
    ct, st = math.cos(tilt), math.sin(tilt)

    def rot(a):
        x1 = a[:, 0] * ca + a[:, 2] * sa
        z1 = -a[:, 0] * sa + a[:, 2] * ca
        y2 = a[:, 1] * ct + z1 * st
        z2 = -a[:, 1] * st + z1 * ct
        return x1, y2, z2

    x1, y2, z2 = rot(p)
    nx, ny, nz = rot(Nrm[keep])

    vis = (nz < 0.0) | (pk == 3)
    L = np.array([-0.35, 0.75, -0.55])
    L /= np.linalg.norm(L)
    dot = nx * L[0] + ny * L[1] + nz * L[2]
    light = 0.45 + 0.65 * np.clip(dot, 0, 1)
    light = np.where(pk == 3, 1.0, light)
    col = np.clip(C[keep] * light[:, None], 0, 255).astype(np.uint8)

    zc = z2 + JARAK
    sc = FOKUS / zc
    xs = (W / 2 + x1 * sc).astype(int)
    ys = (CY - y2 * sc).astype(int)

    idx = np.where(vis)[0]
    idx = idx[np.argsort(-zc[idx])]          
    plot(buf, xs[idx], ys[idx], col[idx], 4, 2)


def ellipse_pts(cx, cy, a, b, ang, n=48):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    xr = x * math.cos(ang) - y * math.sin(ang)
    yr = x * math.sin(ang) + y * math.cos(ang)
    return [(cx + px, cy + py) for px, py in zip(xr, yr)]


def petal(cx, cy, ang, length, width):
    ox = cx + math.cos(ang) * length
    oy = cy + math.sin(ang) * length
    return ellipse_pts(ox, oy, length, width, ang)


def sample(surf, step):
    arr = pygame.surfarray.array3d(surf)
    al = pygame.surfarray.array_alpha(surf)
    gx, gy = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step), indexing="ij")
    m = al[gx, gy] > 128
    xs, ys = gx[m], gy[m]
    return np.column_stack([xs, ys]).astype(float), arr[xs, ys].astype(float)


def build_targets():
    cx, cy = W // 2, 285

    fl = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.line(fl, (70, 175, 95), (cx, cy + 40), (cx, 610), 12)
    for s in (-1, 1):
        pygame.draw.polygon(fl, (80, 190, 105), ellipse_pts(cx + s * 62, 510, 64, 22, -0.5 * s))
    for i in range(10):
        c = (255, 95, 150) if i % 2 == 0 else (255, 120, 170)
        pygame.draw.polygon(fl, c, petal(cx, cy, i * 2 * math.pi / 10, 125, 48))
    for i in range(10):
        pygame.draw.polygon(fl, (255, 160, 195),
                            petal(cx, cy, (i + 0.5) * 2 * math.pi / 10, 90, 36))
    for i in range(8):
        pygame.draw.polygon(fl, (255, 210, 225),
                            petal(cx, cy, i * 2 * math.pi / 8, 55, 24))
    pygame.draw.circle(fl, (255, 205, 70), (cx, cy), 34)
    pygame.draw.circle(fl, (240, 150, 40), (cx, cy), 18)
    fpos, fcol = sample(fl, STEP_BUNGA)

    font = pygame.font.SysFont("georgia,timesnewroman,serif", 96, bold=True, italic=True)
    txt = font.render(TEKS, True, (255, 225, 240))
    tl = pygame.Surface((W, H), pygame.SRCALPHA)
    tl.blit(txt, txt.get_rect(center=(cx, 705)))
    tpos, tcol = sample(tl, STEP_TEKS)
    k = (tpos[:, 0] - tpos[:, 0].min()) / max(1.0, np.ptp(tpos[:, 0]))
    tcol = np.column_stack([np.full(len(k), 255.0), 190 + 65 * k, 215 + 40 * k])

    pos = np.concatenate([fpos, tpos])
    col = np.concatenate([fcol, tcol])
    col *= (0.9 + 0.1 * rng.random((len(col), 1)))
    return pos, col


class Surprise:
    G = 900.0      
    K = 40.0       
    C = 9.0        

    def __init__(self):
        self.box = build_box()
        self.tgt_pos, self.tgt_col = build_targets()
        self.N = len(self.tgt_pos)
        self.ph = rng.uniform(0, 6.28, self.N)
        self.star_pos = rng.uniform([0, 0], [W, H], (140, 2))
        self.star_ph = rng.uniform(0, 6.28, 140)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.ang = 0.5
        self.tilt = 0.35
        self.open_t = None
        self.burst_done = False
        n = 120
        self.rain_pos = np.column_stack([rng.uniform(0, W, n), rng.uniform(-H, 0, n)])
        self.rain_vel = np.column_stack([rng.uniform(-20, 20, n), rng.uniform(70, 170, n)])
        self.rain_col = PALET[rng.integers(0, len(PALET), n)].astype(np.uint8)
        self.rain_ph = rng.uniform(0, 6.28, n)

    def open(self):
        if self.open_t is None:
            self.open_t = 0.0

    def burst(self):
        n = self.N
        ox, oy = W / 2, CY - 0.25 * math.cos(self.tilt) * FOKUS / JARAK - 20
        self.pos = np.tile([ox, oy], (n, 1)) + rng.normal(0, 18, (n, 2))
        a = rng.uniform(-math.pi * 0.85, -math.pi * 0.15, n)
        sp = rng.uniform(250, 900, n)
        self.vel = np.column_stack([np.cos(a) * sp * 0.6, np.sin(a) * sp])
        d = np.hypot(self.tgt_pos[:, 0] - W / 2, self.tgt_pos[:, 1] - 300)
        self.switch = 0.6 + d / d.max() * 0.8 + rng.uniform(0, 0.25, n)
        self.age = -rng.uniform(0, 0.4, n)
        self.conf = PALET[rng.integers(0, len(PALET), n)]
        self.burst_done = True

    def update(self, dt):
        self.t += dt
        if self.open_t is None:
            self.ang += dt * 0.9
        else:
            self.open_t += dt
            speed = max(0.0, 1 - self.open_t / 0.6)
            self.ang += dt * 0.9 * speed
            self.tilt = 0.35 + 0.22 * ease_out(self.open_t / 0.6)
            if self.open_t >= 0.5 and not self.burst_done:
                self.burst()

        if self.burst_done:
            self.age += dt
            active = self.age >= 0
            ball = active & (self.age < self.switch)
            form = active & (self.age >= self.switch)
            self.vel[ball, 1] += self.G * dt
            if form.any():
                ramp = np.clip((self.age[form] - self.switch[form]) / 0.6, 0, 1)
                acc = (self.K * ramp)[:, None] * (self.tgt_pos[form] - self.pos[form]) \
                    - self.C * self.vel[form]
                self.vel[form] += acc * dt
            self.pos[active] += self.vel[active] * dt

        if self.open_t is not None and self.open_t > 3.5:
            self.rain_pos += self.rain_vel * dt
            self.rain_pos[:, 0] += np.sin(self.t * 2 + self.rain_ph) * 0.6
            out = self.rain_pos[:, 1] > H + 10
            self.rain_pos[out, 1] = -10
            self.rain_pos[out, 0] = rng.uniform(0, W, out.sum())

    def draw(self, screen, font):
        buf = np.zeros((W, H, 3), np.uint8)

        b = (110 + 110 * np.sin(self.t * 2 + self.star_ph)).clip(0, 255).astype(np.uint8)
        plot(buf, self.star_pos[:, 0].astype(int), self.star_pos[:, 1].astype(int),
             np.stack([b, b, b], 1), 2, 2)

        draw_box(buf, self.box, self.t, self.ang, self.tilt, self.open_t)

        if self.burst_done:
            act = self.age >= 0
            s = np.clip((self.age - self.switch) / 1.0, 0, 1)[:, None]
            tw = 1 + 0.12 * np.sin(self.t * 3 + self.ph)[:, None] * s
            col = (self.conf * (1 - s) + self.tgt_col * s) * tw
            sway = s * 1.5 * np.column_stack([np.sin(self.t * 1.5 + self.ph),
                                              np.cos(self.t * 1.3 + self.ph)])
            p = self.pos + sway
            plot(buf, p[act, 0].astype(int), p[act, 1].astype(int),
                 col[act].clip(0, 255).astype(np.uint8), 3, 3)

        if self.open_t is not None and self.open_t > 3.5:
            plot(buf, self.rain_pos[:, 0].astype(int), self.rain_pos[:, 1].astype(int),
                 self.rain_col, 4, 3)

        pygame.surfarray.blit_array(screen, buf)

        if self.open_t is None:
            msg = "Klik atau tekan SPASI untuk membuka kado"
            alpha = int(150 + 100 * math.sin(self.t * 3))
        else:
            return
        img = font.render(msg, True, (255, 255, 255))
        img.set_alpha(alpha)
        screen.blit(img, img.get_rect(center=(W // 2, H - 35)))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Kado Surprise")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    show = Surprise()

    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if e.key in (pygame.K_SPACE, pygame.K_RETURN):
                    show.open()
                if e.key == pygame.K_r:
                    show.reset()
            if e.type == pygame.MOUSEBUTTONDOWN:
                show.open()
        show.update(dt)
        show.draw(screen, font)
        pygame.display.flip()


if __name__ == "__main__":
    main()