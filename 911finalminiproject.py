#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pygame
import copy
import random
import numpy as np
import time
import math
from collections import deque # For replay buffera
import json # For saving/loading player profiles
import os # For checking file existence

# --- Actual Imports for DQN (Requires local installation of PyTorch) ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --------------- Game Constants ---------------
GAME_WIDTH = 1000
GAME_HEIGHT = 700
FPS = 30 # Frames per second for game logic and drawing
NUM_AGENTS = 2

# Colors (RGB)
BACKGROUND = (0, 0, 0) # Black for the unexplored areas / deep background
PLAYER_COLORS = [(255, 0, 0), (0, 0, 255)] # Player 1 (Red), Player 2 (Blue)
ENEMY_COLOR = (100, 50, 0) # Goblin brown
FAKE_PLAYER_COLOR = (70, 0, 70) # Dark Wizard purple
GEM_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)] # Cyan, Magenta, Yellow, Red, Green, Blue for gems
SPIKE_COLOR = (150, 150, 150) # Grey for spikes
WALL_COLOR_TREE_TRUNK = (139, 69, 19) # Brown for tree trunks
WALL_COLOR_TREE_LEAVES = (34, 139, 34) # Forest green for tree leaves
PATH_COLOR_DIRT_LIGHT = (139, 69, 19) # Lighter brown for path
PATH_COLOR_DIRT_DARK = (101, 67, 33) # Darker brown for path texture
SECRET_PASSAGE_COLOR = (50, 200, 50) # Greenish glow for secret passages

# Power-up aura colors (also used for pie chart visualization)
STEALTH_COLOR = (100, 100, 100) # Grey for stealth
GHOST_COLOR = (150, 0, 150) # Purple for ghost mode
INVINCIBLE_COLOR = (255, 255, 255) # White for invincibility
BOOST_COLOR = (0, 255, 255) # Cyan for speed boost
DISARM_COLOR = (255, 165, 0) # Orange for disarm

POWERUP_DURATION_SECONDS = 5
GAME_DURATION_SECONDS = 120 # Game duration (2 minutes)
TREASURES_TO_PLACE = 5 # Total number of non-main treasures
FAKE_TREASURE_PENALTY = 100 # Penalty for hitting a fake treasure
NUM_FAKE_TREASURES = 2 # Number of fake treasures to place

MAZE_CELL_SIZE = 50 # Size of each cell in the maze grid
BASE_AGENT_SPEED = 7 # Increased base movement speed
BOOSTED_AGENT_SPEED = 14 # Increased boosted movement speed
PLAYER_SIZE = 25 # Size of the player character
PLAYER_LIGHT_RADIUS = 150 # How far the player's light extends (for AI visibility)
ENEMY_DETECTION_RADIUS = 100 # How far enemies can detect players
FAKE_PLAYER_DETECTION_RADIUS = 120 # How far fake players can detect players

# Treasure values
TREASURE_TYPES = [
    {"name": "Small Gold Chest", "color": (255, 215, 0), "value": 100, "size": 30},
    {"name": "Silver Casket", "color": (192, 192, 192), "value": 200, "size": 35},
    {"name": "Ruby Gemstone", "color": (255, 0, 0), "value": 300, "size": 25},
    {"name": "Emerald Scepter", "color": (0, 255, 0), "value": 400, "size": 25}
]
MAIN_TREASURE = {"name": "Legendary Crown", "color": (255, 255, 0), "value": 1000, "size": 40, "is_main": True}

GEM_VALUE = 50
SPIKE_PENALTY = 75
ENEMY_PENALTY = 50
FAKE_PLAYER_PENALTY = 30
FAKE_PLAYER_DETECT_REWARD = 20

# Power-up energy costs
STEALTH_ENERGY_COST = 10
GHOST_ENERGY_COST = 20
INVINCIBLE_ENERGY_COST = 25
BOOST_ENERGY_COST = 15
DISARM_ENERGY_COST = 20
ENERGY_REGEN_RATE = 2 # Energy per second

# DQN specific parameters
ACTION_SIZE = 9 # UP, DOWN, LEFT, RIGHT, STEALTH, GHOST, INVINCIBLE, BOOST, DISARM
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.998 # Slightly slower decay to allow more exploration
LEARNING_RATE = 0.002 # Slightly increased learning rate
TARGET_UPDATE_FREQ = 10 # Update target network every N episodes/steps
REPLAY_BUFFER_SIZE = 2000

# File for saving player profiles
PLAYER_PROFILES_FILE = "player_profiles.json"
SAVED_GAME_FILE = "saved_game.json"
PLAYER_PROFILES_EXPORT_FILE = "player_profiles_export.json" # New file for exporting


class Particle:
    def __init__(self, x, y, color, radius, velocity, lifetime):
        self.x = x
        self.y = y
        self.base_rgb_color = color # Store the base RGB color
        self.radius = radius
        self.velocity = velocity # (vx, vy)
        self.lifetime = lifetime # in frames
        self.age = 0

    def update(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        self.age += 1
        # Ensure radius doesn't go to zero too quickly or become negative
        self.radius = max(0.1, self.radius - (self.radius / self.lifetime)) # Shrink, ensure radius > 0.1

    def draw(self, surface):
        if self.age < self.lifetime:
            # Calculate alpha for drawing, fading out over lifetime
            alpha = max(0, 255 - int(255 * (self.age / self.lifetime)))
            current_color_with_alpha = (*self.base_rgb_color, alpha)

            # Ensure surface dimensions are at least 1x1 pixel to avoid errors
            surface_dim = max(1, int(self.radius * 2))
            s = pygame.Surface((surface_dim, surface_dim), pygame.SRCALPHA)

            # Draw circle on the temporary surface with RGBA color
            # Ensure center coordinates are integers relative to the small surface
            pygame.draw.circle(s, current_color_with_alpha, (surface_dim // 2, surface_dim // 2), int(self.radius))
            
            # Blit the temporary surface onto the main screen
            surface.blit(s, (int(self.x - self.radius), int(self.y - self.radius)))

class FloatingText:
    def __init__(self, text, x, y, color, lifetime, velocity_y):
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime # in frames
        self.velocity_y = velocity_y # how fast it floats up
        self.age = 0
        self.font = pygame.font.SysFont("Arial", 20, bold=True) # Use a slightly larger, bold font

    def update(self):
        self.y += self.velocity_y
        self.age += 1

    def draw(self, surface):
        if self.age < self.lifetime:
            alpha = max(0, 255 - int(255 * (self.age / self.lifetime)))
            current_color_with_alpha = (*self.color, alpha)
            
            # Render text with alpha
            text_surf = self.font.render(self.text, True, current_color_with_alpha)
            text_rect = text_surf.get_rect(center=(int(self.x), int(self.y)))
            surface.blit(text_surf, text_rect)

# InputBox class for name entry
class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = (255, 255, 255) # White
        self.text = text
        self.font = pygame.font.SysFont("Arial", 20)
        self.active = False
        self.txt_surface = self.font.render(text, True, self.color)
        self.original_w = w # Store original width for resizing

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
            self.color = (255, 255, 0) if self.active else (255, 255, 255)
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False
                    self.color = (255, 255, 255)
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)
                # Resize the box if the text is too long.
                self.rect.w = max(self.original_w, self.txt_surface.get_width() + 10)

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        pygame.draw.rect(screen, self.color, self.rect, 2)

    def get_text(self):
        return self.text

# --- Drawing Functions with Enhanced Realism ---

def draw_treasure(surface, x, y, w, h, color, is_main=False, is_fake=False):
    # Base chest shape with a subtle gradient
    for i in range(h):
        grad_color = (int(color[0] * (1 - i/h * 0.3)), int(color[1] * (1 - i/h * 0.3)), int(color[2] * (1 - i/h * 0.3)))
        pygame.draw.line(surface, grad_color, (x, y + i), (x + w, y + i))

    # Lid, slightly darker with a highlight
    lid_h = h // 3
    pygame.draw.rect(surface, (color[0] * 0.8, color[1] * 0.8, color[2] * 0.8), (x, y, w, lid_h))
    pygame.draw.line(surface, (255, 255, 255, 100), (x, y), (x + w, y), 2) # Top highlight

    if is_main:
        # Draw a crown on top for the main treasure (more detailed)
        crown_points = [
            (x + w//2, y - 10), # Peak
            (x + w//2 - 15, y + 5),
            (x + w//2 - 20, y + 0),
            (x + w//2 - 5, y + 25),
            (x + w//2 + 5, y + 25),
            (x + w//2 + 20, y + 0),
            (x + w//2 + 15, y + 5)
        ]
        pygame.draw.polygon(surface, (255, 223, 0), crown_points) # Gold crown
        # Gems on crown
        pygame.draw.circle(surface, (255, 0, 0), (x + w//2 - 15, y + 3), 4)
        pygame.draw.circle(surface, (0, 0, 255), (x + w//2 + 15, y + 3), 4)
        pygame.draw.circle(surface, (0, 255, 0), (x + w//2, y - 5), 4)
    elif is_fake:
        # Draw a skull or crossbones for fake treasures
        skull_color = (50, 50, 50)
        pygame.draw.circle(surface, skull_color, (int(x + w//2), int(y + h//2)), 10) # Skull head
        pygame.draw.rect(surface, skull_color, (int(x + w//2 - 8), int(y + h//2 + 5), 16, 5)) # Jaw
        pygame.draw.line(surface, skull_color, (int(x + w//2 - 15), int(y + h//2 + 10)), (int(x + w//2 + 15), int(y + h//2 - 5)), 3) # Crossbones
        pygame.draw.line(surface, skull_color, (int(x + w//2 - 15), int(y + h//2 - 5)), (int(x + w//2 + 15), int(y + h//2 + 10)), 3)
    else:
        # Simple lock for regular treasures with shadow/highlight
        lock_x = x + w // 2 - 4
        lock_y = y + h // 3
        lock_w = 8
        lock_h = h // 3
        pygame.draw.rect(surface, (139, 69, 19), (lock_x, lock_y, lock_w, lock_h)) # Lock body
        pygame.draw.rect(surface, (100, 50, 0), (lock_x + 1, lock_y + 1, lock_w - 2, lock_h - 2)) # Inner shadow
        pygame.draw.circle(surface, (200, 150, 100), (lock_x + lock_w // 2, lock_y), lock_w // 2) # Shackle top

    # Gold coins spilling out (if applicable)
    if is_main or (color == (255, 215, 0) and not is_fake): # If main treasure or gold chest and not fake
        for _ in range(5):
            coin_x = x + random.randint(0, w)
            coin_y = y - random.randint(5, 15)
            pygame.draw.circle(surface, (255, 223, 0), (coin_x, coin_y), 4) # Gold coin
            pygame.draw.circle(surface, (200, 170, 0), (coin_x, coin_y), 2) # Coin center

def draw_gem(surface, x, y, size, color):
    # Base shape (diamond-like) with more facets
    points = [
        (x + size//2, y), # Top
        (x + size, y + size//4),
        (x + size, y + size*3//4),
        (x + size//2, y + size), # Bottom
        (x, y + size*3//4),
        (x, y + size//4)
    ]
    pygame.draw.polygon(surface, color, points)

    # Inner sparkle and reflections
    center_x, center_y = x + size // 2, y + size // 2
    pygame.draw.line(surface, (255, 255, 255), (center_x - size//3, center_y), (center_x + size//3, center_y), 2)
    pygame.draw.line(surface, (255, 255, 255), (center_x, center_y - size//3), (center_x, center_y + size//3), 2)
    pygame.draw.circle(surface, (255, 255, 255), (center_x, center_y), size // 8) # Central gleam

    # Small highlights
    pygame.draw.circle(surface, (255, 255, 255), (x + size * 0.7, y + size * 0.3), 2)
    pygame.draw.circle(surface, (255, 255, 255), (x + size * 0.3, y + size * 0.7), 2)


def draw_ninja(surface, x, y, color, is_stealth=False, is_ghost=False, is_invincible=False, is_boosted=False, is_disarming=False, frame=0):
    # Aura effects (drawn first so ninja is on top)
    aura_radius = PLAYER_SIZE + 10
    aura_surface = pygame.Surface((aura_radius * 2, aura_radius * 2), pygame.SRCALPHA)
    aura_center = (aura_radius, aura_radius)

    # Use a fixed alpha for auras for consistency, or calculate based on power-up duration
    aura_alpha = 100 
    if is_stealth:
        pygame.draw.circle(aura_surface, (*STEALTH_COLOR, aura_alpha), aura_center, aura_radius, 0)
    if is_ghost:
        pygame.draw.circle(aura_surface, (*GHOST_COLOR, aura_alpha), aura_center, aura_radius, 0)
    if is_invincible:
        pygame.draw.circle(aura_surface, (*INVINCIBLE_COLOR, aura_alpha), aura_center, aura_radius, 0)
    if is_boosted:
        pygame.draw.circle(aura_surface, (*BOOST_COLOR, aura_alpha), aura_center, aura_radius, 0)
    if is_disarming:
        pygame.draw.circle(aura_surface, (*DISARM_COLOR, aura_alpha), aura_center, aura_radius, 0)
    
    surface.blit(aura_surface, (int(x + PLAYER_SIZE//2 - aura_radius), int(y + PLAYER_SIZE//2 - aura_radius)))

    # Ninja body parts (more detailed and animated)
    head_radius = PLAYER_SIZE // 3
    body_width = PLAYER_SIZE // 2
    body_height = PLAYER_SIZE // 2
    limb_thickness = PLAYER_SIZE // 8

    # Head
    pygame.draw.circle(surface, (255, 224, 189), (int(x + PLAYER_SIZE//2), int(y + PLAYER_SIZE//4)), head_radius)
    # Mask
    pygame.draw.rect(surface, (0, 0, 0), (int(x + PLAYER_SIZE//4), int(y + PLAYER_SIZE//4 + 2), PLAYER_SIZE//2, PLAYER_SIZE//5))
    
    # Body
    pygame.draw.rect(surface, color, (int(x + PLAYER_SIZE//4), int(y + PLAYER_SIZE//2), body_width, body_height))

    # Arms (simple animation)
    arm_offset_y = 0
    arm_offset_x = 0
    if frame == 0: # Default/standing
        arm_offset_y = 0
    elif frame == 1: # Walking frame 1
        arm_offset_y = -2
        arm_offset_x = 2
    elif frame == 2: # Walking frame 2
        arm_offset_y = 2
        arm_offset_x = -2

    # Left arm
    pygame.draw.rect(surface, color, (int(x + PLAYER_SIZE//8 + arm_offset_x), int(y + PLAYER_SIZE//2 + 2 + arm_offset_y), PLAYER_SIZE//4, limb_thickness))
    # Right arm
    pygame.draw.rect(surface, color, (int(x + PLAYER_SIZE * 5//8 - arm_offset_x), int(y + PLAYER_SIZE//2 + 2 - arm_offset_y), PLAYER_SIZE//4, limb_thickness))

    # Legs (simple animation)
    leg_offset_y = 0
    leg_offset_x = 0
    if frame == 0:
        leg_offset_y = 0
    elif frame == 1:
        leg_offset_y = 2
        leg_offset_x = 1
    elif frame == 2:
        leg_offset_y = -2
        leg_offset_x = -1

    # Left leg
    pygame.draw.rect(surface, color, (int(x + PLAYER_SIZE//4 - 2 + leg_offset_x), int(y + PLAYER_SIZE * 3//4 + leg_offset_y), PLAYER_SIZE//4, PLAYER_SIZE//4))
    # Right leg
    pygame.draw.rect(surface, color, (int(x + PLAYER_SIZE * 3//4 - 2 - leg_offset_x), int(y + PLAYER_SIZE * 3//4 - leg_offset_y), PLAYER_SIZE//4, PLAYER_SIZE//4))


def draw_goblin(surface, x, y):
    # Main body (more defined shape)
    pygame.draw.ellipse(surface, ENEMY_COLOR, (int(x), int(y + 10), 30, 20))
    # Head
    pygame.draw.circle(surface, ENEMY_COLOR, (int(x + 15), int(y + 5)), 10)
    # Ears (pointier)
    pygame.draw.polygon(surface, ENEMY_COLOR, [(int(x + 5), int(y + 5)), (int(x), int(y - 2)), (int(x + 10), int(y))])
    pygame.draw.polygon(surface, ENEMY_COLOR, [(int(x + 25), int(y + 5)), (int(x + 30), int(y - 2)), (int(x + 20), int(y))])
    # Eyes (glowing yellow)
    pygame.draw.circle(surface, (255, 255, 0), (int(x + 10), int(y + 5)), 3)
    pygame.draw.circle(surface, (255, 255, 0), (int(x + 20), int(y + 5)), 3)
    pygame.draw.circle(surface, (0, 0, 0), (int(x + 10), int(y + 5)), 1) # Pupils
    pygame.draw.circle(surface, (0, 0, 0), (int(x + 20), int(y + 5)), 1)
    # Mouth (jagged teeth)
    pygame.draw.line(surface, (0, 0, 0), (int(x + 10), int(y + 10)), (int(x + 15), int(y + 12)), 1)
    pygame.draw.line(surface, (0, 0, 0), (int(x + 15), int(y + 12)), (int(x + 20), int(y + 10)), 1)


def draw_dark_wizard(surface, x, y):
    # Robe (more flowing)
    pygame.draw.polygon(surface, FAKE_PLAYER_COLOR, [
        (int(x + 15), int(y + 10)), # Top center
        (int(x), int(y + 15)), # Left shoulder
        (int(x + 5), int(y + 35)), # Left bottom
        (int(x + 25), int(y + 35)), # Right bottom
        (int(x + 30), int(y + 15)) # Right shoulder
    ])
    # Hat (taller, more defined)
    pygame.draw.polygon(surface, (50, 0, 50), [
        (int(x + 15), int(y)), # Peak
        (int(x - 5), int(y + 15)), # Left brim
        (int(x + 35), int(y + 15)) # Right brim
    ])
    # Eyes (glowing red)
    pygame.draw.circle(surface, (255, 0, 0), (int(x + 10), int(y + 18)), 3)
    pygame.draw.circle(surface, (255, 0, 0), (int(x + 20), int(y + 18)), 3)
    # Staff (more detailed)
    pygame.draw.line(surface, (150, 100, 50), (int(x + 25), int(y + 20)), (int(x + 35), int(y + 40)), 3)
    pygame.draw.circle(surface, (255, 255, 0), (int(x + 35), int(y + 40)), 5) # Larger, glowing orb
    pygame.draw.circle(surface, (255, 165, 0), (int(x + 35), int(y + 40)), 3) # Inner glow

def draw_spike_trap(surface, x, y, size):
    # Base (more textured)
    for i in range(size):
        color_val = 100 + int(50 * (i / size))
        pygame.draw.line(surface, (color_val, color_val, color_val), (int(x), int(y + i)), (int(x + size), int(y + i)))
    pygame.draw.rect(surface, (80, 80, 80), (int(x), int(y), size, size), 2) # Border

    # Spikes (sharper, with highlights)
    spike_base_y = int(y + size * 0.7) # Spikes emerge from lower part
    for i in range(4): # More spikes
        spike_x_base = int(x + size//8 + i * size//5)
        spike_tip_y = int(y + size * 0.2)
        pygame.draw.polygon(surface, SPIKE_COLOR, [
            (spike_x_base, spike_base_y),
            (spike_x_base + size//10, spike_tip_y),
            (spike_x_base + size//5, spike_base_y)
        ])
        # Spike highlight
        pygame.draw.line(surface, (200, 200, 200), (spike_x_base, spike_base_y), (spike_x_base + size//10, spike_tip_y), 1)

def draw_maze(surface, maze_grid, secret_passages):
    rows = len(maze_grid)
    cols = len(maze_grid[0])
    for r in range(rows):
        for c in range(cols):
            cell_x = c * MAZE_CELL_SIZE
            cell_y = r * MAZE_CELL_SIZE
            if maze_grid[r][c] == 1:  # It's a wall (tree)
                # Tree Trunk (textured)
                pygame.draw.rect(surface, WALL_COLOR_TREE_TRUNK, (cell_x + MAZE_CELL_SIZE//3, cell_y, MAZE_CELL_SIZE//3, MAZE_CELL_SIZE))
                pygame.draw.line(surface, (100, 50, 0), (cell_x + MAZE_CELL_SIZE//3 + 2, cell_y), (cell_x + MAZE_CELL_SIZE//3 + 2, cell_y + MAZE_CELL_SIZE), 1)
                pygame.draw.line(surface, (100, 50, 0), (cell_x + MAZE_CELL_SIZE*2//3 - 2, cell_y), (cell_x + MAZE_CELL_SIZE*2//3 - 2, cell_y + MAZE_CELL_SIZE), 1)

                # Tree Leaves (more organic shape)
                leaf_surface = pygame.Surface((MAZE_CELL_SIZE * 2, MAZE_CELL_SIZE * 2), pygame.SRCALPHA)
                pygame.draw.ellipse(leaf_surface, (*WALL_COLOR_TREE_LEAVES, 200), (0, 0, MAZE_CELL_SIZE * 2, MAZE_CELL_SIZE * 2))
                pygame.draw.circle(leaf_surface, (*(WALL_COLOR_TREE_LEAVES[0] - 10, WALL_COLOR_TREE_LEAVES[1] - 10, WALL_COLOR_TREE_LEAVES[2] - 10), 200), (MAZE_CELL_SIZE*0.7, MAZE_CELL_SIZE*0.7), MAZE_CELL_SIZE*0.4)
                pygame.draw.circle(leaf_surface, (*(WALL_COLOR_TREE_LEAVES[0] + 10, WALL_COLOR_TREE_LEAVES[1] + 10, WALL_COLOR_TREE_LEAVES[2] + 10), 200), (MAZE_CELL_SIZE*1.3, MAZE_CELL_SIZE*1.3), MAZE_CELL_SIZE*0.3)
                surface.blit(leaf_surface, (cell_x - MAZE_CELL_SIZE//2, cell_y - MAZE_CELL_SIZE//2))

                # Draw subtle glow/crack for secret passages
                if (r, c) in secret_passages:
                    glow_surface = pygame.Surface((MAZE_CELL_SIZE, MAZE_CELL_SIZE), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surface, (*SECRET_PASSAGE_COLOR, 80), (0, 0, MAZE_CELL_SIZE, MAZE_CELL_SIZE), border_radius=5)
                    surface.blit(glow_surface, (cell_x, cell_y))


            else:  # It's a path (dirt)
                # Dirt texture (more granular)
                pygame.draw.rect(surface, PATH_COLOR_DIRT_LIGHT, (cell_x, cell_y, MAZE_CELL_SIZE, MAZE_CELL_SIZE))
                for _ in range(5): # Small dark spots
                    dot_x = random.randint(cell_x, cell_x + MAZE_CELL_SIZE - 1)
                    dot_y = random.randint(cell_y, cell_y + MAZE_CELL_SIZE - 1)
                    pygame.draw.circle(surface, PATH_COLOR_DIRT_DARK, (dot_x, dot_y), 1)


class GameRectangle:
    def __init__(self, x, y, width, height):
        self.x = float(x) # Store as float for sub-pixel movement
        self.y = float(y) # Store as float
        self.width = width
        self.height = height

    def colliderect(self, other):
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height).colliderect(
            pygame.Rect(int(other.x), int(other.y), other.width, other.height))

class AgentState:
    def __init__(self, idx, start_x, start_y, name="Player"):
        self.idx = idx
        self.name = name # Player name
        self.PlayerEntity = GameRectangle(start_x, start_y, PLAYER_SIZE, PLAYER_SIZE)
        self.in_stealth = False
        self.in_ghost = False
        self.invincible = False
        self.boosted = False
        self.disarming = False
        self.stealth_end_time = 0
        self.ghost_end_time = 0
        self.invincible_end_time = 0
        self.boost_end_time = 0
        self.disarm_end_time = 0
        self.total_moves = 0
        self.reward = 0
        self.detected_fakes = 0
        self.commentary = ""
        self.treasures_collected = 0
        self.gems_collected = 0
        self.has_main_treasure = False # New flag for main treasure
        # New attributes to track power-up activations for the graph
        self.stealth_activations = 0
        self.ghost_activations = 0
        self.invincible_activations = 0
        self.boost_activations = 0
        self.disarm_activations = 0
        self.powerups_used = 0 # Total power-ups used for analytics (sum of individual power-ups)

        # Animation variables
        self.current_animation_frame = 0
        self.animation_timer = 0
        self.animation_speed = 0.1 # Frames per second for animation
        self.is_moving = False # To control animation only when moving
        self.last_closest_treasure_dist = float('inf') # For reward shaping
        self.last_pos = (start_x, start_y) # For detecting if agent is stuck
        self.stuck_frames = 0 # Counter for how many frames agent is stuck

        # Resource Meter for Power-Ups
        self.energy = 100.0 # Current energy
        self.max_energy = 100.0 # Maximum energy

        # Agent Footprint Trails
        self.footprints = deque(maxlen=20) # Stores (x, y) positions for trail

class GameState:
    def __init__(self, game_instance): # Accept game_instance
        print("Initializing GameState...")
        self.game = game_instance # Store reference to AbstractGame
        self.maze_rows = GAME_HEIGHT // MAZE_CELL_SIZE
        self.maze_cols = GAME_WIDTH // MAZE_CELL_SIZE
        self.maze, self.secret_passages = self.generate_maze(self.maze_rows, self.maze_cols)
        print(f"Maze generated: {self.maze_rows}x{self.maze_cols}")
        
        self.GoalLocations = []
        # Place main treasure first
        main_treasure_rect = self.place_entity_randomly(MAIN_TREASURE["size"], MAIN_TREASURE["size"])
        self.GoalLocations.append({
            'rect': main_treasure_rect,
            'color': MAIN_TREASURE["color"],
            'value': MAIN_TREASURE["value"],
            'is_main': True,
            'name': MAIN_TREASURE["name"],
            'is_fake': False # Main treasure is never fake
        })
        print(f"Main treasure placed at ({main_treasure_rect.x}, {main_treasure_rect.y})")

        # Place other treasures, some of which are fake
        # Determine which non-main treasures will be fake
        fake_treasure_indices = random.sample(range(TREASURES_TO_PLACE), min(NUM_FAKE_TREASURES, TREASURES_TO_PLACE))

        for i in range(TREASURES_TO_PLACE):
            treasure_type = random.choice(TREASURE_TYPES)
            treasure_rect = self.place_entity_randomly(treasure_type["size"], treasure_type["size"])
            is_fake = (i in fake_treasure_indices)
            self.GoalLocations.append({
                'rect': treasure_rect,
                'color': treasure_type["color"],
                'value': treasure_type["value"],
                'is_main': False,
                'name': treasure_type["name"],
                'is_fake': is_fake
            })

        self.EnemyCollection = [GameRectangle(*self.get_random_open_cell_coords(30, 30), 30, 30) for _ in range(4)]
        self.FakePlayers = [GameRectangle(*self.get_random_open_cell_coords(30, 30), 30, 30) for _ in range(4)]
        self.GemCollection = []
        for _ in range(7):
            gem_rect = GameRectangle(*self.get_random_open_cell_coords(20, 20), 20, 20)
            gem_color = random.choice(GEM_COLORS)
            self.GemCollection.append({'rect': gem_rect, 'color': gem_color})

        self.SpikeTrapCollection = [GameRectangle(*self.get_random_open_cell_coords(MAZE_CELL_SIZE, MAZE_CELL_SIZE), MAZE_CELL_SIZE, MAZE_CELL_SIZE) for _ in range(5)]
        
        # Determine a single starting spot for both players
        start_x, start_y = self.get_random_open_cell_coords(PLAYER_SIZE, PLAYER_SIZE)
        self.Agents = [AgentState(i, start_x, start_y) for i in range(NUM_AGENTS)]
        
        self.game_start_time = time.time()
        self.game_over = False
        self.winner_idx = -1 # -1 for tie, 0 for player 1, 1 for player 2
        self.win_reason = "" # New field for win reason

        self.particles = [] # List to hold active particles
        self.floating_texts = [] # List to hold active floating texts

        # Dynamic Maze Regeneration
        self.maze_reshaped = False
        self.next_reshape_time = self.game_start_time + GAME_DURATION_SECONDS / 2 # Reshape halfway through

        print(f"Total treasures placed: {len(self.GoalLocations)}")
        print(f"Total gems placed: {len(self.GemCollection)}")
        print(f"Player start position: ({self.Agents[0].PlayerEntity.x}, {self.Agents[0].PlayerEntity.y})")
        print("GameState initialization complete.")

    def generate_maze(self, rows, cols):
        # Initialize all cells as walls (1)
        maze = np.ones((rows, cols), dtype=int)
        secret_passages = set() # To store coordinates of secret passages

        def is_valid_carving_pos(r, c):
            # A valid position for carving (a path cell) must be within the inner maze,
            # excluding the border rows/columns.
            return 1 <= r < rows - 1 and 1 <= c < cols - 1

        def carve_path(r, c):
            # Mark the current cell as a path
            maze[r][c] = 0

            # Define directions to move 2 steps (to jump over a wall cell)
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)

            for dr_step, dc_step in directions:
                nr, nc = r + dr_step, c + dc_step # Candidate next cell
                wall_r, wall_c = r + dr_step // 2, c + dc_step // 2 # Wall cell in between

                # Check if the candidate next cell is valid (within inner bounds)
                # and if it has not yet been visited (is still a wall, 1)
                if is_valid_carving_pos(nr, nc) and maze[nr][nc] == 1:
                    maze[wall_r][wall_c] = 0 # Carve the wall between current and next
                    carve_path(nr, nc) # Recursively call for the next cell

        # Find a random starting point for carving within the inner maze.
        # This point must have both row and column indices that are odd,
        # to ensure the recursive backtracking algorithm can reach all cells
        # in the inner grid for typical perfect maze generation.
        # The inner grid spans from (1,1) to (rows-2, cols-2).
        
        # Collect valid odd row/column indices within the inner bounds
        valid_odd_rows = [i for i in range(1, rows - 1) if i % 2 != 0]
        valid_odd_cols = [i for i in range(1, cols - 1) if i % 2 != 0]

        # Ensure there are valid starting points; otherwise, default to (1,1)
        start_r_carve = random.choice(valid_odd_rows) if valid_odd_rows else 1
        start_c_carve = random.choice(valid_odd_cols) if valid_odd_cols else 1
        
        carve_path(start_r_carve, start_c_carve)

        # Add some secret passages (break a few walls)
        num_secret_passages = 5 # Number of secret passages to add
        for _ in range(num_secret_passages):
            while True:
                r = random.randint(1, rows - 2)
                c = random.randint(1, cols - 2)
                if maze[r][c] == 1: # Only break walls
                    # Ensure it connects two path cells
                    # Check horizontal neighbors
                    if c > 0 and c < cols - 1 and maze[r][c-1] == 0 and maze[r][c+1] == 0:
                        maze[r][c] = 0
                        secret_passages.add((r, c))
                        break
                    # Check vertical neighbors
                    if r > 0 and r < rows - 1 and maze[r-1][c] == 0 and maze[r+1][c] == 0:
                        maze[r][c] = 0
                        secret_passages.add((r, c))
                        break
        return maze, secret_passages

    def get_random_open_cell_coords(self, entity_width, entity_height):
        # Ensure placement is within the inner maze boundaries, away from fixed border walls
        min_r = 1
        max_r = self.maze_rows - 2
        min_c = 1
        max_c = self.maze_cols - 2
        
        while True:
            r = random.randint(min_r, max_r) # Pick a row not on the outer border
            c = random.randint(min_c, max_c) # Pick a column not on the outer border
            if self.maze[r][c] == 0: # Checks if the cell is an open path (0)
                return (c * MAZE_CELL_SIZE + (MAZE_CELL_SIZE - entity_width) // 2,
                        r * MAZE_CELL_SIZE + (MAZE_CELL_SIZE - entity_height) // 2)

    def place_entity_randomly(self, width, height):
        x, y = self.get_random_open_cell_coords(width, height)
        return GameRectangle(x, y, width, height)

    def reposition_entities_after_maze_reshape(self):
        print("Repositioning entities after maze reshape...")
        # Reposition players
        for agent in self.Agents:
            agent.PlayerEntity.x, agent.PlayerEntity.y = self.get_random_open_cell_coords(PLAYER_SIZE, PLAYER_SIZE)
            agent.footprints.clear() # Clear old footprints

        # Reposition treasures
        for treasure_data in self.GoalLocations:
            treasure_data['rect'] = self.place_entity_randomly(treasure_data['rect'].width, treasure_data['rect'].height)
        
        # Reposition enemies
        for i in range(len(self.EnemyCollection)):
            self.EnemyCollection[i] = self.place_entity_randomly(30, 30)
        
        # Reposition fake players
        for i in range(len(self.FakePlayers)):
            self.FakePlayers[i] = self.place_entity_randomly(30, 30)

        # Reposition gems
        for gem_data in self.GemCollection:
            gem_data['rect'] = self.place_entity_randomly(gem_data['rect'].width, gem_data['rect'].height)

        # Reposition spike traps
        for i in range(len(self.SpikeTrapCollection)):
            self.SpikeTrapCollection[i] = self.place_entity_randomly(MAZE_CELL_SIZE, MAZE_CELL_SIZE)
        
        # Add a floating text notification
        self.game.floating_texts.append(FloatingText("MAZE RESHAPED!", GAME_WIDTH // 2, GAME_HEIGHT // 2, (255, 255, 0), 60, -1)) # Yellow, 2 seconds, float up slightly


    def is_valid_move(self, rect, dx, dy, agent_in_ghost_mode=False):
        new_x = rect.x + dx
        new_y = rect.y + dy

        temp_rect = pygame.Rect(int(new_x), int(new_y), rect.width, rect.height)

        for r in range(self.maze_rows):
            for c in range(self.maze_cols):
                if self.maze[r][c] == 1: # It's a wall
                    wall_rect = pygame.Rect(c * MAZE_CELL_SIZE, r * MAZE_CELL_SIZE, MAZE_CELL_SIZE, MAZE_CELL_SIZE)
                    if temp_rect.colliderect(wall_rect):
                        # If in ghost mode, check if it's a secret passage
                        if agent_in_ghost_mode and (r, c) in self.secret_passages:
                            return True # Can pass through secret passages in ghost mode
                        return False # Cannot pass through regular walls
        return True

    def update_agent(self, agent: AgentState, action):
        current_time = time.time()
        
        # Update power-up states
        if agent.in_stealth and current_time > agent.stealth_end_time:
            agent.in_stealth = False
            agent.commentary += " - Stealth ended!"
        if agent.in_ghost and current_time > agent.ghost_end_time:
            agent.in_ghost = False
            agent.commentary += " - Ghost mode ended!"
        if agent.invincible and current_time > agent.invincible_end_time:
            agent.invincible = False
            agent.commentary += " - Invincibility ended!"
        if agent.boosted and current_time > agent.boost_end_time:
            agent.boosted = False
            agent.commentary += " - Boost ended!"
        if agent.disarming and current_time > agent.disarm_end_time:
            agent.disarming = False
            agent.commentary += " - Disarm ended!"

        # Energy regeneration
        agent.energy = min(agent.max_energy, agent.energy + ENERGY_REGEN_RATE / FPS)

        speed = BOOSTED_AGENT_SPEED if agent.boosted else BASE_AGENT_SPEED
        dx, dy = 0, 0
        agent.is_moving = False # Reset moving state for animation

        # Initial reward for taking an action
        current_step_reward = -1 # Small penalty for each step to encourage efficiency

        # Handle power-up activations
        # Add a small reward for activating power-ups
        if action == "STEALTH" and not agent.in_stealth:
            if agent.energy >= STEALTH_ENERGY_COST:
                agent.in_stealth = True
                agent.stealth_end_time = current_time + POWERUP_DURATION_SECONDS
                agent.stealth_activations += 1 # Increment counter
                agent.powerups_used += 1 # Increment total power-ups used
                agent.energy -= STEALTH_ENERGY_COST
                agent.commentary = " - Stealth activated!"
                self.spawn_particles(agent.PlayerEntity.x + PLAYER_SIZE//2, agent.PlayerEntity.y + PLAYER_SIZE//2, STEALTH_COLOR, 10, 20)
                self.game.play_sound('powerup') # Play power-up sound
                current_step_reward += 10 # Reward for using power-up
            else:
                agent.commentary = " - Not enough energy for Stealth!"
                current_step_reward -= 5 # Penalty for trying to use power-up without energy
        elif action == "GHOST" and not agent.in_ghost:
            if agent.energy >= GHOST_ENERGY_COST:
                agent.in_ghost = True
                agent.ghost_end_time = current_time + POWERUP_DURATION_SECONDS
                agent.ghost_activations += 1 # Increment counter
                agent.powerups_used += 1 # Increment total power-ups used
                agent.energy -= GHOST_ENERGY_COST
                agent.commentary = " - Ghost mode activated!"
                self.spawn_particles(agent.PlayerEntity.x + PLAYER_SIZE//2, agent.PlayerEntity.y + PLAYER_SIZE//2, GHOST_COLOR, 10, 20)
                self.game.play_sound('powerup') # Play power-up sound
                current_step_reward += 10 # Reward for using power-up
            else:
                agent.commentary = " - Not enough energy for Ghost!"
                current_step_reward -= 5
        elif action == "INVINCIBLE" and not agent.invincible:
            if agent.energy >= INVINCIBLE_ENERGY_COST:
                agent.invincible = True
                agent.invincible_end_time = current_time + POWERUP_DURATION_SECONDS
                agent.invincible_activations += 1 # Increment counter
                agent.powerups_used += 1 # Increment total power-ups used
                agent.energy -= INVINCIBLE_ENERGY_COST
                agent.commentary = " - Invincibility activated!"
                self.spawn_particles(agent.PlayerEntity.x + PLAYER_SIZE//2, agent.PlayerEntity.y + PLAYER_SIZE//2, INVINCIBLE_COLOR, 10, 20)
                self.game.play_sound('powerup') # Play power-up sound
                current_step_reward += 10 # Reward for using power-up
            else:
                agent.commentary = " - Not enough energy for Invincibility!"
                current_step_reward -= 5
        elif action == "BOOST" and not agent.boosted:
            if agent.energy >= BOOST_ENERGY_COST:
                agent.boosted = True
                agent.boost_end_time = current_time + POWERUP_DURATION_SECONDS
                agent.boost_activations += 1 # Increment counter
                agent.powerups_used += 1 # Increment total power-ups used
                agent.energy -= BOOST_ENERGY_COST
                agent.commentary = " - Boost activated!"
                self.spawn_particles(agent.PlayerEntity.x + PLAYER_SIZE//2, agent.PlayerEntity.y + PLAYER_SIZE//2, BOOST_COLOR, 10, 20)
                self.game.play_sound('powerup') # Play power-up sound
                current_step_reward += 10 # Reward for using power-up
            else:
                agent.commentary = " - Not enough energy for Boost!"
                current_step_reward -= 5
        elif action == "DISARM" and not agent.disarming:
            if agent.energy >= DISARM_ENERGY_COST:
                agent.disarming = True
                agent.disarm_end_time = current_time + POWERUP_DURATION_SECONDS
                agent.disarm_activations += 1 # Increment counter
                agent.powerups_used += 1 # Increment total power-ups used
                agent.energy -= DISARM_ENERGY_COST
                agent.commentary = " - Disarm activated!"
                self.spawn_particles(agent.PlayerEntity.x + PLAYER_SIZE//2, agent.PlayerEntity.y + PLAYER_SIZE//2, DISARM_COLOR, 10, 20)
                self.game.play_sound('powerup') # Play power-up sound
                current_step_reward += 10 # Reward for using power-up
            else:
                agent.commentary = " - Not enough energy for Disarm!"
                current_step_reward -= 5

        # Store current position for stuck detection
        current_agent_x, current_agent_y = agent.PlayerEntity.x, agent.PlayerEntity.y

        # Handle movement
        if action == "UP":
            dy = -speed
        elif action == "DOWN":
            dy = speed
        elif action == "LEFT":
            dx = -speed
        elif action == "RIGHT":
            dx = speed

        # Apply movement if valid
        if dx != 0 or dy != 0: # Only apply if a movement action was chosen
            if self.is_valid_move(agent.PlayerEntity, dx, dy, agent.in_ghost):
                agent.PlayerEntity.x += dx
                agent.PlayerEntity.y += dy
                agent.commentary = f"Agent {agent.idx + 1} moved {action.lower()}"
                agent.is_moving = True # Set moving state for animation
                self.game.play_sound('walk_sound') # Play walk sound
                # Add footprint
                agent.footprints.append((agent.PlayerEntity.x + PLAYER_SIZE//2, agent.PlayerEntity.y + PLAYER_SIZE//2))
                # Reset stuck counter if moved successfully
                agent.stuck_frames = 0
            else:
                agent.commentary = f"Agent {agent.idx + 1} hit a tree!"
                current_step_reward -= 15 # Increased penalty for hitting a wall even more
                agent.stuck_frames += 1 # Increment stuck counter
        else: # If no movement action was chosen, or if it was a power-up action
            agent.stuck_frames += 1

        # Force a random move if stuck for too long
        if agent.stuck_frames > FPS * 1: # If stuck for more than 1 second (30 frames)
            current_step_reward -= 100 # Very large penalty for being stuck
            agent.commentary += " - STUCK! FORCING RANDOM MOVE!"
            
            # Try a random movement direction to break free
            random_move_action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            random_dx, random_dy = 0, 0
            if random_move_action == "UP":
                random_dy = -speed
            elif random_move_action == "DOWN":
                random_dy = speed
            elif random_move_action == "LEFT":
                random_dx = -speed
            elif random_move_action == "RIGHT":
                random_dx = speed
            
            if self.is_valid_move(agent.PlayerEntity, random_dx, random_dy, agent.in_ghost):
                agent.PlayerEntity.x += random_dx
                agent.PlayerEntity.y += random_dy
                agent.is_moving = True
                self.game.play_sound('walk_sound')
                agent.commentary += f" - Forced {random_move_action.lower()}!"
                agent.stuck_frames = 0 # Reset after attempting to break free
            else:
                # Even if forced random move failed, still reset stuck_frames to give it a chance
                # and prevent continuous massive penalties without a way out.
                agent.stuck_frames = 0 # Reset to allow it to try another random action next frame if still stuck

        # Update animation frame based on movement
        if agent.is_moving:
            agent.animation_timer += agent.animation_speed
            if agent.animation_timer >= 2: # Two animation frames (0, 1, 2, then loop back to 0)
                agent.animation_timer = 0
            agent.current_animation_frame = int(agent.animation_timer)
        else:
            agent.current_animation_frame = 0 # Default to standing frame

        # --- Reward Shaping for Treasure Pursuit ---
        current_closest_treasure_dist = float('inf')
        for treasure_data in self.GoalLocations:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, treasure_data['rect'], PLAYER_LIGHT_RADIUS)
            if is_visible and distance < current_closest_treasure_dist:
                min_treasure_dist = distance
                closest_visible_treasure = treasure_data['rect']
                current_closest_treasure_dist = distance
        
        # If a treasure was visible last step and is still visible this step
        if agent.last_closest_treasure_dist != float('inf') and current_closest_treasure_dist != float('inf'):
            # If moved closer to the treasure
            if current_closest_treasure_dist < agent.last_closest_treasure_dist:
                current_step_reward += 5.0 # Significantly increased positive reward for progress towards treasure
            # If moved further away from the treasure
            elif current_closest_treasure_dist > agent.last_closest_treasure_dist:
                current_step_reward -= 0.5 # Small penalty for moving away
        
        agent.last_closest_treasure_dist = current_closest_treasure_dist # Update for next step

        # --- Proximity Rewards/Penalties for other entities ---
        # Gems
        for gem_data in self.GemCollection:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, gem_data['rect'], PLAYER_LIGHT_RADIUS)
            if is_visible and distance < MAZE_CELL_SIZE: # If close to a visible gem
                current_step_reward += 0.2 # Small reward for being near a gem

        # Spike Traps
        for trap in self.SpikeTrapCollection:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, trap, PLAYER_LIGHT_RADIUS)
            if is_visible and distance < MAZE_CELL_SIZE and not agent.invincible: # If close to a visible spike and not invincible
                current_step_reward -= 1.0 # Increased penalty for being near a spike

        # Enemies
        for enemy in self.EnemyCollection:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, enemy, PLAYER_LIGHT_RADIUS)
            if is_visible and distance < MAZE_CELL_SIZE * 1.5 and not agent.invincible and not agent.in_ghost: # If near enemy and vulnerable
                current_step_reward -= 1.5 # Increased penalty for being near an enemy

        # Fake Players
        for fake in self.FakePlayers:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, fake, PLAYER_LIGHT_RADIUS)
            if is_visible and distance < MAZE_CELL_SIZE * 1.5 and not agent.invincible and not agent.in_ghost and not agent.in_stealth: # If near fake and vulnerable
                current_step_reward -= 1.2 # Increased penalty for being near a fake player

        # Check for treasure collection
        treasures_collected_in_frame = []
        for treasure_data in self.GoalLocations:
            if agent.PlayerEntity.colliderect(treasure_data['rect']):
                if treasure_data['is_fake']:
                    agent.reward -= FAKE_TREASURE_PENALTY
                    current_step_reward -= FAKE_TREASURE_PENALTY
                    agent.commentary += " - HIT FAKE TREASURE!"
                    self.spawn_particles(treasure_data['rect'].x + treasure_data['rect'].width//2, treasure_data['rect'].y + treasure_data['rect'].height//2, (255,0,0), 20, 40) # Red explosion particles
                    self.game.floating_texts.append(FloatingText(f"-{FAKE_TREASURE_PENALTY}", treasure_data['rect'].x + treasure_data['rect'].width//2, treasure_data['rect'].y + treasure_data['rect'].height//2, (255, 0, 0), 60, -0.5))
                    self.game.play_sound('spike_hit') # Use spike hit sound for explosion
                    treasures_collected_in_frame.append(treasure_data)
                elif treasure_data['is_main']:
                    agent.has_main_treasure = True
                    self.game_over = True # Game over condition
                    self.winner_idx = agent.idx
                    self.win_reason = f"Player {agent.idx + 1} collected the {treasure_data['name']}!"
                    agent.reward += treasure_data['value']
                    current_step_reward += treasure_data['value']
                    agent.commentary += f" - Collected {treasure_data['name']} and WON!"
                    self.spawn_particles(treasure_data['rect'].x + treasure_data['rect'].width//2, treasure_data['rect'].y + treasure_data['rect'].height//2, (255,223,0), 20, 40) # Gold particles
                    self.game.floating_texts.append(FloatingText(f"+{treasure_data['value']}", treasure_data['rect'].x + treasure_data['rect'].width//2, treasure_data['rect'].y + treasure_data['rect'].height//2, (0, 255, 0), 60, -0.5))
                    self.game.play_sound('treasure_main') # Play main treasure sound
                    treasures_collected_in_frame.append(treasure_data) # Mark for removal
                    break # Game ends immediately
                else:
                    agent.treasures_collected += 1
                    agent.reward += treasure_data['value']
                    current_step_reward += treasure_data['value']
                    agent.commentary += f" - Collected {treasure_data['name']}!"
                    self.spawn_particles(treasure_data['rect'].x + treasure_data['rect'].width//2, treasure_data['rect'].y + treasure_data['rect'].height//2, treasure_data['color'], 15, 30)
                    self.game.floating_texts.append(FloatingText(f"+{treasure_data['value']}", treasure_data['rect'].x + treasure_data['rect'].width//2, treasure_data['rect'].y + treasure_data['rect'].height//2, (0, 255, 0), 60, -0.5))
                    self.game.play_sound('treasure_collect') # Play regular treasure sound
                    treasures_collected_in_frame.append(treasure_data)
                
        for treasure_to_remove in treasures_collected_in_frame:
            if treasure_to_remove in self.GoalLocations: # Check if it hasn't been removed by main treasure win
                self.GoalLocations.remove(treasure_to_remove)
            
        # Check for gem collection
        gems_collected_in_frame = []
        for gem_data in self.GemCollection:
            if agent.PlayerEntity.colliderect(gem_data['rect']):
                agent.gems_collected += 1
                agent.reward += GEM_VALUE
                current_step_reward += GEM_VALUE
                agent.commentary += " - GEM COLLECTED!"
                self.spawn_particles(gem_data['rect'].x + gem_data['rect'].width//2, gem_data['rect'].y + gem_data['rect'].height//2, gem_data['color'], 10, 20)
                self.game.floating_texts.append(FloatingText(f"+{GEM_VALUE}", gem_data['rect'].x + gem_data['rect'].width//2, gem_data['rect'].y + gem_data['rect'].height//2, (0, 255, 0), 60, -0.5))
                self.game.play_sound('gem_collect') # Play gem sound
                gems_collected_in_frame.append(gem_data)
        
        for gem_to_remove in gems_collected_in_frame:
            self.GemCollection.remove(gem_to_remove)
            new_gem_rect = GameRectangle(*self.get_random_open_cell_coords(20, 20), 20, 20)
            new_gem_color = random.choice(GEM_COLORS)
            self.GemCollection.append({'rect': new_gem_rect, 'color': new_gem_color})

        # Check for spike trap collision
        for trap in self.SpikeTrapCollection:
            if agent.PlayerEntity.colliderect(trap):
                if not agent.invincible:
                    agent.reward -= SPIKE_PENALTY
                    current_step_reward -= SPIKE_PENALTY
                    agent.commentary += " - HIT SPIKE TRAP!"
                    self.spawn_particles(trap.x + MAZE_CELL_SIZE//2, trap.y + MAZE_CELL_SIZE//2, (200,0,0), 15, 25) # Red/blood particles
                    self.game.floating_texts.append(FloatingText(f"-{SPIKE_PENALTY}", trap.x + MAZE_CELL_SIZE//2, trap.y + MAZE_CELL_SIZE//2, (255, 0, 0), 60, -0.5))
                    self.game.play_sound('spike_hit') # Play spike hit sound
                    # Reset position after hit
                    agent.PlayerEntity.x, agent.PlayerEntity.y = self.get_random_open_cell_coords(PLAYER_SIZE, PLAYER_SIZE)


        # Check for enemy collisions
        enemies_to_remove = []
        for enemy in self.EnemyCollection:
            if agent.PlayerEntity.colliderect(enemy):
                if agent.disarming:
                    agent.commentary += " - GOBLIN DISARMED!"
                    enemies_to_remove.append(enemy)
                    agent.reward += 25
                    current_step_reward += 25
                    self.spawn_particles(enemy.x + 15, enemy.y + 15, (0,200,0), 10, 20) # Green particles for disarm
                    self.game.floating_texts.append(FloatingText(f"+25", enemy.x + 15, enemy.y + 15, (0, 255, 0), 60, -0.5))
                    self.game.play_sound('disarm_enemy') # Play disarm sound
                elif not agent.invincible and not agent.in_ghost:
                    agent.reward -= ENEMY_PENALTY
                    current_step_reward -= ENEMY_PENALTY
                    agent.commentary += " - HIT BY GOBLIN!"
                    self.spawn_particles(enemy.x + 15, enemy.y + 15, (255,100,0), 15, 25) # Orange/brown particles for hit
                    self.game.floating_texts.append(FloatingText(f"-{ENEMY_PENALTY}", enemy.x + 15, enemy.y + 15, (255, 0, 0), 60, -0.5))
                    self.game.play_sound('enemy_hit') # Play enemy hit sound
                    agent.PlayerEntity.x, agent.PlayerEntity.y = self.get_random_open_cell_coords(PLAYER_SIZE, PLAYER_SIZE)
        for enemy_to_remove in enemies_to_remove:
            self.EnemyCollection.remove(enemy_to_remove)
            self.EnemyCollection.append(GameRectangle(*self.get_random_open_cell_coords(30, 30), 30, 30))

        # Check for fake player detection
        fake_players_to_remove = []
        for fake in self.FakePlayers:
            if agent.PlayerEntity.colliderect(fake):
                if agent.disarming:
                    agent.commentary += " - DARK WIZARD DISARMED!"
                    fake_players_to_remove.append(fake)
                    agent.reward += 25
                    current_step_reward += 25
                    self.spawn_particles(fake.x + 15, fake.y + 15, (100,0,100), 10, 20) # Purple particles for disarm
                    self.game.floating_texts.append(FloatingText(f"+25", fake.x + 15, fake.y + 15, (0, 255, 0), 60, -0.5))
                    self.game.play_sound('disarm_enemy') # Play disarm sound
                elif agent.in_stealth:
                    agent.detected_fakes += 1
                    agent.reward += FAKE_PLAYER_DETECT_REWARD
                    current_step_reward += FAKE_PLAYER_DETECT_REWARD
                    agent.commentary += " - DETECTED DARK WIZARD!"
                    self.spawn_particles(fake.x + 15, fake.y + 15, (255,255,255), 10, 20) # White particles for detection
                    self.game.floating_texts.append(FloatingText(f"+{FAKE_PLAYER_DETECT_REWARD}", fake.x + 15, fake.y + 15, (0, 255, 0), 60, -0.5))
                    self.game.play_sound('fake_detect') # Play fake detect sound
                    fake_players_to_remove.append(fake)
                elif not agent.invincible and not agent.in_ghost:
                    agent.reward -= FAKE_PLAYER_PENALTY
                    current_step_reward -= FAKE_PLAYER_PENALTY
                    agent.commentary += " - CAUGHT BY DARK WIZARD!"
                    self.spawn_particles(fake.x + 15, fake.y + 15, (255,0,0), 15, 25) # Red particles for caught
                    self.game.floating_texts.append(FloatingText(f"-{FAKE_PLAYER_PENALTY}", fake.x + 15, fake.y + 15, (255, 0, 0), 60, -0.5))
                    self.game.play_sound('enemy_hit') # Play enemy hit sound
                    agent.PlayerEntity.x, agent.PlayerEntity.y = self.get_random_open_cell_coords(PLAYER_SIZE, PLAYER_SIZE)

        for fake_to_remove in fake_players_to_remove:
            self.FakePlayers.remove(fake_to_remove)
            self.FakePlayers.append(GameRectangle(*self.get_random_open_cell_coords(30, 30), 30, 30))

        return current_step_reward

    def spawn_particles(self, x, y, base_color, num_particles, lifetime):
        for _ in range(num_particles):
            # Random velocity for explosion effect
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            # Slight color variation, clamped to 0-255
            color_var = (
                max(0, min(255, base_color[0] + random.randint(-30, 30))),
                max(0, min(255, base_color[1] + random.randint(-30, 30))),
                max(0, min(255, base_color[2] + random.randint(-30, 30)))
            )
            radius = random.uniform(2, 5)
            # Initial particle color is RGB, alpha is handled in update/draw
            self.particles.append(Particle(x, y, color_var, radius, (vx, vy), lifetime))

    def update_enemy_and_fake(self):
        for enemy in self.EnemyCollection:
            dx, dy = 0, 0
            target_player = None
            min_dist = float('inf')

            # Find closest non-stealthy player
            for agent in self.Agents:
                if not agent.in_stealth:
                    is_visible, distance = self.is_entity_visible(enemy, agent.PlayerEntity, ENEMY_DETECTION_RADIUS)
                    if is_visible and distance < min_dist:
                        min_dist = distance
                        target_player = agent.PlayerEntity
            
            if target_player:
                # Move towards the target player
                if target_player.x > enemy.x:
                    dx = BASE_AGENT_SPEED
                elif target_player.x < enemy.x:
                    dx = -BASE_AGENT_SPEED
                
                if target_player.y > enemy.y:
                    dy = BASE_AGENT_SPEED
                elif target_player.y < enemy.y:
                    dy = -BASE_AGENT_SPEED
            else:
                # Random movement if no target or target is stealthy
                direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
                if direction == "UP": dy = -BASE_AGENT_SPEED
                elif direction == "DOWN": dy = BASE_AGENT_SPEED
                elif direction == "LEFT": dx = -BASE_AGENT_SPEED
                elif direction == "RIGHT": dx = BASE_AGENT_SPEED

            if self.is_valid_move(enemy, dx, dy):
                enemy.x += dx
                enemy.y += dy

        for fake in self.FakePlayers:
            dx, dy = 0, 0
            target_player = None
            min_dist = float('inf')

            # Find closest non-stealthy player
            for agent in self.Agents:
                if not agent.in_stealth: # Fake players also react to non-stealthy players
                    is_visible, distance = self.is_entity_visible(fake, agent.PlayerEntity, FAKE_PLAYER_DETECTION_RADIUS)
                    if is_visible and distance < min_dist:
                        min_dist = distance
                        target_player = agent.PlayerEntity
            
            if target_player:
                # Move towards the target player
                if target_player.x > fake.x:
                    dx = BASE_AGENT_SPEED
                elif target_player.x < fake.x:
                    dx = -BASE_AGENT_SPEED
                
                if target_player.y > fake.y:
                    dy = BASE_AGENT_SPEED
                elif target_player.y < fake.y:
                    dy = -BASE_AGENT_SPEED
            else:
                # Random movement if no target or target is stealthy
                direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
                if direction == "UP": dy = -BASE_AGENT_SPEED
                elif direction == "DOWN": dy = BASE_AGENT_SPEED
                elif direction == "LEFT": dx = -BASE_AGENT_SPEED
                elif direction == "RIGHT": dx = -BASE_AGENT_SPEED

            if self.is_valid_move(fake, dx, dy):
                fake.x += dx
                fake.y += dy

    def check_game_end(self):
        # Game ends immediately if main treasure is collected (handled in update_agent)
        if self.game_over: return

        # Check for time expiry
        current_time = time.time()
        if current_time - self.game_start_time >= GAME_DURATION_SECONDS:
            self.game_over = True
            self.determine_winner_by_score()
            self.win_reason = "Time Expired!"
            return

        # Game ends when all treasures (including main, if not yet collected) are gone
        if not self.GoalLocations:
            self.game_over = True
            self.determine_winner_by_score()
            self.win_reason = "All treasures collected!"

    def determine_winner_by_score(self):
        player1 = self.Agents[0]
        player2 = self.Agents[1]

        score1 = player1.reward # Reward now includes treasure values
        score2 = player2.reward

        if score1 > score2:
            self.winner_idx = 0
            self.game.play_sound('win_sound') # Play win sound
        elif score2 > score1:
            self.winner_idx = 1
            self.game.play_sound('win_sound') # Play win sound
        else:
            self.winner_idx = -1
            self.game.play_sound('tie_sound') # Play tie sound

    def is_entity_visible(self, agent_rect, entity_rect, light_radius):
        """Checks if an entity is within the agent's circular light radius."""
        agent_center_x = agent_rect.x + agent_rect.width / 2
        agent_center_y = agent_rect.y + agent_rect.height / 2
        entity_center_x = entity_rect.x + entity_rect.width / 2
        entity_center_y = entity_rect.y + entity_rect.height / 2

        distance = math.hypot(agent_center_x - entity_center_x, agent_center_y - entity_center_y)
        return distance <= light_radius, distance

    # --- DQN Helper Functions ---
    def get_state_representation(self, agent_idx):
        """
        Converts the current game state into a numerical representation for the DQN.
        This is a simplified example. A more robust state would include:
        - Agent's own position (normalized)
        - Positions of other agents, enemies, fake players, treasures, gems, traps (normalized)
        - Maze layout (flattened or convolved)
        - Agent's current power-up states and remaining durations (normalized)
        - Time left in game (normalized)
        - Visibility information for key entities
        """
        agent = self.Agents[agent_idx]
        
        state_features = []

        # 1. Agent's position (normalized to maze grid coordinates)
        agent_col = agent.PlayerEntity.x // MAZE_CELL_SIZE
        agent_row = agent.PlayerEntity.y // MAZE_CELL_SIZE
        state_features.extend([agent_row / self.maze_rows, agent_col / self.maze_cols])

        # 2. Agent's power-up states and energy
        state_features.extend([
            float(agent.in_stealth), float(agent.in_ghost), float(agent.invincible),
            float(agent.boosted), float(agent.disarming),
            agent.energy / agent.max_energy # Normalized energy
        ])

        # 3. Flattened maze (0 for path, 1 for wall)
        flattened_maze = self.maze.flatten().tolist()
        state_features.extend(flattened_maze)

        # 4. Visible Entities (closest for each type)
        
        # Closest Treasure
        closest_visible_treasure = None
        min_treasure_dist = float('inf')
        for treasure_data in self.GoalLocations:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, treasure_data['rect'], PLAYER_LIGHT_RADIUS)
            if is_visible and distance < min_treasure_dist:
                min_treasure_dist = distance
                closest_visible_treasure = treasure_data['rect']
        
        is_any_treasure_visible = float(closest_visible_treasure is not None)
        state_features.append(is_any_treasure_visible)
        if closest_visible_treasure:
            rel_x = (closest_visible_treasure.x - agent.PlayerEntity.x) / GAME_WIDTH
            rel_y = (closest_visible_treasure.y - agent.PlayerEntity.y) / GAME_HEIGHT
            norm_dist = min_treasure_dist / (GAME_WIDTH + GAME_HEIGHT)
            angle = math.atan2(closest_visible_treasure.y - agent.PlayerEntity.y,
                               closest_visible_treasure.x - agent.PlayerEntity.x)
            state_features.extend([rel_x, rel_y, norm_dist, angle / math.pi])
        else:
            state_features.extend([0.0, 0.0, 0.0, 0.0]) # rel_x, rel_y, norm_dist, angle

        # Closest Gem
        closest_visible_gem = None
        min_gem_dist = float('inf')
        for gem_data in self.GemCollection:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, gem_data['rect'], PLAYER_LIGHT_RADIUS)
            if is_visible and distance < min_gem_dist:
                min_gem_dist = distance
                closest_visible_gem = gem_data['rect']
        
        is_any_gem_visible = float(closest_visible_gem is not None)
        state_features.append(is_any_gem_visible)
        if closest_visible_gem:
            rel_x = (closest_visible_gem.x - agent.PlayerEntity.x) / GAME_WIDTH
            rel_y = (closest_visible_gem.y - agent.PlayerEntity.y) / GAME_HEIGHT
            norm_dist = min_gem_dist / (GAME_WIDTH + GAME_HEIGHT)
            state_features.extend([rel_x, rel_y, norm_dist])
        else:
            state_features.extend([0.0, 0.0, 0.0]) # rel_x, rel_y, norm_dist

        # Closest Enemy
        closest_visible_enemy = None
        min_enemy_dist = float('inf')
        for enemy in self.EnemyCollection:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, enemy, PLAYER_LIGHT_RADIUS)
            if is_visible and distance < min_enemy_dist:
                min_enemy_dist = distance
                closest_visible_enemy = enemy
        
        is_any_enemy_visible = float(closest_visible_enemy is not None)
        state_features.append(is_any_enemy_visible)
        if closest_visible_enemy:
            rel_x = (closest_visible_enemy.x - agent.PlayerEntity.x) / GAME_WIDTH
            rel_y = (closest_visible_enemy.y - agent.PlayerEntity.y) / GAME_HEIGHT
            norm_dist = min_enemy_dist / (GAME_WIDTH + GAME_HEIGHT)
            state_features.extend([rel_x, rel_y, norm_dist])
        else:
            state_features.extend([0.0, 0.0, 0.0]) # rel_x, rel_y, norm_dist

        # Closest Fake Player
        closest_visible_fake = None
        min_fake_dist = float('inf')
        for fake in self.FakePlayers:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, fake, PLAYER_LIGHT_RADIUS)
            if is_visible and distance < min_fake_dist:
                min_fake_dist = distance
                closest_visible_fake = fake
        
        is_any_fake_visible = float(closest_visible_fake is not None)
        state_features.append(is_any_fake_visible)
        if closest_visible_fake:
            rel_x = (closest_visible_fake.x - agent.PlayerEntity.x) / GAME_WIDTH
            rel_y = (closest_visible_fake.y - agent.PlayerEntity.y) / GAME_HEIGHT
            norm_dist = min_fake_dist / (GAME_WIDTH + GAME_HEIGHT)
            state_features.extend([rel_x, rel_y, norm_dist])
        else:
            state_features.extend([0.0, 0.0, 0.0]) # rel_x, rel_y, norm_dist

        # Closest Spike Trap
        closest_visible_spike = None
        min_spike_dist = float('inf')
        for spike in self.SpikeTrapCollection:
            is_visible, distance = self.is_entity_visible(agent.PlayerEntity, spike, PLAYER_LIGHT_RADIUS)
            if is_visible and distance < min_spike_dist:
                min_spike_dist = distance
                closest_visible_spike = spike
        
        is_any_spike_visible = float(closest_visible_spike is not None)
        state_features.append(is_any_spike_visible)
        if closest_visible_spike:
            rel_x = (closest_visible_spike.x - agent.PlayerEntity.x) / GAME_WIDTH
            rel_y = (closest_visible_spike.y - agent.PlayerEntity.y) / GAME_HEIGHT
            norm_dist = min_spike_dist / (GAME_WIDTH + GAME_HEIGHT)
            state_features.extend([rel_x, rel_y, norm_dist])
        else:
            state_features.extend([0.0, 0.0, 0.0]) # rel_x, rel_y, norm_dist

        # 5. Time remaining
        time_left_norm = (GAME_DURATION_SECONDS - (time.time() - self.game_start_time)) / GAME_DURATION_SECONDS
        state_features.append(max(0.0, time_left_norm))
        
        # Ensure state has consistent size. This is crucial for the neural network input layer.
        # Dynamically calculate the expected size based on the features added.
        # Base: 2 (agent pos) + 6 (power-ups + energy) + (maze_rows * maze_cols)
        # Treasure: 1 (is_visible) + 2 (rel_x, rel_y) + 1 (norm_dist) + 1 (angle) = 5
        # Gem: 1 (is_visible) + 2 (rel_x, rel_y) + 1 (norm_dist) = 4
        # Enemy: 1 (is_visible) + 2 (rel_x, rel_y) + 1 (norm_dist) = 4
        # Fake: 1 (is_visible) + 2 (rel_x, rel_y) + 1 (norm_dist) = 4
        # Spike: 1 (is_visible) + 2 (rel_x, rel_y) + 1 (norm_dist) = 4
        # Time: 1
        expected_state_size = (self.maze_rows * self.maze_cols) + 2 + 6 + 5 + 4 + 4 + 4 + 4 + 1
        
        if len(state_features) != expected_state_size:
            # This should ideally not happen if features are consistently added.
            # If it does, padding/truncating might be needed, or the expected_state_size calculation is wrong.
            raise ValueError(f"State size mismatch: Expected {expected_state_size}, got {len(state_features)}")

        return np.array(state_features, dtype=np.float32)

    def get_action_from_index(self, action_index):
        # Map integer action index back to string action
        actions = ["UP", "DOWN", "LEFT", "RIGHT", "STEALTH", "GHOST", "INVINCIBLE", "BOOST", "DISARM"]
        return actions[action_index]

    def get_index_from_action(self, action_string):
        actions = ["UP", "DOWN", "LEFT", "RIGHT", "STEALTH", "GHOST", "INVINCIBLE", "BOOST", "DISARM"]
        return actions.index(action_string)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# PyTorch Q-Network definition
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, ai_profile="EXPLORER"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Initialize learning_rate and other parameters based on profile
        self.ai_profile = ai_profile
        if self.ai_profile == "EXPLORER":
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_min = 0.05
            self.epsilon_decay = 0.995 # Slower decay for more exploration
            self.learning_rate = 0.001
        elif self.ai_profile == "AGGRESSIVE":
            self.gamma = 0.99
            self.epsilon = 0.8 # Start with less exploration
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.998 # Faster decay
            self.learning_rate = 0.003 # Higher learning rate
        elif self.ai_profile == "DEFENSIVE":
            self.gamma = 0.98
            self.epsilon = 1.0
            self.epsilon_min = 0.02
            self.epsilon_decay = 0.997 # Moderate decay
            self.learning_rate = 0.0015
        else: # Default to EXPLORER if profile not recognized
            self.gamma = GAMMA
            self.epsilon = EPSILON_START
            self.epsilon_min = EPSILON_END
            self.epsilon_decay = EPSILON_DECAY
            self.learning_rate = LEARNING_RATE

        self.target_update_counter = 0

        # PyTorch device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main Q-network and Target Q-network
        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() # Mean Squared Error Loss

        self.update_target_model() # Initialize target network weights to Q-network weights

        # Store action mapping for easy use
        self.action_map = ["UP", "DOWN", "LEFT", "RIGHT", "STEALTH", "GHOST", "INVINCIBLE", "BOOST", "DISARM"]

    def _build_q_network(self):
        # This method is now effectively replaced by the QNetwork class.
        # It's kept for structural clarity but the instantiation happens in __init__
        pass

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Convert action string to its index for storage
        action_index = self.action_map.index(action)
        self.memory.add(state, action_index, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_map) # Explore
        
        # Reshape state for model prediction (add batch dimension) and move to device
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Get Q-values from the main model
        with torch.no_grad(): # No gradient calculation needed for inference
            q_values = self.model(state_tensor)
        
        # Get the action with the maximum Q-value
        action_index = torch.argmax(q_values).item()
        return self.action_map[action_index] # Exploit

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return # Not enough experiences to train

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convert numpy arrays to PyTorch tensors and move to device
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device) # Actions are indices
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(int)).float().to(self.device) # Convert bool to int then float

        # Get Q-values for current states from the main model
        current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Get max Q-value for next states from the target model
        with torch.no_grad(): # No gradient calculation for target network
            next_q_values = self.target_model(next_states).max(1)[0] # max(1)[0] gets max value along dim 1

        # Calculate target Q-values using Bellman equation
        # If 'done' is true, target is just the reward (no future Q-value)
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate loss
        loss = self.criterion(current_q_values, targets.detach()) # .detach() important for targets

        # Optimize the model
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward()            # Backpropagation
        # Optional: Clip gradients to prevent exploding gradients
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()     # Update weights

        self.target_update_counter += 1
        if self.target_update_counter % TARGET_UPDATE_FREQ == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AIController:
    def __init__(self, state_size, action_size, ai_profile="EXPLORER"):
        self.dqn_agent = DQNAgent(state_size, action_size, ai_profile)
        self.last_state = None
        self.last_action = None

    def GetAction(self, game_state: GameState, agent_idx, pressed_keys=None, key_down_events=None):
        # AI doesn't use keyboard input, so pressed_keys and key_down_events are ignored
        current_state = game_state.get_state_representation(agent_idx)
        action_string = self.dqn_agent.act(current_state)
        return action_string

    def record_experience_and_train(self, current_state, action_string, reward, next_state, done):
        self.dqn_agent.remember(current_state, action_string, reward, next_state, done)
        self.dqn_agent.replay()


class ManualController:
    def __init__(self, player_num):
        self.player_num = player_num
        # Define movement keys separately from power-up keys
        # Store pygame.K_ constants as values for direct indexing of pressed_keys array
        self.movement_keys_map = {
            1: {
                "UP": pygame.K_UP, "DOWN": pygame.K_DOWN,
                "LEFT": pygame.K_LEFT, "RIGHT": pygame.K_RIGHT
            },
            2: {
                "UP": pygame.K_w, "DOWN": pygame.K_s,
                "LEFT": pygame.K_a, "RIGHT": pygame.K_d
            }
        }
        self.powerup_keys_map = {
            1: {
                "STEALTH": pygame.K_z, "GHOST": pygame.K_x,
                "INVINCIBLE": pygame.K_c, "BOOST": pygame.K_v, "DISARM": pygame.K_b
            },
            2: {
                "STEALTH": pygame.K_t, "GHOST": pygame.K_y,
                "INVINCIBLE": pygame.K_u, "BOOST": pygame.K_i, "DISARM": pygame.K_o
            }
        }

    def GetAction(self, pressed_keys, key_down_events):
        action = "NONE" # Default action if no relevant key is pressed

        # 1. Check for movement keys (continuous input)
        # Iterate through the actions and check if their corresponding key is pressed
        # Use the key code directly to index the pressed_keys array
        for move_action, key_code in self.movement_keys_map[self.player_num].items():
            if pressed_keys[key_code]:
                action = move_action
                break # Prioritize one movement direction if multiple are pressed (e.g., up-left)

        # 2. Check for power-up keys (single press input)
        # Power-up actions should override movement if pressed in the same frame
        for event in key_down_events: # Iterate through only KEYDOWN events
            for powerup_action, key_code in self.powerup_keys_map[self.player_num].items():
                if event.key == key_code:
                    action = powerup_action
                    break # Only one power-up action per frame
            if action != "NONE" and action in self.powerup_keys_map[self.player_num].keys():
                break # If a power-up was found, stop checking events

        return action

class AbstractGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init() # Initialize mixer
        self.screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
        pygame.display.set_caption("Phantom Quest: Hunt or Be Hunted") # Branding
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 18)
        self.large_font = pygame.font.SysFont("Arial", 40, bold=True)
        self.medium_font = pygame.font.SysFont("Arial", 25)
        self.small_font = pygame.font.SysFont("Arial", 14)

        self.sounds = {} # Dictionary to hold sound objects
        self.load_sounds() # Load sounds

        # Game State Modes
        self.game_state_mode = "MAIN_MENU" # MAIN_MENU, SELECT_MODE, ENTER_NAMES, SELECT_AI_PROFILE, PLAYING, PAUSED, INSTRUCTIONS, STATS, GAME_OVER, LEADERBOARD

        # Controllers will be set after game mode selection
        self.controllers = []
        self.player1_is_ai_controlled = False # Flag for Player 1 toggle
        self.current_ai_profile_p1 = "EXPLORER" # Default AI profile for P1
        self.current_ai_profile_p2 = "EXPLORER" # Default AI profile for P2

        # Temporary GameState instance to get the initial state size for AI agents
        temp_game_state = GameState(None)
        self.INITIAL_STATE_SIZE = len(temp_game_state.get_state_representation(0))
        
        # AI controllers initialized with default profiles, updated later
        self.ai_player1_controller = AIController(self.INITIAL_STATE_SIZE, ACTION_SIZE, self.current_ai_profile_p1)
        self.manual_player1_controller = ManualController(1)
        self.ai_player2_controller = AIController(self.INITIAL_STATE_SIZE, ACTION_SIZE, self.current_ai_profile_p2)
        self.manual_player2_controller = ManualController(2)

        self.state = None # GameState object, initialized when game starts
        self.paused_background_surface = None # To store a static image of the game when paused

        self.running = True
        self.episode_count = 0
        self.total_steps = 0
        self.game_start_real_time = 0 # To track actual time spent playing

        # Player Name Entry
        self.player1_name_input = InputBox(GAME_WIDTH // 2 - 100, GAME_HEIGHT // 2 - 30, 200, 32, 'Player 1')
        self.player2_name_input = InputBox(GAME_WIDTH // 2 - 100, GAME_HEIGHT // 2 + 40, 200, 32, 'Player 2')
        self.current_player_name = "Player 1" # Default, can be changed in menu
        self.player2_name = "Player 2" # Default, for two player games

        # Player Profile System
        self.player_profiles = self.load_player_profiles()
        self.current_player_profile = self.get_or_create_profile(self.current_player_name)
        
        # Achievement System
        self.achievements = {
            "First Gem": {"description": "Collect your first gem!", "unlocked": False, "condition": lambda agent: agent.gems_collected >= 1},
            "Gem Collector": {"description": "Collect 10 gems!", "unlocked": False, "condition": lambda agent: agent.gems_collected >= 10},
            "Treasure Hunter": {"description": "Collect 3 treasures!", "unlocked": False, "condition": lambda agent: agent.treasures_collected >= 3},
            "Master Thief": {"description": "Collect the Legendary Crown!", "unlocked": False, "condition": lambda agent: agent.has_main_treasure},
            "Stealth Master": {"description": "Use stealth 5 times!", "unlocked": False, "condition": lambda agent: agent.stealth_activations >= 5},
            "Disarmer": {"description": "Disarm 3 enemies/fake players!", "unlocked": False, "condition": lambda agent: agent.disarm_activations >= 3}
        }
        self.achievement_notifications = deque(maxlen=3) # Store (message, display_end_time) tuples

        # In-Game Analytics
        self.game_metrics = {
            "time_spent_playing": 0,
            "powerups_used_p1": 0,
            "powerups_used_p2": 0,
            "gems_collected_p1": 0,
            "gems_collected_p2": 0,
            "treasures_collected_p1": 0,
            "treasures_collected_p2": 0,
            "enemies_disarmed_p1": 0,
            "enemies_disarmed_p2": 0,
            "fakes_detected_p1": 0,
            "fakes_detected_p2": 0,
        }
        self.current_game_mode = "AI" # Default mode, will be set by menu
        
        # Floating Text (for score, etc.)
        self.floating_texts = []

        # Save/Load Feedback Message
        self.feedback_message = ""
        self.feedback_end_time = 0


    def generate_tone(self, frequency, duration, volume=0.1, harmonics=None, attack=0.01, decay=0.05):
        """
        Generates a sound with multiple harmonics and an ADSR-like envelope.
        frequency: base frequency in Hz
        duration: duration in seconds
        volume: overall volume (0.0 to 1.0)
        harmonics: list of (frequency_multiplier, amplitude_multiplier) tuples.
                   e.g., [(1, 1), (2, 0.5)] for fundamental + octave at half amplitude.
        attack: attack duration as a fraction of total duration
        decay: decay duration as a fraction of total duration
        """
        if harmonics is None:
            harmonics = [(1, 1)] # Default to a single sine wave

        sample_rate = 44100  # samples per second
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        waveform = np.zeros_like(t)
        for freq_mult, amp_mult in harmonics:
            waveform += amp_mult * np.sin(2 * np.pi * frequency * freq_mult * t)
        
        # Normalize waveform to prevent clipping
        max_amp = np.max(np.abs(waveform))
        if max_amp > 0:
            waveform /= max_amp

        # Apply ADSR-like envelope
        envelope = np.ones_like(t)
        
        # Attack phase
        attack_samples = int(attack * num_samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase (after attack, before release)
        decay_samples = int(decay * num_samples)
        # Ensure decay starts after attack and doesn't exceed total samples
        decay_start_idx = attack_samples
        decay_end_idx = min(num_samples, decay_start_idx + decay_samples)
        envelope[decay_start_idx:decay_end_idx] *= np.linspace(1, 0.5, decay_end_idx - decay_start_idx) # Sustain at 0.5
        
        # Release (fade out at the very end) - simple fade out for the rest
        release_samples = int(0.1 * num_samples) # Last 10% of the sound
        envelope[num_samples - release_samples:] *= np.linspace(1, 0, release_samples)

        arr = (waveform * envelope * 32767 * volume).astype(np.int16)
        sound = pygame.mixer.Sound(arr.tobytes())
        return sound

    def load_sounds(self):
        # Power-up sound (a quick, ascending chime)
        self.sounds['powerup'] = self.generate_tone(
            frequency=600, duration=0.15, volume=0.2,
            harmonics=[(1, 1), (1.5, 0.5), (2, 0.3)], attack=0.05, decay=0.1
        )
        # Treasure collect (a bright, short, happy tone)
        self.sounds['treasure_collect'] = self.generate_tone(
            frequency=1000, duration=0.1, volume=0.25,
            harmonics=[(1, 1), (1.2, 0.6), (1.5, 0.4)], attack=0.01, decay=0.08
        )
        # Main treasure collect (a more triumphant, slightly longer chime)
        self.sounds['treasure_main'] = self.generate_tone(
            frequency=1500, duration=0.3, volume=0.35,
            harmonics=[(1, 1), (1.3, 0.7), (1.6, 0.5), (2, 0.3)], attack=0.02, decay=0.15
        )
        # Gem collect (a distinct, quick sparkle)
        self.sounds['gem_collect'] = self.generate_tone(
            frequency=1800, duration=0.07, volume=0.2,
            harmonics=[(1, 1), (0.8, 0.5), (1.2, 0.5)], attack=0.01, decay=0.05
        )
        # Spike hit (a sharp, low, painful thud)
        self.sounds['spike_hit'] = self.generate_tone(
            frequency=100, duration=0.2, volume=0.3,
            harmonics=[(1, 1), (0.5, 0.7)], attack=0.01, decay=0.15
        )
        # Enemy hit/caught (a short, aggressive growl/scream)
        self.sounds['enemy_hit'] = self.generate_tone(
            frequency=250, duration=0.15, volume=0.3,
            harmonics=[(1, 1), (1.1, 0.8), (0.9, 0.6)], attack=0.01, decay=0.1
        )
        # Fake detect/disarm (a quick, positive confirmation)
        self.sounds['fake_detect'] = self.generate_tone(
            frequency=1200, duration=0.1, volume=0.25,
            harmonics=[(1, 1), (0.7, 0.5)], attack=0.01, decay=0.08
        )
        self.sounds['disarm_enemy'] = self.generate_tone(
            frequency=800, duration=0.1, volume=0.25,
            harmonics=[(1, 1), (1.4, 0.6)], attack=0.01, decay=0.08
        )
        # Win sound (a grand, ascending melody)
        self.sounds['win_sound'] = self.generate_tone(
            frequency=440, duration=0.8, volume=0.4,
            harmonics=[(1, 1), (1.25, 0.7), (1.5, 0.5)], attack=0.05, decay=0.2
        )
        # Tie sound (a neutral, slightly descending tone)
        self.sounds['tie_sound'] = self.generate_tone(
            frequency=600, duration=0.4, volume=0.25,
            harmonics=[(1, 1), (0.75, 0.6)], attack=0.03, decay=0.15
        )
        # Walking sound (a subtle, low thud)
        self.sounds['walk_sound'] = self.generate_tone(
            frequency=35, duration=0.1, volume=0.3, # Even deeper frequency, slightly longer duration, increased volume
            harmonics=[(1, 1), (1.5, 0.5), (2.0, 0.3)], # More prominent lower harmonics for a heavier thud
            attack=0.002, decay=0.07 # Very short attack, slightly longer decay for a more resonant thud
        )

    def play_sound(self, sound_name):
        if sound_name in self.sounds:
            self.sounds[sound_name].play()

    def draw_button(self, text, x, y, width, height, inactive_color, active_color, events, action=None):
        mouse = pygame.mouse.get_pos()
        button_rect = pygame.Rect(x, y, width, height)
        
        clicked = False
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                if button_rect.collidepoint(event.pos):
                    clicked = True

        if button_rect.collidepoint(mouse):
            pygame.draw.rect(self.screen, active_color, button_rect, border_radius=10)
            if clicked and action is not None:
                return action
        else:
            pygame.draw.rect(self.screen, inactive_color, button_rect, border_radius=10)

        text_surf = self.medium_font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(x + width / 2, y + height / 2))
        self.screen.blit(text_surf, text_rect)
        return None

    def load_player_profiles(self):
        print("Attempting to load player profiles...")
        if os.path.exists(PLAYER_PROFILES_FILE):
            try:
                with open(PLAYER_PROFILES_FILE, 'r') as f:
                    data = json.load(f)
                    # Ensure the loaded data is a dictionary
                    if isinstance(data, dict):
                        print("Player profiles loaded successfully.")
                        return data
                    else:
                        print(f"Warning: {PLAYER_PROFILES_FILE} contains invalid data format. Resetting profiles.")
            except json.JSONDecodeError:
                print(f"Warning: {PLAYER_PROFILES_FILE} is corrupted. Resetting profiles.")
            except Exception as e:
                print(f"An unexpected error occurred while loading {PLAYER_PROFILES_FILE}: {e}. Resetting profiles.")
        else:
            print(f"Player profiles file '{PLAYER_PROFILES_FILE}' not found. Creating new profiles.")
        return {}

    def save_player_profiles(self):
        try:
            with open(PLAYER_PROFILES_FILE, 'w') as f:
                json.dump(self.player_profiles, f, indent=4)
            print("Player profiles saved successfully.")
        except Exception as e:
            print(f"Error saving player profiles: {e}")

    def export_player_profiles(self):
        try:
            with open(PLAYER_PROFILES_EXPORT_FILE, 'w') as f:
                json.dump(self.player_profiles, f, indent=4)
            self.feedback_message = f"Profiles Exported to {PLAYER_PROFILES_EXPORT_FILE}!"
            self.feedback_end_time = pygame.time.get_ticks() + 2000
            print(self.feedback_message)
        except Exception as e:
            self.feedback_message = f"Error exporting profiles: {e}"
            self.feedback_end_time = pygame.time.get_ticks() + 2000
            print(self.feedback_message)

    def clear_leaderboard_profiles(self):
        """Clears all player profiles from the leaderboard."""
        try:
            if os.path.exists(PLAYER_PROFILES_FILE):
                os.remove(PLAYER_PROFILES_FILE)
                print(f"Leaderboard file '{PLAYER_PROFILES_FILE}' cleared.")
            self.player_profiles = {} # Reset in-memory profiles
            self.feedback_message = "Leaderboard Cleared!"
            self.feedback_end_time = pygame.time.get_ticks() + 2000
        except Exception as e:
            self.feedback_message = f"Error clearing leaderboard: {e}"
            self.feedback_end_time = pygame.time.get_ticks() + 2000
        print(self.feedback_message)

    def get_or_create_profile(self, player_name):
        if player_name not in self.player_profiles:
            print(f"Creating new profile for player: {player_name}")
            self.player_profiles[player_name] = {
                "total_score": 0,
                "games_played": 0,
                "achievements": {name: False for name in self.achievements.keys()},
                "highest_score": 0,
                "total_gems_collected": 0,
                "total_treasures_collected": 0,
                "total_powerups_used": 0,
                "total_enemies_disarmed": 0,
                "total_fakes_detected": 0,
                "total_time_played": 0 # in seconds
            }
            self.save_player_profiles()
        else:
            print(f"Loading existing profile for player: {player_name}")
        return self.player_profiles[player_name]

    def update_player_profile(self):
        if self.state and self.current_player_profile:
            p1_agent = self.state.Agents[0]
            self.current_player_profile["total_score"] += p1_agent.reward
            self.current_player_profile["games_played"] += 1
            self.current_player_profile["highest_score"] = max(self.current_player_profile["highest_score"], p1_agent.reward)
            self.current_player_profile["total_gems_collected"] += p1_agent.gems_collected
            self.current_player_profile["total_treasures_collected"] += p1_agent.treasures_collected
            self.current_player_profile["total_powerups_used"] += p1_agent.powerups_used
            self.current_player_profile["total_enemies_disarmed"] += p1_agent.disarm_activations # Assuming disarm_activations counts successful disarms
            self.current_player_profile["total_fakes_detected"] += p1_agent.detected_fakes
            self.current_player_profile["total_time_played"] += self.game_metrics["time_spent_playing"]

            # Update achievements in profile
            for ach_name, ach_data in self.achievements.items():
                if ach_data["unlocked"] and not self.current_player_profile["achievements"].get(ach_name, False):
                    self.current_player_profile["achievements"][ach_name] = True
            self.save_player_profiles()

    def check_achievements(self, agent):
        for ach_name, ach_data in self.achievements.items():
            if not ach_data["unlocked"] and ach_data["condition"](agent):
                ach_data["unlocked"] = True
                notification_message = f"Achievement Unlocked: {ach_name} - {ach_data['description']}"
                self.achievement_notifications.append((notification_message, pygame.time.get_ticks() + 3000)) # Display for 3 seconds

    def display_achievement_notifications(self):
        current_time = pygame.time.get_ticks()
        y_offset = 10
        notifications_to_remove = []

        for i, (msg, end_time) in enumerate(self.achievement_notifications):
            if current_time < end_time:
                text_surf = self.small_font.render(msg, True, (255, 255, 0)) # Yellow text
                text_rect = text_surf.get_rect(topright=(GAME_WIDTH - 10, y_offset))
                
                # Draw background for notification
                bg_rect = text_rect.inflate(20, 10) # Add padding
                bg_rect.x -= 10 # Adjust x to center text
                pygame.draw.rect(self.screen, (50, 50, 50, 180), bg_rect, border_radius=5) # Dark semi-transparent background
                pygame.draw.rect(self.screen, (255, 255, 0), bg_rect, 2, border_radius=5) # Yellow border

                self.screen.blit(text_surf, text_rect)
                y_offset += text_surf.get_height() + 10
            else:
                notifications_to_remove.append(i)
        
        # Remove old notifications
        for i in sorted(notifications_to_remove, reverse=True):
            del self.achievement_notifications[i]

    def reset_game(self):
        print("Resetting game...")
        # Reset game state
        self.state = GameState(self)
        print("New GameState initialized in reset_game.")
        self.game_start_real_time = time.time()
        self.game_metrics = {
            "time_spent_playing": 0,
            "powerups_used_p1": 0,
            "powerups_used_p2": 0,
            "gems_collected_p1": 0,
            "gems_collected_p2": 0,
            "treasures_collected_p1": 0,
            "treasures_collected_p2": 0,
            "enemies_disarmed_p1": 0,
            "enemies_disarmed_p2": 0,
            "fakes_detected_p1": 0,
            "fakes_detected_p2": 0,
        }
        # Reset achievement tracking for the new game session
        for ach_name in self.achievements:
            self.achievements[ach_name]["unlocked"] = False
        self.achievement_notifications.clear()
        self.floating_texts.clear() # Clear floating texts on reset

        # Re-initialize controllers based on the selected mode and AI profiles
        self.set_game_mode_controllers(self.current_game_mode)
        print("Game reset complete.")

    def set_game_mode_controllers(self, mode):
        print(f"Setting game mode controllers to: {mode}")
        self.current_game_mode = mode # Store the chosen mode
        self.controllers = []

        # Re-instantiate AI controllers with potentially new profiles
        self.ai_player1_controller = AIController(self.INITIAL_STATE_SIZE, ACTION_SIZE, self.current_ai_profile_p1)
        self.ai_player2_controller = AIController(self.INITIAL_STATE_SIZE, ACTION_SIZE, self.current_ai_profile_p2)

        if mode == "MIXED":
            self.controllers.append(self.manual_player1_controller) # Player 1 is manual
            self.controllers.append(self.ai_player2_controller) # Player 2 is AI (DQN)
            self.player1_is_ai_controlled = False
            self.state.Agents[0].name = self.player1_name # Set manual player name
            self.state.Agents[1].name = self.player2_name # Set AI player name
        elif mode == "MANUAL":
            self.controllers.append(self.manual_player1_controller) # Player 1 is manual
            self.controllers.append(self.manual_player2_controller) # Player 2 is manual
            self.player1_is_ai_controlled = False
            self.state.Agents[0].name = self.player1_name
            self.state.Agents[1].name = self.player2_name
        else: # Default to AI if input is not recognized or is "AI"
            self.controllers.append(self.ai_player1_controller) # Player 1 is AI (DQN)
            self.controllers.append(self.ai_player2_controller) # Player 2 is AI (DQN)
            self.player1_is_ai_controlled = True
            self.state.Agents[0].name = self.player1_name
            self.state.Agents[1].name = self.player2_name
        print(f"Controllers set: Player 1 AI controlled: {self.player1_is_ai_controlled}")

    def display_main_menu(self, events):
        self.screen.fill(BACKGROUND)
        
        title_text = self.large_font.render("Phantom Quest: Hunt or Be Hunted", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)

        # Buttons
        button_width, button_height = 200, 50
        start_game_action = self.draw_button("Start Game", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 - 90, button_width, button_height, (0, 100, 0), (0, 150, 0), events, action="START_GAME")
        instructions_action = self.draw_button("Instructions", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 - 20, button_width, button_height, (0, 0, 100), (0, 0, 150), events, action="INSTRUCTIONS")
        leaderboard_action = self.draw_button("Leaderboard", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 50, button_width, button_height, (100, 0, 100), (150, 0, 150), events, action="LEADERBOARD")
        exit_action = self.draw_button("Exit", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 120, button_width, button_height, (100, 0, 0), (150, 0, 0), events, action="EXIT")

        if start_game_action == "START_GAME":
            self.game_state_mode = "SELECT_MODE"
            pygame.time.delay(200) # Small delay after click
            print("Transitioning to SELECT_MODE.")
        if instructions_action == "INSTRUCTIONS":
            self.game_state_mode = "INSTRUCTIONS"
            pygame.time.delay(200) # Small delay after click
            print("Transitioning to INSTRUCTIONS.")
        if leaderboard_action == "LEADERBOARD":
            self.game_state_mode = "LEADERBOARD"
            pygame.time.delay(200) # Small delay after click
            print("Transitioning to LEADERBOARD.")
        if exit_action == "EXIT":
            self.running = False
            pygame.time.delay(200) # Small delay after click
            print("Exiting game.")

        pygame.display.flip()

    def display_select_mode_menu(self, events):
        self.screen.fill(BACKGROUND)
        
        title_text = self.large_font.render("Select Game Mode", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)

        button_width, button_height = 200, 50
        ai_mode_action = self.draw_button("AI Mode", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 - 60, button_width, button_height, (0, 100, 0), (0, 150, 0), events, action="AI_MODE")
        manual_mode_action = self.draw_button("Manual Mode", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 10, button_width, button_height, (0, 0, 100), (0, 0, 150), events, action="MANUAL_MODE")
        mixed_mode_action = self.draw_button("Mixed Mode", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 80, button_width, button_height, (100, 50, 0), (150, 75, 0), events, action="MIXED_MODE")
        back_action = self.draw_button("Back", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 150, button_width, button_height, (70, 70, 70), (100, 100, 100), events, action="BACK")

        if ai_mode_action == "AI_MODE":
            self.current_game_mode = "AI"
            self.game_state_mode = "SELECT_AI_PROFILE" # Transition to AI profile selection
            pygame.time.delay(200)
            print("Selected AI Mode. Transitioning to AI profile selection.")
        if manual_mode_action == "MANUAL_MODE":
            self.current_game_mode = "MANUAL"
            self.game_state_mode = "ENTER_NAMES" # Transition to name entry
            pygame.time.delay(200)
            print("Selected Manual Mode. Transitioning to name entry.")
        if mixed_mode_action == "MIXED_MODE":
            self.current_game_mode = "MIXED"
            self.game_state_mode = "SELECT_AI_PROFILE" # Transition to AI profile selection (for P2)
            pygame.time.delay(200)
            print("Selected Mixed Mode. Transitioning to AI profile selection.")
        if back_action == "BACK":
            self.game_state_mode = "MAIN_MENU"
            pygame.time.delay(200)
            print("Transitioning back to MAIN_MENU.")

        pygame.display.flip()

    def display_enter_names_screen(self, events):
        self.screen.fill(BACKGROUND)

        title_text = self.large_font.render("Enter Player Names", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 4 - 50))
        self.screen.blit(title_text, title_rect)

        # Display current game mode
        mode_text = self.medium_font.render(f"Mode: {self.current_game_mode.replace('_', ' ').title()}", True, (0, 255, 255))
        mode_rect = mode_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 4))
        self.screen.blit(mode_text, mode_rect)

        # Player 1 Name Input
        p1_label = self.medium_font.render("Player 1 Name:", True, (255, 255, 255))
        self.screen.blit(p1_label, (self.player1_name_input.rect.x - p1_label.get_width() - 10, self.player1_name_input.rect.y + 5))
        self.player1_name_input.draw(self.screen)

        # Player 2 Name Input
        p2_label = self.medium_font.render("Player 2 Name:", True, (255, 255, 255))
        self.screen.blit(p2_label, (self.player2_name_input.rect.x - p2_label.get_width() - 10, self.player2_name_input.rect.y + 5))
        self.player2_name_input.draw(self.screen)

        # Update input boxes
        for event in events:
            self.player1_name_input.handle_event(event)
            self.player2_name_input.handle_event(event)

        button_width, button_height = 150, 40
        start_action = self.draw_button("Start Game", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 120, button_width, button_height, (0, 100, 0), (0, 150, 0), events, action="START")
        
        if start_action == "START":
            self.player1_name = self.player1_name_input.get_text()
            self.player2_name = self.player2_name_input.get_text()
            self.current_player_profile = self.get_or_create_profile(self.player1_name) # Update P1 profile
            self.reset_game()
            self.game_state_mode = "PLAYING"
            pygame.time.delay(200)
            print(f"Starting game with P1: {self.player1_name}, P2: {self.player2_name}")

        pygame.display.flip()

    def display_select_ai_profile_screen(self, events):
        self.screen.fill(BACKGROUND)

        title_text = self.large_font.render("Select AI Profile", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 4 - 50))
        self.screen.blit(title_text, title_rect)

        # Display current game mode
        mode_text = self.medium_font.render(f"Mode: {self.current_game_mode.replace('_', ' ').title()}", True, (0, 255, 255))
        mode_rect = mode_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 4))
        self.screen.blit(mode_text, mode_rect)

        button_width, button_height = 200, 50
        explorer_action = self.draw_button("Explorer AI", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 - 60, button_width, button_height, (0, 100, 0), (0, 150, 0), events, action="EXPLORER")
        aggressive_action = self.draw_button("Aggressive AI", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 10, button_width, button_height, (0, 0, 100), (0, 0, 150), events, action="AGGRESSIVE")
        defensive_action = self.draw_button("Defensive AI", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 80, button_width, button_height, (100, 50, 0), (150, 75, 0), events, action="DEFENSIVE")
        back_action = self.draw_button("Back", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 150, button_width, button_height, (70, 70, 70), (100, 100, 100), events, action="BACK")

        selected_profile = None
        if explorer_action == "EXPLORER":
            selected_profile = "EXPLORER"
        elif aggressive_action == "AGGRESSIVE":
            selected_profile = "AGGRESSIVE"
        elif defensive_action == "DEFENSIVE":
            selected_profile = "DEFENSIVE"
        elif back_action == "BACK":
            self.game_state_mode = "SELECT_MODE"
            pygame.time.delay(200)
            print("Transitioning back to SELECT_MODE.")
            return

        if selected_profile:
            # If in AI mode, both players get this profile
            if self.current_game_mode == "AI":
                self.current_ai_profile_p1 = selected_profile
                self.current_ai_profile_p2 = selected_profile
                self.player1_name = "AI Player 1 (" + selected_profile + ")"
                self.player2_name = "AI Player 2 (" + selected_profile + ")"
            # If in Mixed mode, Player 2 (AI) gets this profile
            elif self.current_game_mode == "MIXED":
                self.current_ai_profile_p2 = selected_profile
                self.player2_name = "AI Player 2 (" + selected_profile + ")"
            
            # After AI profile selection, proceed to name entry (for manual player in mixed, or just confirm for AI)
            self.game_state_mode = "ENTER_NAMES"
            pygame.time.delay(200)
            print(f"Selected AI Profile: {selected_profile}. Transitioning to name entry.")

        pygame.display.flip()


    def display_instructions(self, events):
        self.screen.fill(BACKGROUND)
        
        title_text = self.large_font.render("Instructions", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, 50))
        self.screen.blit(title_text, title_rect)

        instructions_text = """
        Welcome to Phantom Quest: Hunt or Be Hunted!
        
        Goal: Collect treasures and gems, avoid enemies and traps.
        The Legendary Crown is the main treasure - collect it to win!
        
        Controls (Player 1 / Player 2):
        Movement: Arrow Keys / WASD
        Stealth: Z / T (Become invisible to enemies, but can't collect)
        Ghost: X / Y (Pass through walls, but vulnerable to enemies)
        Invincible: C / U (Immune to damage for a short time)
        Boost: V / I (Increased speed)
        Disarm: B / O (Temporarily disable enemies/fake players)

        Press 'T' during gameplay to toggle Player 1 between Manual and AI control.
        Press 'Esc' during gameplay to pause the game.

        Enemies (Goblins): Chase non-stealthy players.
        Fake Players (Dark Wizards): Chase non-stealthy players, detect stealthy players.
        Spike Traps: Inflict penalty if stepped on without invincibility.
        Fake Treasures: Look like treasures but inflict penalty if touched!

        Collect gems for points. Collect treasures for more points.
        Be careful, the maze is full of dangers!
        The maze will reshape mid-game!
        """
        
        wrapped_instructions = self.wrap_text(instructions_text, GAME_WIDTH - 100, self.font)
        y_offset = 120
        for line in wrapped_instructions:
            text_surf = self.font.render(line, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(GAME_WIDTH // 2, y_offset))
            self.screen.blit(text_surf, text_rect)
            y_offset += text_surf.get_height() + 2

        back_action = self.draw_button("Back to Main Menu", GAME_WIDTH // 2 - 180 // 2, GAME_HEIGHT - 70, 180, 40, (70, 70, 70), (100, 100, 100), events, action="BACK")
        if back_action == "BACK":
            self.game_state_mode = "MAIN_MENU"
            pygame.time.delay(200) # Small delay after click
            print("Transitioning back to MAIN_MENU from instructions.")

        pygame.display.flip()

    def display_pause_menu(self, events):
        # Blit the captured game screen as background
        if self.paused_background_surface:
            self.screen.blit(self.paused_background_surface, (0, 0))
        else: # Fallback if no background captured (e.g., paused from main menu before game starts)
            self.screen.fill(BACKGROUND)
        
        overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180)) # Semi-transparent black overlay
        self.screen.blit(overlay, (0, 0))

        title_text = self.large_font.render("Game Paused", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)

        button_width, button_height = 200, 50
        resume_action = self.draw_button("Resume", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 - 120, button_width, button_height, (0, 100, 0), (0, 150, 0), events, action="RESUME")
        stats_action = self.draw_button("View Stats", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 - 50, button_width, button_height, (0, 0, 100), (0, 0, 150), events, action="STATS")
        save_action = self.draw_button("Save Game", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 20, button_width, button_height, (100, 50, 0), (150, 75, 0), events, action="SAVE")
        load_action = self.draw_button("Load Game", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 90, button_width, button_height, (100, 50, 0), (150, 75, 0), events, action="LOAD")
        quit_action = self.draw_button("Quit to Main Menu", GAME_WIDTH // 2 - button_width // 2, GAME_HEIGHT // 2 + 160, button_width, button_height, (100, 0, 0), (150, 0, 0), events, action="QUIT")

        if resume_action == "RESUME":
            self.game_state_mode = "PLAYING"
            self.state.game_start_time = time.time() - (self.game_metrics["time_spent_playing"]) # Adjust start time
            pygame.time.delay(200) # Small delay after click
            print("Resuming game.")
        if stats_action == "STATS":
            self.game_state_mode = "STATS"
            pygame.time.delay(200) # Small delay after click
            print("Transitioning to STATS screen.")
        if save_action == "SAVE":
            self.save_current_game_state()
            pygame.time.delay(200) # Small delay after click
            # Set feedback message
            self.feedback_message = "Game Saved!"
            self.feedback_end_time = pygame.time.get_ticks() + 2000 # Display for 2 seconds
            print("Game Saved!")
        if load_action == "LOAD":
            if self.load_game_state():
                self.game_state_mode = "PLAYING"
                pygame.time.delay(200) # Small delay after click
                # Set feedback message
                self.feedback_message = "Game Loaded!"
                self.feedback_end_time = pygame.time.get_ticks() + 2000 # Display for 2 seconds
                print("Game Loaded!")
            else:
                pygame.time.delay(200) # Small delay even if load fails
                # Set feedback message for failure
                self.feedback_message = "Load Failed: No saved game or error!"
                self.feedback_end_time = pygame.time.get_ticks() + 2000
                print("No saved game found or error loading.")
        if quit_action == "QUIT":
            self.game_state_mode = "MAIN_MENU"
            self.state = None # Clear current game state
            pygame.time.delay(200) # Small delay after click
            print("Quitting to Main Menu.")

        # Display feedback message if active
        if pygame.time.get_ticks() < self.feedback_end_time and self.feedback_message:
            feedback_surf = self.medium_font.render(self.feedback_message, True, (255, 255, 0)) # Yellow text
            feedback_rect = feedback_surf.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT - 20))
            self.screen.blit(feedback_surf, feedback_rect)


        pygame.display.flip()

    def display_stats_screen(self, events):
        self.screen.fill(BACKGROUND)
        
        title_text = self.large_font.render("Current Game Statistics", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, 30))
        self.screen.blit(title_text, title_rect)

        if not self.state:
            no_game_text = self.medium_font.render("No active game to display stats.", True, (255, 255, 255))
            no_game_rect = no_game_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 2))
            self.screen.blit(no_game_text, no_game_rect)
        else:
            player1 = self.state.Agents[0]
            player2 = self.state.Agents[1]

            # Player 1 Pie Chart (Left Column)
            p1_pie_center_x = GAME_WIDTH // 4 + 70 # Centered relative to its column
            p1_pie_center_y = GAME_HEIGHT // 2 - 50 # Adjusted for vertical centering
            pie_chart_radius = 100 # Larger radius for more prominence
            self.draw_player_activity_pie_chart(player1, p1_pie_center_x, p1_pie_center_y, pie_chart_radius)

            # Player 2 Pie Chart (Right Column)
            p2_pie_center_x = GAME_WIDTH * 3 // 4 - 70 # Centered relative to its column
            p2_pie_center_y = GAME_HEIGHT // 2 - 50 # Adjusted for vertical centering
            self.draw_player_activity_pie_chart(player2, p2_pie_center_x, p2_pie_center_y, pie_chart_radius)

        # Export Profiles button
        export_action = self.draw_button("Export Profiles", GAME_WIDTH // 2 - 180 // 2, GAME_HEIGHT - 80, 180, 40, (0, 80, 80), (0, 120, 120), events, action="EXPORT_PROFILES")
        if export_action == "EXPORT_PROFILES":
            self.export_player_profiles()
            pygame.time.delay(200)

        # Back button at the bottom center
        back_action = self.draw_button("Back to Pause Menu", GAME_WIDTH // 2 - 180 // 2, GAME_HEIGHT - 40, 180, 40, (70, 70, 70), (100, 100, 100), events, action="BACK")
        if back_action == "BACK":
            self.game_state_mode = "PAUSED"
            pygame.time.delay(200) # Small delay after click
            print("Transitioning back to PAUSED menu from stats.")

        # Display feedback message if active
        if pygame.time.get_ticks() < self.feedback_end_time and self.feedback_message:
            feedback_surf = self.medium_font.render(self.feedback_message, True, (255, 255, 0)) # Yellow text
            feedback_rect = feedback_surf.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 2 + 200)) # Position it below buttons
            self.screen.blit(feedback_surf, feedback_rect)

        pygame.display.flip()

    def display_leaderboard_screen(self, events):
        self.screen.fill(BACKGROUND)

        title_text = self.large_font.render("Leaderboard", True, (255, 255, 0))
        title_rect = title_text.get_rect(center=(GAME_WIDTH // 2, 50))
        self.screen.blit(title_text, title_rect)

        # Sort player profiles by highest_score
        sorted_profiles = sorted(
            [(name, data) for name, data in self.player_profiles.items()],
            key=lambda item: item[1].get("highest_score", 0), # Use .get with default 0 for safety
            reverse=True
        )

        y_offset = 120
        rank = 1
        for name, data in sorted_profiles[:10]: # Display top 10
            highest_score = data.get("highest_score", 0)
            total_score = data.get("total_score", 0)
            games_played = data.get("games_played", 0)
            total_gems = data.get("total_gems_collected", 0)
            total_treasures = data.get("total_treasures_collected", 0)
            total_time = data.get("total_time_played", 0)

            # Display player name and highest score
            name_score_line = f"{rank}. {name}: {highest_score} points (Highest)"
            name_score_surf = self.medium_font.render(name_score_line, True, (255, 255, 255))
            name_score_rect = name_score_surf.get_rect(center=(GAME_WIDTH // 2, y_offset))
            self.screen.blit(name_score_surf, name_score_rect)
            y_offset += name_score_surf.get_height() + 2

            # Display additional stats
            stats_line1 = f"  Total Games: {games_played} | Total Score: {total_score}"
            stats_line2 = f"  Gems: {total_gems} | Treasures: {total_treasures} | Time: {int(total_time)}s"
            
            stats_surf1 = self.font.render(stats_line1, True, (180, 180, 180))
            stats_surf2 = self.font.render(stats_line2, True, (180, 180, 180))

            stats_rect1 = stats_surf1.get_rect(center=(GAME_WIDTH // 2, y_offset))
            stats_rect2 = stats_surf2.get_rect(center=(GAME_WIDTH // 2, y_offset + stats_surf1.get_height() + 2))

            self.screen.blit(stats_surf1, stats_rect1)
            self.screen.blit(stats_surf2, stats_rect2)
            
            y_offset += stats_surf1.get_height() + stats_surf2.get_height() + 15 # Add extra space between players
            rank += 1
        
        if not sorted_profiles:
            no_data_text = self.medium_font.render("No scores recorded yet!", True, (200, 200, 200))
            no_data_rect = no_data_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 2))
            self.screen.blit(no_data_text, no_data_rect)

        # Clear Leaderboard button
        clear_action = self.draw_button("Clear Leaderboard", GAME_WIDTH // 2 - 200 // 2, GAME_HEIGHT - 120, 200, 40, (150, 0, 0), (200, 0, 0), events, action="CLEAR_LEADERBOARD")
        if clear_action == "CLEAR_LEADERBOARD":
            # Show confirmation message box
            if self.draw_message_box("Are you sure you want to clear the leaderboard? This cannot be undone.", "Confirm Clear", events):
                self.clear_leaderboard_profiles()
            pygame.time.delay(200)

        back_action = self.draw_button("Back to Main Menu", GAME_WIDTH // 2 - 180 // 2, GAME_HEIGHT - 70, 180, 40, (70, 70, 70), (100, 100, 100), events, action="BACK")
        if back_action == "BACK":
            self.game_state_mode = "MAIN_MENU"
            pygame.time.delay(200)

        # Display feedback message if active
        if pygame.time.get_ticks() < self.feedback_end_time and self.feedback_message:
            feedback_surf = self.medium_font.render(self.feedback_message, True, (255, 255, 0)) # Yellow text
            feedback_rect = feedback_surf.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT - 20))
            self.screen.blit(feedback_surf, feedback_rect)

        pygame.display.flip()

    def draw_message_box(self, message, title, events):
        """Draws a simple message box with Yes/No buttons and waits for user input."""
        box_width = 400
        box_height = 200
        box_x = (GAME_WIDTH - box_width) // 2
        box_y = (GAME_HEIGHT - box_height) // 2

        # Draw a semi-transparent overlay over the current screen
        overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Draw the message box background
        pygame.draw.rect(self.screen, (50, 50, 100), (box_x, box_y, box_width, box_height), border_radius=15)
        pygame.draw.rect(self.screen, (150, 150, 255), (box_x, box_y, box_width, box_height), 3, border_radius=15)

        # Draw title
        title_surf = self.medium_font.render(title, True, (255, 255, 0))
        title_rect = title_surf.get_rect(center=(GAME_WIDTH // 2, box_y + 30))
        self.screen.blit(title_surf, title_rect)

        # Draw message
        wrapped_message = self.wrap_text(message, box_width - 40, self.font)
        msg_y_offset = box_y + 70
        for line in wrapped_message:
            msg_surf = self.font.render(line, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(center=(GAME_WIDTH // 2, msg_y_offset))
            self.screen.blit(msg_surf, msg_rect)
            msg_y_offset += msg_surf.get_height() + 2

        # Draw Yes/No buttons
        button_width, button_height = 80, 35
        yes_button_x = GAME_WIDTH // 2 - button_width - 20
        no_button_x = GAME_WIDTH // 2 + 20
        button_y = box_y + box_height - 50

        yes_action = self.draw_button("Yes", yes_button_x, button_y, button_width, button_height, (0, 150, 0), (0, 200, 0), events, action="YES")
        no_action = self.draw_button("No", no_button_x, button_y, button_width, button_height, (150, 0, 0), (200, 0, 0), events, action="NO")

        pygame.display.flip() # Update the display to show the message box

        # This function needs to block and wait for a response
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                
                # Re-check button clicks for the message box
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = event.pos
                    yes_rect = pygame.Rect(yes_button_x, button_y, button_width, button_height)
                    no_rect = pygame.Rect(no_button_x, button_y, button_width, button_height)

                    if yes_rect.collidepoint(mouse_pos):
                        return True
                    elif no_rect.collidepoint(mouse_pos):
                        return False
            self.clock.tick(FPS) # Keep the game responsive while waiting for input


    def save_current_game_state(self):
        if not self.state:
            print("No game state to save.")
            return
        
        print("Saving current game state...")
        game_data = {
            "player1_name": self.player1_name,
            "player2_name": self.player2_name,
            "game_mode": self.current_game_mode,
            "time_spent_playing": self.game_metrics["time_spent_playing"],
            "maze": self.state.maze.tolist(), # Convert numpy array to list
            "secret_passages": list(self.state.secret_passages), # Convert set to list
            "goal_locations": [],
            "enemy_collection": [],
            "fake_players": [],
            "gem_collection": [],
            "spike_trap_collection": [],
            "agents": [],
            "player1_is_ai_controlled": self.player1_is_ai_controlled,
            "current_ai_profile_p1": self.current_ai_profile_p1,
            "current_ai_profile_p2": self.current_ai_profile_p2,
            "maze_reshaped": self.state.maze_reshaped,
            "next_reshape_time_offset": self.state.next_reshape_time - self.state.game_start_time # Save offset
        }

        for treasure in self.state.GoalLocations:
            game_data["goal_locations"].append({
                "rect": {"x": treasure["rect"].x, "y": treasure["rect"].y, "width": treasure["rect"].width, "height": treasure["rect"].height},
                "color": treasure["color"],
                "value": treasure["value"],
                "is_main": treasure["is_main"],
                "name": treasure["name"],
                "is_fake": treasure["is_fake"] # Save fake status
            })
        for enemy in self.state.EnemyCollection:
            game_data["enemy_collection"].append({"x": enemy.x, "y": enemy.y, "width": enemy.width, "height": enemy.height})
        for fake in self.state.FakePlayers:
            game_data["fake_players"].append({"x": fake.x, "y": fake.y, "width": fake.width, "height": fake.height})
        for gem in self.state.GemCollection:
            game_data["gem_collection"].append({
                "rect": {"x": gem["rect"].x, "y": gem["rect"].y, "width": gem["rect"].width, "height": gem["rect"].height},
                "color": gem["color"]
            })
        for trap in self.state.SpikeTrapCollection:
            game_data["spike_trap_collection"].append({"x": trap.x, "y": trap.y, "width": trap.width, "height": trap.height})

        for agent in self.state.Agents:
            agent_data = {
                "idx": agent.idx,
                "name": agent.name, # Save agent name
                "player_entity": {"x": agent.PlayerEntity.x, "y": agent.PlayerEntity.y, "width": agent.PlayerEntity.width, "height": agent.PlayerEntity.height},
                "in_stealth": agent.in_stealth,
                "in_ghost": agent.in_ghost,
                "invincible": agent.invincible,
                "boosted": agent.boosted,
                "disarming": agent.disarming,
                "stealth_end_time": agent.stealth_end_time,
                "ghost_end_time": agent.ghost_end_time,
                "invincible_end_time": agent.invincible_end_time,
                "boost_end_time": agent.boost_end_time,
                "disarm_end_time": agent.disarm_end_time,
                "total_moves": agent.total_moves,
                "reward": agent.reward,
                "detected_fakes": agent.detected_fakes,
                "treasures_collected": agent.treasures_collected,
                "gems_collected": agent.gems_collected,
                "has_main_treasure": agent.has_main_treasure,
                "stealth_activations": agent.stealth_activations,
                "ghost_activations": agent.ghost_activations,
                "invincible_activations": agent.invincible_activations,
                "boost_activations": agent.boost_activations,
                "disarm_activations": agent.disarm_activations,
                "powerups_used": agent.powerups_used,
                "last_closest_treasure_dist": agent.last_closest_treasure_dist,
                "stuck_frames": agent.stuck_frames,
                "energy": agent.energy, # Save energy
                "max_energy": agent.max_energy # Save max energy
            }
            game_data["agents"].append(agent_data)
        
        # Save AI models' states if they are AI controllers
        try:
            if isinstance(self.ai_player1_controller, AIController):
                # Save model state dict and epsilon
                torch.save(self.ai_player1_controller.dqn_agent.model.state_dict(), 'player1_ai_model.pth')
                torch.save(self.ai_player1_controller.dqn_agent.target_model.state_dict(), 'player1_ai_target_model.pth')
                with open('player1_ai_epsilon.json', 'w') as f:
                    json.dump({"epsilon": self.ai_player1_controller.dqn_agent.epsilon}, f)
                print("Player 1 AI model saved.")
            if isinstance(self.ai_player2_controller, AIController):
                torch.save(self.ai_player2_controller.dqn_agent.model.state_dict(), 'player2_ai_model.pth')
                torch.save(self.ai_player2_controller.dqn_agent.target_model.state_dict(), 'player2_ai_target_model.pth')
                with open('player2_ai_epsilon.json', 'w') as f:
                    json.dump({"epsilon": self.ai_player2_controller.dqn_agent.epsilon}, f)
                print("Player 2 AI model saved.")
        except Exception as e:
            print(f"Error saving AI models: {e}")


        with open(SAVED_GAME_FILE, "w") as f:
            json.dump(game_data, f, indent=4)
        print(f"Game state saved to {SAVED_GAME_FILE}")

    def load_game_state(self):
        print("Attempting to load game state...")
        if not os.path.exists(SAVED_GAME_FILE):
            print(f"Saved game file '{SAVED_GAME_FILE}' not found.")
            return False
        
        try:
            with open(SAVED_GAME_FILE, "r") as f:
                game_data = json.load(f)

            # Reconstruct GameState
            self.state = GameState(self)
            self.state.maze = np.array(game_data["maze"])
            self.state.secret_passages = set(tuple(sp) for sp in game_data.get("secret_passages", [])) # Load secret passages
            self.state.GoalLocations = []
            for t_data in game_data["goal_locations"]:
                rect_data = t_data.pop("rect")
                self.state.GoalLocations.append({
                    "rect": GameRectangle(rect_data["x"], rect_data["y"], rect_data["width"], rect_data["height"]),
                    **t_data
                })
            self.state.EnemyCollection = [GameRectangle(e["x"], e["y"], e["width"], e["height"]) for e in game_data["enemy_collection"]]
            self.state.FakePlayers = [GameRectangle(f["x"], f["y"], f["width"], f["height"]) for f in game_data["fake_players"]]
            self.state.GemCollection = []
            for g_data in game_data["gem_collection"]:
                rect_data = g_data.pop("rect")
                self.state.GemCollection.append({
                    "rect": GameRectangle(rect_data["x"], rect_data["y"], rect_data["width"], rect_data["height"]),
                    **g_data
                })
            self.state.SpikeTrapCollection = [GameRectangle(s["x"], s["y"], s["width"], s["height"]) for s in game_data["spike_trap_collection"]]

            self.state.Agents = []
            for agent_data in game_data["agents"]:
                agent = AgentState(agent_data["idx"], 0, 0) # Initial dummy coords
                player_entity_data = agent_data.pop("player_entity")
                agent.PlayerEntity = GameRectangle(player_entity_data["x"], player_entity_data["y"], player_entity_data["width"], player_entity_data["height"])
                
                # Restore other agent attributes
                for key, value in agent_data.items():
                    setattr(agent, key, value)
                self.state.Agents.append(agent)

            self.player1_name = game_data.get("player1_name", "Player 1")
            self.player2_name = game_data.get("player2_name", "Player 2")
            self.current_player_profile = self.get_or_create_profile(self.player1_name) # Update P1 profile
            self.game_metrics["time_spent_playing"] = game_data.get("time_spent_playing", 0)
            self.state.game_start_time = time.time() - self.game_metrics["time_spent_playing"] # Adjust start time
            self.player1_is_ai_controlled = game_data.get("player1_is_ai_controlled", False)
            self.current_ai_profile_p1 = game_data.get("current_ai_profile_p1", "EXPLORER")
            self.current_ai_profile_p2 = game_data.get("current_ai_profile_p2", "EXPLORER")
            self.set_game_mode_controllers(game_data.get("game_mode", "AI")) # Re-set controllers based on loaded mode

            self.state.maze_reshaped = game_data.get("maze_reshaped", False)
            self.state.next_reshape_time = self.state.game_start_time + game_data.get("next_reshape_time_offset", GAME_DURATION_SECONDS / 2)


            # Load AI models' states if they were AI controllers
            try:
                # Load to CPU first, then move to device to handle cross-device saving
                if os.path.exists('player1_ai_model.pth'):
                    self.ai_player1_controller.dqn_agent.model.load_state_dict(torch.load('player1_ai_model.pth', map_location=torch.device('cpu')))
                    self.ai_player1_controller.dqn_agent.model.to(self.ai_player1_controller.dqn_agent.device) # Move to correct device
                    self.ai_player1_controller.dqn_agent.target_model.load_state_dict(torch.load('player1_ai_target_model.pth', map_location=torch.device('cpu')))
                    self.ai_player1_controller.dqn_agent.target_model.to(self.ai_player1_controller.dqn_agent.device) # Move to correct device
                    with open('player1_ai_epsilon.json', 'r') as f:
                        self.ai_player1_controller.dqn_agent.epsilon = json.load(f)["epsilon"]
                    print("Player 1 AI model loaded.")
                if os.path.exists('player2_ai_model.pth'):
                    self.ai_player2_controller.dqn_agent.model.load_state_dict(torch.load('player2_ai_model.pth', map_location=torch.device('cpu')))
                    self.ai_player2_controller.dqn_agent.model.to(self.ai_player2_controller.dqn_agent.device)
                    self.ai_player2_controller.dqn_agent.target_model.load_state_dict(torch.load('player2_ai_target_model.pth', map_location=torch.device('cpu')))
                    self.ai_player2_controller.dqn_agent.target_model.to(self.ai_player2_controller.dqn_agent.device)
                    with open('player2_ai_epsilon.json', 'r') as f:
                        self.ai_player2_controller.dqn_agent.epsilon = json.load(f)["epsilon"]
                    print("Player 2 AI model loaded.")
            except Exception as e:
                print(f"Error loading AI models: {e}")

            print("Game state loaded successfully.")
            return True
        except Exception as e:
            print(f"An error occurred while loading game state from {SAVED_GAME_FILE}: {e}")
            return False

    def UpdateFrame(self, events):
        # print("Updating game frame...") # Uncomment for detailed debugging, but can spam console
        # Update time spent playing
        self.game_metrics["time_spent_playing"] = time.time() - self.state.game_start_time

        pressed_keys = pygame.key.get_pressed() # Get state of all keys currently held down

        # Filter for only KEYDOWN events for power-ups and pause
        key_down_events = [event for event in events if event.type == pygame.KEYDOWN]

        # Handle 'Esc' for pause menu
        for event in key_down_events:
            if event.key == pygame.K_ESCAPE:
                self.paused_background_surface = self.screen.copy() # Capture current screen
                self.game_state_mode = "PAUSED"
                print("Game paused.")
                return # Stop updating game logic while paused
            if event.key == pygame.K_t and self.current_game_mode in ["MANUAL", "MIXED"]: # Only allow toggle if player 1 is manual
                if self.controllers[0] == self.manual_player1_controller:
                    self.controllers[0] = self.ai_player1_controller
                    self.player1_is_ai_controlled = True
                    print("Player 1: AI Controlled")
                else:
                    self.controllers[0] = self.manual_player1_controller
                    self.player1_is_ai_controlled = False
                    print("Player 1: Manual Controlled")

        # Dynamic Maze Regeneration Check
        current_time = time.time()
        if not self.state.maze_reshaped and current_time >= self.state.next_reshape_time:
            print("Maze reshaping!")
            self.state.maze, self.state.secret_passages = self.state.generate_maze(self.state.maze_rows, self.state.maze_cols)
            self.state.reposition_entities_after_maze_reshape()
            self.state.maze_reshaped = True # Ensure it only reshapes once

        # Process actions for each agent
        for i, agent in enumerate(self.state.Agents):
            controller_to_use = self.controllers[i]
            
            # If player 1 and toggle is active, use the appropriate controller
            if i == 0 and self.current_game_mode in ["MANUAL", "MIXED"]:
                controller_to_use = self.ai_player1_controller if self.player1_is_ai_controlled else self.manual_player1_controller

            if isinstance(controller_to_use, AIController):
                current_state_before_action = self.state.get_state_representation(i)
                action = controller_to_use.GetAction(self.state, i) 
                
                reward = self.state.update_agent(agent, action)
                agent.total_moves += 1
                next_state = self.state.get_state_representation(i)
                done = self.state.game_over
                
                controller_to_use.record_experience_and_train(
                    current_state_before_action, action, reward, next_state, done
                )
                self.total_steps += 1
            elif isinstance(controller_to_use, ManualController):
                action = controller_to_use.GetAction(pressed_keys, key_down_events)
                if action != "NONE":
                    self.state.update_agent(agent, action)
                    agent.total_moves += 1
            
            # Update analytics metrics
            self.game_metrics[f"powerups_used_p{agent.idx + 1}"] = agent.powerups_used
            self.game_metrics[f"gems_collected_p{agent.idx + 1}"] = agent.gems_collected
            self.game_metrics[f"treasures_collected_p{agent.idx + 1}"] = agent.treasures_collected
            self.game_metrics[f"enemies_disarmed_p{agent.idx + 1}"] = agent.disarm_activations
            self.game_metrics[f"fakes_detected_p{agent.idx + 1}"] = agent.detected_fakes

            # Check achievements for the agent
            self.check_achievements(agent)


        self.state.update_enemy_and_fake()
        
        # Update and clean up particles
        self.state.particles = [p for p in self.state.particles if p.age < p.lifetime]
        for p in self.state.particles:
            p.update()

        # Update and clean up floating texts
        self.floating_texts = [t for t in self.floating_texts if t.age < t.lifetime]
        for t in self.floating_texts:
            t.update()

        self.state.check_game_end()


    def RenderInGameWindow(self):
        # print("Rendering in-game window...") # This might spam the console, use only if needed
        self.screen.fill(BACKGROUND) # Fill with black for fog of war effect

        # Draw the maze first
        draw_maze(self.screen, self.state.maze, self.state.secret_passages)

        # Draw agent footprints
        for agent in self.state.Agents:
            for i, (fx, fy) in enumerate(agent.footprints):
                alpha = int(255 * (i / len(agent.footprints))) # Fade out older footprints
                radius = max(1, int(PLAYER_SIZE / 4 * (i / len(agent.footprints)))) # Shrink older footprints
                pygame.draw.circle(self.screen, (*PLAYER_COLORS[agent.idx], alpha), (int(fx), int(fy)), radius)


        # Create a "fog of war" overlay
        fog_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
        fog_surface.fill((0, 0, 0, 200)) # Semi-transparent black fog

        # Draw transparent circles (light sources) around each player
        for agent in self.state.Agents:
            player_center_x = int(agent.PlayerEntity.x + PLAYER_SIZE // 2)
            player_center_y = int(agent.PlayerEntity.y + PLAYER_SIZE // 2)
            
            # Create a radial gradient effect for light
            light_radius = 150 # How far the light extends
            for r in range(light_radius, 0, -5):
                alpha = int(255 * (r / light_radius) * 0.5) # Fade out towards edge
                pygame.draw.circle(fog_surface, (0, 0, 0, alpha), (player_center_x, player_center_y), r)

        self.screen.blit(fog_surface, (0, 0)) # Apply fog of war

        # Draw game entities (treasures, gems, enemies, fake players, agents)
        for treasure_data in self.state.GoalLocations:
            draw_treasure(self.screen, treasure_data['rect'].x, treasure_data['rect'].y,
                          treasure_data['rect'].width, treasure_data['rect'].height,
                          treasure_data['color'], treasure_data.get('is_main', False), treasure_data.get('is_fake', False))

        for gem_data in self.state.GemCollection:
            draw_gem(self.screen, gem_data['rect'].x, gem_data['rect'].y, 20, gem_data['color'])
        for trap in self.state.SpikeTrapCollection:
            draw_spike_trap(self.screen, trap.x, trap.y, MAZE_CELL_SIZE)
        for enemy in self.state.EnemyCollection:
            draw_goblin(self.screen, enemy.x, enemy.y)
        for fake in self.state.FakePlayers:
            draw_dark_wizard(self.screen, fake.x, fake.y)
        for agent in self.state.Agents:
            draw_ninja(self.screen, agent.PlayerEntity.x, agent.PlayerEntity.y, PLAYER_COLORS[agent.idx],
                       agent.in_stealth, agent.in_ghost, agent.invincible, agent.boosted, agent.disarming,
                       agent.current_animation_frame) # Pass animation frame

        # Draw particles on top of everything else
        for p in self.state.particles:
            p.draw(self.screen)

        # Draw floating texts on top of everything
        for ft in self.floating_texts:
            ft.draw(self.screen)

        agent_1_stats = self.get_agent_stats_text(self.state.Agents[0])
        self.display_text(agent_1_stats, self.font, 10, GAME_HEIGHT - 180, (255, 255, 255)) # White text on dark background
        
        agent_2_stats = self.get_agent_stats_text(self.state.Agents[1])
        # Adjust position for Player 2 stats to ensure it's within bounds
        # Calculate width of the text block to position it correctly from the right edge
        max_width_p2_stats = 0
        for line in self.wrap_text(agent_2_stats, 200, self.font):
            max_width_p2_stats = max(max_width_p2_stats, self.font.size(line)[0])
        
        self.display_text(agent_2_stats, self.font, GAME_WIDTH - max_width_p2_stats - 10, GAME_HEIGHT - 180, (255, 255, 255)) # White text

        # Display time left
        time_left = max(0, GAME_DURATION_SECONDS - int(time.time() - self.state.game_start_time))
        minutes = time_left // 60
        seconds = time_left % 60
        time_text = f"Time Left: {minutes:02d}:{seconds:02d}"
        time_label = self.medium_font.render(time_text, True, (255, 255, 255))
        self.screen.blit(time_label, (GAME_WIDTH // 2 - time_label.get_width() // 2, 10))

        # Display Player 1 control mode
        p1_control_mode = "AI" if self.player1_is_ai_controlled else "Manual"
        control_text = self.small_font.render(f"P1 Control: {p1_control_mode}", True, (200, 200, 200))
        self.screen.blit(control_text, (10, 10))

        # Draw Mini-Map
        self.draw_minimap()

        # Display achievement notifications
        self.display_achievement_notifications()

        # Display energy bars
        self.draw_energy_bars()

        if self.state.game_over:
            # Only draw the game over screen, don't flip display here.
            # The main loop will handle the flip after all drawing is done.
            self.display_game_over_screen()

        pygame.display.flip()

    def draw_minimap(self):
        MINIMAP_WIDTH = GAME_WIDTH // 5
        MINIMAP_HEIGHT = GAME_HEIGHT // 5
        MINIMAP_X = GAME_WIDTH - MINIMAP_WIDTH - 10 # Top-right corner
        MINIMAP_Y = 10
        MINIMAP_CELL_SIZE = MINIMAP_WIDTH / self.state.maze_cols

        # Draw minimap background
        pygame.draw.rect(self.screen, (30, 30, 30, 180), (MINIMAP_X, MINIMAP_Y, MINIMAP_WIDTH, MINIMAP_HEIGHT), border_radius=5)
        pygame.draw.rect(self.screen, (100, 100, 100), (MINIMAP_X, MINIMAP_Y, MINIMAP_WIDTH, MINIMAP_HEIGHT), 2, border_radius=5)

        # Draw maze on minimap
        for r in range(self.state.maze_rows):
            for c in range(self.state.maze_cols):
                cell_x = MINIMAP_X + c * MINIMAP_CELL_SIZE
                cell_y = MINIMAP_Y + r * MINIMAP_CELL_SIZE
                if self.state.maze[r][c] == 1: # Wall
                    color = (50, 50, 50)
                    if (r, c) in self.state.secret_passages: # Highlight secret passages on minimap
                        color = SECRET_PASSAGE_COLOR
                    pygame.draw.rect(self.screen, color, (cell_x, cell_y, MINIMAP_CELL_SIZE, MINIMAP_CELL_SIZE))
                else: # Path
                    pygame.draw.rect(self.screen, (80, 80, 80), (cell_x, cell_y, MINIMAP_CELL_SIZE, MINIMAP_CELL_SIZE))

        # Draw entities on minimap
        scale_x = MINIMAP_WIDTH / GAME_WIDTH
        scale_y = MINIMAP_HEIGHT / GAME_HEIGHT

        # Treasures
        for treasure_data in self.state.GoalLocations:
            mini_x = MINIMAP_X + treasure_data['rect'].x * scale_x
            mini_y = MINIMAP_Y + treasure_data['rect'].y * scale_y
            mini_size = max(2, int(treasure_data['rect'].width * scale_x))
            color = treasure_data['color']
            if treasure_data.get('is_fake', False):
                color = (200, 0, 0) # Red for fake treasures on minimap
            pygame.draw.circle(self.screen, color, (int(mini_x + mini_size/2), int(mini_y + mini_size/2)), mini_size)

        # Gems
        for gem_data in self.state.GemCollection:
            mini_x = MINIMAP_X + gem_data['rect'].x * scale_x
            mini_y = MINIMAP_Y + gem_data['rect'].y * scale_y
            mini_size = max(1, int(gem_data['rect'].width * scale_x))
            pygame.draw.circle(self.screen, gem_data['color'], (int(mini_x + mini_size/2), int(mini_y + mini_size/2)), mini_size)

        # Enemies
        for enemy in self.state.EnemyCollection:
            mini_x = MINIMAP_X + enemy.x * scale_x
            mini_y = MINIMAP_Y + enemy.y * scale_y
            mini_size = max(2, int(enemy.width * scale_x))
            pygame.draw.rect(self.screen, ENEMY_COLOR, (int(mini_x), int(mini_y), mini_size, mini_size))

        # Fake Players
        for fake in self.state.FakePlayers:
            mini_x = MINIMAP_X + fake.x * scale_x
            mini_y = MINIMAP_Y + fake.y * scale_y
            mini_size = max(2, int(fake.width * scale_x))
            pygame.draw.rect(self.screen, FAKE_PLAYER_COLOR, (int(mini_x), int(mini_y), mini_size, mini_size))

        # Players (Agents)
        for agent in self.state.Agents:
            mini_x = MINIMAP_X + agent.PlayerEntity.x * scale_x
            mini_y = MINIMAP_Y + agent.PlayerEntity.y * scale_y
            mini_size = max(3, int(PLAYER_SIZE * scale_x))
            pygame.draw.circle(self.screen, PLAYER_COLORS[agent.idx], (int(mini_x + mini_size/2), int(mini_y + mini_size/2)), mini_size)

    def draw_energy_bars(self):
        bar_width = 100
        bar_height = 15
        padding = 5
        
        # Player 1 Energy Bar
        p1_energy = self.state.Agents[0].energy
        p1_max_energy = self.state.Agents[0].max_energy
        p1_bar_x = 10
        p1_bar_y = 40
        
        pygame.draw.rect(self.screen, (50, 50, 50), (p1_bar_x, p1_bar_y, bar_width, bar_height), border_radius=3) # Background
        fill_width = (p1_energy / p1_max_energy) * bar_width
        pygame.draw.rect(self.screen, (0, 200, 255), (p1_bar_x, p1_bar_y, fill_width, bar_height), border_radius=3) # Fill (Cyan)
        pygame.draw.rect(self.screen, (255, 255, 255), (p1_bar_x, p1_bar_y, bar_width, bar_height), 1, border_radius=3) # Border
        energy_text_p1 = self.small_font.render(f"{self.state.Agents[0].name} Energy: {int(p1_energy)}/{int(p1_max_energy)}", True, (255, 255, 255))
        self.screen.blit(energy_text_p1, (p1_bar_x, p1_bar_y + bar_height + padding))

        # Player 2 Energy Bar
        p2_energy = self.state.Agents[1].energy
        p2_max_energy = self.state.Agents[1].max_energy
        p2_bar_x = GAME_WIDTH - bar_width - 10
        p2_bar_y = 40

        pygame.draw.rect(self.screen, (50, 50, 50), (p2_bar_x, p2_bar_y, bar_width, bar_height), border_radius=3) # Background
        fill_width = (p2_energy / p2_max_energy) * bar_width
        pygame.draw.rect(self.screen, (0, 200, 255), (p2_bar_x, p2_bar_y, fill_width, bar_height), border_radius=3) # Fill (Cyan)
        pygame.draw.rect(self.screen, (255, 255, 255), (p2_bar_x, p2_bar_y, bar_width, bar_height), 1, border_radius=3) # Border
        energy_text_p2 = self.small_font.render(f"{self.state.Agents[1].name} Energy: {int(p2_energy)}/{int(p2_max_energy)}", True, (255, 255, 255))
        # Adjust x position for Player 2 energy text to be right-aligned with the bar
        text_rect_p2 = energy_text_p2.get_rect(topright=(p2_bar_x + bar_width, p2_bar_y + bar_height + padding))
        self.screen.blit(energy_text_p2, text_rect_p2)


    def display_game_over_screen(self):
        overlay = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.state.winner_idx != -1:
            winner_text = f"{self.state.Agents[self.state.winner_idx].name} WINS!"
            winner_color = PLAYER_COLORS[self.state.winner_idx]
        else:
            winner_text = "IT'S A TIE!"
            winner_color = (255, 255, 255)

        game_over_label = self.large_font.render("GAME OVER", True, (255, 0, 0))
        winner_label = self.large_font.render(winner_text, True, winner_color)
        win_reason_label = self.medium_font.render(self.state.win_reason, True, (255, 255, 255))


        self.screen.blit(game_over_label, (GAME_WIDTH // 2 - game_over_label.get_width() // 2, 20))
        self.screen.blit(winner_label, (GAME_WIDTH // 2 - winner_label.get_width() // 2, 70))
        self.screen.blit(win_reason_label, (GAME_WIDTH // 2 - win_reason_label.get_width() // 2, 110))


        player1 = self.state.Agents[0]
        player2 = self.state.Agents[1]

        score1_text = self.medium_font.render(f"{player1.name} Score: {player1.reward}", True, PLAYER_COLORS[0])
        score2_text = self.medium_font.render(f"{player2.name} Score: {player2.reward}", True, PLAYER_COLORS[1])

        self.screen.blit(score1_text, (GAME_WIDTH // 2 - score1_text.get_width() // 2, 160))
        self.screen.blit(score2_text, (GAME_WIDTH // 2 - score2_text.get_width() // 2, 200))
        
        # Draw Player 1 Activity Pie Chart (Left side)
        pie_chart_radius = 70 # Slightly smaller radius
        pie_chart_y_offset = 300 # Adjusted Y for top position
        self.draw_player_activity_pie_chart(player1, GAME_WIDTH // 4, pie_chart_y_offset, pie_chart_radius)

        # Draw Player 2 Activity Pie Chart (Right side)
        self.draw_player_activity_pie_chart(player2, GAME_WIDTH * 3 // 4, pie_chart_y_offset, pie_chart_radius)

        # Line Graph (Bottom-Middle)
        line_graph_width = GAME_WIDTH - 100 # Wider to show more detail
        line_graph_height = 100 # Smaller height to fit
        line_graph_x = (GAME_WIDTH - line_graph_width) // 2 # Center horizontally
        line_graph_y = 550 # Adjusted Y to avoid overlap
        self.draw_player_stats_line_graph(player1, player2, line_graph_x, line_graph_y)

        # Instructions to continue
        continue_text = self.medium_font.render("Press any key to return to Main Menu...", True, (200, 200, 200))
        continue_rect = continue_text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT - 30))
        self.screen.blit(continue_text, continue_rect)


    def get_agent_stats_text(self, agent):
        status_parts = []
        current_time = time.time()
        # Display remaining power-up time if active
        if agent.in_stealth: status_parts.append(f"Stealth ({max(0, round(agent.stealth_end_time - current_time))}s)")
        if agent.in_ghost: status_parts.append(f"Ghost ({max(0, round(agent.ghost_end_time - current_time))}s)")
        if agent.invincible: status_parts.append(f"Invincible ({max(0, round(agent.invincible_end_time - current_time))}s)")
        if agent.boosted: status_parts.append(f"Boosted ({max(0, round(agent.boost_end_time - current_time))}s)")
        if agent.disarming: status_parts.append(f"Disarming ({max(0, round(agent.disarm_end_time - current_time))}s)")
        status_str = ", ".join(status_parts) if status_parts else "Normal"

        return (
            f"{agent.name}\n"
            f"  Moves: {agent.total_moves}\n"
            f"  Treasures: {agent.treasures_collected} | Gems: {agent.gems_collected}\n"
            f"  Reward: {agent.reward}\n"
            f"  Fakes Detected: {agent.detected_fakes}\n"
            f"  Status: {status_str}\n"
            f"  {agent.commentary}"
        )

    def display_text(self, text, font, x, y, color):
        lines = self.wrap_text(text, 200, font)
        y_offset = y
        for line in lines:
            rendered_text = font.render(line, True, color)
            self.screen.blit(rendered_text, (x, y_offset))
            y_offset += font.get_height() + 2

    def wrap_text(self, text, width, font):
        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            if '\n' in word: # Handle explicit newlines
                parts = word.split('\n')
                for i, part in enumerate(parts):
                    if i > 0:
                        lines.append(current_line)
                        current_line = ""
                    test_line = current_line + " " + part if current_line else part
                    text_width, _ = font.size(test_line)
                    if text_width <= width:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = part
                continue # Skip to next word after handling parts

            test_line = current_line + " " + word if current_line else word
            text_width, _ = font.size(test_line)
            if text_width <= width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)
        return [line for line in lines if line.strip()]

    def draw_player_activity_pie_chart(self, player, center_x, center_y, radius):
        # Define the activities and their colors for the pie chart
        activities = [
            ("Gems Collected", player.gems_collected, (0, 255, 255)), # Cyan
            ("Treasures Collected", player.treasures_collected, (255, 215, 0)), # Gold
            ("Fake Players Detected", player.detected_fakes, (200, 0, 200)), # Magenta
            ("Enemies Disarmed", player.disarm_activations, (0, 200, 0)), # Dark Green
            ("Stealth Used", player.stealth_activations, STEALTH_COLOR),
            ("Ghost Used", player.ghost_activations, GHOST_COLOR),
            ("Invincible Used", player.invincible_activations, INVINCIBLE_COLOR),
            ("Boost Used", player.boost_activations, BOOST_COLOR),
            ("Disarm Used", player.disarm_activations, DISARM_COLOR),
            ("Moves Made", player.total_moves, (150, 150, 150)) # Lighter grey for moves, less dominant
        ]

        # Filter out activities with 0 value to avoid drawing tiny, invisible slices
        # and to ensure percentages are calculated only for active contributions.
        active_activities = [(name, value, color) for name, value, color in activities if value > 0]

        total_value = sum(item[1] for item in active_activities)
        
        # Draw title for the pie chart
        title_text = self.medium_font.render(f"{player.name} Activity Breakdown", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(center_x, center_y - radius - 30))
        self.screen.blit(title_text, title_rect)

        # If total_value is 0, draw a single grey slice for "No Activity"
        if total_value == 0:
            pygame.draw.circle(self.screen, (100, 100, 100), (center_x, center_y), radius)
            no_activity_text = self.small_font.render("No Activity", True, (255, 255, 255))
            text_rect = no_activity_text.get_rect(center=(center_x, center_y))
            self.screen.blit(no_activity_text, text_rect)
            
            # Draw legend for "No Activity"
            # Adjust legend_x and legend_y to be relative to the pie chart's position
            legend_x = center_x - radius
            legend_y = center_y + radius + 10
            pygame.draw.rect(self.screen, (100, 100, 100), (legend_x, legend_y, 15, 15))
            legend_label = self.small_font.render("No Activity: 0", True, (255, 255, 255))
            self.screen.blit(legend_label, (legend_x + 20, legend_y))
            return

        start_angle = 0
        for name, value, color in active_activities: # Iterate only over active activities
            percentage = (value / total_value) * 100
            end_angle = start_angle + (value / total_value) * 360

            # Draw the pie slice
            points = [(center_x, center_y)]
            num_segments = 30 # Number of line segments to approximate the arc
            for i in range(num_segments + 1):
                angle = start_angle + (end_angle - start_angle) * (i / num_segments)
                x = center_x + radius * math.cos(math.radians(angle))
                y = center_y + radius * math.sin(math.radians(angle))
                points.append((int(x), int(y)))
            pygame.draw.polygon(self.screen, color, points)

            # Draw percentage text on the slice if slice is large enough
            if percentage > 5: # Only draw percentage if slice is larger than 5% to avoid overlap
                mid_angle = start_angle + (end_angle - start_angle) / 2
                text_x = center_x + (radius / 1.5) * math.cos(math.radians(mid_angle))
                text_y = center_y + (radius / 1.5) * math.sin(math.radians(mid_angle))
                
                percent_text = self.small_font.render(f"{percentage:.1f}%", True, (0, 0, 0)) # Black text for contrast
                percent_rect = percent_text.get_rect(center=(int(text_x), int(text_y)))
                self.screen.blit(percent_text, percent_rect)

            start_angle = end_angle

        # Draw a circle outline for better definition
        pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), radius, 2)

        # Draw legend below the pie chart
        legend_x = center_x - radius # Start legend at the left edge of the circle
        legend_y = center_y + radius + 10 # Below the circle
        for name, value, color in activities:
            # Always show legend for all activities, even if value is 0, for completeness
            pygame.draw.rect(self.screen, color, (legend_x, legend_y, 15, 15))
            legend_label = self.small_font.render(f"{name}: {value}", True, (255, 255, 255))
            self.screen.blit(legend_label, (legend_x + 20, legend_y))
            legend_y += 20 # Move to next line for next legend item


    def draw_player_stats_line_graph(self, player1, player2, graph_x, graph_y):
        # Define the metrics to plot
        stats_map = {
            "Treasures Collected": lambda p: p.treasures_collected,
            "Moves Made": lambda p: p.total_moves,
            "Reward Accuracy": lambda p: p.reward / (p.total_moves + 1e-6) if p.total_moves > 0 else 0, # Add 1e-6 to avoid div by zero
            "Power-ups Used": lambda p: p.stealth_activations + p.ghost_activations + p.invincible_activations + p.boost_activations + p.disarm_activations,
            "Performance Rate": lambda p: (p.reward + p.treasures_collected * 100 + p.gems_collected * 50) / (p.total_moves + 1e-6) if p.total_moves > 0 else 0
        }
        stat_names = list(stats_map.keys())

        graph_width = GAME_WIDTH - 100 # Wider to show more detail
        graph_height = 120 # Smaller height to fit
        padding = 20 # Padding inside the graph area
        
        # Draw graph background and border
        pygame.draw.rect(self.screen, (50, 50, 50, 150), (graph_x - 10, graph_y - 10, graph_width + 20, graph_height + 20), 0)
        pygame.draw.rect(self.screen, (200, 200, 200), (graph_x - 10, graph_y - 10, graph_width + 20, graph_height + 20), 2)

        # Determine max value for scaling Y-axis
        max_value_overall = 0.0
        for stat_name in stat_names:
            max_value_overall = max(max_value_overall, stats_map[stat_name](player1), stats_map[stat_name](player2))
        
        # If all stats are zero, set a small max value to avoid division by zero and allow drawing tiny points
        if max_value_overall == 0:
            max_value_overall = 1.0 # Use float for consistent division

        # Calculate X-coordinates for each stat category
        x_coords = []
        # Ensure there's at least one division for X-coordinates if only one stat exists
        x_step_size = (graph_width - 2 * padding) / (len(stat_names) - 1 if len(stat_names) > 1 else 1)
        for i in range(len(stat_names)):
            x = graph_x + padding + i * x_step_size
            x_coords.append(int(x))

        # Calculate Y-coordinates for each player and draw lines
        player_points = {0: [], 1: []} # List of (x, y) points for each player

        for i, stat_name in enumerate(stat_names):
            p1_val = stats_map[stat_name](player1)
            p2_val = stats_map[stat_name](player2)

            # Scale Y-value
            p1_y = graph_y + graph_height - padding - int((p1_val / max_value_overall) * (graph_height - 2 * padding))
            p2_y = graph_y + graph_height - padding - int((p2_val / max_value_overall) * (graph_height - 2 * padding))
            
            # Clamp Y-coordinates to stay within graph bounds
            p1_y = max(graph_y + padding, min(graph_y + graph_height - padding, p1_y))
            p2_y = max(graph_y + padding, min(graph_y + graph_height - padding, p2_y))


            player_points[0].append((x_coords[i], p1_y))
            player_points[1].append((x_coords[i], p2_y))

            # Draw stat name labels below X-axis
            stat_label = self.font.render(stat_name, True, (255, 255, 255))
            # Adjust position to prevent overlap if labels are too long
            label_x = x_coords[i] - stat_label.get_width() // 2
            # Ensure labels don't go off screen at edges
            label_x = max(graph_x, min(label_x, graph_x + graph_width - stat_label.get_width()))
            self.screen.blit(stat_label, (label_x, graph_y + graph_height + 5))
            
            # Draw actual value labels above points
            # Format values for display, especially for accuracy/rate which might be floats
            p1_value_text = f"{p1_val:.2f}" if "Accuracy" in stat_name or "Rate" in stat_name else str(int(p1_val))
            p2_value_text = f"{p2_val:.2f}" if "Accuracy" in stat_name or "Rate" in stat_name else str(int(p2_val))

            p1_value_label = self.font.render(p1_value_text, True, (255, 255, 255))
            p2_value_label = self.font.render(p2_value_text, True, (255, 255, 255))
            
            # Position value labels
            self.screen.blit(p1_value_label, (x_coords[i] - p1_value_label.get_width() // 2, p1_y - 20))
            self.screen.blit(p2_value_label, (x_coords[i] - p2_value_label.get_width() // 2, p2_y - 20))
            
            # Draw circles at each data point
            pygame.draw.circle(self.screen, PLAYER_COLORS[0], player_points[0][i], 5)
            pygame.draw.circle(self.screen, PLAYER_COLORS[1], player_points[1][i], 5)

        # Draw lines connecting the points for each player
        if len(player_points[0]) > 1:
            pygame.draw.lines(self.screen, PLAYER_COLORS[0], False, player_points[0], 2)
        if len(player_points[1]) > 1:
            pygame.draw.lines(self.screen, PLAYER_COLORS[1], False, player_points[1], 2)

        # Legend for players
        legend_y = graph_y + graph_height + self.font.get_height() + 25
        p1_legend = self.font.render(player1.name, True, PLAYER_COLORS[0])
        p2_legend = self.font.render(player2.name, True, PLAYER_COLORS[1])
        
        # Adjust legend position to be relative to the graph
        legend_x_offset = graph_x + (graph_width // 2) - ((p1_legend.get_width() + p2_legend.get_width() + 20) // 2)
        self.screen.blit(p1_legend, (legend_x_offset, legend_y))
        self.screen.blit(p2_legend, (legend_x_offset + p1_legend.get_width() + 20, legend_y))

        # Draw title for the line graph
        title_text = self.medium_font.render("Player Performance Comparison", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(graph_x + graph_width // 2, graph_y - 30))
        self.screen.blit(title_text, title_rect)


    def Run(self):
        while self.running:
            events = pygame.event.get() # Get all events once per frame

            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
                # Handle key press to exit game over screen
                if self.game_state_mode == "GAME_OVER" and event.type == pygame.KEYDOWN:
                    self.game_state_mode = "MAIN_MENU"
                    self.state = None # Clear current game state
                    # If in AI mode, print current epsilon to show decay
                    if self.current_game_mode == "AI" or self.current_game_mode == "MIXED":
                        for controller in self.controllers:
                            if isinstance(controller, AIController):
                                print(f"Agent Epsilon: {controller.dqn_agent.epsilon:.4f}")
                    pygame.time.delay(200) # Small delay after click
                    print("Transitioning to MAIN_MENU from Game Over screen.")
                    continue # Skip further event processing for this frame

            if self.game_state_mode == "MAIN_MENU":
                self.display_main_menu(events)
            elif self.game_state_mode == "SELECT_MODE":
                self.display_select_mode_menu(events)
            elif self.game_state_mode == "ENTER_NAMES":
                self.display_enter_names_screen(events)
            elif self.game_state_mode == "SELECT_AI_PROFILE":
                self.display_select_ai_profile_screen(events)
            elif self.game_state_mode == "INSTRUCTIONS":
                self.display_instructions(events)
            elif self.game_state_mode == "PAUSED":
                self.display_pause_menu(events)
            elif self.game_state_mode == "STATS":
                self.display_stats_screen(events)
            elif self.game_state_mode == "LEADERBOARD":
                self.display_leaderboard_screen(events)
            elif self.game_state_mode == "PLAYING":
                if self.state.game_over:
                    # Update player profile with end-game stats
                    self.update_player_profile()
                    self.game_state_mode = "GAME_OVER" # Set mode to GAME_OVER to display the screen
                    print("Game over. Transitioning to Game Over screen.")
                    continue # Continue to the next iteration of the while loop to render game over screen
                
                # If the game is not over and in PLAYING mode, proceed with normal frame updates
                self.UpdateFrame(events)
                self.RenderInGameWindow()
                self.clock.tick(FPS)
            elif self.game_state_mode == "GAME_OVER":
                # Only render the game over screen, waiting for user input in the main loop
                self.display_game_over_screen()
                pygame.display.flip() # Ensure the screen is updated
                self.clock.tick(FPS) # Still tick the clock to prevent high CPU usage

        pygame.quit()

# Main Code
# Create a temporary GameState instance to get the initial state size for AI agents
temp_game_state = GameState(None) # Pass None for temporary GameState initialization
INITIAL_STATE_SIZE = len(temp_game_state.get_state_representation(0)) # Get size from first agent's state

# The game object now handles controller initialization based on menu selection
game = AbstractGame()
game.Run()


# In[ ]:




