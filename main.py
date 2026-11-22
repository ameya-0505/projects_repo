import heapq
import math
import sys
import time
import networkx as nx
import numpy as np
import pygame

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
start_x, start_y = 100.0, 100.0
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Topological Memory vs. Grid A* Benchmark")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 102, 204)
RED = (204, 0, 0)
GREEN = (0, 170, 0)
ORANGE = (235, 130, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (70, 70, 70)

# Obstacles in the arena (pygame.Rect)
obstacles = [
    pygame.Rect(260, 140, 60, 260),
    pygame.Rect(480, 200, 60, 280),
]


def collides_with_obstacles(x, y, radius=0.0):
  probe = pygame.Rect(
      int(x - radius),
      int(y - radius),
      int(max(1, 2 * radius)),
      int(max(1, 2 * radius)),
  )
  return any(rect.colliderect(probe) for rect in obstacles)


def segment_intersects_obstacles(p1, p2, steps=12):
  x1, y1 = p1
  x2, y2 = p2
  for i in range(steps + 1):
    t = i / steps
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    if collides_with_obstacles(px, py, radius=2.0):
      return True
  return False


class Robot:
  def __init__(self, x, y):
    self.x = float(x)
    self.y = float(y)
    self.speed = 3.0
    self.radius = 12

  def move(self, dx, dy):
    new_x = max(self.radius, min(WIDTH - self.radius, self.x + dx * self.speed))
    new_y = max(self.radius, min(HEIGHT - self.radius, self.y + dy * self.speed))

    # Move along each axis independently so sliding along walls feels smooth
    if not collides_with_obstacles(new_x, self.y, self.radius):
      self.x = new_x
    if not collides_with_obstacles(self.x, new_y, self.radius):
      self.y = new_y

  def draw(self, surface):
    pygame.draw.circle(surface, BLUE, (int(self.x), int(self.y)), self.radius)


class TopologicalMemory:
  def __init__(self, start_pos, step_threshold=45.0, connect_radius=80.0):
    self.graph = nx.Graph()
    self.step_threshold = float(step_threshold)
    self.connect_radius = float(connect_radius)
    self.current_node = 0
    self.graph.add_node(self.current_node, pos=(float(start_pos[0]), float(start_pos[1])))
    self.last_query_ms = None

  def update(self, x, y):
    current_pos = np.array([float(x), float(y)])
    last_pos = np.array(self.graph.nodes[self.current_node]["pos"], dtype=float)
    distance_traveled = float(np.linalg.norm(current_pos - last_pos))

    if distance_traveled >= self.step_threshold:
      new_node = self.graph.number_of_nodes()
      new_pos = (float(x), float(y))
      self.graph.add_node(new_node, pos=new_pos)

      # Primary odometry edge
      self.graph.add_edge(self.current_node, new_node, weight=distance_traveled)

      # Loop-closure shortcut edges to nearby earlier nodes if line-of-sight is clear
      for node_id, data in self.graph.nodes(data=True):
        if node_id == new_node:
          continue
        other_pos = tuple(data["pos"])
        dist = float(np.linalg.norm(current_pos - np.array(other_pos, dtype=float)))
        if (
            dist <= self.connect_radius
            and not self.graph.has_edge(new_node, node_id)
            and not segment_intersects_obstacles(new_pos, other_pos)
        ):
          self.graph.add_edge(new_node, node_id, weight=dist)

      self.current_node = new_node
      return True

    return False

  def find_shortest_path(self, source_node, target_node):
    if source_node not in self.graph or target_node not in self.graph:
      return [], 0.0, 0.0

    t0 = time.perf_counter()
    try:
      path = nx.shortest_path(self.graph, source=source_node, target=target_node, weight="weight")
      cost = nx.path_weight(self.graph, path, weight="weight") if len(path) > 1 else 0.0
    except nx.NetworkXNoPath:
      path, cost = [], 0.0
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    self.last_query_ms = elapsed_ms
    return path, cost, elapsed_ms

  def log_status(self, prefix="STEP"):
    v_count = self.graph.number_of_nodes()
    e_count = self.graph.number_of_edges()
    q_str = f"{self.last_query_ms:.4f} ms" if self.last_query_ms is not None else "N/A"
    print(f"[{prefix}] Topo V={v_count}, E={e_count}, Last Query={q_str}")


class GridAStar:
  def __init__(self, width, height, cell_size=20, obstacle_rects=None, clearance=0):
    self.width = width
    self.height = height
    self.cell_size = cell_size
    self.cols = width // cell_size
    self.rows = height // cell_size
    self.total_cells = self.rows * self.cols
    self.last_query_ms = None

    # 0 = free, 1 = blocked
    self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
    for rect in obstacle_rects or []:
      inflated = rect.inflate(clearance * 2, clearance * 2)
      c_min = max(0, inflated.left // cell_size)
      c_max = min(self.cols - 1, inflated.right // cell_size)
      r_min = max(0, inflated.top // cell_size)
      r_max = min(self.rows - 1, inflated.bottom // cell_size)
      self.grid[r_min : r_max + 1, c_min : c_max + 1] = 1

  def world_to_cell(self, x, y):
    col = max(0, min(self.cols - 1, int(x // self.cell_size)))
    row = max(0, min(self.rows - 1, int(y // self.cell_size)))
    return (row, col)

  def cell_to_world(self, row, col):
    cx = (col + 0.5) * self.cell_size
    cy = (row + 0.5) * self.cell_size
    return (cx, cy)

  def _nearest_free_cell(self, cell):
    if self.grid[cell] == 0:
      return cell
    r0, c0 = cell
    best_cell = None
    best_dist = float("inf")
    for r in range(self.rows):
      for c in range(self.cols):
        if self.grid[r, c] == 0:
          d = math.hypot(r - r0, c - c0)
          if d < best_dist:
            best_dist = d
            best_cell = (r, c)
    return best_cell

  def find_path(self, start_xy, goal_xy):
    t0 = time.perf_counter()
    start = self._nearest_free_cell(self.world_to_cell(start_xy[0], start_xy[1]))
    goal = self._nearest_free_cell(self.world_to_cell(goal_xy[0], goal_xy[1]))

    if start is None or goal is None:
      self.last_query_ms = (time.perf_counter() - t0) * 1000.0
      return [], 0.0, self.last_query_ms

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    open_heap = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
      _, current = heapq.heappop(open_heap)
      if current in closed:
        continue
      closed.add(current)

      if current == goal:
        cells = []
        curr = current
        while curr in came_from:
          cells.append(curr)
          curr = came_from[curr]
        cells.append(start)
        cells.reverse()

        world_pts = [self.cell_to_world(r, c) for r, c in cells]
        pixel_cost = sum(
            math.hypot(world_pts[i + 1][0] - world_pts[i][0], world_pts[i + 1][1] - world_pts[i][1])
            for i in range(len(world_pts) - 1)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_query_ms = elapsed_ms
        return world_pts, pixel_cost, elapsed_ms

      r, c = current
      for dr, dc, step_cost in neighbors:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
          continue
        if self.grid[nr, nc] == 1 or (nr, nc) in closed:
          continue
        # Prevent corner-cutting diagonally across blocked cells
        if dr != 0 and dc != 0:
          if self.grid[r + dr, c] == 1 or self.grid[r, c + dc] == 1:
            continue

        tentative_g = g_score[current] + step_cost
        neighbor = (nr, nc)
        if tentative_g < g_score.get(neighbor, float("inf")):
          came_from[neighbor] = current
          g_score[neighbor] = tentative_g
          h = math.hypot(goal[0] - nr, goal[1] - nc)
          heapq.heappush(open_heap, (tentative_g + h, neighbor))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    self.last_query_ms = elapsed_ms
    return [], 0.0, elapsed_ms


# Instantiate Robot, Topological Memory, and Grid A* Baseline
robot = Robot(start_x, start_y)
memory = TopologicalMemory((robot.x, robot.y), step_threshold=45.0, connect_radius=80.0)
astar = GridAStar(WIDTH, HEIGHT, cell_size=20, obstacle_rects=obstacles, clearance=robot.radius)

topo_path_nodes = []
astar_path_pts = []

# --- MAIN SIMULATION LOOP ---
running = True
while running:
  screen.fill(WHITE)

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
      topo_path_nodes, topo_cost, topo_ms = memory.find_shortest_path(0, memory.current_node)
      astar_path_pts, astar_cost, astar_ms = astar.find_path(
          (start_x, start_y),
          memory.graph.nodes[memory.current_node]["pos"],
      )
      print(
          f"[BENCHMARK] Topo: cost={topo_cost:.1f} px, time={topo_ms:.4f} ms "
          f"(V={memory.graph.number_of_nodes()}, E={memory.graph.number_of_edges()}) | "
          f"Grid A*: cost={astar_cost:.1f} px, time={astar_ms:.4f} ms "
          f"(cells={astar.total_cells})"
      )

  keys = pygame.key.get_pressed()
  dx, dy = 0, 0
  if keys[pygame.K_LEFT]:
    dx -= 1
  if keys[pygame.K_RIGHT]:
    dx += 1
  if keys[pygame.K_UP]:
    dy -= 1
  if keys[pygame.K_DOWN]:
    dy += 1

  if dx != 0 or dy != 0:
    length = math.hypot(dx, dy)
    robot.move(dx / length, dy / length)
    if memory.update(robot.x, robot.y):
      memory.log_status()

  # Draw obstacles
  for rect in obstacles:
    pygame.draw.rect(screen, DARK_GRAY, rect)

  # Draw topological edges
  for u, v in memory.graph.edges():
    pygame.draw.line(screen, GRAY, memory.graph.nodes[u]["pos"], memory.graph.nodes[v]["pos"], 2)

  # Draw Grid A* path in orange
  if len(astar_path_pts) > 1:
    for i in range(len(astar_path_pts) - 1):
      pygame.draw.line(screen, ORANGE, astar_path_pts[i], astar_path_pts[i + 1], 3)

  # Draw Topological shortest path in green
  if len(topo_path_nodes) > 1:
    for i in range(len(topo_path_nodes) - 1):
      p1 = memory.graph.nodes[topo_path_nodes[i]]["pos"]
      p2 = memory.graph.nodes[topo_path_nodes[i + 1]]["pos"]
      pygame.draw.line(screen, GREEN, p1, p2, 4)

  # Draw place-cell nodes
  for node in memory.graph.nodes():
    pos = memory.graph.nodes[node]["pos"]
    pygame.draw.circle(screen, RED, (int(pos[0]), int(pos[1])), 5)

  robot.draw(screen)
  pygame.display.flip()
  clock.tick(60)

pygame.quit()
sys.exit()