#!/usr/bin/env python3
import rospy
import heapq
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

class AStarPlanner:
    def __init__(self):
        rospy.init_node('a_star_planner', anonymous=True)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
        
        # Grid and scene parameters
        self.width, self.height = 50, 50
        self.resolution = 1.0
        self.start = (5, 5)
        self.goal = (45, 45)
        self.obstacles = set([
            (i, 20) for i in range(10, 40)
        ] + [
            (30, i) for i in range(10, 40)
        ] + [
            (i, 40) for i in range(15, 25)
        ])

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self):
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = { (x,y): float('inf') for x in range(self.width) for y in range(self.height) }
        g_score[self.start] = 0
        f_score = { (x,y): float('inf') for x in range(self.width) for y in range(self.height) }
        f_score[self.start] = self.heuristic(self.start, self.goal)

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == self.goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]: # 4-way connectivity
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height):
                    continue
                if neighbor in self.obstacles:
                    continue
                
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None # No path found

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def publish_markers(self, path):
        marker_array = MarkerArray()

        # Grid, Obstacles, Start, Goal
        for x in range(self.width):
            for y in range(self.height):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "grid"
                marker.id = x * self.width + y
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = x * self.resolution
                marker.pose.position.y = y * self.resolution
                marker.pose.position.z = 0
                marker.scale.x = self.resolution
                marker.scale.y = self.resolution
                marker.scale.z = 0.1
                marker.color.a = 0.5
                if (x, y) in self.obstacles:
                    marker.color.r = 1.0 # Red for obstacles
                elif (x, y) == self.start:
                    marker.color.g = 1.0 # Green for start
                elif (x, y) == self.goal:
                    marker.color.b = 1.0 # Blue for goal
                else:
                    marker.color.r = 0.8
                    marker.color.g = 0.8
                    marker.color.b = 0.8 # Gray for grid
                marker_array.markers.append(marker)
        
        # Path
        if path:
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.header.stamp = rospy.Time.now()
            path_marker.ns = "path"
            path_marker.id = self.width * self.height
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.5
            path_marker.color.a = 1.0
            path_marker.color.g = 1.0
            path_marker.color.b = 1.0
            for node in path:
                p = Point()
                p.x = node[0] * self.resolution
                p.y = node[1] * self.resolution
                p.z = 0.2
                path_marker.points.append(p)
            marker_array.markers.append(path_marker)

        self.marker_pub.publish(marker_array)

    def run(self):
        rospy.sleep(1.0)
        path = self.plan()
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.publish_markers(path)
            rate.sleep()

if __name__ == '__main__':
    try:
        planner = AStarPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
