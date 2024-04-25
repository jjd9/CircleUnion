"""
Code to calculate the exact area covered by N circles.
The core idea is that the area of N intersecting circles can be
decomposed into X polygons and Y arcs. 
The polygons are formed by cycles in the undirected graph formed by the centers of the circles and the intersection points between the circles.
The arcs are the set of arcs (formed by the circle intersections and the associated circle centers) which are not contained (even partially) by other circles.

This is conceptually satisfying, but unless your set of circles is extremely large and you need very high accuacy, approx_circle_union.py is way better.

Author: John D'Angelo
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patch
from math import pi, atan2, fmod, pow, sqrt, cos, sin
import random

import networkx as nx

def signum(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_list(self):
        return (self.x, self.y)

class Circle:
    def __init__(self, center, r):
        self.center = center
        self.r = r
        self.r_squared = r*r
        self.area = pi * self.r_squared

    def area(self):
        """
        returns the area of this circle
        """
        return pi * self.r * self.r
    
    def is_point_inside(self, point):
        """
        Check if this circle contains a point
        """
        center_dist_sq = pow(point.x - self.center.x, 2) + pow(point.y - self.center.y, 2)
        return (center_dist_sq < self.r_squared)

    def contains_arc(self, arc):
        """
        determine if all or part of the arc is inside of this circle
        """
        (pt1, pt2, arc_center, arc_radius) = arc

        center_dx = self.center.x - arc_center.x
        center_dy = self.center.y - arc_center.y
        center_distance = sqrt((center_dx)**2 + (center_dy)**2)
        if (center_distance == 0.0) and (abs(self.r - arc_radius) == 0.0):
            return False

        # the vector connecting the circle centers must be in the arc
        center_vec_theta = angle_c = atan2(center_dy, center_dx)
        angle_a = atan2(pt1.y - arc_center.y,  pt1.x - arc_center.x)
        angle_b = atan2(pt2.y - arc_center.y, pt2.x - arc_center.x)
        a_to_b = (angle_b - angle_a)
        a_to_c = (angle_c - angle_a)

        if a_to_b < 0:
            a_to_b += 2 * pi  # 2 * pi

        if a_to_c < 0:
            a_to_c += 2 * pi  # 2 * pi

        arc_point = Point(arc_center.x + arc_radius * cos(center_vec_theta), arc_center.y + arc_radius * sin(center_vec_theta))

        if signum(a_to_b) == signum(a_to_c) and a_to_b >= a_to_c:
            # if it is, the arc intersects the circle if the point on the arc along that vector is in the circle
            return self.is_point_inside(arc_point)
        else:
            return False


    def contains(self, other_circle):
        """
        Check if this circle contains another circle
        """
        dist = sqrt(pow(other_circle.center.x - self.center.x, 2) + pow(other_circle.center.y - self.center.y, 2))
        if dist + other_circle.r < self.r:
            return True
        else:
            return False

def sort_vertices_ccw(vertices):
    """
    Sort the vertices of a 2D polygon in counterclockwise (CCW) order.
    
    Parameters:
    vertices (list of tuples): Unsorted list of (x, y) coordinates of vertices.
    
    Returns:
    list of tuples: Vertices sorted in CCW order.
    """
    
    # Calculate centroid
    centroid = [sum(x) / len(x) for x in zip(*vertices)]
    
    # Sort vertices based on polar angle from centroid
    vertices.sort(key=lambda vertex: (atan2(vertex[1] - centroid[1], vertex[0] - centroid[0]), vertex[0], vertex[1]))
    
    return vertices

def polygon_area(pts):
    """
    Calculate the area of a 2D polygon given its vertices using the Shoelace formula.
    
    Parameters:
    vertices (list of tuples): Unsorted list of (x, y) coordinates of vertices.
    
    Returns:
    float: Area of the polygon.
    """
    
    # Check if the polygon has at least 3 vertices
    if len(pts) < 3:
        return 0.0
    
    list_pts = []
    for pt in pts:
        list_pts.append([pt.x, pt.y])

    # Sort vertices in CCW order
    vertices = sort_vertices_ccw(list_pts)
    
    # Add the first vertex at the end to complete the cycle
    vertices = vertices + [vertices[0]]
    
    # Calculate the area using the Shoelace formula
    area = 0.0
    for i in range(len(vertices) - 1):
        x1, y1 = vertices[i]
        x2, y2 = vertices[i + 1]
        area += (x1 * y2) - (x2 * y1)
    
    # Take the absolute value and divide by 2
    area = abs(area) / 2.0
    
    return area

def normalize_angle_positive(angle):
    """
    Normalizes angle between 0 and 2pi
    """
    result = fmod(angle, 2.0*pi)
    if result < 0:
        return result + 2.0*pi
    return result;    

def arc_area(point1, point2, center, radius):
    """
    Calculate the area of an arc or circle sector defined by its start/end point, center, and radius
    """

    th1 = atan2(point1.y - center.y, point1.x - center.x)
    th2 = atan2(point2.y - center.y, point2.x - center.x)

    return pi * (radius * radius) * normalize_angle_positive(th2-th1) / (2*pi)

def circle_intersection(circle0, circle1):
    """
    Return the intersection points between two circles.
    if there is no intersection, an empty list is returned
    """

    x0 = circle0.center.x
    y0 = circle0.center.y
    r0 = circle0.r
    x1 = circle1.center.x
    y1 = circle1.center.y
    r1 = circle1.r

    d=sqrt((x1-x0)**2 + (y1-y0)**2)
    if d + r0 < r1:
        # circle 1 contains circle 0
        return []
    elif d + r1 < r0:
        # circle 0 contains circle 1
        return []
    elif d < 0.001:
        # circles are on top of each other
        return []
    
    a=(r0**2-r1**2+d**2)/(2*d)
    h_sq = r0**2-a**2
    if h_sq < 0:
        return []
    h=sqrt(h_sq)
    x2=x0+a*(x1-x0)/d   
    y2=y0+a*(y1-y0)/d   
    # intersection 1
    x3_a=x2+h*(y1-y0)/d
    y3_a=y2-h*(x1-x0)/d
    # intersection 2
    x3_b=x2-h*(y1-y0)/d
    y3_b=y2+h*(x1-x0)/d
    return [Point(x3_a, y3_a), Point(x3_b, y3_b)]

class Graph:
    """
    This object stores the circle centers and circle intersections points 
    as an undirected graph.    
    """


    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, circle, intersection_point):
        if circle in self.adjacency_list:
            self.adjacency_list[circle].append(intersection_point)
        else:
            self.adjacency_list[circle] = [intersection_point]

        if intersection_point in self.adjacency_list:
            self.adjacency_list[intersection_point].append(circle)
        else:
            self.adjacency_list[intersection_point] = [circle]

    def find_polygons(self):
        """
        find polygons in the circle intersection graph. polygons are cycles in the graph 
        starting at a circle node.
        polygons are lists of points/vertices
        """
        polygons = []

        g = nx.Graph()
        g.add_nodes_from(self.adjacency_list.keys())

        for k, v in self.adjacency_list.items():
            g.add_edges_from(([(k, t) for t in v]))

        # find all the cycles in the graph and return them as sequences of points
        cycles = nx.simple_cycles(g)
        for c in cycles:
            poly = []
            for element in c:
                if isinstance(element, Circle):
                    poly.append(element.center)
                else:
                    poly.append(element)
            polygons.append(poly)

        return polygons

    def find_arcs(self):
        """
        find arcs in the circle intersection graph. an arc exists between each pair of neighbors
        of a circle node. 
        arcs are tuples of (point1, point2, center, radius)
        """

        arcs = []
        for node in self.adjacency_list:
            if isinstance(node, Circle):
                neighbors = self.adjacency_list[node]
                for i in range(len(neighbors)):
                    neighbor1 = neighbors[i]
                    if isinstance(neighbor1, Circle):
                        pt1 = neighbor1.center
                    else:
                        pt1 = neighbor1
                    for j in range(i+1, len(neighbors)):
                        neighbor2 = neighbors[j]
                        if isinstance(neighbor2, Circle):
                            pt2 = neighbor2.center
                        else:
                            pt2 = neighbor2
                        # we dont know yet which direction the arc should go
                        arcs.append([pt1, pt2, node.center, node.r])            
                        arcs.append([pt2, pt1, node.center, node.r])            

        return arcs

def generate_random_circles(n):
    """
    Return a list of n random circles.
    """
    circles = []
    for _ in range(n):
        center = Point(random.uniform(-1, 1)*5, random.uniform(-1, 1)*5)
        radius = random.uniform(0.1, 5)
        circle = Circle(center, radius)
        circles.append(circle)
    return circles

def generate_circle_of_circles(num_circles, radius, radius2):
    import numpy as np

    # Calculate positions for smaller circles
    theta = np.linspace(0, 2 * np.pi, num_circles, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Plot smaller circles
    circles = []
    for i in range(num_circles):
        circles.append(Circle(Point(x[i], y[i]), radius2))
    return circles

if __name__ == "__main__":
    # center0 = Point(0,0)
    # center1 = Point(0.3, 0.4)
    # center2 = Point(0.3, 1.6)
    # center3 = Point(0.7, -0.8)
    # circle0 = Circle(center0, 1)
    # circle1 = Circle(center1, 1)
    # circle2 = Circle(center2, 0.5)
    # circle3 = Circle(center3, 0.5)

    # circles = [circle0, circle1, circle2, circle3]

    # Example: generate 4 random circles
    # n = 10
    # circles = generate_random_circles(n)
    circles = generate_circle_of_circles(10, 0.2, 0.3)

    num_circles = len(circles)

    total_area = 0.0

    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    graph = Graph()

    # if a circle is fully contained by another circle, it should be ignored
    ignore_circle = [False for _ in range(num_circles)]
    for i in range(num_circles):
        for j in range(num_circles):
            if i != j:
                if circles[i].contains(circles[j]):
                    ignore_circle[j] = True
    
    # for each circle, add its intersection points with other circles to the graph
    print("Build graph")
    has_no_intersections = [True for _ in range(num_circles)]
    for i in range(num_circles):
        if ignore_circle[i]:
            has_no_intersections[i] = False
            continue
        for j in range(i+1, num_circles):
            if ignore_circle[j]:
                has_no_intersections[j] = False
                continue

            if i == j:
                continue

            # get the intersection points
            intersection_pts = circle_intersection(circles[i], circles[j])
            for pt in intersection_pts:
                # check that they arent in any of the circles
                pt_in_circle = False
                for k in range(num_circles):
                    if k == i or k == j:
                        continue
                    if circles[k].is_point_inside(pt):
                        pt_in_circle = True
                        break
                if not pt_in_circle:
                    has_no_intersections[i] = False
                    has_no_intersections[j] = False
                    graph.add_edge(circles[i], pt)
                    graph.add_edge(circles[j], pt)

    # account for isolated circles (not inside another circle and not intersecting with another circle)
    for i in range(num_circles):
        if has_no_intersections[i]:
            total_area += circles[i].area

    # find the polygons
    print("Find polygons")
    polygons = graph.find_polygons()
    print(f"Num polygons: {len(polygons)}")
    for poly in polygons:
        poly_area = polygon_area(poly)
        total_area += poly_area
    print(f"Total polygon area: {total_area}")

    # find all the arcs
    print("Find arcs")
    all_arcs = graph.find_arcs()
    print(f"Number of arcs before filtering: {len(all_arcs)}")
    # filter the arcs to remove those that are (at least partially) inside of other circles
    arcs = []
    for c_arc in all_arcs:
        is_free_arc = True
        for circle in circles:
            if circle.contains_arc(c_arc):
                is_free_arc = False
                break
        if is_free_arc:
            arcs.append(c_arc)

    total_arc_area = 0.0
    for arc in arcs:
        temp_arc_area = arc_area(*arc)
        total_arc_area = temp_arc_area
        total_area += temp_arc_area

    print(f"Total arc area: {total_arc_area}")
    print(f"Num arcs after filtering: {len(arcs)}")

    print(f"Total area: {total_area}")

    # Visualize the graph edges
    for node in graph.adjacency_list:
        if isinstance(node, Circle):
            node_pt = node.center
        else:
            node_pt = node

        for neighbor in graph.adjacency_list[node]:
            if isinstance(neighbor, Circle):
                neighbor_pt = neighbor.center
            else:
                neighbor_pt = neighbor
            ax.plot([node_pt.x, neighbor_pt.x],[node_pt.y, neighbor_pt.y],'b-')

    # Visualize the circles
    for circle in circles:
        c0 = plt.Circle(circle.center.to_list(), circle.r, color='r', fill=False)
        ax.plot([circle.center.x],[circle.center.y],'ro')
        ax.add_patch(c0)

    # Visualize the arcs
    for j,arc in enumerate(arcs):
        (pt1, pt2, center, r) = arc
        theta1 = atan2(pt1.y - center.y, pt1.x - center.x) * 180 / pi
        theta2 = atan2(pt2.y - center.y, pt2.x - center.x) * 180 / pi
        ax.add_patch(patch.Arc([center.x, center.y], 2*r, 2*r, 0, theta1, theta2))

    for poly in polygons:
        x = [pt.x for pt in poly]
        y = [pt.y for pt in poly]
        x.append(x[0])
        y.append(y[0])

    ax.set_aspect("equal")
    ax.grid()
    plt.show()